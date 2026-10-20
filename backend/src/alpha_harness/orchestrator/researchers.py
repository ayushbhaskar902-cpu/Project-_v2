"""Multi-LLM Researcher Agent orchestration layer.

Coordinates heterogeneous LLM quantitative researcher personas (e.g. Momentum,
Mean Reversion, Value, Volatility specialists), dispatches generation tasks in parallel
up to the round capacity (80 simulations across 8 BRAIN slots with 10 multi-sim batch items),
injects versioned PromptStrategy context, and persists candidate lineage to SQLite.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import timedelta
import re
import time
from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import func, select, update

from ..brain.schemas import SimulationRequest, SimulationSettings, SimulationType
from ..db.models import (
    CandidateStatus,
    GenerationAttempt,
    GenerationAttemptStatus,
    PromptStrategy,
    PromptVersion,
    ResearchCandidate,
    ResearchResearcher,
    ResearchSession,
    ResearchStatus,
    SimStatus,
    SimulationRecord,
    utcnow,
)
from ..discovery.catalog import (
    DatasetAllocationPolicy,
    DatasetEligibilityError,
    DatasetNotFoundError,
    DatasetRegistry,
    DatasetSelectionContext,
)
from ..generator.pipeline import (
    AlphaGenerationPipeline,
    GeneratedCandidate,
    GenerationBatch,
    validate_alpha_language,
)
from ..labs.fastexpr import extract_candidate_fingerprint, operator_names, parse
from .diversity import CandidateItem, DiversityConfig, DiversitySelector
from .simulation_settings import (
    CandidateSimulationSettings,
    PlatformSchemaValidator,
    SimulationSettingsResolutionError,
    SimulationSettingsValidationError,
    compute_candidate_provenance_hash,
    resolve_candidate_simulation_settings,
    validate_candidate_dataset_preflight,
)
from ..learning.memory import ResearchMemory
from ..prompts.library import StrategyCategory, normalize_category
from ..prompts.versioning import (
    PromptIntegrityError,
    PromptResolutionError,
    compute_prompt_content_hash,
    normalize_prompt_components,
    verify_prompt_integrity,
)

if TYPE_CHECKING:
    from ..db.sqlite import Database
    from ..engine.slots import BatchEngine
    from ..llm.service import LLMService

log = structlog.get_logger(__name__)

#: WorldQuant BRAIN core slots and multi-sim packing limits
SLOT_CAP: int = 8
ITEMS_PER_CORE: int = 10
ROUND_CAPACITY: int = SLOT_CAP * ITEMS_PER_CORE  # 80 candidates per round
MAX_RESEARCHER_RETRIES: int = 2


@dataclass(slots=True)
class ResearcherDispatchOutcome:
    """Result of one logical researcher dispatch, including an explicit failure outcome."""

    attempt_id: int
    researcher_id: int
    model_ref: str
    strategy_id: int | None
    prompt_version_id: int | None
    resolution_tier: str | None
    temperature: float
    selection_context: DatasetSelectionContext | None
    batch: GenerationBatch | None
    status: str
    retry_count: int
    latency_seconds: float | None
    error_class: str | None = None
    error_message: str | None = None
    validation_outcome: str | None = None


class GenerationRoundResult(list[int]):
    """Backward-compatible list of valid candidate IDs with additive round accounting."""

    def __init__(self, candidate_ids: list[int], summary: dict[str, Any]) -> None:
        super().__init__(candidate_ids)
        self.summary = summary


def _safe_error(exc: BaseException, limit: int = 500) -> str:
    """Return a bounded provider error without leaking credentials or raw payloads."""
    message = " ".join(str(exc).split())
    message = re.sub(
        r"(?i)(api[_ -]?key|authorization|bearer|token|secret)\s*[:=]\s*\S+",
        r"\1=[redacted]",
        message,
    )
    return message[:limit] or exc.__class__.__name__


def _is_retryable(exc: BaseException) -> bool:
    """Retry only errors that explicitly advertise transient, safe retry semantics."""
    declared = getattr(exc, "retryable", None)
    if declared is not None:
        return bool(declared)
    status = getattr(exc, "status", None)
    return status in {429, 500, 502, 503, 504} or isinstance(
        exc, (TimeoutError, ConnectionError, asyncio.TimeoutError)
    )


@dataclass(slots=True)
class ResearcherConfig:
    """Configuration for one LLM researcher persona."""

    model_ref: str
    label: str
    strategy_focus: StrategyCategory = StrategyCategory.GENERAL
    temperature: float = 0.7
    system_prompt: str | None = None
    target_dataset_id: str | None = None
    dataset_allocation_policy: str | None = None
    description: str | None = None
    enabled: bool = True


DEFAULT_RESEARCHER_ROSTER: list[ResearcherConfig] = [
    ResearcherConfig(
        model_ref="google:gemini-2.5-pro",
        label="Gemini Trend & Momentum",
        strategy_focus=StrategyCategory.MOMENTUM,
        temperature=0.7,
        description="Specialist in multi-horizon time-series trends, moving average breakouts, and cross-sectional momentum.",
    ),
    ResearcherConfig(
        model_ref="google:gemini-2.5-flash",
        label="Gemini Fast Reversion",
        strategy_focus=StrategyCategory.MEAN_REVERSION,
        temperature=0.6,
        description="Specialist in short-term mean reversion, RSI oscillators, and liquidity shocks.",
    ),
    ResearcherConfig(
        model_ref="openai:gpt-4o",
        label="GPT Fundamental Value",
        strategy_focus=StrategyCategory.VALUE,
        temperature=0.7,
        description="Specialist in balance sheet quality, earnings yield, cash flow ratios, and fundamental anomalies.",
    ),
    ResearcherConfig(
        model_ref="deepseek:deepseek-chat",
        label="DeepSeek Multi-Factor",
        strategy_focus=StrategyCategory.GENERAL,
        temperature=0.8,
        description="Specialist in composite multi-factor models blending volume, volatility, and price interaction.",
    ),
]


class MultiLLMOrchestrator:
    """Orchestrates generation rounds across configured LLM researchers."""

    def __init__(
        self,
        db: Database,
        pipeline: AlphaGenerationPipeline,
        memory: ResearchMemory | None = None,
        settings_schema: dict[str, Any] | None = None,
    ) -> None:
        self.db = db
        self.pipeline = pipeline
        self.memory = memory or ResearchMemory(db)
        self.settings_schema = settings_schema

    async def initialize_session_researchers(
        self,
        session_id: int,
        custom_roster: list[ResearcherConfig] | None = None,
    ) -> list[int]:
        """Creates Researcher agent records in SQLite for a new research session and seeds strategies."""
        roster = custom_roster or DEFAULT_RESEARCHER_ROSTER
        researcher_ids: list[int] = []

        async with self.db.session() as session:
            # 1. Create researcher agent personas
            for cfg in roster:
                focus_val = normalize_category(cfg.strategy_focus)
                researcher = ResearchResearcher(
                    session_id=session_id,
                    model_ref=cfg.model_ref,
                    label=cfg.label,
                    strategy_focus=focus_val,
                    system_prompt=cfg.system_prompt,
                    temperature=cfg.temperature,
                    target_dataset_id=cfg.target_dataset_id,
                    dataset_allocation_policy=cfg.dataset_allocation_policy,
                    enabled=cfg.enabled,
                    total_generated=0,
                    total_valid=0,
                    total_simulated=0,
                    total_accepted=0,
                    total_tokens=0,
                )
                session.add(researcher)
                await session.flush()
                researcher_ids.append(researcher.id)

            # 2. Seed active PromptStrategy rows for this session if not already existing
            existing_strats = await session.execute(
                select(PromptStrategy.category).where(PromptStrategy.session_id == session_id)
            )
            seeded_categories = {
                normalize_category(c) for c in existing_strats.scalars().all() if c
            }

            for cfg in roster:
                norm_cat = normalize_category(cfg.strategy_focus)
                if norm_cat not in seeded_categories:
                    tmpl = self.pipeline.library.get_template(norm_cat)
                    t_text, n_guidance, n_examples, n_anti_patterns = normalize_prompt_components(
                        template_text=tmpl.template_text,
                        guidance={
                            "strategy": tmpl.strategy_guidance,
                            "operators": tmpl.operator_guidance,
                        },
                        examples=tmpl.default_examples,
                        anti_patterns=tmpl.anti_patterns,
                    )
                    c_hash = compute_prompt_content_hash(
                        template_text=t_text,
                        guidance=n_guidance,
                        examples=n_examples,
                        anti_patterns=n_anti_patterns,
                    )
                    prompt_strat = PromptStrategy(
                        session_id=session_id,
                        name=f"Seed {tmpl.name}",
                        version=1,
                        category=norm_cat,
                        template_text=tmpl.template_text,
                        guidance={
                            "strategy": tmpl.strategy_guidance,
                            "operators": tmpl.operator_guidance,
                        },
                        examples=tmpl.default_examples,
                        anti_patterns=tmpl.anti_patterns,
                        is_active=True,
                    )
                    session.add(prompt_strat)
                    await session.flush()
                    pv = PromptVersion(
                        origin_session_id=session_id,
                        origin_strategy_id=prompt_strat.id,
                        category=norm_cat,
                        version=1,
                        authoring_source="SEED",
                        template_text=t_text,
                        guidance=n_guidance,
                        examples=n_examples,
                        anti_patterns=n_anti_patterns,
                        content_hash=c_hash,
                    )
                    session.add(pv)
                    await session.flush()
                    prompt_strat.current_prompt_version_id = pv.id
                    seeded_categories.add(norm_cat)

        log.info(
            "orchestrator.researchers_initialized",
            session_id=session_id,
            count=len(researcher_ids),
        )
        return researcher_ids

    async def get_session_researchers(
        self,
        session_id: int,
    ) -> list[ResearchResearcher]:
        """Fetches all researcher personas configured for a session."""
        async with self.db.session() as session:
            stmt = (
                select(ResearchResearcher)
                .where(ResearchResearcher.session_id == session_id)
                .order_by(ResearchResearcher.id.asc())
            )
            res = await session.execute(stmt)
            return list(res.scalars().all())

    async def add_researcher(
        self,
        session_id: int,
        config: ResearcherConfig,
    ) -> int:
        """Adds a new researcher persona to an existing session."""
        focus_val = config.strategy_focus.value if isinstance(config.strategy_focus, StrategyCategory) else str(config.strategy_focus)
        async with self.db.session() as session:
            researcher = ResearchResearcher(
                session_id=session_id,
                model_ref=config.model_ref,
                label=config.label,
                strategy_focus=focus_val,
                system_prompt=config.system_prompt,
                temperature=config.temperature,
                target_dataset_id=config.target_dataset_id,
                dataset_allocation_policy=config.dataset_allocation_policy,
                enabled=config.enabled,
                total_generated=0,
                total_valid=0,
                total_simulated=0,
                total_accepted=0,
                total_tokens=0,
            )
            session.add(researcher)
            await session.flush()
            rid = researcher.id

        log.info("orchestrator.researcher_added", session_id=session_id, researcher_id=rid, label=config.label)
        return rid

    async def update_researcher(
        self,
        researcher_id: int,
        *,
        enabled: bool | None = None,
        temperature: float | None = None,
        strategy_focus: str | StrategyCategory | None = None,
        system_prompt: str | None = None,
    ) -> bool:
        """Updates researcher persona settings."""
        async with self.db.session() as session:
            stmt = select(ResearchResearcher).where(ResearchResearcher.id == researcher_id)
            res = await session.execute(stmt)
            researcher = res.scalar_one_or_none()
            if not researcher:
                return False

            if enabled is not None:
                researcher.enabled = enabled
            if temperature is not None:
                researcher.temperature = temperature
            if strategy_focus is not None:
                researcher.strategy_focus = (
                    strategy_focus.value if isinstance(strategy_focus, StrategyCategory) else str(strategy_focus)
                )
            if system_prompt is not None:
                researcher.system_prompt = system_prompt
            researcher.updated_at = utcnow()
            return True

    async def run_generation_round(
        self,
        session_id: int,
        total_target_candidates: int = ROUND_CAPACITY,
        auto_repair: bool = True,
        max_repairs: int = 2,
    ) -> GenerationRoundResult:
        """Runs a concurrent generation round across all enabled researchers in the session.

        Distributes candidate quota evenly across researchers, taking remainders into account:
        e.g. 80 / 4 = 20 each; 80 / 3 = 27, 27, 26.
        Injects active PromptStrategy examples and anti-patterns for each category.
        Persists all generated candidates to SQLite with complete lineage tracking.
        Returns the list of valid candidate IDs ready for simulation.
        """
        await self.reconcile_generation_attempts()
        async with self.db.session() as session:
            # 1. Fetch session details
            s_stmt = select(ResearchSession).where(ResearchSession.id == session_id)
            res_s = await session.execute(s_stmt)
            res_session = res_s.scalar_one_or_none()
            if not res_session:
                raise ValueError(f"ResearchSession {session_id} not found")

            # P1-15: Explicit session configuration and profile identification
            region = res_session.region
            universe = res_session.universe
            delay = res_session.delay
            current_gen = res_session.current_generation or 0
            session_constraints = res_session.constraints or {}
            settings_profile_id = (
                getattr(res_session, "settings_profile_id", None)
                or session_constraints.get("settings_profile_id")
                or "standard_us_equity_v1"
            )
            alloc_region = region or "USA"
            alloc_universe = universe or "TOP3000"
            alloc_delay = delay if delay is not None else 1

            # 2. Fetch active researchers
            r_stmt = (
                select(ResearchResearcher)
                .where(
                    ResearchResearcher.session_id == session_id,
                    ResearchResearcher.enabled.is_(True),
                )
                .order_by(ResearchResearcher.id.asc())
            )
            res_r = await session.execute(r_stmt)
            researchers = list(res_r.scalars().all())

            if not researchers:
                raise ValueError(f"No active researchers found for session {session_id}")

            # 3. Calculate candidate allotment per researcher with remainder distribution
            num_researchers = len(researchers)
            base_count = total_target_candidates // num_researchers
            remainder = total_target_candidates % num_researchers
            allotments = [
                max(1, base_count + (1 if i < remainder else 0))
                for i in range(num_researchers)
            ]

            # 4. Fetch active PromptStrategy and immutable PromptVersion using exact 5-tier fallback precedence
            strategy_map: dict[str, tuple[PromptStrategy | None, PromptVersion, str]] = {}
            for r in researchers:
                cat_norm = normalize_category(r.strategy_focus)
                if cat_norm not in strategy_map:
                    strat, pv, tier = await self._resolve_strategy_version(session, session_id, cat_norm)
                    if pv is None:
                        raise PromptResolutionError(
                            f"No valid immutable PromptVersion resolved for category {cat_norm} in session {session_id}"
                        )
                    strategy_map[cat_norm] = (strat, pv, tier)

        # 5. Build dispatch specifications outside the provider call and transaction.
        dispatch_specs: list[dict[str, Any]] = []
        for idx, r in enumerate(researchers):
            cat_norm = normalize_category(r.strategy_focus)
            cat = (
                StrategyCategory(cat_norm)
                if cat_norm in StrategyCategory._value2member_map_
                else StrategyCategory.GENERAL
            )
            active_strat, pv, res_tier = strategy_map[cat_norm]
            winning_ex = list(pv.examples) if pv.examples else None
            anti_pats = list(pv.anti_patterns) if pv.anti_patterns else None

            # Formulate learning feedback from immutable snapshot guidance
            feedback_str = "No prior generation data. Explore baseline anomalies."
            if pv.guidance:
                try:
                    import json
                    g = json.loads(pv.guidance) if isinstance(pv.guidance, str) else pv.guidance
                except Exception:
                    g = {"strategy": pv.guidance}
                if isinstance(g, dict):
                    pref = g.get("preferred_operators")
                    avoid = g.get("avoid_operators")
                    strat_tips = g.get("strategy")
                    parts = []
                    if pref:
                        parts.append(f"Preferred operators: {', '.join(pref)}")
                    if avoid:
                        parts.append(f"Operators to avoid: {', '.join(avoid)}")
                    if strat_tips:
                        parts.append(f"Guidance: {strat_tips}")
                    if parts:
                        feedback_str = " | ".join(parts)
                else:
                    feedback_str = str(pv.guidance)

            r_target_ds = getattr(r, "target_dataset_id", None)
            # P1-12: Inject persistent researcher-specific episodic memory context
            try:
                mem_ctx = await self.memory.build_researcher_memory_context(
                    researcher_id=r.id,
                    strategy_focus=cat_norm,
                    dataset_id=r_target_ds,
                    include_shared=True,
                )
                if mem_ctx:
                    feedback_str = f"{feedback_str}\n\n{mem_ctx}"
            except Exception as mem_err:
                log.warning("orchestrator.memory_context_failed", researcher_id=r.id, error=str(mem_err))

            # Authoritative dataset selection per researcher persona
            if (
                self.pipeline
                and hasattr(self.pipeline, "discovery")
                and self.pipeline.discovery
                and hasattr(self.pipeline.discovery, "registry")
                and isinstance(self.pipeline.discovery.registry, DatasetRegistry)
            ):
                registry = self.pipeline.discovery.registry
            else:
                registry = DatasetRegistry()
            r_target_ds = getattr(r, "target_dataset_id", None)
            r_policy = getattr(r, "dataset_allocation_policy", None)
            if r_target_ds:
                sel_ctx = registry.allocate_dataset(
                    policy=DatasetAllocationPolicy.EXPLICIT,
                    target_dataset_id=r_target_ds,
                    region=alloc_region,
                    universe=alloc_universe,
                    delay=alloc_delay,
                    session_id=session_id,
                    researcher_id=r.id,
                    generation=current_gen,
                    researcher_index=idx,
                    num_researchers=num_researchers,
                )
            elif r_policy == DatasetAllocationPolicy.CONTROLLED_EXPLORATION.value:
                sel_ctx = registry.allocate_dataset(
                    policy=DatasetAllocationPolicy.CONTROLLED_EXPLORATION,
                    region=alloc_region,
                    universe=alloc_universe,
                    delay=alloc_delay,
                    session_id=session_id,
                    researcher_id=r.id,
                    generation=current_gen,
                    researcher_index=idx,
                    num_researchers=num_researchers,
                )
            else:
                sel_ctx = registry.allocate_dataset(
                    policy=DatasetAllocationPolicy.CATEGORY_ALIGNED,
                    strategy_category=cat_norm,
                    region=alloc_region,
                    universe=alloc_universe,
                    delay=alloc_delay,
                    session_id=session_id,
                    researcher_id=r.id,
                    generation=current_gen,
                    researcher_index=idx,
                    num_researchers=num_researchers,
                )

            dispatch_specs.append(
                {
                    "researcher_id": r.id,
                    "model_ref": r.model_ref or "google:gemini-2.5-flash",
                    "category": cat,
                    "count": allotments[idx],
                    "region": alloc_region,
                    "universe": alloc_universe,
                    "delay": alloc_delay,
                    "generation": current_gen,
                    "temperature": r.temperature or 0.7,
                    "winning_examples": winning_ex,
                    "anti_patterns": anti_pats,
                    "learning_feedback": feedback_str,
                    "auto_repair": auto_repair,
                    "max_repairs": max_repairs,
                    "strategy_id": active_strat.id if active_strat else None,
                    "prompt_version_id": pv.id,
                    "template_text": pv.template_text,
                    "resolution_tier": res_tier,
                    "selection_context": sel_ctx,
                }
            )

        # Create the durable logical attempts before any provider coroutine starts.
        async with self.db.session() as session:
            for spec in dispatch_specs:
                started_at = utcnow()
                attempt = GenerationAttempt(
                    session_id=session_id,
                    researcher_id=spec["researcher_id"],
                    prompt_strategy_id=spec["strategy_id"],
                    prompt_version_id=spec["prompt_version_id"],
                    resolution_tier=spec["resolution_tier"],
                    generation=current_gen,
                    model_ref=spec["model_ref"],
                    temperature=spec["temperature"],
                    target_dataset_id=(
                        spec["selection_context"].selected_dataset_id
                        if isinstance(spec["selection_context"], DatasetSelectionContext)
                        else None
                    ),
                    catalog_version=(
                        spec["selection_context"].catalog_version
                        if isinstance(spec["selection_context"], DatasetSelectionContext)
                        else None
                    ),
                    dataset_selection_policy=(
                        spec["selection_context"].allocation_policy
                        if isinstance(spec["selection_context"], DatasetSelectionContext)
                        else None
                    ),
                    dataset_selection_reason=(
                        spec["selection_context"].selection_reason
                        if isinstance(spec["selection_context"], DatasetSelectionContext)
                        else None
                    ),
                    datafield_catalog_version=(
                        spec["selection_context"].datafield_catalog_version
                        if isinstance(spec["selection_context"], DatasetSelectionContext)
                        else None
                    ),
                    allowed_fields=(
                        list(spec["selection_context"].allowed_field_names)
                        if isinstance(spec["selection_context"], DatasetSelectionContext)
                        else None
                    ),
                    status=GenerationAttemptStatus.RUNNING,
                    started_at=started_at,
                    retry_count=0,
                    candidate_count=0,
                    output_count=0,
                )
                session.add(attempt)
                await session.flush()
                spec["attempt_id"] = attempt.id

        # 6. Run parallel generation outside DB transaction to avoid locking.
        tasks = [
            self._dispatch_researcher(**spec)
            for spec in dispatch_specs
        ]
        dispatch_results = await asyncio.gather(*tasks, return_exceptions=True)

        outcomes: list[ResearcherDispatchOutcome] = []
        for spec, item in zip(dispatch_specs, dispatch_results, strict=True):
            if isinstance(item, ResearcherDispatchOutcome):
                outcomes.append(item)
            elif isinstance(item, BaseException):
                error_message = _safe_error(item)
                log.error(
                    "orchestrator.unexpected_dispatch_exception",
                    researcher_id=spec["researcher_id"],
                    attempt_id=spec["attempt_id"],
                    error=error_message,
                )
                outcomes.append(
                    ResearcherDispatchOutcome(
                        attempt_id=spec["attempt_id"],
                        researcher_id=spec["researcher_id"],
                        model_ref=spec["model_ref"],
                        strategy_id=spec["strategy_id"],
                        prompt_version_id=spec["prompt_version_id"],
                        resolution_tier=spec["resolution_tier"],
                        temperature=spec["temperature"],
                        selection_context=spec["selection_context"],
                        batch=None,
                        status=GenerationAttemptStatus.CANCELLED
                        if isinstance(item, asyncio.CancelledError)
                        else GenerationAttemptStatus.FAILED,
                        retry_count=0,
                        latency_seconds=None,
                        error_class=item.__class__.__name__,
                        error_message=error_message,
                        validation_outcome="ORCHESTRATION_EXCEPTION",
                    )
                )

        # 6. Persist candidates and update researcher stats in SQLite
        valid_candidate_ids: list[int] = []
        async with self.db.session() as session:
            # Query existing session candidate hashes for session-wide exact deduplication
            existing_hashes_stmt = (
                select(ResearchCandidate.expression_hash)
                .where(
                    ResearchCandidate.session_id == session_id,
                    ResearchCandidate.expression_hash.is_not(None),
                )
            )
            existing_hashes_res = await session.execute(existing_hashes_stmt)
            seen_hashes = set(existing_hashes_res.scalars().all())

            r_focus_map: dict[int, str] = {r.id: str(r.strategy_focus) for r in researchers}
            candidate_items: list[CandidateItem] = []
            cand_rows_by_id: dict[int, ResearchCandidate] = {}

            for outcome in outcomes:
                r_id = outcome.researcher_id
                strat_id = outcome.strategy_id
                pv_id = outcome.prompt_version_id
                res_tier = outcome.resolution_tier
                temp = outcome.temperature
                sel_ctx = outcome.selection_context
                batch = outcome.batch
                attempt = await session.get(GenerationAttempt, outcome.attempt_id)
                if attempt is not None:
                    attempt.status = outcome.status
                    attempt.completed_at = utcnow()
                    attempt.retry_count = outcome.retry_count
                    attempt.latency_seconds = outcome.latency_seconds
                    attempt.error_class = outcome.error_class
                    attempt.error_message = outcome.error_message
                    attempt.validation_outcome = outcome.validation_outcome
                    if batch is not None:
                        attempt.model_ref = batch.model_ref
                        attempt.system_prompt = batch.system_prompt
                        attempt.user_prompt = batch.user_prompt
                        attempt.response_schema = batch.response_schema
                        attempt.candidate_count = len(batch.candidates)
                        attempt.output_count = len(batch.candidates)
                        attempt.token_usage = batch.total_tokens if batch.total_tokens > 0 else None
                        attempt.provider_request_id = getattr(batch, "provider_request_id", None)
                        attempt.target_dataset_id = batch.target_dataset_id or attempt.target_dataset_id
                        attempt.catalog_version = batch.catalog_version or attempt.catalog_version
                        attempt.dataset_selection_policy = (
                            batch.dataset_selection_policy or attempt.dataset_selection_policy
                        )
                        attempt.dataset_selection_reason = (
                            batch.dataset_selection_reason or attempt.dataset_selection_reason
                        )
                        attempt.datafield_catalog_version = (
                            batch.datafield_catalog_version or attempt.datafield_catalog_version
                        )
                        attempt.allowed_fields = batch.allowed_fields or attempt.allowed_fields
                        attempt.operator_catalog_version = (
                            getattr(batch, "operator_catalog_version", None)
                            or attempt.operator_catalog_version
                        )

                if batch is None:
                    continue
                raw_t_ds = getattr(batch, "target_dataset_id", None) or (
                    sel_ctx.selected_dataset_id if isinstance(sel_ctx, DatasetSelectionContext) else None
                )
                raw_cat_ver = getattr(batch, "catalog_version", None) or (
                    sel_ctx.catalog_version if isinstance(sel_ctx, DatasetSelectionContext) else None
                )
                raw_pol = getattr(batch, "dataset_selection_policy", None) or (
                    sel_ctx.allocation_policy if isinstance(sel_ctx, DatasetSelectionContext) else None
                )
                raw_reason = getattr(batch, "dataset_selection_reason", None) or (
                    sel_ctx.selection_reason if isinstance(sel_ctx, DatasetSelectionContext) else None
                )

                raw_df_cat_ver = getattr(batch, "datafield_catalog_version", None) or (
                    sel_ctx.datafield_catalog_version if isinstance(sel_ctx, DatasetSelectionContext) else None
                )
                raw_allowed_flds = getattr(batch, "allowed_fields", None) or (
                    list(sel_ctx.allowed_field_names) if isinstance(sel_ctx, DatasetSelectionContext) else None
                )
                raw_op_cat_ver = getattr(batch, "operator_catalog_version", None)

                t_ds = str(raw_t_ds) if (raw_t_ds is not None and not hasattr(raw_t_ds, "_mock_return_value")) else None
                cat_ver = str(raw_cat_ver) if (raw_cat_ver is not None and not hasattr(raw_cat_ver, "_mock_return_value")) else None
                pol = str(raw_pol) if (raw_pol is not None and not hasattr(raw_pol, "_mock_return_value")) else None
                reason = str(raw_reason) if (raw_reason is not None and not hasattr(raw_reason, "_mock_return_value")) else None
                df_cat_ver = str(raw_df_cat_ver) if (raw_df_cat_ver is not None and not hasattr(raw_df_cat_ver, "_mock_return_value")) else None
                allowed_flds = list(raw_allowed_flds) if (raw_allowed_flds is not None and not hasattr(raw_allowed_flds, "_mock_return_value")) else None
                op_cat_ver = str(raw_op_cat_ver) if (raw_op_cat_ver is not None and not hasattr(raw_op_cat_ver, "_mock_return_value")) else None

                # Update researcher counters only for a returned batch. Provider failures
                # are not successful zero-output generations.
                u_stmt = (
                    update(ResearchResearcher)
                    .where(ResearchResearcher.id == r_id)
                    .values(
                        total_generated=ResearchResearcher.total_generated + len(batch.candidates),
                        total_valid=ResearchResearcher.total_valid + batch.valid_count,
                        total_tokens=ResearchResearcher.total_tokens + batch.total_tokens,
                        updated_at=utcnow(),
                    )
                )
                await session.execute(u_stmt)

                # Persist each candidate with full lineage
                for c in batch.candidates:
                    status = CandidateStatus.VALID if c.validation.is_valid else CandidateStatus.INVALID
                    raw_expr = str(c.raw_response.get("expression") or c.expression) if c.raw_response else c.expression
                    rep_hist = list(c.validation.repair_history) if (c.validation and c.validation.repair_history) else []

                    # Classify field families where available
                    field_fams: dict[str, str] = {}
                    if c.validation and c.validation.fields_used:
                        df_cat = registry.get_datafield_catalog(
                            dataset=sel_ctx.dataset if isinstance(sel_ctx, DatasetSelectionContext) else None,
                            region=alloc_region,
                            universe=alloc_universe,
                            delay=alloc_delay,
                        )
                        for f_used in c.validation.fields_used:
                            fam = df_cat.get_field_family(f_used)
                            if fam:
                                field_fams[f_used] = fam

                    # Determine operators used from validation result or expression AST
                    used_operators: list[str] = []
                    if c.validation and getattr(c.validation, "operators_used", None):
                        used_operators = list(c.validation.operators_used)
                    elif c.expression:
                        try:
                            used_operators = sorted(operator_names(parse(c.expression)))
                        except Exception:
                            used_operators = []

                    # P1-14: Deterministic AST fingerprinting and skeleton abstraction
                    fp = extract_candidate_fingerprint(c.expression or "")

                    # P1-15: Resolve explicit, candidate-specific simulation settings
                    c_overrides = getattr(c, "simulation_settings_overrides", None)
                    if not c_overrides and c.raw_response:
                        c_overrides = (
                            c.raw_response.get("simulation_settings_overrides")
                            or c.raw_response.get("simulation_settings")
                            or c.raw_response.get("settings")
                        )

                    cand_settings: CandidateSimulationSettings | None = None
                    try:
                        cand_settings = resolve_candidate_simulation_settings(
                            candidate_overrides=c_overrides,
                            session_region=region,
                            session_universe=universe,
                            session_delay=delay,
                            session_constraints=session_constraints,
                            profile=settings_profile_id,
                        )
                    except (SimulationSettingsResolutionError, SimulationSettingsValidationError) as set_err:
                        cand_settings = None
                        status = CandidateStatus.INVALID
                        c.validation.is_valid = False
                        c.validation.error = str(set_err)

                    if cand_settings is not None:
                        # Preflight dataset & field coverage check
                        is_ds_valid, ds_err = validate_candidate_dataset_preflight(
                            expression=c.expression or "",
                            settings=cand_settings,
                            target_dataset_id=t_ds,
                            dataset_registry=registry,
                            datafield_catalog=df_cat if "df_cat" in locals() else None,
                        )
                        # Platform schema options validation
                        schema_val = PlatformSchemaValidator(
                            schema=self.settings_schema,
                            schema_source="CACHED" if self.settings_schema else "FALLBACK",
                        )
                        is_schema_valid, schema_problems, v_status = schema_val.validate(
                            cand_settings, allow_fallback=True
                        )

                        if not is_ds_valid:
                            cand_settings.verification_status = "INVALID"
                            status = CandidateStatus.INVALID
                            c.validation.is_valid = False
                            c.validation.error = ds_err
                        elif not is_schema_valid:
                            cand_settings.verification_status = "INVALID"
                            status = CandidateStatus.INVALID
                            c.validation.is_valid = False
                            c.validation.error = f"ERR_PREFLIGHT_SETTINGS_INCOMPATIBLE: {'; '.join(schema_problems)}"
                        else:
                            cand_settings.verification_status = v_status

                    sim_settings_dict = cand_settings.to_dict() if cand_settings else None
                    settings_h = cand_settings.compute_hash() if cand_settings else None
                    cand_prov_h = (
                        compute_candidate_provenance_hash(
                            expression_hash=fp.expr_hash,
                            settings_hash=settings_h or "",
                            model_ref=batch.model_ref,
                            prompt_version_id=pv_id,
                            target_dataset_id=t_ds,
                            catalog_version=cat_ver,
                        )
                        if cand_settings
                        else None
                    )

                    cand_row = ResearchCandidate(
                        session_id=session_id,
                        researcher_id=r_id,
                        prompt_strategy_id=strat_id,
                        prompt_version_id=pv_id,
                        generation_attempt_id=attempt.id,
                        generation=current_gen,
                        model_ref=batch.model_ref,
                        temperature=temp,
                        hypothesis=c.hypothesis,
                        expression=c.expression,
                        raw_expression=raw_expr,
                        repair_history=rep_hist,
                        language=validate_alpha_language(getattr(batch, "language", None) or "FASTEXPR"),
                        target_dataset_id=t_ds,
                        catalog_version=cat_ver,
                        dataset_selection_policy=pol,
                        dataset_selection_reason=reason,
                        datafield_catalog_version=df_cat_ver,
                        field_families=field_fams if field_fams else None,
                        operator_catalog_version=op_cat_ver,
                        operators_used=used_operators,
                        datasets_used=[c.target_dataset] if c.target_dataset else [],
                        fields_used=c.validation.fields_used,
                        operator_count=c.validation.operator_count,
                        field_count=c.validation.field_count,
                        is_valid=c.validation.is_valid,
                        validation_error=c.validation.error,
                        retry_count=c.validation.retry_count,
                        sim_region=cand_settings.region if cand_settings else (region or "USA"),
                        sim_universe=cand_settings.universe if cand_settings else (universe or "TOP3000"),
                        sim_neutralization=cand_settings.neutralization if cand_settings else "NONE",
                        sim_decay=cand_settings.decay if cand_settings else 0,
                        sim_delay=cand_settings.delay if cand_settings else 1,
                        simulation_settings=sim_settings_dict,
                        settings_hash=settings_h,
                        settings_profile_id=cand_settings.profile_id if cand_settings else None,
                        settings_profile_version=cand_settings.profile_version if cand_settings else None,
                        settings_schema_version=cand_settings.schema_version if cand_settings else None,
                        settings_verification_status=cand_settings.verification_status if cand_settings else "INVALID",
                        candidate_provenance_hash=cand_prov_h,
                        status=status,
                        expression_hash=fp.expr_hash,
                        ast_skeleton=fp.skeleton,
                    )
                    session.add(cand_row)
                    await session.flush()
                    cand_rows_by_id[cand_row.id] = cand_row

                    c_cat = r_focus_map.get(r_id, "GENERAL")
                    item = CandidateItem(
                        expression=c.expression or "",
                        hypothesis=c.hypothesis or "",
                        category=c_cat,
                        target_dataset_id=t_ds,
                        fields_used=c.validation.fields_used or [],
                        field_families=field_fams,
                        operators_used=used_operators,
                        is_valid=bool(c.validation.is_valid),
                        candidate_id=cand_row.id,
                        metadata={"validation_error": c.validation.error},
                    )
                    candidate_items.append(item)

            # P1-14: Apply deterministic upstream deduplication, near-duplicate pruning, and diversity allocation
            diversity_selector = DiversitySelector()
            selection_result = diversity_selector.select_candidates(
                candidate_items,
                capacity=max(total_target_candidates, sum(allotments)),
                session_seen_hashes=seen_hashes,
            )

            for rec in selection_result.audit_records:
                if rec.candidate_id and rec.candidate_id in cand_rows_by_id:
                    crow = cand_rows_by_id[rec.candidate_id]
                    if rec.decision == "DISPATCHED":
                        crow.status = CandidateStatus.VALID
                        crow.pruning_reason = None
                        valid_candidate_ids.append(crow.id)
                    elif rec.decision in ("EXACT_DUPLICATE_PRUNED", "NEAR_DUPLICATE_PRUNED"):
                        crow.status = CandidateStatus.INVALID
                        crow.pruning_reason = rec.decision
                        crow.validation_error = rec.reason
                    elif rec.decision == "RESERVED_OVER_CAPACITY":
                        crow.status = CandidateStatus.VALID
                        crow.pruning_reason = rec.decision
                        crow.validation_error = rec.reason
                    elif rec.decision == "INVALID_EXPRESSION":
                        crow.status = CandidateStatus.INVALID
                        crow.pruning_reason = rec.decision

            # Update session generation count and status
            sess_stmt = (
                update(ResearchSession)
                .where(ResearchSession.id == session_id)
                .values(
                    status=ResearchStatus.RUNNING,
                    updated_at=utcnow(),
                )
            )
            await session.execute(sess_stmt)

        log.info(
            "orchestrator.generation_round_complete",
            session_id=session_id,
            generation=current_gen,
            total_generated=sum(
                len(item.batch.candidates) for item in outcomes if item.batch is not None
            ),
            valid_candidates=len(valid_candidate_ids),
        )
        failed = [
            item
            for item in outcomes
            if item.status in (GenerationAttemptStatus.FAILED, GenerationAttemptStatus.CANCELLED)
        ]
        summary = {
            "attemptedResearchers": len(outcomes),
            "successfulResearchers": sum(
                item.status == GenerationAttemptStatus.SUCCEEDED for item in outcomes
            ),
            "failedResearchers": sum(
                item.status == GenerationAttemptStatus.FAILED for item in outcomes
            ),
            "cancelledResearchers": sum(
                item.status == GenerationAttemptStatus.CANCELLED for item in outcomes
            ),
            "emptyOutputResearchers": sum(
                item.validation_outcome == "EMPTY_OUTPUT" for item in outcomes
            ),
            "outputValidationFailures": sum(
                item.validation_outcome == "OUTPUT_VALIDATION_FAILED" for item in outcomes
            ),
            "partialSuccess": bool(failed),
            "generatedCandidates": sum(
                len(item.batch.candidates) for item in outcomes if item.batch is not None
            ),
            "failureDetails": [
                {
                    "attemptId": item.attempt_id,
                    "researcherId": item.researcher_id,
                    "modelRef": item.model_ref,
                    "status": item.status,
                    "retryCount": item.retry_count,
                    "errorClass": item.error_class,
                    "errorMessage": item.error_message,
                    "validationOutcome": item.validation_outcome,
                }
                for item in failed
            ],
        }
        return GenerationRoundResult(valid_candidate_ids, summary)

    async def reconcile_generation_attempts(self, stale_after_seconds: int = 900) -> int:
        """Mark abandoned RUNNING attempts as failed after a bounded stale period."""
        cutoff = utcnow() - timedelta(seconds=stale_after_seconds)
        now = utcnow()
        stmt = (
            update(GenerationAttempt)
            .where(
                GenerationAttempt.status == GenerationAttemptStatus.RUNNING,
                GenerationAttempt.started_at.is_not(None),
                GenerationAttempt.started_at < cutoff,
            )
            .values(
                status=GenerationAttemptStatus.FAILED,
                completed_at=now,
                error_class="StaleAttemptRecovery",
                error_message="Attempt recovered after exceeding the stale RUNNING timeout.",
                validation_outcome="STALE_RECOVERED",
            )
        )
        async with self.db.session() as session:
            result = await session.execute(stmt)
            return int(result.rowcount or 0)

    async def _resolve_strategy_version(
        self, session: Any, session_id: int, category: str
    ) -> tuple[PromptStrategy | None, PromptVersion | None, str]:
        """Resolves active prompt strategy and verified immutable PromptVersion using strict 5 tiers:
        1. EXACT_SESSION: Active strategy for requested category in current session
        2. GLOBAL_CATEGORY: Active global baseline for requested category (session_id IS NULL)
        3. SESSION_GENERAL: Active general strategy in current session
        4. GLOBAL_GENERAL: Active global general baseline (session_id IS NULL)
        5. TEMPLATE_FALLBACK: Canonical global seed PromptVersion fallback
        """
        cat_norm = normalize_category(category)

        # Tier 1: EXACT_SESSION
        stmt_a = (
            select(PromptStrategy)
            .where(
                PromptStrategy.session_id == session_id,
                func.lower(func.trim(PromptStrategy.category)) == cat_norm,
                PromptStrategy.is_active.is_(True),
            )
            .order_by(PromptStrategy.version.desc(), PromptStrategy.id.desc())
        )
        res_a = await session.execute(stmt_a)
        for strat_a in res_a.scalars().all():
            if strat_a.current_prompt_version_id:
                pv = await session.get(PromptVersion, strat_a.current_prompt_version_id)
                if pv:
                    verify_prompt_integrity(pv)
                    return strat_a, pv, "EXACT_SESSION"
            log.warning(
                "prompt_strategy.legacy_unversioned_skipped",
                strategy_id=strat_a.id,
                category=cat_norm,
                tier="EXACT_SESSION",
            )

        # Tier 2: GLOBAL_CATEGORY
        stmt_b = (
            select(PromptStrategy)
            .where(
                PromptStrategy.session_id.is_(None),
                func.lower(func.trim(PromptStrategy.category)) == cat_norm,
                PromptStrategy.is_active.is_(True),
            )
            .order_by(PromptStrategy.version.desc(), PromptStrategy.id.desc())
        )
        res_b = await session.execute(stmt_b)
        for strat_b in res_b.scalars().all():
            if strat_b.current_prompt_version_id:
                pv = await session.get(PromptVersion, strat_b.current_prompt_version_id)
                if pv:
                    verify_prompt_integrity(pv)
                    return strat_b, pv, "GLOBAL_CATEGORY"
            log.warning(
                "prompt_strategy.legacy_unversioned_skipped",
                strategy_id=strat_b.id,
                category=cat_norm,
                tier="GLOBAL_CATEGORY",
            )

        # Tier 3: SESSION_GENERAL
        stmt_c = (
            select(PromptStrategy)
            .where(
                PromptStrategy.session_id == session_id,
                func.lower(func.trim(PromptStrategy.category)) == StrategyCategory.GENERAL.value,
                PromptStrategy.is_active.is_(True),
            )
            .order_by(PromptStrategy.version.desc(), PromptStrategy.id.desc())
        )
        res_c = await session.execute(stmt_c)
        for strat_c in res_c.scalars().all():
            if strat_c.current_prompt_version_id:
                pv = await session.get(PromptVersion, strat_c.current_prompt_version_id)
                if pv:
                    verify_prompt_integrity(pv)
                    return strat_c, pv, "SESSION_GENERAL"
            log.warning(
                "prompt_strategy.legacy_unversioned_skipped",
                strategy_id=strat_c.id,
                category=cat_norm,
                tier="SESSION_GENERAL",
            )

        # Tier 4: GLOBAL_GENERAL
        stmt_d = (
            select(PromptStrategy)
            .where(
                PromptStrategy.session_id.is_(None),
                func.lower(func.trim(PromptStrategy.category)) == StrategyCategory.GENERAL.value,
                PromptStrategy.is_active.is_(True),
            )
            .order_by(PromptStrategy.version.desc(), PromptStrategy.id.desc())
        )
        res_d = await session.execute(stmt_d)
        for strat_d in res_d.scalars().all():
            if strat_d.current_prompt_version_id:
                pv = await session.get(PromptVersion, strat_d.current_prompt_version_id)
                if pv:
                    verify_prompt_integrity(pv)
                    return strat_d, pv, "GLOBAL_GENERAL"
            log.warning(
                "prompt_strategy.legacy_unversioned_skipped",
                strategy_id=strat_d.id,
                category=cat_norm,
                tier="GLOBAL_GENERAL",
            )

        # Tier 5: TEMPLATE_FALLBACK / Canonical Global Seed Version
        stmt_e = (
            select(PromptVersion)
            .where(
                PromptVersion.origin_session_id.is_(None),
                PromptVersion.origin_strategy_id.is_(None),
                PromptVersion.category == cat_norm,
                PromptVersion.version == 1,
            )
            .limit(1)
        )
        res_e = await session.execute(stmt_e)
        seed_pv = res_e.scalars().first()
        if not seed_pv:
            stmt_e_gen = (
                select(PromptVersion)
                .where(
                    PromptVersion.origin_session_id.is_(None),
                    PromptVersion.origin_strategy_id.is_(None),
                    PromptVersion.category == StrategyCategory.GENERAL.value,
                    PromptVersion.version == 1,
                )
                .limit(1)
            )
            res_e_gen = await session.execute(stmt_e_gen)
            seed_pv = res_e_gen.scalars().first()

        if seed_pv:
            verify_prompt_integrity(seed_pv)
            log.warning(
                "prompt_strategy.database_miss_template_fallback",
                requested_category=cat_norm,
                session_id=session_id,
            )
            return None, seed_pv, "TEMPLATE_FALLBACK"

        return None, None, "TEMPLATE_FALLBACK"

    async def _resolve_strategy(
        self,
        session: Any,
        session_id: int,
        category: str,
        include_version: bool = False,
    ) -> Any:
        """Resolves active prompt strategy using strict 5-tier fallback precedence.
        If include_version is True, delegates to _resolve_strategy_version returning (strat, pv, tier).
        Otherwise returns standard 2-tuple (strat, tier) for full backwards-compatibility.
        """
        if include_version:
            return await self._resolve_strategy_version(session, session_id, category)

        cat_norm = normalize_category(category)

        # Tier 1: EXACT_SESSION
        stmt_a = (
            select(PromptStrategy)
            .where(
                PromptStrategy.session_id == session_id,
                func.lower(func.trim(PromptStrategy.category)) == cat_norm,
                PromptStrategy.is_active.is_(True),
            )
            .order_by(PromptStrategy.version.desc(), PromptStrategy.id.desc())
        )
        strat_a = (await session.execute(stmt_a)).scalars().first()
        if strat_a:
            return strat_a, "EXACT_SESSION"

        # Tier 2: GLOBAL_CATEGORY
        stmt_b = (
            select(PromptStrategy)
            .where(
                PromptStrategy.session_id.is_(None),
                func.lower(func.trim(PromptStrategy.category)) == cat_norm,
                PromptStrategy.is_active.is_(True),
            )
            .order_by(PromptStrategy.version.desc(), PromptStrategy.id.desc())
        )
        strat_b = (await session.execute(stmt_b)).scalars().first()
        if strat_b:
            return strat_b, "GLOBAL_CATEGORY"

        # Tier 3: SESSION_GENERAL
        stmt_c = (
            select(PromptStrategy)
            .where(
                PromptStrategy.session_id == session_id,
                func.lower(func.trim(PromptStrategy.category)) == StrategyCategory.GENERAL.value,
                PromptStrategy.is_active.is_(True),
            )
            .order_by(PromptStrategy.version.desc(), PromptStrategy.id.desc())
        )
        strat_c = (await session.execute(stmt_c)).scalars().first()
        if strat_c:
            return strat_c, "SESSION_GENERAL"

        # Tier 4: GLOBAL_GENERAL
        stmt_d = (
            select(PromptStrategy)
            .where(
                PromptStrategy.session_id.is_(None),
                func.lower(func.trim(PromptStrategy.category)) == StrategyCategory.GENERAL.value,
                PromptStrategy.is_active.is_(True),
            )
            .order_by(PromptStrategy.version.desc(), PromptStrategy.id.desc())
        )
        strat_d = (await session.execute(stmt_d)).scalars().first()
        if strat_d:
            return strat_d, "GLOBAL_GENERAL"

        # Tier 5: TEMPLATE_FALLBACK
        log.warning(
            "prompt_strategy.database_miss_template_fallback",
            requested_category=cat_norm,
            session_id=session_id,
        )
        return None, "TEMPLATE_FALLBACK"

    async def _dispatch_researcher(
        self,
        researcher_id: int,
        model_ref: str,
        category: StrategyCategory,
        count: int,
        region: str,
        universe: str,
        delay: int,
        generation: int,
        temperature: float,
        attempt_id: int,
        winning_examples: list[dict[str, Any]] | None = None,
        anti_patterns: list[dict[str, Any]] | None = None,
        learning_feedback: str = "No prior generation data. Explore baseline anomalies.",
        auto_repair: bool = True,
        max_repairs: int = 2,
        strategy_id: int | None = None,
        prompt_version_id: int | None = None,
        template_text: str | None = None,
        resolution_tier: str | None = None,
        selection_context: DatasetSelectionContext | None = None,
    ) -> ResearcherDispatchOutcome:
        """Dispatch one logical researcher attempt with bounded transient retries."""
        started = time.monotonic()
        retry_count = 0
        while True:
            try:
                batch = await self.pipeline.generate_batch(
                    model_ref=model_ref,
                    category=category,
                    candidate_count=count,
                    region=region,
                    universe=universe,
                    delay=delay,
                    generation=generation,
                    temperature=temperature,
                    winning_examples=winning_examples,
                    anti_patterns=anti_patterns,
                    learning_feedback=learning_feedback,
                    target_dataset_id=selection_context.selected_dataset_id if selection_context else None,
                    selection_context=selection_context,
                    auto_repair=auto_repair,
                    max_repairs=max_repairs,
                    template_text=template_text,
                    allow_synthetic_fallback=False,
                )
                if not batch.candidates:
                    status = GenerationAttemptStatus.SUCCEEDED
                    validation_outcome = "EMPTY_OUTPUT"
                elif batch.valid_count == 0:
                    status = GenerationAttemptStatus.FAILED
                    validation_outcome = "OUTPUT_VALIDATION_FAILED"
                elif batch.valid_count < len(batch.candidates):
                    status = GenerationAttemptStatus.SUCCEEDED
                    validation_outcome = "PARTIAL_VALIDATION"
                else:
                    status = GenerationAttemptStatus.SUCCEEDED
                    validation_outcome = "VALIDATED"
                return ResearcherDispatchOutcome(
                    attempt_id=attempt_id,
                    researcher_id=researcher_id,
                    model_ref=model_ref,
                    strategy_id=strategy_id,
                    prompt_version_id=prompt_version_id,
                    resolution_tier=resolution_tier,
                    temperature=temperature,
                    selection_context=selection_context,
                    batch=batch,
                    status=status,
                    retry_count=retry_count,
                    latency_seconds=time.monotonic() - started,
                    validation_outcome=validation_outcome,
                )
            except asyncio.CancelledError as exc:
                return ResearcherDispatchOutcome(
                    attempt_id=attempt_id,
                    researcher_id=researcher_id,
                    model_ref=model_ref,
                    strategy_id=strategy_id,
                    prompt_version_id=prompt_version_id,
                    resolution_tier=resolution_tier,
                    temperature=temperature,
                    selection_context=selection_context,
                    batch=None,
                    status=GenerationAttemptStatus.CANCELLED,
                    retry_count=retry_count,
                    latency_seconds=time.monotonic() - started,
                    error_class=exc.__class__.__name__,
                    error_message=_safe_error(exc),
                    validation_outcome="CANCELLED",
                )
            except Exception as exc:
                if _is_retryable(exc) and retry_count < MAX_RESEARCHER_RETRIES:
                    retry_count += 1
                    await asyncio.sleep(min(0.25 * (2 ** (retry_count - 1)), 2.0))
                    continue
                error_message = _safe_error(exc)
                log.warning(
                    "orchestrator.researcher_dispatch_failed",
                    researcher_id=researcher_id,
                    attempt_id=attempt_id,
                    model_ref=model_ref,
                    retry_count=retry_count,
                    error_class=exc.__class__.__name__,
                    error=error_message,
                )
                return ResearcherDispatchOutcome(
                    attempt_id=attempt_id,
                    researcher_id=researcher_id,
                    model_ref=model_ref,
                    strategy_id=strategy_id,
                    prompt_version_id=prompt_version_id,
                    resolution_tier=resolution_tier,
                    temperature=temperature,
                    selection_context=selection_context,
                    batch=None,
                    status=GenerationAttemptStatus.FAILED,
                    retry_count=retry_count,
                    latency_seconds=time.monotonic() - started,
                    error_class=exc.__class__.__name__,
                    error_message=error_message,
                    validation_outcome="PROVIDER_FAILURE",
                )

    async def enqueue_candidates_for_simulation(
        self,
        session_id: int,
        candidate_ids: list[int] | None = None,
        task: str | None = None,
        batch_engine: BatchEngine | None = None,
    ) -> list[int]:
        """Transitions valid candidates to QUEUED status and optionally submits to BatchEngine."""
        async with self.db.session() as session:
            stmt = (
                select(ResearchCandidate)
                .where(
                    ResearchCandidate.session_id == session_id,
                    ResearchCandidate.is_valid.is_(True),
                    ResearchCandidate.status == CandidateStatus.VALID,
                )
            )
            if candidate_ids is not None:
                stmt = stmt.where(ResearchCandidate.id.in_(candidate_ids))

            res = await session.execute(stmt)
            candidates = list(res.scalars().all())

            if not candidates:
                return []

            queued_ids: list[int] = []
            if batch_engine is not None:
                # P1-07: Ensure every candidate strictly uses supported language
                for c in candidates:
                    validate_alpha_language(getattr(c, "language", None) or "FASTEXPR")

                task_name = task or f"research_session_{session_id}"
                sim_requests = [
                    SimulationRequest(
                        type=SimulationType.REGULAR,
                        settings=c.to_simulation_settings(),
                        regular=c.expression,
                    )
                    for c in candidates
                ]
                enqueue_result = await batch_engine.enqueue(sim_requests, task=task_name)
                outcomes = enqueue_result.get("outcomes", [])

                for idx, c in enumerate(candidates):
                    c.status = CandidateStatus.QUEUED
                    if idx < len(outcomes):
                        outcome = outcomes[idx]
                        c.simulation_record_id = outcome.get("recordId")
                        if outcome.get("alphaId"):
                            c.alpha_id = outcome.get("alphaId")
                    queued_ids.append(c.id)
            else:
                for c in candidates:
                    c.status = CandidateStatus.QUEUED
                    queued_ids.append(c.id)

        log.info(
            "orchestrator.candidates_enqueued",
            session_id=session_id,
            count=len(queued_ids),
        )
        return queued_ids

    async def sync_session_simulations(self, session_id: int) -> dict[str, int]:
        """Synchronizes candidate records with linked SimulationRecord outcomes from BatchEngine/BRAIN."""
        async with self.db.session() as session:
            stmt = (
                select(ResearchCandidate, SimulationRecord)
                .join(
                    SimulationRecord,
                    ResearchCandidate.simulation_record_id == SimulationRecord.id,
                )
                .where(ResearchCandidate.session_id == session_id)
            )
            res = await session.execute(stmt)
            rows = res.all()

            updated = 0
            completed = 0
            failed = 0
            pending = 0

            for cand, sim in rows:
                changed = False
                if sim.alpha_id and cand.alpha_id != sim.alpha_id:
                    cand.alpha_id = sim.alpha_id
                    changed = True

                if sim.status in (SimStatus.COMPLETE, SimStatus.WARNING):
                    completed += 1
                    # Candidate has completed backtest and is ready for evaluation
                    if cand.status in (CandidateStatus.QUEUED, CandidateStatus.SIMULATING, CandidateStatus.VALID):
                        cand.status = CandidateStatus.SIMULATING
                        changed = True
                elif sim.status in (SimStatus.ERROR, SimStatus.CANCELLED):
                    failed += 1
                    if cand.status in (CandidateStatus.QUEUED, CandidateStatus.SIMULATING):
                        cand.status = CandidateStatus.REJECTED
                        cand.validation_error = sim.message or "Simulation failed on platform"
                        changed = True
                else:
                    pending += 1

            sess_row = await session.get(ResearchSession, session_id)
            if sess_row is not None:
                sess_row.simulations_used = completed

            await session.commit()

        log.info(
            "orchestrator.simulations_synced",
            session_id=session_id,
            completed=completed,
            failed=failed,
            pending=pending,
            updated=updated,
        )
        return {
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "updated": updated,
        }

