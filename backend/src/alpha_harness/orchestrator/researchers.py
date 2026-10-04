"""Multi-LLM Researcher Agent orchestration layer.

Coordinates heterogeneous LLM quantitative researcher personas (e.g. Momentum,
Mean Reversion, Value, Volatility specialists), dispatches generation tasks in parallel
up to the round capacity (80 simulations across 8 BRAIN slots with 10 multi-sim batch items),
injects versioned PromptStrategy context, and persists candidate lineage to SQLite.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import select, update

from ..brain.schemas import SimulationRequest, SimulationSettings, SimulationType
from ..db.models import (
    CandidateStatus,
    PromptStrategy,
    ResearchCandidate,
    ResearchResearcher,
    ResearchSession,
    ResearchStatus,
    SimStatus,
    SimulationRecord,
    utcnow,
)
from ..generator.pipeline import AlphaGenerationPipeline, GeneratedCandidate, GenerationBatch
from ..prompts.library import StrategyCategory

if TYPE_CHECKING:
    from ..db.sqlite import Database
    from ..engine.slots import BatchEngine
    from ..llm.service import LLMService

log = structlog.get_logger(__name__)

#: WorldQuant BRAIN core slots and multi-sim packing limits
SLOT_CAP: int = 8
ITEMS_PER_CORE: int = 10
ROUND_CAPACITY: int = SLOT_CAP * ITEMS_PER_CORE  # 80 candidates per round


@dataclass(slots=True)
class ResearcherConfig:
    """Configuration for one LLM researcher persona."""

    model_ref: str
    label: str
    strategy_focus: StrategyCategory = StrategyCategory.GENERAL
    temperature: float = 0.7
    system_prompt: str | None = None
    target_dataset_id: str | None = None
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
    ) -> None:
        self.db = db
        self.pipeline = pipeline

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
                focus_val = cfg.strategy_focus.value if isinstance(cfg.strategy_focus, StrategyCategory) else str(cfg.strategy_focus)
                researcher = ResearchResearcher(
                    session_id=session_id,
                    model_ref=cfg.model_ref,
                    label=cfg.label,
                    strategy_focus=focus_val,
                    system_prompt=cfg.system_prompt,
                    temperature=cfg.temperature,
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
            seeded_categories = set(existing_strats.scalars().all())

            for cfg in roster:
                cat_enum = (
                    cfg.strategy_focus
                    if isinstance(cfg.strategy_focus, StrategyCategory)
                    else StrategyCategory(cfg.strategy_focus)
                    if cfg.strategy_focus in StrategyCategory._value2member_map_
                    else StrategyCategory.GENERAL
                )
                if cat_enum.value not in seeded_categories:
                    tmpl = self.pipeline.library.get_template(cat_enum)
                    prompt_strat = PromptStrategy(
                        session_id=session_id,
                        name=f"Seed {tmpl.name}",
                        version=1,
                        category=cat_enum.value,
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
                    seeded_categories.add(cat_enum.value)

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
    ) -> list[int]:
        """Runs a concurrent generation round across all enabled researchers in the session.

        Distributes candidate quota evenly across researchers, taking remainders into account:
        e.g. 80 / 4 = 20 each; 80 / 3 = 27, 27, 26.
        Injects active PromptStrategy examples and anti-patterns for each category.
        Persists all generated candidates to SQLite with complete lineage tracking.
        Returns the list of valid candidate IDs ready for simulation.
        """
        async with self.db.session() as session:
            # 1. Fetch session details
            s_stmt = select(ResearchSession).where(ResearchSession.id == session_id)
            res_s = await session.execute(s_stmt)
            res_session = res_s.scalar_one_or_none()
            if not res_session:
                raise ValueError(f"ResearchSession {session_id} not found")

            region = res_session.region or "USA"
            universe = res_session.universe or "TOP3000"
            delay = res_session.delay or 1
            current_gen = res_session.current_generation or 0

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

            # 4. Fetch active PromptStrategy for each category to inject few-shot context & guidance
            strategy_map: dict[str, PromptStrategy] = {}
            for r in researchers:
                cat_str = r.strategy_focus or StrategyCategory.GENERAL.value
                if cat_str not in strategy_map:
                    # Query session-specific or active strategy
                    ps_stmt = (
                        select(PromptStrategy)
                        .where(
                            PromptStrategy.session_id == session_id,
                            PromptStrategy.category == cat_str,
                            PromptStrategy.is_active.is_(True),
                        )
                        .order_by(PromptStrategy.version.desc())
                        .limit(1)
                    )
                    ps_res = await session.execute(ps_stmt)
                    active_ps = ps_res.scalars().first()
                    if not active_ps:
                        # Fallback to any active strategy for this category
                        fallback_stmt = (
                            select(PromptStrategy)
                            .where(
                                PromptStrategy.category == cat_str,
                                PromptStrategy.is_active.is_(True),
                            )
                            .order_by(PromptStrategy.version.desc())
                            .limit(1)
                        )
                        fallback_res = await session.execute(fallback_stmt)
                        active_ps = fallback_res.scalars().first()
                    if active_ps:
                        strategy_map[cat_str] = active_ps

        # 5. Run parallel generation outside DB transaction to avoid locking
        tasks = []
        for idx, r in enumerate(researchers):
            cat = (
                StrategyCategory(r.strategy_focus)
                if r.strategy_focus in StrategyCategory._value2member_map_
                else StrategyCategory.GENERAL
            )
            cat_str = r.strategy_focus or StrategyCategory.GENERAL.value
            active_strat = strategy_map.get(cat_str)
            winning_ex = (active_strat.examples or []) if active_strat else None
            anti_pats = (active_strat.anti_patterns or []) if active_strat else None
            
            # Formulate learning feedback from strategy guidance
            feedback_str = "No prior generation data. Explore baseline anomalies."
            if active_strat and active_strat.guidance:
                g = active_strat.guidance
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

            tasks.append(
                self._dispatch_researcher(
                    researcher_id=r.id,
                    model_ref=r.model_ref or "google:gemini-2.5-flash",
                    category=cat,
                    count=allotments[idx],
                    region=region,
                    universe=universe,
                    delay=delay,
                    generation=current_gen,
                    temperature=r.temperature or 0.7,
                    winning_examples=winning_ex,
                    anti_patterns=anti_pats,
                    learning_feedback=feedback_str,
                    auto_repair=auto_repair,
                    max_repairs=max_repairs,
                    strategy_id=active_strat.id if active_strat else None,
                )
            )

        dispatch_results = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[tuple[int, int | None, GenerationBatch]] = []
        for item in dispatch_results:
            if isinstance(item, tuple) and len(item) == 3:
                results.append(item)
            elif isinstance(item, Exception):
                log.error("orchestrator.unexpected_dispatch_exception", error=str(item))

        # 6. Persist candidates and update researcher stats in SQLite
        valid_candidate_ids: list[int] = []
        async with self.db.session() as session:
            for r_id, strat_id, batch in results:
                # Update researcher counters
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

                # Persist each candidate
                for c in batch.candidates:
                    status = CandidateStatus.VALID if c.validation.is_valid else CandidateStatus.INVALID
                    cand_row = ResearchCandidate(
                        session_id=session_id,
                        researcher_id=r_id,
                        prompt_strategy_id=strat_id,
                        generation=current_gen,
                        hypothesis=c.hypothesis,
                        expression=c.expression,
                        language="FASTEXPR",
                        datasets_used=[c.target_dataset] if c.target_dataset else [],
                        fields_used=c.validation.fields_used,
                        operator_count=c.validation.operator_count,
                        field_count=c.validation.field_count,
                        is_valid=c.validation.is_valid,
                        validation_error=c.validation.error,
                        retry_count=c.validation.retry_count,
                        sim_region=region,
                        sim_universe=universe,
                        sim_neutralization="INDUSTRY",
                        sim_decay=4,
                        sim_delay=delay,
                        status=status,
                    )
                    session.add(cand_row)
                    await session.flush()

                    if c.validation.is_valid:
                        valid_candidate_ids.append(cand_row.id)

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
            total_generated=sum(len(b.candidates) for _, _, b in results),
            valid_candidates=len(valid_candidate_ids),
        )
        return valid_candidate_ids

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
        winning_examples: list[dict[str, Any]] | None = None,
        anti_patterns: list[dict[str, Any]] | None = None,
        learning_feedback: str = "No prior generation data. Explore baseline anomalies.",
        auto_repair: bool = True,
        max_repairs: int = 2,
        strategy_id: int | None = None,
    ) -> tuple[int, int | None, GenerationBatch]:
        """Dispatches generation for a single researcher persona with error isolation."""
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
                auto_repair=auto_repair,
                max_repairs=max_repairs,
            )
            return researcher_id, strategy_id, batch
        except Exception as exc:
            log.warning(
                "orchestrator.researcher_dispatch_failed",
                researcher_id=researcher_id,
                model_ref=model_ref,
                error=str(exc),
            )
            # Return empty batch so orchestrator does not crash
            return researcher_id, strategy_id, GenerationBatch(
                summary=f"Generation failed: {exc}",
                candidates=[],
                model_ref=model_ref,
                total_tokens=0,
                duration_seconds=0.0,
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
                task_name = task or f"research_session_{session_id}"
                sim_requests = [
                    SimulationRequest(
                        type=SimulationType.REGULAR,
                        settings=SimulationSettings(
                            instrument_type="EQUITY",
                            region=c.sim_region or "USA",
                            universe=c.sim_universe or "TOP3000",
                            delay=c.sim_delay or 1,
                            decay=c.sim_decay or 4,
                            neutralization=c.sim_neutralization or "INDUSTRY",
                            truncation=0.08,
                            pasteurization="ON",
                            unit_handling="VERIFY",
                            nan_handling="OFF",
                            language="FASTEXPR",
                            visualization=False,
                        ),
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

