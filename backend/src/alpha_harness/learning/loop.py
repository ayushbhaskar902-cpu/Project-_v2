"""Closed-loop self-learning and prompt strategy mutation engine.

Executes after every X simulation rounds (default 80 across 8 BRAIN slots):
1. Partitions candidate results into top winners and failure modes
2. Performs operator and field statistical attribution
3. Derives quantitative rules and insights via evidence-grounded LLM reflection
4. Persists versioned insights into SQLite (ResearchInsight)
5. Mutates and updates prompt templates and few-shot examples (PromptStrategy)
6. Advances research session generation and triggers leaderboard snapshots
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx
import structlog
from sqlalchemy import func, select, update

from ..db.models import (
    LeaderboardSnapshot,
    PromptLifecycleEvent,
    PromptStrategy,
    PromptVersion,
    ResearchCandidate,
    ResearchEvaluation,
    ResearchInsight,
    ResearchResearcher,
    ResearchSession,
    utcnow,
)
from ..llm.keys import BudgetExhaustedError, LLMError
from ..llm.openai_compat import ProviderError
from ..prompts.library import (
    ALLOWED_CITATION_FIELDS,
    REFLECTION_RESPONSE_SCHEMA,
    REFLECTION_SYSTEM_PROMPT,
    SEED_TEMPLATES,
    StrategyCategory,
    normalize_category,
    render_reflection_evidence_prompt,
)
from ..prompts.versioning import (
    compute_prompt_content_hash,
    normalize_prompt_components,
)
from .memory import ResearchMemory

if TYPE_CHECKING:
    from ..db.sqlite import Database
    from ..llm.service import LLMService

log = structlog.get_logger(__name__)


class CitationValidationError(ValueError):
    """Raised when an LLM citation fails strict evidence grounding validation."""


class SchemaValidationError(ValueError):
    """Raised when an LLM reflection response fails structural validation."""


def _extract_json(text: str) -> dict[str, Any] | None:
    """Extracts JSON from model response text, stripping markdown codeblocks if present."""
    text = text.strip()
    if not text:
        return None

    # Try direct parse
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Try markdown fence extraction
    fence_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
    match = fence_pattern.search(text)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    # Try substring from first { to last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    return None


def _validate_reflection_schema(data: dict[str, Any]) -> None:
    for req in ("summary", "hypotheses", "recommendations", "anti_patterns"):
        if req not in data:
            raise SchemaValidationError(f"Missing required key '{req}' in reflection response")
    if not isinstance(data["summary"], str) or not data["summary"].strip():
        raise SchemaValidationError("Reflection 'summary' must be a non-empty string")
    if not isinstance(data["hypotheses"], list):
        raise SchemaValidationError("'hypotheses' must be a list")
    if not isinstance(data["recommendations"], list):
        raise SchemaValidationError("'recommendations' must be a list")
    if not isinstance(data["anti_patterns"], list):
        raise SchemaValidationError("'anti_patterns' must be a list")


def validate_citations(
    structured_reflection: dict[str, Any],
    evidence_candidates: list[dict[str, Any]],
) -> None:
    evidence_map = {c["candidate_id"]: c for c in evidence_candidates}

    sections = [
        ("hypotheses", structured_reflection.get("hypotheses", [])),
        ("recommendations", structured_reflection.get("recommendations", [])),
        ("anti_patterns", structured_reflection.get("anti_patterns", [])),
    ]

    for section_name, items in sections:
        if not isinstance(items, list):
            raise CitationValidationError(f"Section {section_name} must be a list")
        for item_idx, item in enumerate(items):
            if not isinstance(item, dict):
                raise CitationValidationError(f"Item {item_idx} in {section_name} is not a dict")
            citations = item.get("citations")
            if not citations or not isinstance(citations, list) or len(citations) == 0:
                raise CitationValidationError(
                    f"Item {item_idx} in {section_name} has empty or missing citations"
                )
            for cit_idx, cit in enumerate(citations):
                if not isinstance(cit, dict):
                    raise CitationValidationError(
                        f"Citation {cit_idx} in {section_name}[{item_idx}] is not a dict"
                    )
                cid = cit.get("candidate_id")
                field = cit.get("field")
                if cid is None or cid not in evidence_map:
                    raise CitationValidationError(
                        f"Citation in {section_name}[{item_idx}] references invalid candidate_id {cid}"
                    )
                if not field or field not in ALLOWED_CITATION_FIELDS:
                    raise CitationValidationError(
                        f"Citation in {section_name}[{item_idx}] references unpermitted field '{field}'"
                    )
                cand = evidence_map[cid]
                if field not in cand or cand[field] is None:
                    raise CitationValidationError(
                        f"Candidate {cid} does not contain cited field '{field}'"
                    )
                # Anti-patterns check: expression must match the cited candidate expression
                if section_name == "anti_patterns":
                    ap_expr = item.get("expression", "").strip()
                    cand_expr = cand.get("expression", "").strip()
                    if ap_expr != cand_expr:
                        raise CitationValidationError(
                            f"Anti-pattern expression '{ap_expr}' does not match cited candidate {cid} expression '{cand_expr}'"
                        )


@dataclass(slots=True)
class LearningCycleResult:
    """Outcome of one category-scoped self-learning reflection cycle."""

    session_id: int
    generation: int
    batch_size: int
    winner_count: int
    loser_count: int
    avg_sharpe: float
    best_sharpe: float
    insights_generated: list[dict[str, Any]]
    updated_prompt_strategy_id: int | None = None
    category: str = "general"
    status: str = "COMPLETED"
    error_message: str | None = None


@dataclass(slots=True)
class MultiCategoryLearningResult:
    """Aggregated outcome of learning cycles across multiple categories in a session generation."""

    session_id: int
    generation: int
    categories_evolved: list[str]
    category_results: dict[str, LearningCycleResult]
    updated_prompt_strategy_ids: list[int]
    category_statuses: dict[str, str] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return sum(r.batch_size for r in self.category_results.values())

    @property
    def winner_count(self) -> int:
        return sum(r.winner_count for r in self.category_results.values())

    @property
    def loser_count(self) -> int:
        return sum(r.loser_count for r in self.category_results.values())

    @property
    def avg_sharpe(self) -> float:
        total_candidates = sum(r.batch_size for r in self.category_results.values())
        if total_candidates == 0:
            return 0.0
        weighted_sum = sum(r.avg_sharpe * r.batch_size for r in self.category_results.values())
        return round(weighted_sum / total_candidates, 3)

    @property
    def best_sharpe(self) -> float:
        if not self.category_results:
            return 0.0
        return max(r.best_sharpe for r in self.category_results.values())

    @property
    def insights_generated(self) -> list[dict[str, Any]]:
        combined: list[dict[str, Any]] = []
        for r in self.category_results.values():
            combined.extend(r.insights_generated)
        return combined

    @property
    def updated_prompt_strategy_id(self) -> int | None:
        return self.updated_prompt_strategy_ids[0] if self.updated_prompt_strategy_ids else None


class SelfLearningEngine:
    """Closed-loop quantitative reflection and strategy optimization engine."""

    def __init__(
        self,
        db: Database,
        memory: ResearchMemory | None = None,
        llm: LLMService | None = None,
    ) -> None:
        self.db = db
        self.memory = memory or ResearchMemory(db)
        self.llm = llm

    async def execute_learning_cycle(
        self,
        session_id: int,
        category: str | None = None,
        min_candidates: int = 1,
        require_validation: bool = False,
    ) -> LearningCycleResult | MultiCategoryLearningResult:
        """Executes reflection and prompt evolution for a specific category or all active session categories."""
        if category is not None:
            return await self._execute_single_category_cycle(
                session_id=session_id,
                category=category,
                min_candidates=min_candidates,
                advance_generation=True,
                require_validation=require_validation,
            )

        # Multi-category discovery when category is None
        async with self.db.session() as session:
            s_stmt = select(ResearchSession).where(ResearchSession.id == session_id)
            res = await session.execute(s_stmt)
            research_session = res.scalar_one_or_none()
            if not research_session:
                raise ValueError(f"ResearchSession {session_id} not found")
            gen = research_session.current_generation or 0

        # Discover all candidate categories in this generation
        gen_candidates = await self.memory.get_eligible_candidates(session_id=session_id, generation=gen)
        if not gen_candidates:
            gen_candidates = await self.memory.get_eligible_candidates(session_id=session_id, generation=None)

        discovered_categories = {c["category"] for c in gen_candidates if c.get("category")}

        # If no candidates present yet, discover from configured researchers
        if not discovered_categories:
            async with self.db.session() as session:
                r_stmt = select(ResearchResearcher.strategy_focus).where(ResearchResearcher.session_id == session_id)
                r_res = await session.execute(r_stmt)
                discovered_categories = {
                    normalize_category(c) for c in r_res.scalars().all() if c
                }

        if not discovered_categories:
            discovered_categories = {"general"}

        # Deterministic alphanumeric order
        sorted_categories = sorted(discovered_categories)

        category_results: dict[str, LearningCycleResult] = {}
        categories_evolved: list[str] = []
        updated_strategy_ids: list[int] = []
        category_statuses: dict[str, str] = {}

        for cat in sorted_categories:
            cat_result = await self._execute_single_category_cycle(
                session_id=session_id,
                category=cat,
                min_candidates=min_candidates,
                advance_generation=False,
                require_validation=require_validation,
            )
            category_results[cat] = cat_result
            category_statuses[cat] = cat_result.status
            if cat_result.status == "COMPLETED" and cat_result.updated_prompt_strategy_id:
                categories_evolved.append(cat)
                updated_strategy_ids.append(cat_result.updated_prompt_strategy_id)

        # Advance session generation once for the session if at least one category evolved
        if categories_evolved:
            async with self.db.session() as session:
                sess_up = (
                    update(ResearchSession)
                    .where(ResearchSession.id == session_id)
                    .values(
                        current_generation=gen + 1,
                        updated_at=utcnow(),
                    )
                )
                await session.execute(sess_up)

        return MultiCategoryLearningResult(
            session_id=session_id,
            generation=gen,
            categories_evolved=categories_evolved,
            category_results=category_results,
            updated_prompt_strategy_ids=updated_strategy_ids,
            category_statuses=category_statuses,
        )

    async def _execute_single_category_cycle(
        self,
        session_id: int,
        category: str,
        min_candidates: int = 1,
        advance_generation: bool = True,
        require_validation: bool = False,
    ) -> LearningCycleResult:
        """Executes a category-scoped reflection and prompt evolution cycle."""
        cat_norm = normalize_category(category)

        async with self.db.session() as session:
            s_stmt = select(ResearchSession).where(ResearchSession.id == session_id)
            res = await session.execute(s_stmt)
            research_session = res.scalar_one_or_none()
            if not research_session:
                raise ValueError(f"ResearchSession {session_id} not found")

            gen = research_session.current_generation or 0
            model_ref = (
                research_session.model_refs[0]
                if (research_session.model_refs and len(research_session.model_refs) > 0)
                else None
            )

            # Query active antecedent strategy for this specific category (session-scoped first)
            strat_stmt = (
                select(PromptStrategy)
                .where(
                    PromptStrategy.session_id == session_id,
                    func.lower(func.trim(PromptStrategy.category)) == cat_norm,
                    PromptStrategy.is_active.is_(True),
                )
                .order_by(PromptStrategy.version.desc(), PromptStrategy.id.desc())
            )
            res_strat = await session.execute(strat_stmt)
            curr_strat = res_strat.scalars().first()

            if not curr_strat:
                # Query global baseline for this category
                strat_stmt_global = (
                    select(PromptStrategy)
                    .where(
                        PromptStrategy.session_id.is_(None),
                        func.lower(func.trim(PromptStrategy.category)) == cat_norm,
                        PromptStrategy.is_active.is_(True),
                    )
                    .order_by(PromptStrategy.version.desc(), PromptStrategy.id.desc())
                )
                res_strat_global = await session.execute(strat_stmt_global)
                curr_strat = res_strat_global.scalars().first()

            antecedent_strategy_id = curr_strat.id if curr_strat else None

        # 1. Fetch eligible candidates and partition winners / losers strictly for this category
        eligible = await self.memory.get_eligible_candidates(
            session_id=session_id, generation=gen, category=cat_norm
        )
        if not eligible:
            eligible = await self.memory.get_eligible_candidates(
                session_id=session_id, generation=None, category=cat_norm
            )

        winners, losers = await self.memory.get_session_performance_split(
            session_id=session_id, generation=gen, category=cat_norm
        )
        if not winners and not losers:
            winners, losers = await self.memory.get_session_performance_split(
                session_id=session_id, generation=None, category=cat_norm
            )

        all_candidates = eligible if eligible else (winners + losers)
        batch_size = len(all_candidates)
        sharpes = [float(c["sharpe"]) for c in all_candidates]
        avg_sharpe = round(sum(sharpes) / len(sharpes), 3) if sharpes else 0.0
        best_sharpe = max(sharpes) if sharpes else 0.0

        # P1-12: Record persistent researcher-specific episodic memory
        if all_candidates:
            try:
                await self.memory.record_batch_episodes_from_candidates(
                    session_id=session_id,
                    generation=gen,
                    candidates_with_evals=all_candidates,
                    selection_split=(winners, losers),
                )
            except Exception as ep_exc:
                log.warning("learning.record_episodes_failed", session_id=session_id, error=str(ep_exc))

        # P1-09: Operator statistics with controlled attribution, shrinkage, and evidence gating
        op_stats = await self.memory.compute_operator_stats(session_id, category=cat_norm)
        # P1-09: Select preferred operators by Lower Confidence Bound (conservative estimate of positive lift)
        # and avoid operators by Upper Confidence Bound (conservative estimate of negative lift) with minimum sample gate >= 3
        eligible_ops = [op for op in op_stats.values() if op.appearances >= 3]
        best_ops = sorted(
            [op for op in eligible_ops if op.evidence_status == "POSITIVE_ASSOCIATION"],
            key=lambda o: (o.lower_confidence_bound if o.lower_confidence_bound is not None else o.shrunk_lift),
            reverse=True,
        )[:3]
        worst_ops = sorted(
            [op for op in eligible_ops if op.evidence_status == "NEGATIVE_ASSOCIATION"],
            key=lambda o: (o.upper_confidence_bound if o.upper_confidence_bound is not None else o.shrunk_lift),
        )[:3]

        # Check Insufficient Evidence threshold (batch_size < min_candidates)
        if batch_size < min_candidates:
            log.info(
                "learning.insufficient_evidence",
                session_id=session_id,
                generation=gen,
                category=cat_norm,
                batch_size=batch_size,
                min_candidates=min_candidates,
            )
            async with self.db.session() as session:
                ins_row = ResearchInsight(
                    session_id=session_id,
                    generation=gen,
                    insight_type="insufficient_evidence",
                    summary=f"Insufficient evidence for {cat_norm} in generation {gen}: batch size {batch_size} < {min_candidates}.",
                    rules={},
                    evidence={},
                    batch_size=batch_size,
                    batch_avg_sharpe=avg_sharpe,
                    batch_best_sharpe=best_sharpe,
                    batch_valid_rate=1.0 if batch_size > 0 else 0.0,
                    applied=False,
                    prompt_strategy_id=antecedent_strategy_id,
                    evolved_strategy_id=None,
                    model_ref=model_ref,
                    execution_status="INSUFFICIENT_EVIDENCE",
                    is_fallback=False,
                    evidence_candidate_ids=[c["candidate_id"] for c in all_candidates],
                    error_details=None,
                )
                session.add(ins_row)

            return LearningCycleResult(
                session_id=session_id,
                generation=gen,
                category=cat_norm,
                batch_size=batch_size,
                winner_count=len(winners),
                loser_count=len(losers),
                avg_sharpe=avg_sharpe,
                best_sharpe=best_sharpe,
                insights_generated=[{
                    "insight_type": "insufficient_evidence",
                    "summary": f"Insufficient evidence for {cat_norm}: batch size {batch_size} < {min_candidates}",
                }],
                updated_prompt_strategy_id=None,
                status="INSUFFICIENT_EVIDENCE",
            )

        # 2. Attempt genuine LLM reflection
        structured_reflection: dict[str, Any] | None = None
        reflection_error: str | None = None
        is_fallback = False

        if self.llm is not None:
            factual_stats = {
                "batch_size": batch_size,
                "avg_sharpe": avg_sharpe,
                "best_sharpe": best_sharpe,
                "valid_rate": 1.0,
                # P1-10: Multiplicity context for reflection prompt
                "multiplicity_trials_count": winners[0]["multiplicity_trials_count"] if winners else batch_size,
                "cohort_baseline_sharpe": winners[0]["cohort_baseline_sharpe"] if winners else avg_sharpe,
                "winner_selection_method": "p1_10_bh_fdr",
                "fdr_threshold": 0.15,
                "operator_stats": [
                    {
                        "name": op.name,
                        "appearances": op.appearances,
                        "avg_sharpe": op.avg_sharpe,
                        "shrunk_lift": op.shrunk_lift,
                        "confidence_level": op.confidence_level,
                        "evidence_status": op.evidence_status,
                    }
                    for op in op_stats.values()
                ],
            }
            user_prompt = render_reflection_evidence_prompt(
                session_id=session_id,
                generation=gen,
                eligible_candidates=all_candidates,
                factual_stats=factual_stats,
            )
            try:
                answer = await self.llm.generate(
                    system=REFLECTION_SYSTEM_PROMPT,
                    user=user_prompt,
                    model_ref=model_ref,
                    response_schema=REFLECTION_RESPONSE_SCHEMA,
                    temperature=0.2,
                )
                parsed = _extract_json(answer.text)
                if not parsed:
                    raise SchemaValidationError(f"Could not extract JSON from reflection output: {answer.text[:200]}")
                _validate_reflection_schema(parsed)
                validate_citations(parsed, all_candidates)
                structured_reflection = parsed
            except (
                BudgetExhaustedError,
                LLMError,
                ProviderError,
                httpx.TimeoutException,
                httpx.HTTPError,
                asyncio.TimeoutError,
                json.JSONDecodeError,
                CitationValidationError,
                SchemaValidationError,
            ) as exc:
                log.warning("learning.reflection_fallback", error=str(exc))
                reflection_error = f"{type(exc).__name__}: {str(exc)}"
                is_fallback = True
        else:
            is_fallback = True
            reflection_error = "No LLM service configured"

        # 3. Formulate insights and evolve prompt strategy atomically
        updated_strategy_id: int | None = None
        insights_data: list[dict[str, Any]] = []

        async with self.db.session() as session:
            # P1-10: Few-shot examples sourced strictly from FDR-verified SIGNIFICANT_WINNERs
            few_shot_examples = [
                {
                    "hypothesis": w["hypothesis"],
                    "expression": w["expression"],
                    "sharpe": w["sharpe"],
                    "selection_tier": "SIGNIFICANT_WINNER",
                    "fdr_q_value": w.get("fdr_q_value"),
                }
                for w in winners[:3]
                if w.get("selection_tier") == "SIGNIFICANT_WINNER"
            ]

            # P1-10: Identify provisional candidates for separate, explicitly labeled learning mechanism
            provisional_candidates = [
                c for c in all_candidates if c.get("selection_tier") == "PROVISIONAL"
            ]
            provisional_examples = [
                {
                    "expression": p["expression"],
                    "hypothesis": p.get("hypothesis"),
                    "sharpe": p.get("sharpe"),
                    "selection_tier": "PROVISIONAL",
                    "label": "[Provisional Candidate / Insufficient Statistical Evidence]",
                }
                for p in provisional_candidates[:3]
            ]

            if not is_fallback and structured_reflection is not None:
                execution_status = "LLM_GROUNDED"
                reflection_summary = structured_reflection.get("summary", "")
                hypotheses = structured_reflection.get("hypotheses", [])
                recommendations = structured_reflection.get("recommendations", [])
                model_anti_patterns = structured_reflection.get("anti_patterns", [])

                anti_patterns = [
                    {
                        "flaw": f"[Unverified Hypothesis] {ap.get('flaw', 'Underperforming pattern')}",
                        "expression": ap.get("expression", ""),
                    }
                    for ap in model_anti_patterns
                ]
                if not anti_patterns and losers:
                    anti_patterns = [
                        {"flaw": "High turnover or decay", "expression": l["expression"]}
                        for l in losers[:3]
                    ]

                unverified_recs = [
                    f"[Unverified Recommendation] {r.get('action')}: {r.get('rationale')}"
                    for r in recommendations
                ]
                guidance = {
                    "preferred_operators": [o.name for o in best_ops],
                    "avoid_operators": [o.name for o in worst_ops],
                    "unverified_recommendations": unverified_recs,
                    "strategy": " | ".join(unverified_recs) if unverified_recs else None,
                    "operator_evidence_method": "p1_09_v1",
                    "winner_selection_method": "p1_10_bh_fdr",
                    "provisional_hypotheses": provisional_examples,
                }

                insight_row = ResearchInsight(
                    session_id=session_id,
                    generation=gen,
                    insight_type="llm_reflection",
                    summary=reflection_summary,
                    rules={"hypotheses": hypotheses, "recommendations": recommendations},
                    evidence={"anti_patterns": model_anti_patterns},
                    batch_size=batch_size,
                    batch_avg_sharpe=avg_sharpe,
                    batch_best_sharpe=best_sharpe,
                    batch_valid_rate=1.0,
                    applied=True,
                    prompt_strategy_id=antecedent_strategy_id,
                    model_ref=model_ref,
                    execution_status=execution_status,
                    is_fallback=False,
                    evidence_candidate_ids=[c["candidate_id"] for c in all_candidates],
                    structured_output=structured_reflection,
                    error_details=None,
                )
                session.add(insight_row)
                insights_data.append({
                    "insight_type": "llm_reflection",
                    "summary": reflection_summary,
                    "hypotheses_count": len(hypotheses),
                    "recommendations_count": len(recommendations),
                })
                strat_name = f"LLM Evolved {cat_norm.title()} Strategy Gen {gen+1}"
            else:
                execution_status = "FALLBACK_DETERMINISTIC"
                anti_patterns = [
                    {
                        "flaw": "High turnover or decay",
                        "expression": l["expression"],
                    }
                    for l in losers[:3]
                ]
                guidance = {
                    "preferred_operators": [o.name for o in best_ops],
                    "avoid_operators": [o.name for o in worst_ops],
                    "operator_evidence_method": "p1_09_v1",
                    "winner_selection_method": "p1_10_bh_fdr",
                    "provisional_hypotheses": provisional_examples,
                }

                fb_summary = f"Generation {gen} deterministic fallback reflection for {cat_norm}."
                if winners:
                    fb_summary += f" Top signal Sharpe {winners[0]['sharpe']:.2f}."
                if losers:
                    fb_summary += f" Observed {len(losers)} underperforming signals."

                insight_row = ResearchInsight(
                    session_id=session_id,
                    generation=gen,
                    insight_type="fallback_heuristic",
                    summary=fb_summary,
                    rules=guidance,
                    evidence={"top_expression": winners[0]["expression"] if winners else None},
                    batch_size=batch_size,
                    batch_avg_sharpe=avg_sharpe,
                    batch_best_sharpe=best_sharpe,
                    batch_valid_rate=1.0,
                    applied=True,
                    prompt_strategy_id=antecedent_strategy_id,
                    model_ref=model_ref,
                    execution_status=execution_status,
                    is_fallback=True,
                    evidence_candidate_ids=[c["candidate_id"] for c in all_candidates],
                    structured_output=None,
                    error_details=reflection_error,
                )
                session.add(insight_row)
                insights_data.append({
                    "insight_type": "fallback_heuristic",
                    "summary": fb_summary,
                })
                strat_name = f"Evolved {cat_norm.title()} Strategy Gen {gen+1} (Fallback)"

            # Atomically deactivate prior active strategies only if promoting directly
            if not require_validation:
                await session.execute(
                    update(PromptStrategy)
                    .where(
                        PromptStrategy.session_id == session_id,
                        func.lower(func.trim(PromptStrategy.category)) == cat_norm,
                        PromptStrategy.is_active.is_(True),
                    )
                    .values(is_active=False, updated_at=utcnow())
                )

            new_version = (curr_strat.version + 1) if curr_strat and curr_strat.version else 1

            # Resolve template text from prior strategy or canonical seed
            base_tmpl_text = curr_strat.template_text if (curr_strat and curr_strat.template_text) else None
            if not base_tmpl_text:
                cat_enum = (
                    StrategyCategory(cat_norm)
                    if cat_norm in StrategyCategory._value2member_map_
                    else StrategyCategory.GENERAL
                )
                seed_tmpl = SEED_TEMPLATES.get(cat_enum) or SEED_TEMPLATES[StrategyCategory.GENERAL]
                base_tmpl_text = seed_tmpl.template_text

            t_text, n_guidance, n_examples, n_anti_patterns = normalize_prompt_components(
                template_text=base_tmpl_text,
                guidance=guidance,
                examples=few_shot_examples,
                anti_patterns=anti_patterns,
            )
            c_hash = compute_prompt_content_hash(
                template_text=t_text,
                guidance=n_guidance,
                examples=n_examples,
                anti_patterns=n_anti_patterns,
            )

            new_strategy = PromptStrategy(
                session_id=session_id,
                name=strat_name,
                version=new_version,
                category=cat_norm,
                parent_id=curr_strat.id if curr_strat else None,
                template_text=t_text,
                guidance=guidance,
                examples=few_shot_examples,
                anti_patterns=anti_patterns,
                avg_sharpe=avg_sharpe,
                is_active=False if require_validation else True,
                validation_status="EXPERIMENTAL" if require_validation else "PROMOTED",
                previous_promoted_strategy_id=curr_strat.id if (curr_strat and not require_validation) else None,
                previous_promoted_version_id=curr_strat.current_prompt_version_id if (curr_strat and not require_validation) else None,
                promoted_at=utcnow() if not require_validation else None,
            )
            session.add(new_strategy)
            await session.flush()

            parent_pv_id = curr_strat.current_prompt_version_id if curr_strat else None
            new_pv = PromptVersion(
                origin_session_id=session_id,
                origin_strategy_id=new_strategy.id,
                category=cat_norm,
                version=new_version,
                authoring_source="EVOLUTION",
                template_text=t_text,
                guidance=n_guidance,
                examples=n_examples,
                anti_patterns=n_anti_patterns,
                content_hash=c_hash,
                parent_version_id=parent_pv_id,
            )
            session.add(new_pv)
            await session.flush()

            new_strategy.current_prompt_version_id = new_pv.id
            updated_strategy_id = new_strategy.id

            # Traceable lifecycle audit event
            event = PromptLifecycleEvent(
                session_id=session_id,
                category=cat_norm,
                strategy_id=new_strategy.id,
                prompt_version_id=new_pv.id,
                event_type="CREATED_EXPERIMENTAL" if require_validation else "PROMOTED",
                from_status=None if require_validation else "EXPERIMENTAL",
                to_status="EXPERIMENTAL" if require_validation else "PROMOTED",
                reason=(
                    "Evolved prompt generated as EXPERIMENTAL pending matched-cohort validation."
                    if require_validation
                    else "Evolved prompt promoted directly."
                ),
            )
            session.add(event)

            # Set evolved_strategy_id on the insight row
            insight_row.evolved_strategy_id = updated_strategy_id

            if advance_generation:
                # Advance session generation
                sess_up = (
                    update(ResearchSession)
                    .where(ResearchSession.id == session_id)
                    .values(
                        current_generation=gen + 1,
                        updated_at=utcnow(),
                    )
                )
                await session.execute(sess_up)

        log.info(
            "learning.cycle_complete",
            session_id=session_id,
            category=cat_norm,
            generation=gen,
            execution_status=execution_status,
            is_fallback=is_fallback,
            winners=len(winners),
            losers=len(losers),
            insights=len(insights_data),
        )

        return LearningCycleResult(
            session_id=session_id,
            generation=gen,
            category=cat_norm,
            batch_size=batch_size,
            winner_count=len(winners),
            loser_count=len(losers),
            avg_sharpe=avg_sharpe,
            best_sharpe=best_sharpe,
            insights_generated=insights_data,
            updated_prompt_strategy_id=updated_strategy_id,
            status="COMPLETED",
        )
