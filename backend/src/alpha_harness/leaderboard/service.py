"""Multi-axis quantitative leaderboard and experimental research ranking service.

Treats the leaderboard as a live experimental instrument ranking research assets across
four dimensions:
1. Alpha Candidate Leaderboard (Composite Score, After-Cost Sharpe, OOS Stability)
2. LLM Researcher Leaderboard (Yield rate, top-decile alpha rate, avg Sharpe)
3. Prompt Strategy Leaderboard (Prompt version progression and Sharpe lift)
4. Historical Trend Archival (LeaderboardSnapshot persistence across generations)
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import desc, func, or_, select

from ..db.models import (
    CandidateStatus,
    LeaderboardSnapshot,
    PromptStrategy,
    PromptVersion,
    ResearchCandidate,
    ResearchEvaluation,
    ResearchResearcher,
    ResearchSession,
    utcnow,
)
from ..learning.memory import _estimate_observation_years, _multiplicity_corrected_split
from ..prompts.library import normalize_category
from ..prompts.versioning import verify_prompt_integrity

if TYPE_CHECKING:
    from ..db.sqlite import Database

log = structlog.get_logger(__name__)


@dataclass(slots=True)
class AlphaRankEntry:
    rank: int
    candidate_id: int
    expression: str
    hypothesis: str
    model_label: str
    generation: int
    composite_score: float
    full_sharpe: float
    after_cost_sharpe: float
    oos_sharpe: float
    sharpe_decay: float
    turnover: float
    fitness: float
    is_submission_grade: bool
    evaluation_scheme: str = "TWO_WAY_LEGACY"
    is_graduated: bool = False
    val_composite_score: float | None = None
    composite_score_spec: dict[str, Any] | None = None
    val_sharpe: float | None = None
    val_after_cost_sharpe: float | None = None
    val_fitness: float | None = None
    val_turnover: float | None = None
    holdout_sharpe: float | None = None


@dataclass(slots=True)
class ResearcherRankEntry:
    # Existing fields preserved for backward compatibility
    rank: int | None
    researcher_id: int
    label: str
    model_ref: str
    strategy_focus: str
    total_generated: int
    total_valid: int
    valid_rate: float
    avg_sharpe: float
    best_sharpe: float
    submission_grade_count: int
    top_decile_rate: float

    # P1-11 Fair Benchmarking & Multi-Stage Funnel Fields
    status: str = "RANKED"  # "RANKED", "INSUFFICIENT_EVIDENCE", "INCOMPARABLE"
    cohort_id: str | None = None
    sample_count: int = 0
    mean_validation_sharpe: float | None = None
    cohort_baseline_sharpe: float | None = None
    relative_lift: float | None = None
    shrunk_relative_lift: float | None = None
    sharpe_se: float | None = None
    lcb_sharpe: float | None = None

    # Transparent Multi-Stage Funnel
    ast_valid_count: int = 0
    ast_valid_rate: float | None = None
    preflight_pass_count: int = 0
    preflight_pass_rate: float | None = None
    simulated_count: int = 0
    simulation_pass_count: int = 0
    simulation_pass_rate: float | None = None
    statistical_winner_count: int = 0
    statistical_discovery_rate: float | None = None
    verified_provenance_count: int = 0
    verified_provenance_rate: float | None = None

    # Efficiency & Resource Metrics (Separated from Statistical Quality)
    total_tokens: int = 0
    efficiency_cost_usd: float | None = None
    efficiency_latency_sec: float | None = None
    incomparability_reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PromptStrategyRankEntry:
    rank: int
    strategy_id: int
    name: str
    version: int
    category: str
    candidate_count: int
    avg_sharpe: float
    top_sharpe: float
    valid_rate: float


@dataclass(slots=True)
class SimulationCohort:
    cohort_id: str
    category: str
    region: str
    universe: str
    neutralization: str
    decay: int
    delay: int
    truncation: float
    nan_handling: str
    pasteurization: str
    unit_handling: str
    cost_bps: float
    evaluation_scheme: str
    val_start: str
    val_end: str
    datasets: list[str]
    candidate_count: int


@dataclass(slots=True)
class PromptFunnelMetrics:
    generated: int
    valid: int
    evaluated: int
    accepted: int
    valid_pct: float
    evaluated_pct: float
    accepted_pct: float


@dataclass(slots=True)
class PromptValidationMetrics:
    median_sharpe: float | None
    mean_sharpe: float | None
    best_sharpe: float | None
    median_fitness: float | None
    median_turnover: float | None


@dataclass(slots=True)
class PromptLeaderboardEntry:
    prompt_version_id: int
    category: str
    version: int
    authoring_source: str
    content_hash: str
    parent_version_id: int | None
    cohort_id: str | None
    rank: int | None
    status: str  # "RANKED", "INSUFFICIENT_SAMPLE", "INCOMPARABLE"
    incomparability_reasons: list[str]
    funnel: PromptFunnelMetrics
    validation_metrics: PromptValidationMetrics


@dataclass(slots=True)
class PromptLeaderboardResult:
    active_cohorts: list[SimulationCohort]
    rankings: list[PromptLeaderboardEntry]
    incomparable_records: list[PromptLeaderboardEntry]


def compute_cohort_fingerprint(
    category: str,
    eval_row: ResearchEvaluation,
    cand: ResearchCandidate,
) -> tuple[str | None, list[str], dict[str, Any]]:
    """Computes deterministic 15-dimension cohort fingerprint from canonical_spec and evaluation row.

    Returns:
        (cohort_id, incomparability_reasons, cohort_dimensions_dict)
        If incomplete or conflicting, cohort_id is None and reasons contains specific diagnostics.
    """
    reasons: list[str] = []
    spec = eval_row.canonical_spec or {}
    sim = spec.get("sim_settings", {})
    dates = spec.get("dates", {})

    # 1. Category
    norm_category = normalize_category(category)

    # 2. Simulation settings
    region = sim.get("region") or cand.sim_region
    universe = sim.get("universe") or cand.sim_universe
    neutralization = sim.get("neutralization") or cand.sim_neutralization
    decay = sim.get("decay") if "decay" in sim else cand.sim_decay
    delay = sim.get("delay") if "delay" in sim else cand.sim_delay
    truncation = sim.get("truncation")
    nan_handling = sim.get("nan_handling")
    pasteurization = sim.get("pasteurization")
    unit_handling = sim.get("unit_handling")

    # 3. Cost assumptions with conflict detection
    spec_cost = spec.get("cost_bps")
    eval_cost = eval_row.cost_bps
    if spec_cost is not None and eval_cost is not None and abs(float(spec_cost) - float(eval_cost)) > 1e-4:
        reasons.append(f"conflicting:cost_bps (canonical={spec_cost}, eval={eval_cost})")
        cost_bps = None
    else:
        cost_bps = spec_cost if spec_cost is not None else eval_cost

    # 4. Evaluation scheme & dates
    eval_scheme = spec.get("evaluation_scheme") or eval_row.evaluation_scheme
    val_start = dates.get("val_start")
    val_end = dates.get("val_end")

    if not val_start:
        if eval_scheme in ("SPLIT_3WAY_FINAL", "SPLIT_3WAY_DEV") and eval_row.is_end_date:
            val_start = eval_row.is_end_date.isoformat()
        else:
            reasons.append("missing:val_start")

    if not val_end:
        if eval_row.val_end_date:
            val_end = eval_row.val_end_date.isoformat()
        else:
            reasons.append("missing:val_end")

    # 5. Datasets scope
    raw_datasets = spec.get("datasets_used") or cand.datasets_used
    if raw_datasets is None:
        reasons.append("missing:datasets_used")
        datasets = None
    else:
        datasets = tuple(sorted(str(d) for d in raw_datasets))

    # Check for missing required dimensions
    dim_checks = [
        ("region", region),
        ("universe", universe),
        ("neutralization", neutralization),
        ("decay", decay),
        ("delay", delay),
        ("truncation", truncation),
        ("nan_handling", nan_handling),
        ("pasteurization", pasteurization),
        ("unit_handling", unit_handling),
        ("cost_bps", cost_bps),
        ("eval_scheme", eval_scheme),
    ]
    for name, val in dim_checks:
        if val is None:
            reasons.append(f"missing:{name}")

    cohort_dict = {
        "category": norm_category,
        "cost_bps": float(cost_bps) if cost_bps is not None else 5.0,
        "datasets": list(datasets) if datasets is not None else [],
        "decay": int(decay) if decay is not None else 4,
        "delay": int(delay) if delay is not None else 1,
        "evaluation_scheme": str(eval_scheme) if eval_scheme is not None else "SPLIT_3WAY_FINAL",
        "nan_handling": str(nan_handling) if nan_handling is not None else "OFF",
        "neutralization": str(neutralization) if neutralization is not None else "INDUSTRY",
        "pasteurization": str(pasteurization) if pasteurization is not None else "ON",
        "region": str(region) if region is not None else "USA",
        "truncation": float(truncation) if truncation is not None else 0.08,
        "unit_handling": str(unit_handling) if unit_handling is not None else "VERIFY",
        "universe": str(universe) if universe is not None else "TOP3000",
        "val_end": str(val_end) if val_end is not None else "",
        "val_start": str(val_start) if val_start is not None else "",
    }

    if reasons:
        return None, reasons, cohort_dict

    payload = {
        "category": norm_category,
        "cost_bps": float(cost_bps),
        "datasets": list(datasets),
        "decay": int(decay),
        "delay": int(delay),
        "eval_scheme": str(eval_scheme),
        "nan_handling": str(nan_handling),
        "neutralization": str(neutralization),
        "pasteurization": str(pasteurization),
        "region": str(region),
        "truncation": float(truncation),
        "unit_handling": str(unit_handling),
        "universe": str(universe),
        "val_end": str(val_end),
        "val_start": str(val_start),
    }
    cohort_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]

    return cohort_hash, [], cohort_dict


class LeaderboardService:
    """Computes dynamic multi-dimensional rankings and persists generation snapshots."""

    def __init__(self, db: Database) -> None:
        self.db = db

    async def get_alpha_leaderboard(
        self,
        session_id: int | None = None,
        min_sharpe: float | None = None,
        submission_grade_only: bool = False,
        limit: int = 50,
    ) -> list[AlphaRankEntry]:
        """Ranks individual alpha candidates by Composite Score descending."""
        async with self.db.session() as session:
            score_order = func.coalesce(
                ResearchEvaluation.val_composite_score,
                ResearchEvaluation.composite_score,
                0.0,
            )
            stmt = (
                select(
                    ResearchCandidate,
                    ResearchEvaluation,
                    ResearchResearcher.label.label("researcher_label"),
                )
                .join(
                    ResearchEvaluation,
                    ResearchEvaluation.candidate_id == ResearchCandidate.id,
                )
                .outerjoin(
                    ResearchResearcher,
                    ResearchResearcher.id == ResearchCandidate.researcher_id,
                )
                .where(
                    ResearchEvaluation.is_verified.is_(True),
                    score_order >= -1.7976931348623157e308,
                    score_order <= 1.7976931348623157e308,
                    ResearchCandidate.status.in_(
                        [
                            CandidateStatus.EVALUATED,
                            CandidateStatus.ACCEPTED,
                            CandidateStatus.GRADUATED,
                        ]
                    ),
                )
            )

            if session_id is not None:
                stmt = stmt.where(ResearchCandidate.session_id == session_id)
            if min_sharpe is not None:
                stmt = stmt.where(
                    func.coalesce(ResearchEvaluation.val_sharpe, ResearchEvaluation.full_sharpe)
                    >= min_sharpe
                )
            if submission_grade_only:
                stmt = stmt.where(
                    or_(ResearchEvaluation.internal_checks.is_not(None), ResearchEvaluation.checks.is_(None)),
                    ResearchEvaluation.checks_passed == ResearchEvaluation.checks_total,
                    ResearchEvaluation.checks_total > 0,
                )

            stmt = stmt.order_by(
                desc(score_order),
                ResearchCandidate.id.asc(),
            ).limit(limit)
            res = await session.execute(stmt)
            rows = res.all()

            entries: list[AlphaRankEntry] = []
            for idx, (cand, ev, r_label) in enumerate(rows, start=1):
                is_3way = ev.evaluation_scheme in ("THREE_WAY_CHRONOLOGICAL", "SPLIT_3WAY_FINAL")
                comp_score = (
                    ev.val_composite_score
                    if (is_3way and ev.val_composite_score is not None)
                    else (ev.composite_score or 0.0)
                )
                if not math.isfinite(float(comp_score)):
                    log.warning("leaderboard.skip_non_finite_composite", candidate_id=cand.id)
                    continue
                after_cost = (
                    ev.val_after_cost_sharpe
                    if (is_3way and ev.val_after_cost_sharpe is not None)
                    else (ev.after_cost_sharpe or 0.0)
                )
                oos_s = (
                    ev.val_sharpe
                    if (is_3way and ev.val_sharpe is not None)
                    else (ev.oos_sharpe or 0.0)
                )
                decay = (
                    ev.is_val_decay
                    if (is_3way and ev.is_val_decay is not None)
                    else (ev.sharpe_decay or 0.0)
                )
                to_val = (
                    ev.val_turnover
                    if (is_3way and ev.val_turnover is not None)
                    else (ev.is_turnover or 0.0)
                )
                fit_val = (
                    ev.val_fitness
                    if (is_3way and ev.val_fitness is not None)
                    else (ev.full_fitness or 0.0)
                )
                ranking_metrics = (
                    comp_score,
                    ev.full_sharpe or 0.0,
                    after_cost,
                    oos_s,
                    decay,
                    to_val,
                    fit_val,
                )
                if any(not math.isfinite(float(value)) for value in ranking_metrics) or to_val < 0:
                    log.warning("leaderboard.skip_invalid_metrics", candidate_id=cand.id)
                    continue
                # Final holdout metrics are revealed ONLY if candidate is graduated
                h_sharpe = ev.holdout_sharpe if cand.status == CandidateStatus.GRADUATED else None
                if h_sharpe is not None and not math.isfinite(float(h_sharpe)):
                    h_sharpe = None

                entries.append(
                    AlphaRankEntry(
                        rank=idx,
                        candidate_id=cand.id,
                        expression=cand.expression or "",
                        hypothesis=cand.hypothesis or "",
                        model_label=r_label or "Unknown Researcher",
                        generation=cand.generation or 0,
                        composite_score=round(float(comp_score), 3),
                        full_sharpe=round(float(ev.full_sharpe or 0.0), 2),
                        after_cost_sharpe=round(float(after_cost), 2),
                        oos_sharpe=round(float(oos_s), 2),
                        sharpe_decay=round(float(decay), 2),
                        turnover=round(float(to_val), 3),
                        fitness=round(float(fit_val), 2),
                        is_submission_grade=bool(
                            (ev.internal_checks is not None or ev.checks is None)
                            and ev.checks_passed == ev.checks_total and (ev.checks_total or 0) > 0
                        ),
                        evaluation_scheme=ev.evaluation_scheme or "TWO_WAY_LEGACY",
                        is_graduated=(cand.status == CandidateStatus.GRADUATED),
                        val_composite_score=(
                            round(float(ev.val_composite_score), 3)
                            if ev.val_composite_score is not None
                            else None
                        ),
                        composite_score_spec=ev.composite_score_spec,
                        val_sharpe=round(float(ev.val_sharpe), 2) if ev.val_sharpe is not None else None,
                        val_after_cost_sharpe=(
                            round(float(ev.val_after_cost_sharpe), 2)
                            if ev.val_after_cost_sharpe is not None
                            else None
                        ),
                        val_fitness=(
                            round(float(ev.val_fitness), 2)
                            if ev.val_fitness is not None
                            else None
                        ),
                        val_turnover=(
                            round(float(ev.val_turnover), 3)
                            if ev.val_turnover is not None
                            else None
                        ),
                        holdout_sharpe=round(float(h_sharpe), 2) if h_sharpe is not None else None,
                    )
                )

            for rank, entry in enumerate(entries, start=1):
                entry.rank = rank
            return entries

    async def get_researcher_leaderboard(
        self,
        session_id: int | None = None,
        cohort_id: str | None = None,
        min_sample_size: int = 5,
        shrinkage_k: float = 5.0,
    ) -> list[ResearcherRankEntry]:
        """P1-11: Ranks LLM researchers using matched-cohort relative performance,
        Empirical Bayes shrinkage, and uncertainty-aware Lower Confidence Bounds (LCB95).

        Acceptance Criteria Enforced:
        1. Matched Cohort Comparisons: Researchers are compared within genuinely comparable
           SimulationCohorts (same universe, region, neutralization, decay, delay, evaluation window).
           If researchers have no matched cohort or insufficient overlap, they are marked
           INSUFFICIENT_EVIDENCE or INCOMPARABLE rather than silently pooled.
        2. Volume-Bias Elimination: High candidate volume alone cannot elevate a model's rank.
           Ranking is determined by conservative LCB95 of shrunk relative lift, and discovery rate,
           NOT raw submission_grade_count.
        3. Multi-Stage Evaluation Funnel: Transparent counts and explicit rates for AST validity,
           preflight validation, simulation passage, statistical discovery, and verified provenance.
        4. Uncertainty Quantification: Empirical Bayes shrinkage towards zero prior (k=5.0) and
           LCB95 = shrunk_lift - 1.96 * SE prevent small-sample fluke rankings. Small samples (N < 5)
           are explicitly flagged as INSUFFICIENT_EVIDENCE.
        5. Quality / Efficiency Separation: Token usage, latency, and resource costs are reported
           independently and not blended into an opaque quality score.
        6. Provenance & Holdout Protection (P0-01, P0-03): Only verified BRAIN evaluations
           (is_verified=True) and validation partition metrics (val_sharpe) are used. Holdout data
           is strictly quarantined.
        """
        async with self.db.session() as session:
            stmt = select(ResearchResearcher)
            if session_id is not None:
                stmt = stmt.where(ResearchResearcher.session_id == session_id)

            res = await session.execute(stmt)
            researchers = list(res.scalars().all())
            if not researchers:
                return []

            # Query all candidates and evaluations for these researchers
            cand_stmt = (
                select(ResearchCandidate, ResearchEvaluation)
                .outerjoin(
                    ResearchEvaluation,
                    ResearchEvaluation.candidate_id == ResearchCandidate.id,
                )
                .where(ResearchCandidate.researcher_id.in_([r.id for r in researchers]))
            )
            if session_id is not None:
                cand_stmt = cand_stmt.where(ResearchCandidate.session_id == session_id)

            c_res = await session.execute(cand_stmt)
            cand_eval_pairs = list(c_res.all())

            # Group candidates by researcher
            cands_by_researcher: dict[int, list[tuple[ResearchCandidate, ResearchEvaluation | None]]] = {
                r.id: [] for r in researchers
            }
            for c, ev in cand_eval_pairs:
                if c.researcher_id in cands_by_researcher:
                    cands_by_researcher[c.researcher_id].append((c, ev))

            # Step 1: Compute transparent evaluation funnel and efficiency metrics per researcher
            funnel_data: dict[int, dict[str, Any]] = {}
            for r in researchers:
                pairs = cands_by_researcher[r.id]
                tot_gen = r.total_generated if (r.total_generated and r.total_generated >= len(pairs)) else len(pairs)

                ast_valid = sum(1 for c, _ in pairs if c.is_valid is True)
                ast_valid_rate = round(ast_valid / tot_gen, 4) if tot_gen > 0 else None

                preflight_pass = sum(
                    1 for c, _ in pairs
                    if c.is_valid is True and c.status in (
                        CandidateStatus.EVALUATED,
                        CandidateStatus.ACCEPTED,
                        CandidateStatus.GRADUATED,
                    )
                )
                preflight_pass_rate = round(preflight_pass / ast_valid, 4) if ast_valid > 0 else None

                simulated = sum(1 for _, ev in pairs if ev is not None)
                simulation_pass = sum(
                    1 for _, ev in pairs
                    if ev is not None and ev.checks_passed == ev.checks_total and (ev.checks_total or 0) > 0
                )
                simulation_pass_rate = round(simulation_pass / simulated, 4) if simulated > 0 else None

                # P0-01 Provenance check: verified BRAIN evaluations
                verified_prov = sum(
                    1 for _, ev in pairs
                    if ev is not None and ev.is_verified is True and ev.provenance == "BRAIN_VAULT"
                )
                verified_prov_rate = round(verified_prov / simulated, 4) if simulated > 0 else None

                tot_tokens = r.total_tokens or 0
                cost_usd = round(tot_tokens * 0.000002, 4) if tot_tokens > 0 else None

                funnel_data[r.id] = {
                    "total_generated": tot_gen,
                    "total_valid": ast_valid,
                    "valid_rate": round(ast_valid / tot_gen, 2) if tot_gen else 0.0,
                    "ast_valid_count": ast_valid,
                    "ast_valid_rate": ast_valid_rate,
                    "preflight_pass_count": preflight_pass,
                    "preflight_pass_rate": preflight_pass_rate,
                    "simulated_count": simulated,
                    "simulation_pass_count": simulation_pass,
                    "simulation_pass_rate": simulation_pass_rate,
                    "verified_provenance_count": verified_prov,
                    "verified_provenance_rate": verified_prov_rate,
                    "total_tokens": tot_tokens,
                    "efficiency_cost_usd": cost_usd,
                    "efficiency_latency_sec": None,
                }

            # Step 2: Stratified Simulation Cohort Mapping
            # Group valid, verified evaluations by cohort fingerprint
            cohort_candidates: dict[str, list[tuple[int, ResearchCandidate, ResearchEvaluation, float]]] = {}
            incomparable_reasons_by_r: dict[int, list[str]] = {r.id: [] for r in researchers}

            for r in researchers:
                pairs = cands_by_researcher[r.id]
                for c, ev in pairs:
                    if not ev or not ev.is_verified:
                        continue
                    # P0-03 holdout isolation: strictly val_sharpe (or is_sharpe for unpartitioned legacy)
                    v_sharpe = ev.val_sharpe if ev.val_sharpe is not None else (ev.is_sharpe if ev.is_sharpe is not None else None)
                    if v_sharpe is None:
                        continue

                    c_hash, reasons, _ = compute_cohort_fingerprint(r.strategy_focus or "general", ev, c)
                    if c_hash is not None:
                        cohort_candidates.setdefault(c_hash, []).append((r.id, c, ev, float(v_sharpe)))
                    else:
                        incomparable_reasons_by_r[r.id].extend(reasons)

            # Step 3: Cohort Selection and Overlap Check
            target_cohort: str | None = None
            if cohort_id is not None:
                if cohort_id in cohort_candidates:
                    target_cohort = cohort_id
            elif cohort_candidates:
                cohort_researchers = {
                    ch: set(item[0] for item in items) for ch, items in cohort_candidates.items()
                }
                best_cohort = max(
                    cohort_candidates.keys(),
                    key=lambda ch: (len(cohort_researchers[ch]), len(cohort_candidates[ch])),
                )
                # Only use matched cohort if at least 2 researchers overlap or only 1 researcher exists in session
                if len(cohort_researchers[best_cohort]) >= min(2, len(researchers)):
                    target_cohort = best_cohort

            # Step 4: Multiplicity correction and statistical discovery in target cohort
            cohort_baseline: float | None = None
            cohort_stat_winners: set[int] = set()

            if target_cohort and target_cohort in cohort_candidates:
                items = cohort_candidates[target_cohort]
                cohort_sharpes = [it[3] for it in items]
                cohort_baseline = statistics.median(cohort_sharpes) if cohort_sharpes else 0.0

                # Run P1-10 multiplicity-corrected split on all cohort candidates
                cohort_records = [
                    {
                        "candidate_id": it[1].id,
                        "researcher_id": it[0],
                        "sharpe": it[3],
                        "t_years": _estimate_observation_years(it[2]),
                        "composite_score": it[2].val_composite_score or 0.0,
                    }
                    for it in items
                ]
                winners, _ = _multiplicity_corrected_split(cohort_records)
                cohort_stat_winners = {
                    w["candidate_id"]
                    for w in winners
                    if w.get("selection_tier") == "SIGNIFICANT_WINNER"
                }

            # Step 5: Build ResearcherRankEntry objects
            entries: list[ResearcherRankEntry] = []
            for r in researchers:
                fd = funnel_data[r.id]

                r_cohort_items = [
                    it for it in cohort_candidates.get(target_cohort, [])
                    if it[0] == r.id
                ] if target_cohort else []

                sample_count = len(r_cohort_items)
                stat_winner_count = sum(1 for it in r_cohort_items if it[1].id in cohort_stat_winners)
                stat_discovery_rate = round(stat_winner_count / sample_count, 4) if sample_count > 0 else None

                all_r_sharpes = [
                    float(ev.val_sharpe if ev.val_sharpe is not None else ev.is_sharpe)
                    for _, ev in cands_by_researcher[r.id]
                    if ev and ev.is_verified and (ev.val_sharpe is not None or ev.is_sharpe is not None)
                ]
                legacy_avg_sharpe = round(statistics.mean(all_r_sharpes), 2) if all_r_sharpes else round(float(r.avg_sharpe or 0.0), 2)
                legacy_best_sharpe = round(max(all_r_sharpes), 2) if all_r_sharpes else round(float(r.best_sharpe or 0.0), 2)

                mean_val_sharpe: float | None = None
                relative_lift: float | None = None
                shrunk_lift: float | None = None
                sharpe_se: float | None = None
                lcb_sharpe: float | None = None

                if target_cohort is None:
                    status = "INSUFFICIENT_EVIDENCE" if not cohort_candidates else "INCOMPARABLE"
                elif sample_count == 0:
                    status = "INCOMPARABLE" if any(it[0] == r.id for items in cohort_candidates.values() for it in items) else "INSUFFICIENT_EVIDENCE"
                elif sample_count < min_sample_size:
                    status = "INSUFFICIENT_EVIDENCE"
                    r_sharpes = [it[3] for it in r_cohort_items]
                    mean_val_sharpe = round(statistics.mean(r_sharpes), 4)
                    if cohort_baseline is not None:
                        relative_lift = round(mean_val_sharpe - cohort_baseline, 4)
                        shrunk_lift = round((sample_count / (sample_count + shrinkage_k)) * relative_lift, 4)
                    s_dev = statistics.stdev(r_sharpes) if sample_count >= 2 else 0.50
                    sharpe_se = round(max(s_dev, 0.10) / (sample_count ** 0.5), 4)
                    if shrunk_lift is not None:
                        lcb_sharpe = round(shrunk_lift - 1.96 * sharpe_se, 4)
                else:
                    status = "RANKED"
                    r_sharpes = [it[3] for it in r_cohort_items]
                    mean_val_sharpe = round(statistics.mean(r_sharpes), 4)
                    if cohort_baseline is not None:
                        relative_lift = round(mean_val_sharpe - cohort_baseline, 4)
                        shrunk_lift = round((sample_count / (sample_count + shrinkage_k)) * relative_lift, 4)

                    s_dev = statistics.stdev(r_sharpes) if sample_count >= 2 else 0.50
                    sharpe_se = round(max(s_dev, 0.05) / (sample_count ** 0.5), 4)
                    if shrunk_lift is not None:
                        lcb_sharpe = round(shrunk_lift - 1.96 * sharpe_se, 4)

                reasons = list(set(incomparable_reasons_by_r.get(r.id, [])))

                entries.append(
                    ResearcherRankEntry(
                        rank=None,
                        researcher_id=r.id,
                        label=r.label or "Researcher",
                        model_ref=r.model_ref or "",
                        strategy_focus=r.strategy_focus or "general",
                        total_generated=fd["total_generated"],
                        total_valid=fd["total_valid"],
                        valid_rate=fd["valid_rate"],
                        avg_sharpe=legacy_avg_sharpe,
                        best_sharpe=legacy_best_sharpe,
                        submission_grade_count=sum(
                            1 for _, ev in cands_by_researcher[r.id]
                            if ev is not None
                            and (ev.internal_checks is not None or ev.checks is None)
                            and ev.checks_passed == ev.checks_total
                            and (ev.checks_total or 0) > 0
                        ),
                        top_decile_rate=round(fd["simulation_pass_count"] / fd["total_generated"], 2) if fd["total_generated"] else 0.0,
                        # P1-11 Fair Benchmarking fields
                        status=status,
                        cohort_id=target_cohort,
                        sample_count=sample_count,
                        mean_validation_sharpe=mean_val_sharpe,
                        cohort_baseline_sharpe=round(cohort_baseline, 4) if cohort_baseline is not None else None,
                        relative_lift=relative_lift,
                        shrunk_relative_lift=shrunk_lift,
                        sharpe_se=sharpe_se,
                        lcb_sharpe=lcb_sharpe,
                        ast_valid_count=fd["ast_valid_count"],
                        ast_valid_rate=fd["ast_valid_rate"],
                        preflight_pass_count=fd["preflight_pass_count"],
                        preflight_pass_rate=fd["preflight_pass_rate"],
                        simulated_count=fd["simulated_count"],
                        simulation_pass_count=fd["simulation_pass_count"],
                        simulation_pass_rate=fd["simulation_pass_rate"],
                        statistical_winner_count=stat_winner_count,
                        statistical_discovery_rate=stat_discovery_rate,
                        verified_provenance_count=fd["verified_provenance_count"],
                        verified_provenance_rate=fd["verified_provenance_rate"],
                        total_tokens=fd["total_tokens"],
                        efficiency_cost_usd=fd["efficiency_cost_usd"],
                        efficiency_latency_sec=fd["efficiency_latency_sec"],
                        incomparability_reasons=reasons,
                    )
                )

            # Step 6: Deterministic Ranking (Volume-Bias Eliminated)
            ranked_entries = [e for e in entries if e.status == "RANKED"]
            unranked_entries = [e for e in entries if e.status != "RANKED"]

            ranked_entries.sort(
                key=lambda e: (
                    e.lcb_sharpe if e.lcb_sharpe is not None else -999.0,
                    e.shrunk_relative_lift if e.shrunk_relative_lift is not None else -999.0,
                    e.statistical_discovery_rate if e.statistical_discovery_rate is not None else -999.0,
                    e.mean_validation_sharpe if e.mean_validation_sharpe is not None else -999.0,
                    -e.researcher_id,
                ),
                reverse=True,
            )

            for idx, entry in enumerate(ranked_entries, start=1):
                entry.rank = idx

            unranked_entries.sort(key=lambda e: (e.sample_count, -e.researcher_id), reverse=True)

            return ranked_entries + unranked_entries

    async def capture_snapshot(
        self, session_id: int, generation: int
    ) -> LeaderboardSnapshot:
        """Captures a snapshot of current rankings for generation trend charting."""
        alpha_ranks = await self.get_alpha_leaderboard(session_id=session_id, limit=20)
        researcher_ranks = await self.get_researcher_leaderboard(session_id=session_id)

        entries_payload = {
            "top_alphas": [
                {
                    "rank": a.rank,
                    "candidate_id": a.candidate_id,
                    "expression": a.expression,
                    "composite_score": a.composite_score,
                    "sharpe": a.full_sharpe,
                    "after_cost_sharpe": a.after_cost_sharpe,
                    "is_submission_grade": a.is_submission_grade,
                }
                for a in alpha_ranks
            ],
            "researchers": [
                {
                    "rank": r.rank,
                    "researcher_id": r.researcher_id,
                    "label": r.label,
                    "model_ref": r.model_ref,
                    "submission_count": r.submission_grade_count,
                    "avg_sharpe": r.avg_sharpe,
                    "status": r.status,
                    "cohort_id": r.cohort_id,
                    "shrunk_relative_lift": r.shrunk_relative_lift,
                    "lcb_sharpe": r.lcb_sharpe,
                    "statistical_winner_count": r.statistical_winner_count,
                    "discovery_rate": r.statistical_discovery_rate,
                }
                for r in researcher_ranks
            ],
        }

        async with self.db.session() as session:
            snap = LeaderboardSnapshot(
                session_id=session_id,
                generation=generation,
                dimension="composite",
                entries=[entries_payload],
                snapshot_at=utcnow(),
            )
            session.add(snap)
            await session.flush()

        log.info(
            "leaderboard.snapshot_captured",
            session_id=session_id,
            generation=generation,
            top_alphas=len(alpha_ranks),
        )
        return snap

    async def get_prompt_strategy_leaderboard(
        self,
        session_id: int | None = None,
        cohort_id: str | None = None,
        category: str | None = None,
        min_evaluated: int = 3,
    ) -> PromptLeaderboardResult:
        """Ranks PromptVersion snapshots across comparable cohorts with pure validation metrics."""
        async with self.db.session() as session:
            # 1. Fetch PromptVersions (filtered by category if specified)
            pv_stmt = select(PromptVersion).order_by(PromptVersion.category.asc(), PromptVersion.version.asc())
            if category is not None:
                pv_stmt = pv_stmt.where(PromptVersion.category == normalize_category(category))
            pv_res = await session.execute(pv_stmt)
            prompt_versions = list(pv_res.scalars().all())

            # Fail-closed integrity check on all queried versions
            for pv in prompt_versions:
                verify_prompt_integrity(pv)

            # 2. Fetch candidates with evaluations (strictly validation metrics only)
            cand_stmt = (
                select(ResearchCandidate)
                .where(ResearchCandidate.prompt_version_id.is_not(None))
            )
            if session_id is not None:
                cand_stmt = cand_stmt.where(ResearchCandidate.session_id == session_id)
            cand_res = await session.execute(cand_stmt)
            all_candidates = list(cand_res.scalars().all())

            # Fetch evaluations for these candidates
            eval_stmt = (
                select(ResearchEvaluation)
                .where(
                    ResearchEvaluation.candidate_id.in_([c.id for c in all_candidates if c.id is not None])
                )
                .order_by(ResearchEvaluation.evaluated_at.desc(), ResearchEvaluation.id.desc())
            ) if all_candidates else None

            eval_map: dict[int, ResearchEvaluation] = {}
            if eval_stmt is not None:
                eval_res = await session.execute(eval_stmt)
                for ev in eval_res.scalars().all():
                    if ev.candidate_id is not None and ev.candidate_id not in eval_map:
                        eval_map[ev.candidate_id] = ev

            # Group candidates by prompt_version_id
            candidates_by_pv: dict[int, list[ResearchCandidate]] = {}
            for c in all_candidates:
                if c.prompt_version_id is not None:
                    candidates_by_pv.setdefault(c.prompt_version_id, []).append(c)

            active_cohorts_map: dict[str, SimulationCohort] = {}
            cohort_pv_evaluated: dict[str, dict[int, list[tuple[ResearchCandidate, ResearchEvaluation]]]] = {}
            incomparable_by_pv: dict[int, list[tuple[ResearchCandidate, ResearchEvaluation, list[str]]]] = {}

            for pv in prompt_versions:
                cands = candidates_by_pv.get(pv.id, [])
                for c in cands:
                    # Authoritative validity rule: is_valid is True
                    if not c.is_valid:
                        continue

                    # Evaluation completion check
                    if c.status not in (
                        CandidateStatus.EVALUATED,
                        CandidateStatus.ACCEPTED,
                        CandidateStatus.GRADUATED,
                    ):
                        continue

                    ev = eval_map.get(c.id)
                    # Provenance & non-imputation check (P0-01)
                    if not ev or not ev.is_verified:
                        continue
                    if ev.val_sharpe is None or ev.val_fitness is None:
                        continue

                    c_hash, reasons, c_dict = compute_cohort_fingerprint(pv.category, ev, c)
                    if c_hash is not None:
                        if c_hash not in active_cohorts_map:
                            active_cohorts_map[c_hash] = SimulationCohort(
                                cohort_id=c_hash,
                                category=c_dict["category"],
                                region=c_dict["region"],
                                universe=c_dict["universe"],
                                neutralization=c_dict["neutralization"],
                                decay=c_dict["decay"],
                                delay=c_dict["delay"],
                                truncation=c_dict["truncation"],
                                nan_handling=c_dict["nan_handling"],
                                pasteurization=c_dict["pasteurization"],
                                unit_handling=c_dict["unit_handling"],
                                cost_bps=c_dict["cost_bps"],
                                evaluation_scheme=c_dict["evaluation_scheme"],
                                val_start=c_dict["val_start"],
                                val_end=c_dict["val_end"],
                                datasets=c_dict["datasets"],
                                candidate_count=0,
                            )
                        cohort_pv_evaluated.setdefault(c_hash, {}).setdefault(pv.id, []).append((c, ev))
                        active_cohorts_map[c_hash].candidate_count += 1
                    else:
                        incomparable_by_pv.setdefault(pv.id, []).append((c, ev, reasons))

            rankings: list[PromptLeaderboardEntry] = []
            pv_map = {pv.id: pv for pv in prompt_versions}

            # Build rankings per cohort
            for c_id, pv_evals in cohort_pv_evaluated.items():
                if cohort_id is not None and c_id != cohort_id:
                    continue

                cohort_entries: list[PromptLeaderboardEntry] = []
                for pv_id, pairs in pv_evals.items():
                    pv = pv_map[pv_id]
                    all_cands = candidates_by_pv.get(pv_id, [])
                    generated_count = len(all_cands)
                    valid_count = sum(1 for c in all_cands if c.is_valid is True)
                    eval_count = len(pairs)
                    accepted_count = sum(
                        1 for c, _ in pairs if c.status in (CandidateStatus.ACCEPTED, CandidateStatus.GRADUATED)
                    )

                    val_sharpes = [float(ev.val_sharpe) for _, ev in pairs if ev.val_sharpe is not None]
                    val_fits = [float(ev.val_fitness) for _, ev in pairs if ev.val_fitness is not None]
                    val_tos = [float(ev.val_turnover) for _, ev in pairs if ev.val_turnover is not None]

                    med_sharpe = round(statistics.median(val_sharpes), 3) if val_sharpes else None
                    mean_sharpe = round(statistics.mean(val_sharpes), 3) if val_sharpes else None
                    best_sharpe = round(max(val_sharpes), 3) if val_sharpes else None
                    med_fit = round(statistics.median(val_fits), 3) if val_fits else None
                    med_to = round(statistics.median(val_tos), 3) if val_tos else None

                    funnel = PromptFunnelMetrics(
                        generated=generated_count,
                        valid=valid_count,
                        evaluated=eval_count,
                        accepted=accepted_count,
                        valid_pct=round(100.0 * valid_count / generated_count, 1) if generated_count else 0.0,
                        evaluated_pct=round(100.0 * eval_count / generated_count, 1) if generated_count else 0.0,
                        accepted_pct=round(100.0 * accepted_count / eval_count, 1) if eval_count else 0.0,
                    )
                    v_metrics = PromptValidationMetrics(
                        median_sharpe=med_sharpe,
                        mean_sharpe=mean_sharpe,
                        best_sharpe=best_sharpe,
                        median_fitness=med_fit,
                        median_turnover=med_to,
                    )

                    status = "RANKED" if eval_count >= min_evaluated else "INSUFFICIENT_SAMPLE"
                    cohort_entries.append(
                        PromptLeaderboardEntry(
                            prompt_version_id=pv.id,
                            category=pv.category,
                            version=pv.version,
                            authoring_source=pv.authoring_source,
                            content_hash=pv.content_hash,
                            parent_version_id=pv.parent_version_id,
                            cohort_id=c_id,
                            rank=None,
                            status=status,
                            incomparability_reasons=[],
                            funnel=funnel,
                            validation_metrics=v_metrics,
                        )
                    )

                # Sort by median_sharpe DESC
                cohort_entries.sort(
                    key=lambda e: (
                        1 if e.status == "RANKED" else 0,
                        e.validation_metrics.median_sharpe if e.validation_metrics.median_sharpe is not None else -999.0,
                    ),
                    reverse=True,
                )
                rank_counter = 1
                for entry in cohort_entries:
                    if entry.status == "RANKED":
                        entry.rank = rank_counter
                        rank_counter += 1
                    else:
                        entry.rank = None

                rankings.extend(cohort_entries)

            # Build incomparable records
            incomparable_records: list[PromptLeaderboardEntry] = []
            for pv_id, pairs_with_reasons in incomparable_by_pv.items():
                pv = pv_map[pv_id]
                all_cands = candidates_by_pv.get(pv_id, [])
                generated_count = len(all_cands)
                valid_count = sum(1 for c in all_cands if c.is_valid is True)
                eval_count = len(pairs_with_reasons)
                accepted_count = sum(
                    1 for c, _, _ in pairs_with_reasons if c.status in (CandidateStatus.ACCEPTED, CandidateStatus.GRADUATED)
                )

                val_sharpes = [float(ev.val_sharpe) for _, ev, _ in pairs_with_reasons if ev.val_sharpe is not None]
                val_fits = [float(ev.val_fitness) for _, ev, _ in pairs_with_reasons if ev.val_fitness is not None]
                val_tos = [float(ev.val_turnover) for _, ev, _ in pairs_with_reasons if ev.val_turnover is not None]

                med_sharpe = round(statistics.median(val_sharpes), 3) if val_sharpes else None
                mean_sharpe = round(statistics.mean(val_sharpes), 3) if val_sharpes else None
                best_sharpe = round(max(val_sharpes), 3) if val_sharpes else None
                med_fit = round(statistics.median(val_fits), 3) if val_fits else None
                med_to = round(statistics.median(val_tos), 3) if val_tos else None

                all_reasons = sorted({r for _, _, r_list in pairs_with_reasons for r in r_list})

                funnel = PromptFunnelMetrics(
                    generated=generated_count,
                    valid=valid_count,
                    evaluated=eval_count,
                    accepted=accepted_count,
                    valid_pct=round(100.0 * valid_count / generated_count, 1) if generated_count else 0.0,
                    evaluated_pct=round(100.0 * eval_count / generated_count, 1) if generated_count else 0.0,
                    accepted_pct=round(100.0 * accepted_count / eval_count, 1) if eval_count else 0.0,
                )
                v_metrics = PromptValidationMetrics(
                    median_sharpe=med_sharpe,
                    mean_sharpe=mean_sharpe,
                    best_sharpe=best_sharpe,
                    median_fitness=med_fit,
                    median_turnover=med_to,
                )

                incomparable_records.append(
                    PromptLeaderboardEntry(
                        prompt_version_id=pv.id,
                        category=pv.category,
                        version=pv.version,
                        authoring_source=pv.authoring_source,
                        content_hash=pv.content_hash,
                        parent_version_id=pv.parent_version_id,
                        cohort_id=None,
                        rank=None,
                        status="INCOMPARABLE",
                        incomparability_reasons=all_reasons,
                        funnel=funnel,
                        validation_metrics=v_metrics,
                    )
                )

            # If a PromptVersion has no evaluated candidates at all, provide unranked entry if cohort filter is None
            if cohort_id is None:
                covered_pv_ids = {e.prompt_version_id for e in rankings} | {e.prompt_version_id for e in incomparable_records}
                for pv in prompt_versions:
                    if pv.id not in covered_pv_ids:
                        all_cands = candidates_by_pv.get(pv.id, [])
                        gen_cnt = len(all_cands)
                        val_cnt = sum(1 for c in all_cands if c.is_valid is True)
                        funnel = PromptFunnelMetrics(
                            generated=gen_cnt,
                            valid=val_cnt,
                            evaluated=0,
                            accepted=0,
                            valid_pct=round(100.0 * val_cnt / gen_cnt, 1) if gen_cnt else 0.0,
                            evaluated_pct=0.0,
                            accepted_pct=0.0,
                        )
                        v_metrics = PromptValidationMetrics(None, None, None, None, None)
                        rankings.append(
                            PromptLeaderboardEntry(
                                prompt_version_id=pv.id,
                                category=pv.category,
                                version=pv.version,
                                authoring_source=pv.authoring_source,
                                content_hash=pv.content_hash,
                                parent_version_id=pv.parent_version_id,
                                cohort_id=None,
                                rank=None,
                                status="INSUFFICIENT_SAMPLE",
                                incomparability_reasons=[],
                                funnel=funnel,
                                validation_metrics=v_metrics,
                            )
                        )

            return PromptLeaderboardResult(
                active_cohorts=list(active_cohorts_map.values()),
                rankings=rankings,
                incomparable_records=incomparable_records,
            )
