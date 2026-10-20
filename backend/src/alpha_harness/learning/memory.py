"""Long-term quantitative memory and feature attribution store.

Tracks statistical efficacy of Fast Expression operators, data fields, categories, and
prompt versions across generations, providing empirical evidence for the self-learning loop.
"""

from __future__ import annotations

import math
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import and_, delete, func, or_, select, update

from ..db.models import (
    CandidateStatus,
    OperatorEfficacyRecord,
    PromptStrategy,
    ResearchCandidate,
    ResearchEpisode,
    ResearchEvaluation,
    ResearchInsight,
    ResearchResearcher,
    ResearchSession,
    utcnow,
)
from ..labs.fastexpr import extract_ast_operators
from ..prompts.library import normalize_category

if TYPE_CHECKING:
    from ..db.sqlite import Database

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# P1-10: Statistical helper functions for multiplicity control
# ---------------------------------------------------------------------------

#: Default FDR threshold for Benjamini-Hochberg procedure
BH_FDR_THRESHOLD: float = 0.15

#: Conservative default observation length (in years) when partition dates
#: are unavailable (e.g., legacy 2-way evaluations or unit test fixtures).
DEFAULT_OBSERVATION_YEARS: float = 4.0

#: Trading days per year (BRAIN convention, measured)
TRADING_DAYS_PER_YEAR: int = 250


def _normal_cdf(x: float) -> float:
    """Standard normal CDF Φ(x) using the complementary error function."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _sharpe_standard_error(
    sharpe: float,
    t_years: float,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Standard error of an **annualized** Sharpe ratio (Lo 2002, Theorem 1).

    The vault computes  SR_a = (mean_daily / std_daily) × √q  where q is
    trading days per year (250 by BRAIN convention).  Under IID-normal daily
    returns the asymptotic variance of the sample annualized Sharpe is:

        Var(SR̂_a) ≈ (1 + SR_a² / (2q)) / Y

    where Y is the observation duration in years and q = trading_days_per_year.

    Derivation (Lo 2002 §2, adapted for annualisation):
        daily:  Var(SR̂_d) ≈ (1 + ½ SR_d²) / N       (N daily observations)
        annual: SR̂_a = √q · SR̂_d
                Var(SR̂_a) = q · Var(SR̂_d)
                           = q (1 + ½ SR_d²) / N
        Substituting SR_d = SR_a/√q  and  N = qY :
                           = q (1 + SR_a²/(2q)) / (qY)
                           = (1 + SR_a²/(2q)) / Y

    Assumptions / limitations (documented, not addressed):
        • IID normality — ignores serial correlation and fat tails.
        • No Newey–West or HAC correction.
        • For fat-tailed returns the true SE is larger, making this estimate
          anti-conservative for *accepting* winners.  The additional LCB₉₅ > 0
          gate partially mitigates this.
        • See deferred-issues log for planned improvements.
    """
    if t_years <= 0:
        return float("inf")
    q = float(trading_days_per_year)
    return math.sqrt((1.0 + (sharpe * sharpe) / (2.0 * q)) / t_years)


def _estimate_observation_years(
    ev: Any,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Estimate the observation duration Y (in years) for a candidate's selection metric.

    For 3-way evaluated candidates the selection metric is val_sharpe, measured
    strictly over the validation sample window [is_end_date, val_end_date). This window
    is the validation sample itself (typically the middle 20% of trading days), NOT a
    gap or buffer between the in-sample and validation periods.

    Resolution Priority:
    1. Actual valid observation count: If `val_observations`, `n_observations`, or
       `val_trading_days` is present, uses `n_obs / trading_days_per_year`. This prefers
       the actual number of valid observations and their frequency over calendar-date
       approximations. Raises ValueError if n_obs <= 0 (window/count mismatch).
    2. Exact sample window dates: If `is_end_date` and `val_end_date` are present for
       a val_sharpe metric, calculates trading years from the calendar span. Raises
       ValueError if `val_end_date <= is_end_date` (mismatch between metric window
       and uncertainty estimate).
    3. Legacy fallback: `DEFAULT_OBSERVATION_YEARS` for legacy unpartitioned records
       where exact validation sample dates are missing.
    """
    def _get(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    val_sharpe = _get(ev, "val_sharpe")

    # 1. Prefer actual number of valid observations and their frequency
    for obs_attr in ("val_observations", "n_observations", "val_trading_days"):
        n_obs = _get(ev, obs_attr)
        if n_obs is not None:
            if n_obs <= 0:
                raise ValueError(
                    f"Observation window mismatch: non-positive observation count ({n_obs}) "
                    f"for uncertainty estimate."
                )
            return max(0.1, float(n_obs) / float(trading_days_per_year))

    # 2. Validation window dates [is_end_date, val_end_date)
    is_end = _get(ev, "is_end_date")
    val_end = _get(ev, "val_end_date")

    if val_sharpe is not None and is_end is not None and val_end is not None:
        if val_end <= is_end:
            raise ValueError(
                f"Observation window mismatch: val_end_date ({val_end}) must be strictly "
                f"after is_end_date ({is_end}) for validation uncertainty estimate."
            )
        cal_days = (val_end - is_end).days
        # Convert calendar days to approximate trading days, then to years (~250 trading days per 365 calendar days)
        trading_days = cal_days * float(trading_days_per_year) / 365.0
        return max(0.1, trading_days / float(trading_days_per_year))

    return DEFAULT_OBSERVATION_YEARS


def _benjamini_hochberg(p_values: list[float], fdr_level: float = BH_FDR_THRESHOLD) -> list[float]:
    """Benjamini-Hochberg FDR correction.

    Given raw one-sided p-values, returns adjusted q-values.
    A candidate with q ≤ fdr_level is a statistically significant discovery.
    """
    m = len(p_values)
    if m == 0:
        return []

    # Sort indices by ascending p-value
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    q_values = [1.0] * m

    # BH adjustment: q_(k) = min(p_(k) * M / k, q_(k+1))
    prev_q = 1.0
    for rank_idx in range(m - 1, -1, -1):
        orig_idx, p_val = indexed[rank_idx]
        bh_rank = rank_idx + 1  # 1-based rank
        adjusted = min(p_val * m / bh_rank, 1.0)
        adjusted = min(adjusted, prev_q)  # Enforce monotonicity
        q_values[orig_idx] = adjusted
        prev_q = adjusted

    return q_values


def _multiplicity_corrected_split(
    records: list[dict[str, Any]],
    fdr_level: float = BH_FDR_THRESHOLD,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """P1-10: Multiplicity-corrected winner/loser selection.

    1. Compute cohort baseline (median Sharpe across all candidates).
    2. Estimate SE for each candidate using Lo (2002) simplified formula.
    3. Compute z-scores testing H₀: SR_i ≤ baseline.
    4. Derive one-sided p-values and apply Benjamini-Hochberg FDR.
    5. Winners: q ≤ fdr_level AND LCB₉₅ > 0.
    6. Losers: regime-relative (z ≤ -1.645 below baseline, or in bottom
       quartile with Sharpe below baseline).
    7. Small batch (M < 5): conservative fallback requiring t ≥ 1.96 vs zero.
    8. Attach multiplicity metadata to every record.
    """
    m = len(records)
    if m == 0:
        return [], []

    # Stable score order; candidate identity breaks ties independently of query order.
    records.sort(
        key=lambda r: (-float(r.get("composite_score") or 0.0), int(r.get("candidate_id", 0)))
    )

    # 1. Cohort baseline: median Sharpe (robust to outliers)
    sharpes = sorted(float(r["sharpe"]) for r in records)
    if m % 2 == 1:
        cohort_baseline = sharpes[m // 2]
    else:
        cohort_baseline = (sharpes[m // 2 - 1] + sharpes[m // 2]) / 2.0

    # 2. SE and z-score for each candidate
    for r in records:
        sr = float(r["sharpe"])
        t_y = float(r["t_years"])
        se = _sharpe_standard_error(sr, t_y)
        r["sharpe_se"] = round(se, 4)
        r["t_years_used"] = round(t_y, 2)

        # t-statistic against zero: H₀: SR_i ≤ 0
        if se > 1e-9:
            t_zero = sr / se
            p_zero = 1.0 - _normal_cdf(t_zero)
        else:
            t_zero = 0.0
            p_zero = 0.5

        r["t_vs_zero"] = round(t_zero, 3)
        r["p_value"] = round(p_zero, 6)

        # Cohort-relative z-score testing H₀: SR_i ≤ cohort_baseline
        if se > 1e-9:
            z_cohort = (sr - cohort_baseline) / se
        else:
            z_cohort = 0.0
        r["z_score"] = round(z_cohort, 3)

        # Lower endpoint for conventional two-sided 95% confidence interval:
        # LCB₉₅ = SR - 1.96 * SE (positive lower endpoint requires t_vs_zero >= 1.96)
        r["lcb95"] = round(sr - 1.96 * se, 4)

    # 3. BH-FDR correction
    raw_p_values = [float(r["p_value"]) for r in records]
    q_values = _benjamini_hochberg(raw_p_values, fdr_level)

    for r, q in zip(records, q_values):
        r["fdr_q_value"] = round(q, 6)

    # 4. Multiplicity metadata common to all records
    for r in records:
        r["multiplicity_trials_count"] = m
        r["cohort_baseline_sharpe"] = round(cohort_baseline, 4)

    # 5. Classify selection tier strictly on statistical evidence (no rank-based bypass)
    for r in records:
        sr = float(r["sharpe"])
        se = float(r["sharpe_se"])
        q = float(r["fdr_q_value"])
        lcb = float(r["lcb95"])
        t_zero = float(r["t_vs_zero"])
        z_cohort = float(r["z_score"])

        # Genuine statistical significance:
        # 1. q-value passes FDR threshold (controls family-wise false discoveries across candidate cohort)
        # 2. Positive lower endpoint for conventional two-sided 95% confidence interval:
        #    lcb > 0.0  <=>  t_vs_zero >= 1.96  (does not weaken significance gate)
        # 3. Sharpe point estimate is strictly positive
        is_statistically_significant = (q <= fdr_level) and (lcb > 0.0) and (sr > 0)

        if is_statistically_significant:
            r["selection_tier"] = "SIGNIFICANT_WINNER"
        elif se > 1e-9 and z_cohort <= -1.645:
            r["selection_tier"] = "SIGNIFICANT_LOSER"
        elif sr < cohort_baseline:
            r["selection_tier"] = "CANDIDATE_POOL_UNDERPERFORMER"
        elif sr > 0:
            # Positive point estimate, but lacks statistical certainty (short sample, high SE, or marginal p-value).
            # Excluded from SIGNIFICANT_WINNER and marked PROVISIONAL.
            r["selection_tier"] = "PROVISIONAL"
        else:
            r["selection_tier"] = "CANDIDATE_POOL"

    # 6. Partition into winners and losers: only top-ranked candidates meeting SIGNIFICANT_WINNER qualify
    cutoff_top = max(1, m // 4)
    winners = [r for r in records[:cutoff_top] if r["selection_tier"] == "SIGNIFICANT_WINNER"]
    losers = [r for r in records[cutoff_top:] if r["selection_tier"] in ("SIGNIFICANT_LOSER", "CANDIDATE_POOL_UNDERPERFORMER")]

    # Sort winners: primary by composite_score descending (preserves stable order when composite_score is not set)
    winners.sort(
        key=lambda r: (-float(r.get("composite_score") or 0.0), int(r.get("candidate_id", 0)))
    )

    # Sort losers by z-score ascending (worst relative performers first)
    losers.sort(key=lambda r: float(r["z_score"]))

    return winners, losers



@dataclass(slots=True)
class OperatorEfficacy:
    """Historical performance and controlled statistical efficacy of a Fast Expression operator."""

    name: str
    appearances: int = 0
    winner_count: int = 0
    loser_count: int = 0
    avg_sharpe: float = 0.0
    sharpe_values: list[float] = field(default_factory=list)

    # P1-09: Controlled attribution and Empirical Bayes shrinkage fields
    cohort_baseline: float = 0.0
    observed_lift: float = 0.0
    marginal_lift: float = 0.0
    shrunk_lift: float = 0.0
    std_error: float = 0.0
    t_stat: float | None = None
    lower_confidence_bound: float | None = None
    upper_confidence_bound: float | None = None
    confidence_level: str = "INSUFFICIENT_SAMPLE"  # HIGH | MEDIUM | LOW | INSUFFICIENT_SAMPLE
    evidence_status: str = "INSUFFICIENT_EVIDENCE"  # POSITIVE_ASSOCIATION | NEGATIVE_ASSOCIATION | INCONCLUSIVE | INSUFFICIENT_EVIDENCE
    co_occurring_operators: list[str] = field(default_factory=list)
    adjusted_for_fields: bool = False
    adjusted_for_complexity: bool = True
    method_version: str = "p1_09_v1"

    @property
    def win_rate(self) -> float:
        return (self.winner_count / self.appearances) if self.appearances else 0.0

    @property
    def sample_count(self) -> int:
        return self.appearances

    @property
    def is_statistically_significant(self) -> bool:
        return self.evidence_status in ("POSITIVE_ASSOCIATION", "NEGATIVE_ASSOCIATION") and (
            abs(self.t_stat) >= 1.96 if self.t_stat is not None else False
        )


@dataclass(slots=True)
class FieldEfficacy:
    """Historical performance of a specific data field."""

    field_id: str
    dataset_id: str | None = None
    appearances: int = 0
    winner_count: int = 0
    avg_sharpe: float = 0.0
    sharpe_values: list[float] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return (self.winner_count / self.appearances) if self.appearances else 0.0


class ResearchMemory:
    """Manages empirical memory of alpha performance, operator utility, and field efficacy."""

    def __init__(self, db: Database, operator_catalog: Any | None = None) -> None:
        self.db = db
        self._operator_catalog = operator_catalog

    async def get_session_performance_split(
        self,
        session_id: int,
        generation: int | None = None,
        category: str | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Separates candidates into top performers (winners) and underperformers (losers).

        Returns: (winners, losers) list of dicts with expression, hypothesis, and metrics.
        """
        async with self.db.session() as session:
            stmt = (
                select(
                    ResearchCandidate,
                    ResearchEvaluation,
                    PromptStrategy.category,
                    ResearchResearcher.strategy_focus,
                )
                .join(
                    ResearchEvaluation,
                    ResearchEvaluation.candidate_id == ResearchCandidate.id,
                )
                .outerjoin(
                    PromptStrategy,
                    PromptStrategy.id == ResearchCandidate.prompt_strategy_id,
                )
                .outerjoin(
                    ResearchResearcher,
                    ResearchResearcher.id == ResearchCandidate.researcher_id,
                )
                .where(
                    ResearchCandidate.session_id == session_id,
                    ResearchEvaluation.is_verified.is_(True),
                    ResearchCandidate.status.in_(
                        [
                            CandidateStatus.EVALUATED,
                            CandidateStatus.ACCEPTED,
                            CandidateStatus.GRADUATED,
                        ]
                    ),
                )
            )
            if generation is not None:
                stmt = stmt.where(ResearchCandidate.generation == generation)

            res = await session.execute(stmt)
            rows = res.all()

            if not rows:
                return [], []

            # Query session's active strategy category as fallback for legacy candidates without lineage
            active_strat_stmt = (
                select(PromptStrategy.category)
                .where(PromptStrategy.session_id == session_id, PromptStrategy.is_active.is_(True))
                .order_by(PromptStrategy.version.desc(), PromptStrategy.id.desc())
                .limit(1)
            )
            act_res = await session.execute(active_strat_stmt)
            sess_cat = act_res.scalars().first()

            records = []
            target_cat = normalize_category(category) if category is not None else None
            for cand, ev, strat_cat, researcher_focus in rows:
                cand_cat = normalize_category(strat_cat if strat_cat else (researcher_focus if researcher_focus else sess_cat))
                if target_cat is not None and cand_cat != target_cat:
                    continue

                # P0-03: fall back to is_sharpe, and only full_sharpe for unpartitioned legacy records
                v_sharpe = (
                    ev.val_sharpe
                    if ev.val_sharpe is not None
                    else (ev.is_sharpe if ev.is_sharpe is not None else (ev.full_sharpe or 0.0))
                )
                v_fitness = (
                    ev.val_fitness
                    if ev.val_fitness is not None
                    else (ev.is_fitness if ev.is_fitness is not None else (ev.full_fitness or 0.0))
                )
                v_after_cost = (
                    ev.val_after_cost_sharpe
                    if ev.val_after_cost_sharpe is not None
                    else (ev.after_cost_sharpe or 0.0)
                )
                v_decay = ev.is_val_decay if ev.is_val_decay is not None else (ev.sharpe_decay or 0.0)
                v_turnover = ev.val_turnover if ev.val_turnover is not None else (ev.is_turnover or 0.0)
                v_composite = (
                    ev.val_composite_score
                    if ev.val_composite_score is not None
                    else (ev.composite_score or 0.0)
                )
                score_metrics = (
                    v_composite,
                    v_sharpe,
                    v_fitness,
                    v_after_cost,
                    v_decay,
                    v_turnover,
                )
                invalid_metrics = any(
                    not math.isfinite(float(value)) for value in score_metrics
                ) or v_turnover < 0
                if invalid_metrics:
                    log.warning("memory.skip_invalid_metrics", candidate_id=cand.id)
                    continue

                # P1-10: Estimate observation length T (in years) for SE calculation
                t_years = _estimate_observation_years(ev)

                records.append(
                    {
                        "candidate_id": cand.id,
                        "researcher_id": cand.researcher_id,
                        "prompt_strategy_id": cand.prompt_strategy_id,
                        "category": cand_cat,
                        "generation": cand.generation,
                        "hypothesis": cand.hypothesis,
                        "expression": cand.expression,
                        "fields": cand.fields_used or [],
                        "operator_count": cand.operator_count,
                        "sharpe": v_sharpe,
                        "fitness": v_fitness,
                        "after_cost_sharpe": v_after_cost,
                        "sharpe_decay": v_decay,
                        "turnover": v_turnover,
                        "composite_score": v_composite,
                        "composite_score_spec": ev.composite_score_spec,
                        "is_submission_grade": (
                            (ev.internal_checks is not None or ev.checks is None)
                            and ev.checks_passed == ev.checks_total
                            and (ev.checks_total or 0) > 0
                        ),
                        # P1-10: observation metadata for SE and multiplicity
                        "t_years": t_years,
                    }
                )

            if not records:
                return [], []

            # P1-10: Multiplicity-corrected winner/loser selection
            winners, losers = _multiplicity_corrected_split(records)
            return winners, losers

    async def get_eligible_candidates(
        self,
        session_id: int,
        generation: int | None = None,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieves all verified, eligible research candidates for a session/generation."""
        async with self.db.session() as session:
            stmt = (
                select(
                    ResearchCandidate,
                    ResearchEvaluation,
                    PromptStrategy.category,
                    ResearchResearcher.strategy_focus,
                )
                .join(
                    ResearchEvaluation,
                    ResearchEvaluation.candidate_id == ResearchCandidate.id,
                )
                .outerjoin(
                    PromptStrategy,
                    PromptStrategy.id == ResearchCandidate.prompt_strategy_id,
                )
                .outerjoin(
                    ResearchResearcher,
                    ResearchResearcher.id == ResearchCandidate.researcher_id,
                )
                .where(
                    ResearchCandidate.session_id == session_id,
                    ResearchEvaluation.is_verified.is_(True),
                    ResearchCandidate.status.in_(
                        [
                            CandidateStatus.EVALUATED,
                            CandidateStatus.ACCEPTED,
                            CandidateStatus.GRADUATED,
                        ]
                    ),
                )
            )
            if generation is not None:
                stmt = stmt.where(ResearchCandidate.generation == generation)

            res = await session.execute(stmt)
            rows = res.all()

            # Query session's active strategy category as fallback for legacy candidates without lineage
            active_strat_stmt = (
                select(PromptStrategy.category)
                .where(PromptStrategy.session_id == session_id, PromptStrategy.is_active.is_(True))
                .order_by(PromptStrategy.version.desc(), PromptStrategy.id.desc())
                .limit(1)
            )
            act_res = await session.execute(active_strat_stmt)
            sess_cat = act_res.scalars().first()

            records = []
            target_cat = normalize_category(category) if category is not None else None
            for cand, ev, strat_cat, researcher_focus in rows:
                cand_cat = normalize_category(strat_cat if strat_cat else (researcher_focus if researcher_focus else sess_cat))
                if target_cat is not None and cand_cat != target_cat:
                    continue

                # P0-03: fall back to is_sharpe, and only full_sharpe for unpartitioned legacy records
                v_sharpe = (
                    ev.val_sharpe
                    if ev.val_sharpe is not None
                    else (ev.is_sharpe if ev.is_sharpe is not None else (ev.full_sharpe or 0.0))
                )
                v_fitness = (
                    ev.val_fitness
                    if ev.val_fitness is not None
                    else (ev.is_fitness if ev.is_fitness is not None else (ev.full_fitness or 0.0))
                )
                v_after_cost = (
                    ev.val_after_cost_sharpe
                    if ev.val_after_cost_sharpe is not None
                    else (ev.after_cost_sharpe or 0.0)
                )
                v_decay = ev.is_val_decay if ev.is_val_decay is not None else (ev.sharpe_decay or 0.0)
                v_turnover = ev.val_turnover if ev.val_turnover is not None else (ev.is_turnover or 0.0)
                v_composite = (
                    ev.val_composite_score
                    if ev.val_composite_score is not None
                    else (ev.composite_score or 0.0)
                )
                score_metrics = (
                    v_composite,
                    v_sharpe,
                    v_fitness,
                    v_after_cost,
                    v_decay,
                    v_turnover,
                )
                invalid_metrics = any(
                    not math.isfinite(float(value)) for value in score_metrics
                ) or v_turnover < 0
                if invalid_metrics:
                    log.warning("memory.skip_invalid_metrics", candidate_id=cand.id)
                    continue

                # P1-10: Estimate observation length T (in years) for SE calculation
                t_years = _estimate_observation_years(ev)

                records.append(
                    {
                        "candidate_id": cand.id,
                        "researcher_id": cand.researcher_id,
                        "prompt_strategy_id": cand.prompt_strategy_id,
                        "category": cand_cat,
                        "generation": cand.generation,
                        "hypothesis": cand.hypothesis,
                        "expression": cand.expression,
                        "fields": cand.fields_used or [],
                        "operator_count": cand.operator_count,
                        "sharpe": v_sharpe,
                        "fitness": v_fitness,
                        "after_cost_sharpe": v_after_cost,
                        "sharpe_decay": v_decay,
                        "turnover": v_turnover,
                        "composite_score": v_composite,
                        "composite_score_spec": ev.composite_score_spec,
                        "is_submission_grade": (
                            (ev.internal_checks is not None or ev.checks is None)
                            and ev.checks_passed == ev.checks_total
                            and (ev.checks_total or 0) > 0
                        ),
                        "t_years": t_years,
                    }
                )
            if records:
                _multiplicity_corrected_split(records)
            records.sort(
                key=lambda r: (-float(r["composite_score"]), int(r.get("candidate_id", 0)))
            )
            return records

    async def compute_operator_stats(
        self,
        session_id: int,
        category: str | None = None,
        shrinkage_k: float = 3.0,
        min_sample_threshold: int = 3,
    ) -> dict[str, OperatorEfficacy]:
        """Aggregates controlled, deconfounded operator-level Sharpe lift, shrinkage, and win rates.

        P1-09 Methodology:
        1. Ingestion: Loads only verified genuine BRAIN evaluations (is_verified=True). Holdout metrics
           are strictly protected and excluded.
        2. AST Extraction: Extracts true operator calls from Fast Expression AST nodes using
           extract_ast_operators(), validated against OperatorCatalog. Rejects non-call names,
           constants, and unknown operators. Deduplicates operators per candidate so each candidate
           contributes at most 1 observation per operator.
        3. Cohort Baseline: Calculates mean performance across the comparable cohort (category or session).
        4. Confounder Controls:
           - Complexity Weighting: Discounts credit in multi-operator expressions by 1 / sqrt(operator_count).
           - Field Quality Covariate: Adjusts for average performance of referenced data fields where available.
        5. Empirical Bayes Shrinkage: Shrinks marginal lift toward the zero-prior using:
           shrunk_lift = (N / (N + k)) * marginal_lift
        6. Uncertainty Quantification: Computes standard error, t-stat, lower confidence bound (LCB),
           and upper confidence bound (UCB).
        7. Evidence Classification: Requires appearances >= min_sample_threshold (default 3) for eligibility.
           Distinguishes POSITIVE_ASSOCIATION, NEGATIVE_ASSOCIATION, INCONCLUSIVE, and INSUFFICIENT_EVIDENCE.
        8. Co-occurrence: Identifies highly collinear operator pairs (correlation >= 0.85, count >= 3).
        """
        async with self.db.session() as session:
            stmt = (
                select(
                    ResearchCandidate.id,
                    ResearchCandidate.expression,
                    # P0-03: fall back to is_sharpe, and only full_sharpe for unpartitioned legacy records
                    func.coalesce(ResearchEvaluation.val_sharpe, ResearchEvaluation.is_sharpe, ResearchEvaluation.full_sharpe),
                    ResearchCandidate.fields_used,
                    PromptStrategy.category,
                    ResearchResearcher.strategy_focus,
                )
                .join(
                    ResearchEvaluation,
                    ResearchEvaluation.candidate_id == ResearchCandidate.id,
                )
                .outerjoin(
                    PromptStrategy,
                    PromptStrategy.id == ResearchCandidate.prompt_strategy_id,
                )
                .outerjoin(
                    ResearchResearcher,
                    ResearchResearcher.id == ResearchCandidate.researcher_id,
                )
                .where(
                    ResearchCandidate.session_id == session_id,
                    ResearchEvaluation.is_verified.is_(True),
                    ResearchCandidate.status.in_(
                        [
                            CandidateStatus.EVALUATED,
                            CandidateStatus.ACCEPTED,
                            CandidateStatus.GRADUATED,
                        ]
                    ),
                )
            )
            res = await session.execute(stmt)
            rows = res.all()

            if not rows:
                return {}

            # Query session's active strategy category as fallback for legacy candidates without lineage
            active_strat_stmt = (
                select(PromptStrategy.category)
                .where(PromptStrategy.session_id == session_id, PromptStrategy.is_active.is_(True))
                .order_by(PromptStrategy.version.desc(), PromptStrategy.id.desc())
                .limit(1)
            )
            act_res = await session.execute(active_strat_stmt)
            sess_cat = act_res.scalars().first()

            target_cat = normalize_category(category) if category is not None else None
            candidates_by_id: dict[int, dict[str, Any]] = {}
            for cid, expr, sharpe, fields_used, strat_cat, researcher_focus in rows:
                if cid in candidates_by_id:
                    continue
                cand_cat = normalize_category(strat_cat if strat_cat else (researcher_focus if researcher_focus else sess_cat))
                if target_cat is not None and cand_cat != target_cat:
                    continue
                candidates_by_id[cid] = {
                    "id": cid,
                    "expression": expr or "",
                    "sharpe": float(sharpe or 0.0),
                    "fields": fields_used or [],
                    "category": cand_cat,
                }

            if not candidates_by_id:
                return {}

            # 1. AST Operator Extraction per Candidate
            candidate_ops: dict[int, list[str]] = {}
            for cid, cdata in candidates_by_id.items():
                expr = cdata["expression"]
                if expr:
                    ops = extract_ast_operators(
                        expr, catalog=self._operator_catalog, deduplicate=True
                    )
                else:
                    ops = []
                candidate_ops[cid] = ops

            # 2. Cohort Baseline Sharpe
            all_sharpes = [c["sharpe"] for c in candidates_by_id.values()]
            cohort_baseline = round(sum(all_sharpes) / len(all_sharpes), 3) if all_sharpes else 0.0

            # 3. Field Quality Covariate Calculation
            field_sharpes: dict[str, list[float]] = defaultdict(list)
            for cid, cdata in candidates_by_id.items():
                s_val = cdata["sharpe"]
                for f in cdata["fields"]:
                    if isinstance(f, str) and f:
                        field_sharpes[f].append(s_val)

            field_lifts: dict[str, float] = {}
            for f, f_vals in field_sharpes.items():
                if f_vals:
                    field_lifts[f] = (sum(f_vals) / len(f_vals)) - cohort_baseline

            # Candidate adjusted excess Sharpe
            candidate_adjusted_y: dict[int, float] = {}
            adjusted_for_fields_flag = bool(field_lifts)
            for cid, cdata in candidates_by_id.items():
                s_val = cdata["sharpe"]
                excess = s_val - cohort_baseline
                c_fields = [f for f in cdata["fields"] if isinstance(f, str) and f]
                if c_fields and adjusted_for_fields_flag:
                    avg_field_lift = sum(field_lifts.get(f, 0.0) for f in c_fields) / len(c_fields)
                    # Controlling for field quality: subtract attenuated field contribution
                    candidate_adjusted_y[cid] = excess - 0.5 * avg_field_lift
                else:
                    candidate_adjusted_y[cid] = excess

            # 4. Group by Operator and calculate Co-occurrences
            op_to_candidates: dict[str, list[int]] = defaultdict(list)
            co_occurrence_counts: dict[tuple[str, str], int] = defaultdict(int)
            for cid, ops in candidate_ops.items():
                for op in ops:
                    op_to_candidates[op].append(cid)
                sorted_ops = sorted(ops)
                for i in range(len(sorted_ops)):
                    for j in range(i + 1, len(sorted_ops)):
                        co_occurrence_counts[(sorted_ops[i], sorted_ops[j])] += 1

            stats_map: dict[str, OperatorEfficacy] = {}
            for op, c_ids in op_to_candidates.items():
                n_obs = len(c_ids)
                op_sharpes = [candidates_by_id[cid]["sharpe"] for cid in c_ids]
                raw_avg_sharpe = round(sum(op_sharpes) / n_obs, 3)
                observed_lift = round(raw_avg_sharpe - cohort_baseline, 3)

                winners = sum(1 for s in op_sharpes if s >= 1.25)
                losers = sum(1 for s in op_sharpes if s < 0.5)

                # Complexity-weighted marginal lift: discounts credit in multi-operator expressions by 1 / sqrt(operator_count)
                adj_y_values: list[float] = []
                for cid in c_ids:
                    n_ops_in_cand = max(1, len(candidate_ops[cid]))
                    complexity_discount = 1.0 / math.sqrt(n_ops_in_cand)
                    y_adj = candidate_adjusted_y[cid] * complexity_discount
                    adj_y_values.append(y_adj)

                marginal_lift = round(sum(adj_y_values) / n_obs, 4) if n_obs > 0 else 0.0

                # Empirical Bayes shrinkage
                shrunk_lift = round((n_obs / (n_obs + shrinkage_k)) * marginal_lift, 4)

                # Uncertainty: Standard Error and t-stat
                if n_obs >= 2:
                    mean_adj = sum(adj_y_values) / n_obs
                    sample_var = sum((y - mean_adj) ** 2 for y in adj_y_values) / (n_obs - 1)
                    std_error = round(math.sqrt(max(0.0, sample_var)) / math.sqrt(n_obs), 4)
                else:
                    std_error = 0.0

                t_stat = round(marginal_lift / std_error, 2) if std_error > 1e-5 else None

                # Confidence bounds based on standard error
                lcb = round(shrunk_lift - 1.96 * std_error, 4) if std_error > 0 else round(shrunk_lift, 4)
                ucb = round(shrunk_lift + 1.96 * std_error, 4) if std_error > 0 else round(shrunk_lift, 4)

                # Evidence status & confidence level
                if n_obs < min_sample_threshold:
                    confidence_level = "INSUFFICIENT_SAMPLE"
                    evidence_status = "INSUFFICIENT_EVIDENCE"
                else:
                    if n_obs >= 10:
                        confidence_level = "HIGH"
                    elif n_obs >= 5:
                        confidence_level = "MEDIUM"
                    else:
                        confidence_level = "LOW"

                    if lcb > 0.0 and shrunk_lift > 0.0:
                        evidence_status = "POSITIVE_ASSOCIATION"
                    elif ucb < 0.0 and shrunk_lift < 0.0:
                        evidence_status = "NEGATIVE_ASSOCIATION"
                    else:
                        evidence_status = "INCONCLUSIVE"

                # Find highly co-occurring operators (ratio >= 0.85 and count >= 3)
                co_occurring = []
                for other_op in op_to_candidates.keys():
                    if other_op == op:
                        continue
                    pair_key = (min(op, other_op), max(op, other_op))
                    co_count = co_occurrence_counts.get(pair_key, 0)
                    if co_count >= 3 and (co_count / min(n_obs, len(op_to_candidates[other_op]))) >= 0.85:
                        co_occurring.append(other_op)

                stats_map[op] = OperatorEfficacy(
                    name=op,
                    appearances=n_obs,
                    winner_count=winners,
                    loser_count=losers,
                    avg_sharpe=raw_avg_sharpe,
                    sharpe_values=op_sharpes,
                    cohort_baseline=cohort_baseline,
                    observed_lift=observed_lift,
                    marginal_lift=marginal_lift,
                    shrunk_lift=shrunk_lift,
                    std_error=std_error,
                    t_stat=t_stat,
                    lower_confidence_bound=lcb,
                    upper_confidence_bound=ucb,
                    confidence_level=confidence_level,
                    evidence_status=evidence_status,
                    co_occurring_operators=sorted(co_occurring),
                    adjusted_for_fields=adjusted_for_fields_flag,
                    adjusted_for_complexity=True,
                    method_version="p1_09_v1",
                )

            return stats_map

    async def persist_operator_stats(
        self,
        session_id: int,
        stats: dict[str, OperatorEfficacy],
        generation: int | None = None,
        category: str | None = None,
        provenance: str | None = None,
    ) -> list[OperatorEfficacyRecord]:
        """Persists computed OperatorEfficacy records into the database."""
        records: list[OperatorEfficacyRecord] = []
        async with self.db.session() as session:
            for op_name, eff in stats.items():
                rec = OperatorEfficacyRecord(
                    session_id=session_id,
                    generation=generation,
                    operator_name=op_name,
                    category=category,
                    sample_count=eff.appearances,
                    winner_count=eff.winner_count,
                    loser_count=eff.loser_count,
                    raw_avg_sharpe=eff.avg_sharpe,
                    baseline_sharpe=eff.cohort_baseline,
                    observed_lift=eff.observed_lift,
                    marginal_lift=eff.marginal_lift,
                    shrunk_lift=eff.shrunk_lift,
                    std_error=eff.std_error,
                    t_stat=eff.t_stat,
                    lower_confidence_bound=eff.lower_confidence_bound,
                    upper_confidence_bound=eff.upper_confidence_bound,
                    confidence_level=eff.confidence_level,
                    evidence_status=eff.evidence_status,
                    method_version=eff.method_version,
                    provenance=provenance,
                    co_occurring_operators=eff.co_occurring_operators,
                    details={
                        "adjusted_for_fields": eff.adjusted_for_fields,
                        "adjusted_for_complexity": eff.adjusted_for_complexity,
                    },
                )
                session.add(rec)
                records.append(rec)
            await session.commit()
        return records

    async def get_persisted_operator_stats(
        self,
        session_id: int,
        generation: int | None = None,
        category: str | None = None,
    ) -> dict[str, OperatorEfficacy]:
        """Reloads persisted operator efficacy records from the database."""
        async with self.db.session() as session:
            stmt = select(OperatorEfficacyRecord).where(
                OperatorEfficacyRecord.session_id == session_id
            )
            if generation is not None:
                stmt = stmt.where(OperatorEfficacyRecord.generation == generation)
            if category is not None:
                stmt = stmt.where(OperatorEfficacyRecord.category == normalize_category(category))
            stmt = stmt.order_by(OperatorEfficacyRecord.created_at.desc())
            res = await session.execute(stmt)
            rows = res.scalars().all()

            loaded: dict[str, OperatorEfficacy] = {}
            for r in rows:
                if r.operator_name in loaded:
                    continue
                loaded[r.operator_name] = OperatorEfficacy(
                    name=r.operator_name,
                    appearances=r.sample_count,
                    winner_count=r.winner_count,
                    loser_count=r.loser_count,
                    avg_sharpe=r.raw_avg_sharpe,
                    cohort_baseline=r.baseline_sharpe,
                    observed_lift=r.observed_lift,
                    marginal_lift=r.marginal_lift,
                    shrunk_lift=r.shrunk_lift,
                    std_error=r.std_error,
                    t_stat=r.t_stat,
                    lower_confidence_bound=r.lower_confidence_bound,
                    upper_confidence_bound=r.upper_confidence_bound,
                    confidence_level=r.confidence_level,
                    evidence_status=r.evidence_status,
                    co_occurring_operators=r.co_occurring_operators or [],
                    method_version=r.method_version,
                )
            return loaded

    # ---------------------------------------------------------------------------
    # P1-12: Persistent Researcher-Specific Episodic Memory
    # ---------------------------------------------------------------------------

    async def record_research_episode(
        self,
        *,
        session_id: int,
        researcher_id: int,
        candidate_id: int | None = None,
        prompt_version_id: int | None = None,
        generation: int = 0,
        strategy_focus: str | None = None,
        dataset_id: str | None = None,
        region: str | None = None,
        universe: str | None = None,
        delay: int | None = None,
        neutralization: str | None = None,
        decay: int | None = None,
        hypothesis_family: str | None = None,
        hypothesis: str | None = None,
        rationale: str | None = None,
        expression: str | None = None,
        funnel_stage: str = "GENERATED",
        outcome: str = "PENDING",
        selection_tier: str | None = None,
        provenance: str | None = "LOCAL_PROXY",
        brain_simulation_id: str | None = None,
        # Strictly non-holdout validation metrics (P0-03 Holdout Isolation)
        validation_sharpe: float | None = None,
        validation_fitness: float | None = None,
        validation_turnover: float | None = None,
        validation_decay: float | None = None,
        in_sample_sharpe: float | None = None,
        t_years: float | None = None,
        val_observations: int | None = None,
        # Statistical evidence
        sharpe_se: float | None = None,
        t_stat: float | None = None,
        p_value: float | None = None,
        lcb95: float | None = None,
        z_score: float | None = None,
        fdr_q_value: float | None = None,
        is_statistically_significant: bool = False,
        # Failure classification & diagnostic learning
        failure_category: str | None = None,
        error_details: dict[str, Any] | None = None,
        lessons_learned: str | None = None,
        catalog_version: str | None = None,
        method_version: str = "p1_12_v1",
        is_shared_knowledge: bool = False,
        **kwargs: Any,
    ) -> ResearchEpisode:
        """Persists a structured research episode scoped to a specific researcher persona.

        Enforces strict P0-03 holdout quarantine: rejects any attempt to pass holdout metrics.
        Validates researcher and session identities.
        """
        if not researcher_id or researcher_id <= 0:
            raise ValueError(f"Valid positive researcher_id is required; got {researcher_id!r}")
        if not session_id or session_id <= 0:
            raise ValueError(f"Valid positive session_id is required; got {session_id!r}")

        # P0-03 Holdout Isolation: Refuse any holdout metrics from entering episodic memory
        for k in kwargs:
            if "holdout" in k.lower():
                raise ValueError(f"Holdout metric '{k}' is strictly prohibited from entering episodic memory (P0-03)")

        expr_clean = expression.strip() if expression else None
        expr_hash = hashlib.sha256(expr_clean.encode("utf-8")).hexdigest()[:32] if expr_clean else None

        details = dict(error_details) if error_details else {}

        # Deduplication check: check if this researcher already ran this exact expression
        if expr_hash:
            async with self.db.session() as check_session:
                dup_stmt = select(func.count(ResearchEpisode.id)).where(
                    ResearchEpisode.researcher_id == researcher_id,
                    ResearchEpisode.expression_hash == expr_hash,
                )
                dup_res = await check_session.execute(dup_stmt)
                prior_count = dup_res.scalar() or 0
                if prior_count > 0:
                    details["is_repeated_execution"] = True
                    details["prior_execution_count"] = prior_count

        episode = ResearchEpisode(
            session_id=session_id,
            researcher_id=researcher_id,
            candidate_id=candidate_id,
            prompt_version_id=prompt_version_id,
            generation=generation,
            strategy_focus=normalize_category(strategy_focus) if strategy_focus else None,
            dataset_id=dataset_id,
            region=region,
            universe=universe,
            delay=delay,
            neutralization=neutralization,
            decay=decay,
            hypothesis_family=hypothesis_family,
            hypothesis=hypothesis,
            rationale=rationale,
            expression=expr_clean,
            expression_hash=expr_hash,
            funnel_stage=funnel_stage,
            outcome=outcome,
            selection_tier=selection_tier,
            provenance=provenance,
            brain_simulation_id=brain_simulation_id,
            validation_sharpe=validation_sharpe,
            validation_fitness=validation_fitness,
            validation_turnover=validation_turnover,
            validation_decay=validation_decay,
            in_sample_sharpe=in_sample_sharpe,
            t_years=t_years,
            val_observations=val_observations,
            sharpe_se=sharpe_se,
            t_stat=t_stat,
            p_value=p_value,
            lcb95=lcb95,
            z_score=z_score,
            fdr_q_value=fdr_q_value,
            is_statistically_significant=is_statistically_significant,
            failure_category=failure_category,
            error_details=details if details else None,
            lessons_learned=lessons_learned,
            catalog_version=catalog_version,
            method_version=method_version,
            is_shared_knowledge=is_shared_knowledge,
        )

        async with self.db.session() as session:
            session.add(episode)
            await session.commit()
            await session.refresh(episode)

        return episode

    async def get_researcher_episodes(
        self,
        researcher_id: int,
        *,
        include_shared: bool = False,
        strategy_focus: str | None = None,
        dataset_id: str | None = None,
        region: str | None = None,
        universe: str | None = None,
        hypothesis_family: str | None = None,
        prompt_version_id: int | None = None,
        catalog_version: str | None = None,
        min_generation: int | None = None,
        max_generation: int | None = None,
        outcomes: list[str] | None = None,
        only_validated_discoveries: bool = False,
        only_failures: bool = False,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 50,
        deduplicate_by_expression: bool = True,
    ) -> list[ResearchEpisode]:
        """Retrieves episodic memory strictly scoped to the specified researcher persona.

        Private isolation rule:
        - Researcher A can ONLY view episodes where researcher_id == A.
        - Episodes belonging to Researcher B are NEVER returned unless explicitly marked
          `is_shared_knowledge == True` AND `include_shared == True`.
        - Missing or invalid researcher_id raises ValueError immediately.
        """
        if not researcher_id or researcher_id <= 0:
            raise ValueError(f"Valid positive researcher_id is required; got {researcher_id!r}")

        async with self.db.session() as session:
            stmt = select(ResearchEpisode)

            # Strict private isolation condition
            if include_shared:
                stmt = stmt.where(
                    or_(
                        ResearchEpisode.researcher_id == researcher_id,
                        ResearchEpisode.is_shared_knowledge.is_(True),
                    )
                )
            else:
                stmt = stmt.where(ResearchEpisode.researcher_id == researcher_id)

            if strategy_focus is not None:
                stmt = stmt.where(ResearchEpisode.strategy_focus == normalize_category(strategy_focus))
            if dataset_id is not None:
                stmt = stmt.where(ResearchEpisode.dataset_id == dataset_id)
            if region is not None:
                stmt = stmt.where(func.upper(ResearchEpisode.region) == region.upper())
            if universe is not None:
                stmt = stmt.where(func.upper(ResearchEpisode.universe) == universe.upper())
            if hypothesis_family is not None:
                stmt = stmt.where(func.lower(ResearchEpisode.hypothesis_family) == hypothesis_family.lower())
            if prompt_version_id is not None:
                stmt = stmt.where(ResearchEpisode.prompt_version_id == prompt_version_id)
            if catalog_version is not None:
                stmt = stmt.where(ResearchEpisode.catalog_version == catalog_version)
            if min_generation is not None:
                stmt = stmt.where(ResearchEpisode.generation >= min_generation)
            if max_generation is not None:
                stmt = stmt.where(ResearchEpisode.generation <= max_generation)
            if outcomes:
                stmt = stmt.where(ResearchEpisode.outcome.in_(outcomes))
            if only_validated_discoveries:
                stmt = stmt.where(
                    ResearchEpisode.outcome == "STATISTICAL_WINNER",
                    ResearchEpisode.is_statistically_significant.is_(True),
                )
            if only_failures:
                stmt = stmt.where(
                    ResearchEpisode.outcome.in_(
                        [
                            "SYNTAX_ERROR",
                            "PREFLIGHT_ERROR",
                            "SIMULATION_ERROR",
                            "STATISTICAL_LOSER",
                        ]
                    )
                )
            if since is not None:
                stmt = stmt.where(ResearchEpisode.created_at >= since)
            if until is not None:
                stmt = stmt.where(ResearchEpisode.created_at <= until)

            stmt = stmt.order_by(
                ResearchEpisode.generation.desc(),
                ResearchEpisode.created_at.desc(),
                ResearchEpisode.id.desc(),
            )

            res = await session.execute(stmt)
            rows = list(res.scalars().all())

        if not rows:
            return []

        # Deduplication by expression hash if requested
        if deduplicate_by_expression:
            seen_hashes: set[str] = set()
            deduped: list[ResearchEpisode] = []
            for ep in rows:
                if ep.expression_hash:
                    if ep.expression_hash in seen_hashes:
                        continue
                    seen_hashes.add(ep.expression_hash)
                deduped.append(ep)
                if len(deduped) >= limit:
                    break
            return deduped

        return rows[:limit]

    async def update_researcher_episode(
        self,
        researcher_id: int,
        episode_id: int,
        *,
        lessons_learned: str | None = None,
        rationale: str | None = None,
        hypothesis_family: str | None = None,
        failure_category: str | None = None,
        error_details: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> bool:
        """Updates an episode strictly scoped to researcher ownership.

        Cross-researcher mutation is strictly prohibited: returns False if the episode
        does not belong to researcher_id.
        """
        if not researcher_id or researcher_id <= 0:
            raise ValueError(f"Valid positive researcher_id is required; got {researcher_id!r}")
        for k in kwargs:
            if "holdout" in k.lower():
                raise ValueError(f"Holdout metric '{k}' is strictly prohibited from entering episodic memory (P0-03)")

        async with self.db.session() as session:
            stmt = select(ResearchEpisode).where(
                ResearchEpisode.id == episode_id,
                ResearchEpisode.researcher_id == researcher_id,
            )
            res = await session.execute(stmt)
            episode = res.scalar_one_or_none()
            if not episode:
                log.warning(
                    "memory.cross_researcher_update_rejected",
                    researcher_id=researcher_id,
                    episode_id=episode_id,
                )
                return False

            if lessons_learned is not None:
                episode.lessons_learned = lessons_learned
            if rationale is not None:
                episode.rationale = rationale
            if hypothesis_family is not None:
                episode.hypothesis_family = hypothesis_family
            if failure_category is not None:
                episode.failure_category = failure_category
            if error_details is not None:
                episode.error_details = error_details

            episode.updated_at = utcnow()
            await session.commit()
            return True

    async def delete_researcher_episode(
        self,
        researcher_id: int,
        episode_id: int,
    ) -> bool:
        """Deletes an episode strictly scoped to researcher ownership.

        Cross-researcher deletion is rejected: returns False if the episode
        does not belong to researcher_id.
        """
        if not researcher_id or researcher_id <= 0:
            raise ValueError(f"Valid positive researcher_id is required; got {researcher_id!r}")

        async with self.db.session() as session:
            stmt = select(ResearchEpisode).where(
                ResearchEpisode.id == episode_id,
                ResearchEpisode.researcher_id == researcher_id,
            )
            res = await session.execute(stmt)
            episode = res.scalar_one_or_none()
            if not episode:
                log.warning(
                    "memory.cross_researcher_delete_rejected",
                    researcher_id=researcher_id,
                    episode_id=episode_id,
                )
                return False

            await session.delete(episode)
            await session.commit()
            return True

    async def promote_to_shared_knowledge(
        self,
        researcher_id: int,
        episode_id: int,
    ) -> bool:
        """Promotes an episode to shared knowledge, verifying researcher ownership."""
        if not researcher_id or researcher_id <= 0:
            raise ValueError(f"Valid positive researcher_id is required; got {researcher_id!r}")

        async with self.db.session() as session:
            stmt = select(ResearchEpisode).where(
                ResearchEpisode.id == episode_id,
                ResearchEpisode.researcher_id == researcher_id,
            )
            res = await session.execute(stmt)
            episode = res.scalar_one_or_none()
            if not episode:
                return False

            episode.is_shared_knowledge = True
            episode.updated_at = utcnow()
            await session.commit()
            return True

    async def build_researcher_memory_context(
        self,
        researcher_id: int,
        *,
        max_discoveries: int = 4,
        max_failures: int = 4,
        max_recent_hypotheses: int = 4,
        include_shared: bool = True,
        strategy_focus: str | None = None,
        dataset_id: str | None = None,
    ) -> str:
        """Assembles structured, evidence-grounded episodic memory context for an LLM prompt.

        Contains:
        1. Statistically Validated Discoveries (what works, strictly validation metrics).
        2. Known Failure Modes & Anti-Patterns (what to avoid).
        3. Recently Explored Hypotheses (to avoid redundant duplication).

        Guarantees zero holdout contamination (P0-03).
        """
        if not researcher_id or researcher_id <= 0:
            return ""

        # 1. Validated discoveries
        discoveries = await self.get_researcher_episodes(
            researcher_id,
            include_shared=include_shared,
            strategy_focus=strategy_focus,
            dataset_id=dataset_id,
            only_validated_discoveries=True,
            limit=max_discoveries,
        )

        # 2. Critical failure modes
        failures = await self.get_researcher_episodes(
            researcher_id,
            include_shared=include_shared,
            strategy_focus=strategy_focus,
            dataset_id=dataset_id,
            only_failures=True,
            limit=max_failures,
        )

        # 3. Recent experiments
        recent = await self.get_researcher_episodes(
            researcher_id,
            include_shared=False,  # Private recent history only
            strategy_focus=strategy_focus,
            dataset_id=dataset_id,
            limit=max_recent_hypotheses,
        )

        if not discoveries and not failures and not recent:
            return ""

        sections: list[str] = [
            "### PERSISTENT RESEARCHER EPISODIC MEMORY (Scored on Validation Partition Only)"
        ]

        if discoveries:
            sections.append("#### Statistically Validated Discoveries (Efficacy Proven):")
            for ep in discoveries:
                sharpe_str = f"Val Sharpe {ep.validation_sharpe:.2f}" if ep.validation_sharpe is not None else "Val Sharpe N/A"
                t_str = f"t-stat {ep.t_stat:.2f}" if ep.t_stat is not None else ""
                stat_meta = ", ".join(filter(None, [sharpe_str, t_str]))
                sections.append(
                    f"- [Episode #{ep.id} | Gen {ep.generation}] `{ep.expression}` ({stat_meta})\n"
                    f"  Rationale: {ep.hypothesis or ep.rationale or 'N/A'}"
                )

        if failures:
            sections.append("#### Known Failure Modes & Pitfalls (Patterns to Avoid):")
            for ep in failures:
                cat_str = ep.failure_category or ep.outcome
                sections.append(
                    f"- [Episode #{ep.id} | Gen {ep.generation} | {cat_str}] `{ep.expression or 'N/A'}`\n"
                    f"  Issue: {ep.lessons_learned or (ep.error_details.get('error') if ep.error_details else None) or ep.outcome}"
                )

        if recent:
            sections.append("#### Recently Explored Hypotheses (Avoid Exact Re-testing):")
            for ep in recent:
                sections.append(
                    f"- [Episode #{ep.id}] {ep.hypothesis or 'Baseline anomaly exploration'} -> Outcome: {ep.outcome}"
                )

        return "\n\n".join(sections)

    async def record_batch_episodes_from_candidates(
        self,
        session_id: int,
        generation: int,
        candidates_with_evals: list[dict[str, Any]],
        selection_split: tuple[list[dict[str, Any]], list[dict[str, Any]]] | None = None,
    ) -> list[ResearchEpisode]:
        """Batch records research episodes from evaluated candidate records in a generation.

        Maps multiplicity tiers (P1-10) to episodic outcomes and attributes to specific researchers.
        """
        winners_map = {}
        losers_map = {}
        if selection_split:
            win_list, lose_list = selection_split
            winners_map = {w["candidate_id"]: w for w in win_list}
            losers_map = {l["candidate_id"]: l for l in lose_list}

        episodes: list[ResearchEpisode] = []
        for c in candidates_with_evals:
            cid = c.get("candidate_id") or c.get("id")
            rid = c.get("researcher_id")
            if not rid or rid <= 0:
                continue

            # Determine outcome from statistical split
            is_stat_sig = False
            if cid in winners_map:
                w_rec = winners_map[cid]
                outcome = "STATISTICAL_WINNER"
                tier = w_rec.get("selection_tier", "SIGNIFICANT_WINNER")
                is_stat_sig = True
                sharpe_se = w_rec.get("sharpe_se")
                t_stat = w_rec.get("t_vs_zero")
                p_val = w_rec.get("p_value")
                lcb = w_rec.get("lcb95")
                z_sc = w_rec.get("z_score")
                q_val = w_rec.get("fdr_q_value")
                fail_cat = None
                lesson = "Strong risk-adjusted excess returns on validation partition."
            elif cid in losers_map:
                l_rec = losers_map[cid]
                outcome = "STATISTICAL_LOSER"
                tier = l_rec.get("selection_tier", "SIGNIFICANT_LOSER")
                sharpe_se = l_rec.get("sharpe_se")
                t_stat = l_rec.get("t_vs_zero")
                p_val = l_rec.get("p_value")
                lcb = l_rec.get("lcb95")
                z_sc = l_rec.get("z_score")
                q_val = l_rec.get("fdr_q_value")
                fail_cat = "STATISTICAL_UNDERPERFORMANCE"
                lesson = "Statistically significant underperformance vs cohort baseline."
            else:
                outcome = "PROVISIONAL"
                tier = c.get("selection_tier", "CANDIDATE_POOL")
                sharpe_se = c.get("sharpe_se")
                t_stat = c.get("t_vs_zero")
                p_val = c.get("p_value")
                lcb = c.get("lcb95")
                z_sc = c.get("z_score")
                q_val = c.get("fdr_q_value")
                fail_cat = None
                lesson = None

            ep = await self.record_research_episode(
                session_id=session_id,
                researcher_id=rid,
                candidate_id=cid,
                prompt_version_id=c.get("prompt_version_id"),
                generation=generation,
                strategy_focus=c.get("category"),
                dataset_id=c.get("dataset_id"),
                region=c.get("region"),
                universe=c.get("universe"),
                delay=c.get("delay"),
                hypothesis_family=c.get("hypothesis_family"),
                hypothesis=c.get("hypothesis"),
                rationale=c.get("rationale"),
                expression=c.get("expression"),
                funnel_stage="EVALUATED",
                outcome=outcome,
                selection_tier=tier,
                provenance=c.get("provenance", "LOCAL_PROXY"),
                validation_sharpe=c.get("sharpe"),
                validation_fitness=c.get("fitness"),
                validation_turnover=c.get("turnover"),
                validation_decay=c.get("sharpe_decay"),
                t_years=c.get("t_years"),
                sharpe_se=sharpe_se,
                t_stat=t_stat,
                p_value=p_val,
                lcb95=lcb,
                z_score=z_sc,
                fdr_q_value=q_val,
                is_statistically_significant=is_stat_sig,
                failure_category=fail_cat,
                lessons_learned=lesson,
            )
            episodes.append(ep)

        return episodes
