"""Quantitative evaluation engine for alpha simulation results.

Enforces strict three-way chronological partitioning:
- In-Sample (IS): First 60% of trading days (hypothesis development and diagnostic baseline)
- Validation: Middle 20% of trading days (candidate selection, model tournament, prompt evolution)
- Final Holdout: Final 20% of trading days (sealed, immutable one-way graduation)

Candidate selection score is computed 100% on Validation partition data.
Historical holdout assessments are registered program-wide in HoldoutAssessment to prevent
holdout reuse, cross-market peeking, and chronological regression under the strict walk-forward rule.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
from sqlalchemy import func, select, update

from ..db.models import (
    CandidateStatus,
    HoldoutAssessment,
    ResearchCandidate,
    ResearchEvaluation,
    ResearchResearcher,
    ResearchSession,
    PlatformEligibilityStatus,
    Submission,
    utcnow,
)
from ..vault.metrics import BOOK, YEAR, after_cost, after_cost_sharpe, stats

if TYPE_CHECKING:
    from ..db.sqlite import Database
    from ..vault.store import AlphaVault

log = structlog.get_logger(__name__)

# Standard WorldQuant BRAIN submission thresholds
DEFAULT_MIN_SHARPE = 1.25
DEFAULT_MIN_FITNESS = 1.00
DEFAULT_MIN_TURNOVER = 0.01
DEFAULT_MAX_TURNOVER = 0.70
DEFAULT_MAX_DRAWDOWN = 0.50
DEFAULT_MIN_MARGIN = 0.0005


@dataclass(slots=True)
class EvaluationScorecard:
    """Detailed quantitative performance breakdown of an alpha candidate."""

    candidate_id: int

    # In-sample metrics (typically first 60% of trading days)
    is_sharpe: float | None = None
    is_fitness: float | None = None
    is_turnover: float | None = None
    is_returns: float | None = None
    is_drawdown: float | None = None
    is_margin: float | None = None

    # Pure Validation partition metrics (typically middle 20% of trading days)
    val_sharpe: float | None = None
    val_fitness: float | None = None
    val_turnover: float | None = None
    val_returns: float | None = None
    val_drawdown: float | None = None
    val_after_cost_sharpe: float | None = None
    val_composite_score: float | None = None

    # Diagnostic only: IS-versus-Validation decay (strictly excluded from candidate selection)
    is_val_decay: float | None = None

    # Final Holdout metrics (strictly uncomputed / None during development; unsealed only at graduation)
    holdout_sharpe: float | None = None
    holdout_fitness: float | None = None
    holdout_turnover: float | None = None
    holdout_returns: float | None = None
    holdout_drawdown: float | None = None
    holdout_after_cost_sharpe: float | None = None
    holdout_evaluated_at: datetime | None = None

    # Out-of-sample metrics (legacy 2-way backward-compatibility)
    oos_sharpe: float | None = None
    oos_fitness: float | None = None
    oos_turnover: float | None = None
    oos_returns: float | None = None
    oos_drawdown: float | None = None

    # Full period metrics (legacy 2-way backward-compatibility)
    full_sharpe: float | None = None
    full_fitness: float | None = None
    after_cost_sharpe: float | None = None

    # Overfitting signals
    is_oos_gap: float | None = None
    sharpe_decay: float | None = None

    # Composite score (in 3-way split, mirrors val_composite_score for backward compatibility)
    composite_score: float | None = None

    # Validation check results
    checks: dict[str, bool] = field(default_factory=dict)
    checks_passed: int = 0
    checks_total: int = 0
    meets_internal_thresholds: bool = False

    @property
    def is_submission_grade(self) -> bool:
        """Deprecated compatibility alias; this is not BRAIN eligibility."""
        return self.meets_internal_thresholds

    # Exact quantitative evaluation parameters & dates
    cost_bps: float = 5.0
    split_date: date | None = None
    is_end_date: date | None = None
    val_end_date: date | None = None
    evaluation_scheme: str = "SPLIT_3WAY_DEV"

    # Frozen configuration fingerprint
    config_hash: str | None = None
    canonical_spec: dict[str, Any] | None = None


def compute_validation_composite_score(
    val_after_cost_sharpe_val: float | None,
    fitness_val: float | None,
    turnover_val: float | None,
) -> float:
    """Computes validation composite score purely from validation partition statistics.

    Formula:
      0.50 * ValAfterCostSharpe + 0.35 * ValFitness - 0.15 * min(ValTurnover, 1.0)
    All components are calculated exclusively from validation partition data.
    """
    ac_sharpe = val_after_cost_sharpe_val or 0.0
    fit = fitness_val or 0.0
    to = turnover_val if turnover_val is not None else 0.20
    score = 0.50 * ac_sharpe + 0.35 * fit - 0.15 * min(to, 1.0)
    return round(float(score), 4)


def compute_composite_score(
    after_cost_sharpe_val: float | None,
    fitness_val: float | None,
    sharpe_decay_val: float | None,
    turnover_val: float | None,
) -> float:
    """Computes legacy composite score for 2-way historical evaluation records.

    Formula:
      0.40 * AfterCostSharpe + 0.30 * Fitness + 0.20 * (1 - clamped_decay) - 0.10 * Turnover
    """
    ac_sharpe = after_cost_sharpe_val or 0.0
    fit = fitness_val or 0.0
    decay = sharpe_decay_val if sharpe_decay_val is not None else 0.5
    clamped_decay = min(max(decay, 0.0), 1.0)
    to = turnover_val or 0.20

    score = 0.40 * ac_sharpe + 0.30 * fit + 0.20 * (1.0 - clamped_decay) - 0.10 * min(to, 1.0)
    return round(float(score), 4)


def evaluate_pnl_series(
    candidate_id: int,
    dates: list[date],
    pnl_series: list[float],
    turnover_series: list[float],
    split_date: date | None = None,
    is_end_date: date | None = None,
    val_end_date: date | None = None,
    cost_bps: float = 5.0,
    evaluate_holdout: bool = False,
    legacy_2way: bool = False,
) -> EvaluationScorecard:
    """Computes quantitative metrics from daily PnL and turnover arrays.

    By default, executes strict 3-way chronological partitioning:
      IS: [0 .. is_end_date) (first 60% by default)
      Validation: [is_end_date .. val_end_date) (middle 20% by default)
      Holdout: [val_end_date .. end] (final 20% by default)

    When evaluate_holdout is False, holdout metrics remain strictly None.
    When legacy_2way is True, runs pre-P0-03 2-way evaluation for backward compatibility.
    """
    scorecard = EvaluationScorecard(candidate_id=candidate_id)

    if not dates:
        return scorecard

    pnl_arr = np.array(pnl_series, dtype=np.float64)
    to_arr = np.array(turnover_series, dtype=np.float64)

    # 1. Legacy 2-Way Evaluation Mode
    if legacy_2way:
        if len(dates) < 20:
            return scorecard

        full_st = stats(pnl_arr, to_arr)
        if full_st:
            scorecard.full_sharpe = round(full_st.sharpe, 4) if full_st.sharpe is not None else None
            scorecard.full_fitness = round(full_st.fitness, 4) if full_st.fitness is not None else None

        ac_pnl = after_cost(pnl_arr, to_arr, cost_bps)
        ac_st = stats(ac_pnl, to_arr)
        scorecard.after_cost_sharpe = round(ac_st.sharpe, 4) if ac_st and ac_st.sharpe is not None else scorecard.full_sharpe

        if split_date is None:
            split_idx = int(len(dates) * 0.80)
            split_date = dates[split_idx]

        scorecard.split_date = split_date
        scorecard.cost_bps = cost_bps
        scorecard.evaluation_scheme = "LEGACY_2WAY"

        d_arr = np.array(dates, dtype="datetime64[D]")
        cut = np.datetime64(split_date, "D")
        is_mask = d_arr < cut
        oos_mask = d_arr >= cut

        if np.any(is_mask):
            is_st = stats(pnl_arr[is_mask], to_arr[is_mask])
            if is_st:
                scorecard.is_sharpe = round(is_st.sharpe, 4) if is_st.sharpe is not None else None
                scorecard.is_fitness = round(is_st.fitness, 4) if is_st.fitness is not None else None
                scorecard.is_turnover = round(is_st.turnover, 4)
                scorecard.is_returns = round(is_st.returns, 4)
                scorecard.is_drawdown = round(is_st.drawdown, 4)
                scorecard.is_margin = round(is_st.margin, 6)

        if np.any(oos_mask):
            oos_st = stats(pnl_arr[oos_mask], to_arr[oos_mask])
            if oos_st:
                scorecard.oos_sharpe = round(oos_st.sharpe, 4) if oos_st.sharpe is not None else None
                scorecard.oos_fitness = round(oos_st.fitness, 4) if oos_st.fitness is not None else None
                scorecard.oos_turnover = round(oos_st.turnover, 4)
                scorecard.oos_returns = round(oos_st.returns, 4)
                scorecard.oos_drawdown = round(oos_st.drawdown, 4)

        if scorecard.is_sharpe is not None and scorecard.oos_sharpe is not None:
            gap = scorecard.is_sharpe - scorecard.oos_sharpe
            scorecard.is_oos_gap = round(gap, 4)
            scorecard.sharpe_decay = round(gap / scorecard.is_sharpe, 4) if scorecard.is_sharpe > 0 else 1.0
        else:
            scorecard.sharpe_decay = 0.0

        scorecard.composite_score = compute_composite_score(
            scorecard.after_cost_sharpe,
            scorecard.full_fitness,
            scorecard.sharpe_decay,
            scorecard.is_turnover or (full_st.turnover if full_st else None),
        )

        checks: dict[str, bool] = {}
        f_sharpe = scorecard.full_sharpe or 0.0
        f_fitness = scorecard.full_fitness or 0.0
        f_turnover = scorecard.is_turnover or (full_st.turnover if full_st else 0.0)
        f_drawdown = scorecard.is_drawdown or (full_st.drawdown if full_st else 1.0)
        f_margin = scorecard.is_margin or (full_st.margin if full_st else 0.0)

        checks["sharpe_gte_1_25"] = f_sharpe >= DEFAULT_MIN_SHARPE
        checks["fitness_gte_1_00"] = f_fitness >= DEFAULT_MIN_FITNESS
        checks["turnover_valid"] = DEFAULT_MIN_TURNOVER <= f_turnover <= DEFAULT_MAX_TURNOVER
        checks["drawdown_lte_50pct"] = f_drawdown <= DEFAULT_MAX_DRAWDOWN
        checks["margin_gte_5bps"] = f_margin >= DEFAULT_MIN_MARGIN
        checks["oos_sharpe_positive"] = (scorecard.oos_sharpe or 0.0) > 0.0

        passed = sum(1 for v in checks.values() if v)
        scorecard.checks = checks
        scorecard.checks_passed = passed
        scorecard.checks_total = len(checks)
        scorecard.meets_internal_thresholds = passed == len(checks)
        return scorecard

    # 2. Strict 3-Way Partitioning Mode
    if len(dates) < 60:
        raise InsufficientDataError(f"Series has {len(dates)} observations; minimum 60 required for 3-way split.")

    # Determine boundaries (default: 60% IS, 20% Validation, 20% Holdout)
    if is_end_date is None:
        if split_date is not None:
            is_end_date = split_date
        else:
            is_idx = int(len(dates) * 0.60)
            is_end_date = dates[is_idx]

    if val_end_date is None:
        val_idx = int(len(dates) * 0.80)
        val_end_date = dates[val_idx]

    d_arr = np.array(dates, dtype="datetime64[D]")
    is_cut = np.datetime64(is_end_date, "D")
    val_cut = np.datetime64(val_end_date, "D")

    if is_cut >= val_cut:
        raise InsufficientDataError(f"is_end_date ({is_end_date}) must be strictly before val_end_date ({val_end_date}).")

    is_mask = d_arr < is_cut
    val_mask = (d_arr >= is_cut) & (d_arr < val_cut)
    holdout_mask = d_arr >= val_cut

    if np.sum(is_mask) < 20 or np.sum(val_mask) < 20 or np.sum(holdout_mask) < 20:
        raise InsufficientDataError(
            f"Each partition must contain at least 20 observations (IS: {np.sum(is_mask)}, Val: {np.sum(val_mask)}, Holdout: {np.sum(holdout_mask)})."
        )

    scorecard.is_end_date = is_end_date
    scorecard.val_end_date = val_end_date
    scorecard.split_date = is_end_date  # Backward-compatible split_date mapping
    scorecard.cost_bps = cost_bps

    # A. In-Sample Partition Statistics
    is_st = stats(pnl_arr[is_mask], to_arr[is_mask])
    if is_st:
        scorecard.is_sharpe = round(is_st.sharpe, 4) if is_st.sharpe is not None else None
        scorecard.is_fitness = round(is_st.fitness, 4) if is_st.fitness is not None else None
        scorecard.is_turnover = round(is_st.turnover, 4)
        scorecard.is_returns = round(is_st.returns, 4)
        scorecard.is_drawdown = round(is_st.drawdown, 4)
        scorecard.is_margin = round(is_st.margin, 6)

    # B. Pure Validation Partition Statistics
    val_st = stats(pnl_arr[val_mask], to_arr[val_mask])
    if val_st:
        scorecard.val_sharpe = round(val_st.sharpe, 4) if val_st.sharpe is not None else None
        scorecard.val_fitness = round(val_st.fitness, 4) if val_st.fitness is not None else None
        scorecard.val_turnover = round(val_st.turnover, 4)
        scorecard.val_returns = round(val_st.returns, 4)
        scorecard.val_drawdown = round(val_st.drawdown, 4)

    val_ac_pnl = after_cost(pnl_arr[val_mask], to_arr[val_mask], cost_bps)
    val_ac_st = stats(val_ac_pnl, to_arr[val_mask])
    scorecard.val_after_cost_sharpe = round(val_ac_st.sharpe, 4) if val_ac_st and val_ac_st.sharpe is not None else scorecard.val_sharpe

    # Validation Selection Score (100% Validation partition variables)
    scorecard.val_composite_score = compute_validation_composite_score(
        scorecard.val_after_cost_sharpe,
        scorecard.val_fitness,
        scorecard.val_turnover,
    )
    scorecard.composite_score = scorecard.val_composite_score

    # Diagnostic only: IS-versus-Validation Decay
    if scorecard.is_sharpe is not None and scorecard.val_sharpe is not None:
        gap = scorecard.is_sharpe - scorecard.val_sharpe
        scorecard.is_val_decay = round(gap / scorecard.is_sharpe, 4) if scorecard.is_sharpe > 0 else 1.0
    else:
        scorecard.is_val_decay = 0.0

    # C. Final Holdout Partition Statistics (gated strictly by evaluate_holdout)
    if evaluate_holdout:
        h_st = stats(pnl_arr[holdout_mask], to_arr[holdout_mask])
        if h_st:
            scorecard.holdout_sharpe = round(h_st.sharpe, 4) if h_st.sharpe is not None else None
            scorecard.holdout_fitness = round(h_st.fitness, 4) if h_st.fitness is not None else None
            scorecard.holdout_turnover = round(h_st.turnover, 4)
            scorecard.holdout_returns = round(h_st.returns, 4)
            scorecard.holdout_drawdown = round(h_st.drawdown, 4)

        h_ac_pnl = after_cost(pnl_arr[holdout_mask], to_arr[holdout_mask], cost_bps)
        h_ac_st = stats(h_ac_pnl, to_arr[holdout_mask])
        scorecard.holdout_after_cost_sharpe = round(h_ac_st.sharpe, 4) if h_ac_st and h_ac_st.sharpe is not None else scorecard.holdout_sharpe
        scorecard.holdout_evaluated_at = utcnow()
        scorecard.evaluation_scheme = "SPLIT_3WAY_FINAL"
    else:
        scorecard.evaluation_scheme = "SPLIT_3WAY_DEV"

    # D. Development Gating Checks (governed strictly by Validation + fixed IS sanity guard)
    checks: dict[str, bool] = {}
    v_sharpe = scorecard.val_sharpe or 0.0
    v_fitness = scorecard.val_fitness or 0.0
    v_turnover = scorecard.val_turnover or 0.0
    v_drawdown = scorecard.val_drawdown if scorecard.val_drawdown is not None else 1.0
    i_sharpe = scorecard.is_sharpe or 0.0

    checks["is_sharpe_positive"] = i_sharpe > 0.0  # Fixed binary eligibility guard
    checks["val_sharpe_gte_1_25"] = v_sharpe >= DEFAULT_MIN_SHARPE
    checks["val_fitness_gte_1_00"] = v_fitness >= DEFAULT_MIN_FITNESS
    checks["val_turnover_valid"] = DEFAULT_MIN_TURNOVER <= v_turnover <= DEFAULT_MAX_TURNOVER
    checks["val_drawdown_lte_50pct"] = v_drawdown <= DEFAULT_MAX_DRAWDOWN

    passed = sum(1 for v in checks.values() if v)
    scorecard.checks = checks
    scorecard.checks_passed = passed
    scorecard.checks_total = len(checks)
    scorecard.meets_internal_thresholds = passed == len(checks)

    return scorecard


class InsufficientDataError(RuntimeError):
    """Raised when an alpha candidate lacks verified BRAIN daily series in the vault."""


class HoldoutChronologyError(RuntimeError):
    """Raised when holdout evaluation violates the strict forward-only walk-forward policy."""


class HoldoutOverlapError(RuntimeError):
    """Raised when a requested holdout window overlaps an already revealed historical period."""


class HoldoutIndeterminacyError(RuntimeError):
    """Raised when unresolved legacy contamination blocks holdout evaluation program-wide."""


class HoldoutImmutabilityError(RuntimeError):
    """Raised when attempting to modify a frozen, graduated candidate or its assessment."""


def build_canonical_spec(
    candidate: ResearchCandidate,
    scorecard: EvaluationScorecard,
    dates: list[date],
) -> dict[str, Any]:
    """Constructs a deterministic canonical specification dictionary for hashing."""
    return {
        "alpha_id": candidate.alpha_id,
        "candidate_id": candidate.id,
        "cost_bps": scorecard.cost_bps,
        "dates": {
            "is_start": dates[0].isoformat(),
            "is_end": scorecard.is_end_date.isoformat() if scorecard.is_end_date else None,
            "val_start": scorecard.is_end_date.isoformat() if scorecard.is_end_date else None,
            "val_end": scorecard.val_end_date.isoformat() if scorecard.val_end_date else None,
            "holdout_start": scorecard.val_end_date.isoformat() if scorecard.val_end_date else None,
            "holdout_end": dates[-1].isoformat(),
        },
        "evaluation_scheme": "SPLIT_3WAY_FINAL",
        "expression": candidate.expression,
        "language": candidate.language or "FASTEXPR",
        "methodology_version": "P0-03-V1",
        "sim_settings": candidate.effective_simulation_settings(),
        "total_observations": len(dates),
    }


class EvaluationEngine:
    """Persists and manages quantitative candidate evaluations."""

    def __init__(self, db: Database, vault: AlphaVault | None = None) -> None:
        self.db = db
        self.vault = vault
        self._graduation_lock = asyncio.Lock()

    async def evaluate_candidate(
        self,
        candidate_id: int,
        split_date: date | None = None,
        is_end_date: date | None = None,
        val_end_date: date | None = None,
        cost_bps: float = 5.0,
        legacy_2way: bool = False,
    ) -> EvaluationScorecard:
        """Evaluates an alpha candidate using verified series loaded directly from AlphaVault.

        Always runs in Development Mode (evaluate_holdout=False). Final holdout metrics remain None.
        """
        if self.vault is None:
            raise RuntimeError("AlphaVault is required for candidate evaluation.")

        async with self.db.session() as session:
            cand = await session.get(ResearchCandidate, candidate_id)
            if cand is None or not cand.alpha_id:
                raise InsufficientDataError(f"Candidate {candidate_id} has no completed BRAIN alpha_id.")
            if cand.status == CandidateStatus.GRADUATED:
                raise HoldoutImmutabilityError(
                    f"Candidate {candidate_id} is already GRADUATED and immutable. Re-evaluation is prohibited."
                )
            alpha_id = cand.alpha_id

        # 1. Retrieve verified series strictly through AlphaVault boundary
        verified = await self.vault.load_verified_series(alpha_id)
        if verified is None:
            raise InsufficientDataError(f"Candidate {candidate_id} (alpha {alpha_id}) lacks verified BRAIN series in vault.")

        dates, pnl_series, turnover_series = verified
        if len(dates) < 60:
            legacy_2way = True

        scorecard = evaluate_pnl_series(
            candidate_id=candidate_id,
            dates=dates,
            pnl_series=pnl_series,
            turnover_series=turnover_series,
            split_date=split_date,
            is_end_date=is_end_date,
            val_end_date=val_end_date,
            cost_bps=cost_bps,
            evaluate_holdout=False,
            legacy_2way=legacy_2way,
        )

        async with self.db.session() as session:
            # 2. Upsert evaluation record atomically with verified provenance
            eval_stmt = select(ResearchEvaluation).where(
                ResearchEvaluation.candidate_id == candidate_id
            )
            res = await session.execute(eval_stmt)
            existing_eval = res.scalar_one_or_none()

            if existing_eval is None:
                eval_row = ResearchEvaluation(
                    candidate_id=candidate_id,
                    is_sharpe=scorecard.is_sharpe,
                    is_fitness=scorecard.is_fitness,
                    is_turnover=scorecard.is_turnover,
                    is_returns=scorecard.is_returns,
                    is_drawdown=scorecard.is_drawdown,
                    is_margin=scorecard.is_margin,
                    val_sharpe=scorecard.val_sharpe,
                    val_fitness=scorecard.val_fitness,
                    val_turnover=scorecard.val_turnover,
                    val_returns=scorecard.val_returns,
                    val_drawdown=scorecard.val_drawdown,
                    val_after_cost_sharpe=scorecard.val_after_cost_sharpe,
                    val_composite_score=scorecard.val_composite_score,
                    is_val_decay=scorecard.is_val_decay,
                    holdout_sharpe=None,
                    holdout_fitness=None,
                    holdout_turnover=None,
                    holdout_returns=None,
                    holdout_drawdown=None,
                    holdout_after_cost_sharpe=None,
                    holdout_evaluated_at=None,
                    oos_sharpe=scorecard.oos_sharpe,
                    oos_fitness=scorecard.oos_fitness,
                    oos_turnover=scorecard.oos_turnover,
                    oos_returns=scorecard.oos_returns,
                    oos_drawdown=scorecard.oos_drawdown,
                    full_sharpe=scorecard.full_sharpe,
                    full_fitness=scorecard.full_fitness,
                    after_cost_sharpe=scorecard.after_cost_sharpe,
                    is_oos_gap=scorecard.is_oos_gap,
                    sharpe_decay=scorecard.sharpe_decay,
                    composite_score=scorecard.composite_score,
                    internal_checks=scorecard.checks,
                    checks_passed=scorecard.checks_passed,
                    checks_total=scorecard.checks_total,
                    cost_bps=scorecard.cost_bps,
                    split_date=scorecard.split_date,
                    is_end_date=scorecard.is_end_date,
                    val_end_date=scorecard.val_end_date,
                    evaluation_scheme=scorecard.evaluation_scheme,
                    is_verified=True,
                    provenance="BRAIN_VAULT",
                )
                session.add(eval_row)
            else:
                existing_eval.is_sharpe = scorecard.is_sharpe
                existing_eval.is_fitness = scorecard.is_fitness
                existing_eval.is_turnover = scorecard.is_turnover
                existing_eval.is_returns = scorecard.is_returns
                existing_eval.is_drawdown = scorecard.is_drawdown
                existing_eval.is_margin = scorecard.is_margin
                existing_eval.val_sharpe = scorecard.val_sharpe
                existing_eval.val_fitness = scorecard.val_fitness
                existing_eval.val_turnover = scorecard.val_turnover
                existing_eval.val_returns = scorecard.val_returns
                existing_eval.val_drawdown = scorecard.val_drawdown
                existing_eval.val_after_cost_sharpe = scorecard.val_after_cost_sharpe
                existing_eval.val_composite_score = scorecard.val_composite_score
                existing_eval.is_val_decay = scorecard.is_val_decay
                existing_eval.oos_sharpe = scorecard.oos_sharpe
                existing_eval.oos_fitness = scorecard.oos_fitness
                existing_eval.oos_turnover = scorecard.oos_turnover
                existing_eval.oos_returns = scorecard.oos_returns
                existing_eval.oos_drawdown = scorecard.oos_drawdown
                existing_eval.full_sharpe = scorecard.full_sharpe
                existing_eval.full_fitness = scorecard.full_fitness
                existing_eval.after_cost_sharpe = scorecard.after_cost_sharpe
                existing_eval.is_oos_gap = scorecard.is_oos_gap
                existing_eval.sharpe_decay = scorecard.sharpe_decay
                existing_eval.composite_score = scorecard.composite_score
                existing_eval.internal_checks = scorecard.checks
                existing_eval.checks_passed = scorecard.checks_passed
                existing_eval.checks_total = scorecard.checks_total
                existing_eval.cost_bps = scorecard.cost_bps
                existing_eval.split_date = scorecard.split_date
                existing_eval.is_end_date = scorecard.is_end_date
                existing_eval.val_end_date = scorecard.val_end_date
                existing_eval.evaluation_scheme = scorecard.evaluation_scheme
                existing_eval.is_verified = True
                existing_eval.provenance = "BRAIN_VAULT"
                existing_eval.evaluated_at = utcnow()

            # 3. Update candidate development status
            target_sharpe = scorecard.val_sharpe if scorecard.val_sharpe is not None else scorecard.full_sharpe
            new_status = (
                CandidateStatus.ACCEPTED
                if scorecard.is_submission_grade
                else (CandidateStatus.EVALUATED if (target_sharpe or 0) > 0 else CandidateStatus.REJECTED)
            )

            cand_stmt = (
                update(ResearchCandidate)
                .where(ResearchCandidate.id == candidate_id)
                .values(
                    status=new_status,
                    finished_at=utcnow(),
                )
            )
            await session.execute(cand_stmt)

        log.info(
            "evaluation.candidate_evaluated",
            candidate_id=candidate_id,
            val_sharpe=scorecard.val_sharpe,
            val_composite=scorecard.val_composite_score,
            submission_grade=scorecard.is_submission_grade,
        )
        return scorecard

    async def record_platform_eligibility(
        self,
        candidate_id: int,
        platform_checks: dict[str, Any],
        *,
        offline: bool = False,
    ) -> str:
        """Persist an actual BRAIN check response without changing research quality.

        ``offline`` is explicit test/offline evidence and is never reported as live
        platform eligibility. Live eligibility additionally requires verified provenance
        and complete, verified simulation settings.
        """
        async with self.db.session() as session:
            cand = await session.get(ResearchCandidate, candidate_id)
            ev = (await session.execute(
                select(ResearchEvaluation).where(ResearchEvaluation.candidate_id == candidate_id)
            )).scalar_one_or_none()
            if cand is None or ev is None:
                raise InsufficientDataError(f"Candidate {candidate_id} has no evaluation.")
            if offline:
                status = PlatformEligibilityStatus.EXEMPT_OFFLINE
            elif not ev.is_verified or ev.provenance != "BRAIN_VAULT":
                status = PlatformEligibilityStatus.INELIGIBLE
            elif not cand.is_simulation_settings_complete:
                status = PlatformEligibilityStatus.INELIGIBLE
            else:
                results = [str(v.get("result", v)).upper() for v in platform_checks.values() if isinstance(v, dict)]
                status = (
                    PlatformEligibilityStatus.ELIGIBLE
                    if results and all(r in {"PASS", "WARNING"} for r in results)
                    else PlatformEligibilityStatus.INELIGIBLE
                )
            cand.platform_eligibility_status = status
            cand.platform_eligibility_checked_at = utcnow()
            ev.checks = platform_checks
            return status

    async def graduate_candidate(self, candidate_id: int) -> EvaluationScorecard:
        """Executes explicit, one-way final holdout evaluation for an eligible candidate.

        Enforces:
        1. Eligibility: Candidate must be ACCEPTED.
        2. Idempotency: Repeated graduation returns existing evaluation without recomputing.
        3. Program-Wide Safeguard: Rejects graduation if any unresolved LEGACY_CONTAMINATED_UNKNOWN exists.
        4. Strict Walk-Forward Rule: holdout_start_date must be strictly after the latest consumed date.
        5. Zero Overlap: Rejects any overlapping historical holdout window across the program.
        6. Immutability: Creates frozen HoldoutAssessment record and transitions candidate to GRADUATED.
        """
        if self.vault is None:
            raise RuntimeError("AlphaVault is required for candidate graduation.")

        async with self._graduation_lock:
            async with self.db.session() as session:
                cand = await session.get(ResearchCandidate, candidate_id)
            if cand is None or not cand.alpha_id:
                raise InsufficientDataError(f"Candidate {candidate_id} has no completed BRAIN alpha_id.")

            # Idempotent return for already-graduated candidates
            if cand.status == CandidateStatus.GRADUATED:
                eval_stmt = select(ResearchEvaluation).where(ResearchEvaluation.candidate_id == candidate_id)
                res_eval = await session.execute(eval_stmt)
                existing_eval = res_eval.scalar_one_or_none()
                if existing_eval is not None and existing_eval.holdout_evaluated_at is not None:
                    return EvaluationScorecard(
                        candidate_id=candidate_id,
                        val_sharpe=existing_eval.val_sharpe,
                        val_fitness=existing_eval.val_fitness,
                        val_turnover=existing_eval.val_turnover,
                        val_returns=existing_eval.val_returns,
                        val_drawdown=existing_eval.val_drawdown,
                        val_after_cost_sharpe=existing_eval.val_after_cost_sharpe,
                        val_composite_score=existing_eval.val_composite_score,
                        is_val_decay=existing_eval.is_val_decay,
                        holdout_sharpe=existing_eval.holdout_sharpe,
                        holdout_fitness=existing_eval.holdout_fitness,
                        holdout_turnover=existing_eval.holdout_turnover,
                        holdout_returns=existing_eval.holdout_returns,
                        holdout_drawdown=existing_eval.holdout_drawdown,
                        holdout_after_cost_sharpe=existing_eval.holdout_after_cost_sharpe,
                        holdout_evaluated_at=existing_eval.holdout_evaluated_at,
                        is_end_date=existing_eval.is_end_date,
                        val_end_date=existing_eval.val_end_date,
                        evaluation_scheme=existing_eval.evaluation_scheme,
                        config_hash=existing_eval.config_hash,
                        canonical_spec=existing_eval.canonical_spec,
                    )

            if cand.status != CandidateStatus.ACCEPTED:
                raise HoldoutChronologyError(
                    f"Candidate {candidate_id} has status '{cand.status}'; must be 'ACCEPTED' to graduate."
                )

            # Program-Wide Unknown Legacy Contamination Guard
            unknown_stmt = select(HoldoutAssessment).where(
                HoldoutAssessment.assessment_type == "LEGACY_CONTAMINATED_UNKNOWN"
            )
            has_unknown = (await session.execute(unknown_stmt)).scalars().first()
            if has_unknown is not None:
                raise HoldoutIndeterminacyError(
                    "Program-wide final holdout graduation is blocked: unresolved LEGACY_CONTAMINATED_UNKNOWN "
                    f"record exists in registry (Candidate {has_unknown.candidate_id}). "
                    "All legacy contamination must be resolved before independent final assessments are permitted."
                )

            # Retrieve verified series
            verified = await self.vault.load_verified_series(cand.alpha_id)
            if verified is None:
                raise InsufficientDataError(f"Candidate {candidate_id} lacks verified series in vault.")

            dates, pnl_series, turnover_series = verified
            if len(dates) < 60:
                raise InsufficientDataError("At least 60 trading days required for graduation evaluation.")

            val_idx = int(len(dates) * 0.80)
            h_start_date = dates[val_idx]
            h_end_date = dates[-1]

            async with self.db.session() as session:
                # Strict Walk-Forward Advancement Check
                watermark_stmt = select(func.max(HoldoutAssessment.holdout_end_date)).where(
                    HoldoutAssessment.holdout_end_date.is_not(None)
                )
                latest_consumed = (await session.execute(watermark_stmt)).scalar_one_or_none()

                if latest_consumed is not None and h_start_date <= latest_consumed:
                    raise HoldoutChronologyError(
                        f"Holdout start date {h_start_date} is not strictly after latest consumed research date {latest_consumed}. "
                        "Under the strict walk-forward policy, historical or overlapping periods cannot be evaluated as an independent final holdout."
                    )

                # Program-Wide Interval Overlap Check
                overlap_stmt = select(HoldoutAssessment).where(
                    HoldoutAssessment.holdout_start_date <= h_end_date,
                    HoldoutAssessment.holdout_end_date >= h_start_date,
                )
                matched = (await session.execute(overlap_stmt)).scalars().first()
                if matched is not None:
                    raise HoldoutOverlapError(
                        f"Requested holdout window [{h_start_date} to {h_end_date}] overlaps existing revealed holdout "
                        f"[{matched.holdout_start_date} to {matched.holdout_end_date}] in scope '{matched.market_scope}'."
                    )

            # Run Holdout Evaluation
            scorecard = evaluate_pnl_series(
                candidate_id=candidate_id,
                dates=dates,
                pnl_series=pnl_series,
                turnover_series=turnover_series,
                cost_bps=5.0,
                evaluate_holdout=True,
            )

            canonical_spec = build_canonical_spec(cand, scorecard, dates)
            canonical_bytes = json.dumps(canonical_spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
            config_hash = hashlib.sha256(canonical_bytes).hexdigest()

            scorecard.config_hash = config_hash
            scorecard.canonical_spec = canonical_spec

            async with self.db.session() as session:
                # Commit immutable HoldoutAssessment registry record
                market_scope = f"{cand.sim_region or 'USA'}:{cand.sim_universe or 'TOP3000'}"
                assessment_row = HoldoutAssessment(
                    assessment_type="FINAL_ASSESSMENT",
                    holdout_start_date=h_start_date,
                    holdout_end_date=h_end_date,
                    observation_count=len([d for d in dates if d >= h_start_date]),
                    market_scope=market_scope,
                    candidate_id=candidate_id,
                    session_id=cand.session_id,
                    alpha_id=cand.alpha_id,
                    config_hash=config_hash,
                    canonical_spec=canonical_spec,
                    holdout_sharpe=scorecard.holdout_sharpe,
                    holdout_fitness=scorecard.holdout_fitness,
                    holdout_turnover=scorecard.holdout_turnover,
                    holdout_returns=scorecard.holdout_returns,
                    holdout_drawdown=scorecard.holdout_drawdown,
                    evaluated_at=utcnow(),
                )
                session.add(assessment_row)

                # Update ResearchEvaluation
                eval_stmt = select(ResearchEvaluation).where(ResearchEvaluation.candidate_id == candidate_id)
                res_eval = await session.execute(eval_stmt)
                eval_row = res_eval.scalar_one_or_none()

                if eval_row is not None:
                    eval_row.holdout_sharpe = scorecard.holdout_sharpe
                    eval_row.holdout_fitness = scorecard.holdout_fitness
                    eval_row.holdout_turnover = scorecard.holdout_turnover
                    eval_row.holdout_returns = scorecard.holdout_returns
                    eval_row.holdout_drawdown = scorecard.holdout_drawdown
                    eval_row.holdout_after_cost_sharpe = scorecard.holdout_after_cost_sharpe
                    eval_row.holdout_evaluated_at = scorecard.holdout_evaluated_at
                    eval_row.evaluation_scheme = "SPLIT_3WAY_FINAL"
                    eval_row.config_hash = config_hash
                    eval_row.canonical_spec = canonical_spec

                # Lock candidate as GRADUATED
                cand_row = await session.get(ResearchCandidate, candidate_id)
                if cand_row:
                    cand_row.status = CandidateStatus.GRADUATED

            log.info(
                "evaluation.candidate_graduated",
                candidate_id=candidate_id,
                holdout_sharpe=scorecard.holdout_sharpe,
                holdout_start=h_start_date,
                holdout_end=h_end_date,
            )
            return scorecard

    async def backfill_legacy_holdout_contamination(self) -> int:
        """Identifies pre-P0-03 legacy evaluation records and backfills contamination registry.

        Registers full historical date span [min_date, max_date] for series in DuckDB.
        Registers LEGACY_CONTAMINATED_UNKNOWN without fabricated dates when series is missing.
        Idempotent: skips candidates already registered in HoldoutAssessment.
        """
        backfilled = 0
        async with self.db.session() as session:
            # Find legacy evaluations
            stmt = (
                select(ResearchEvaluation, ResearchCandidate)
                .join(ResearchCandidate, ResearchCandidate.id == ResearchEvaluation.candidate_id)
                .where(
                    ResearchEvaluation.evaluation_scheme == "LEGACY_2WAY"
                )
            )
            res = await session.execute(stmt)
            legacy_rows = res.all()

            for ev, cand in legacy_rows:
                # Check if already registered
                check_stmt = select(HoldoutAssessment).where(HoldoutAssessment.candidate_id == cand.id)
                if (await session.execute(check_stmt)).scalar_one_or_none() is not None:
                    continue

                min_date: date | None = None
                max_date: date | None = None
                obs_count: int | None = None

                # Query DuckDB if vault is available
                if self.vault and self.vault.catalog and cand.alpha_id:
                    try:
                        duck_res = await self.vault.catalog.query(
                            """
                            SELECT min(date) as min_d, max(date) as max_d, count(*) as cnt
                            FROM alpha_pnl
                            WHERE alpha_id = ? AND pnl IS NOT NULL AND turnover IS NOT NULL
                            """,
                            [cand.alpha_id],
                        )
                        if duck_res and duck_res[0].get("min_d") is not None:
                            min_date = duck_res[0]["min_d"]
                            max_date = duck_res[0]["max_d"]
                            obs_count = duck_res[0]["cnt"]
                    except Exception as exc:
                        log.warning("migration.duckdb_query_failed", alpha_id=cand.alpha_id, error=str(exc))

                market_scope = f"{cand.sim_region or 'USA'}:{cand.sim_universe or 'TOP3000'}"

                if min_date is not None and max_date is not None:
                    assessment = HoldoutAssessment(
                        assessment_type="LEGACY_CONTAMINATED",
                        holdout_start_date=min_date,
                        holdout_end_date=max_date,
                        observation_count=obs_count,
                        market_scope=market_scope,
                        candidate_id=cand.id,
                        session_id=cand.session_id,
                        alpha_id=cand.alpha_id,
                        evaluated_at=ev.evaluated_at or utcnow(),
                    )
                else:
                    assessment = HoldoutAssessment(
                        assessment_type="LEGACY_CONTAMINATED_UNKNOWN",
                        holdout_start_date=None,
                        holdout_end_date=None,
                        observation_count=None,
                        market_scope=market_scope,
                        candidate_id=cand.id,
                        session_id=cand.session_id,
                        alpha_id=cand.alpha_id,
                        evaluated_at=ev.evaluated_at or utcnow(),
                    )

                session.add(assessment)
                backfilled += 1

        log.info("migration.legacy_contamination_backfilled", count=backfilled)
        return backfilled

