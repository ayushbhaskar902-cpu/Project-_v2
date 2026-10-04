"""Quantitative evaluation engine for alpha simulation results.

Separates In-Sample (IS) from Out-of-Sample (OOS) performance, quantifies strategy
overfitting via Sharpe decay and IS/OOS divergence, evaluates full 10-year After-Cost
Sharpe, verifies WorldQuant BRAIN submission thresholds, and computes multi-metric
composite ranking scores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
from sqlalchemy import select, update

from ..db.models import (
    CandidateStatus,
    ResearchCandidate,
    ResearchEvaluation,
    ResearchResearcher,
    ResearchSession,
    utcnow,
)
from ..vault.metrics import BOOK, YEAR, after_cost, after_cost_sharpe, stats

if TYPE_CHECKING:
    from ..db.sqlite import Database

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
    # In-sample metrics (typically first 8 of 10 years)
    is_sharpe: float | None = None
    is_fitness: float | None = None
    is_turnover: float | None = None
    is_returns: float | None = None
    is_drawdown: float | None = None
    is_margin: float | None = None

    # Out-of-sample metrics (typically last 2 of 10 years)
    oos_sharpe: float | None = None
    oos_fitness: float | None = None
    oos_turnover: float | None = None
    oos_returns: float | None = None
    oos_drawdown: float | None = None

    # Full period metrics (all 10 years)
    full_sharpe: float | None = None
    full_fitness: float | None = None
    after_cost_sharpe: float | None = None

    # Overfitting signals
    is_oos_gap: float | None = None
    sharpe_decay: float | None = None  # (IS - OOS) / IS

    # Composite leaderboard score
    composite_score: float | None = None

    # BRAIN submission check results
    checks: dict[str, bool] = field(default_factory=dict)
    checks_passed: int = 0
    checks_total: int = 0
    is_submission_grade: bool = False


def compute_composite_score(
    after_cost_sharpe_val: float | None,
    fitness_val: float | None,
    sharpe_decay_val: float | None,
    turnover_val: float | None,
) -> float:
    """Computes composite leaderboard score balancing risk-adjusted return and overfitting.

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
    cost_bps: float = 5.0,
) -> EvaluationScorecard:
    """Computes comprehensive IS, OOS, and full-period metrics from daily PnL and turnover arrays."""
    scorecard = EvaluationScorecard(candidate_id=candidate_id)

    if not dates or len(dates) < 20:
        return scorecard

    pnl_arr = np.array(pnl_series, dtype=np.float64)
    to_arr = np.array(turnover_series, dtype=np.float64)

    # Full period stats
    full_st = stats(pnl_arr, to_arr)
    if full_st:
        scorecard.full_sharpe = round(full_st.sharpe, 4) if full_st.sharpe is not None else None
        scorecard.full_fitness = round(full_st.fitness, 4) if full_st.fitness is not None else None

    # After-cost Sharpe
    ac_pnl = after_cost(pnl_arr, to_arr, cost_bps)
    ac_st = stats(ac_pnl, to_arr)
    if ac_st and ac_st.sharpe is not None:
        scorecard.after_cost_sharpe = round(ac_st.sharpe, 4)
    else:
        scorecard.after_cost_sharpe = scorecard.full_sharpe

    # Determine split date for IS / OOS (default: 80% IS, 20% OOS)
    if split_date is None:
        split_idx = int(len(dates) * 0.80)
        split_date = dates[split_idx]

    d_arr = np.array(dates, dtype="datetime64[D]")
    cut = np.datetime64(split_date, "D")

    is_mask = d_arr < cut
    oos_mask = d_arr >= cut

    # In-Sample stats
    if np.any(is_mask):
        is_st = stats(pnl_arr[is_mask], to_arr[is_mask])
        if is_st:
            scorecard.is_sharpe = round(is_st.sharpe, 4) if is_st.sharpe is not None else None
            scorecard.is_fitness = round(is_st.fitness, 4) if is_st.fitness is not None else None
            scorecard.is_turnover = round(is_st.turnover, 4)
            scorecard.is_returns = round(is_st.returns, 4)
            scorecard.is_drawdown = round(is_st.drawdown, 4)
            scorecard.is_margin = round(is_st.margin, 6)

    # Out-of-Sample stats
    if np.any(oos_mask):
        oos_st = stats(pnl_arr[oos_mask], to_arr[oos_mask])
        if oos_st:
            scorecard.oos_sharpe = round(oos_st.sharpe, 4) if oos_st.sharpe is not None else None
            scorecard.oos_fitness = round(oos_st.fitness, 4) if oos_st.fitness is not None else None
            scorecard.oos_turnover = round(oos_st.turnover, 4)
            scorecard.oos_returns = round(oos_st.returns, 4)
            scorecard.oos_drawdown = round(oos_st.drawdown, 4)

    # Overfitting signals
    if scorecard.is_sharpe is not None and scorecard.oos_sharpe is not None:
        gap = scorecard.is_sharpe - scorecard.oos_sharpe
        scorecard.is_oos_gap = round(gap, 4)
        if scorecard.is_sharpe > 0:
            scorecard.sharpe_decay = round(gap / scorecard.is_sharpe, 4)
        else:
            scorecard.sharpe_decay = 1.0
    else:
        scorecard.sharpe_decay = 0.0

    # Composite score
    scorecard.composite_score = compute_composite_score(
        scorecard.after_cost_sharpe,
        scorecard.full_fitness,
        scorecard.sharpe_decay,
        scorecard.is_turnover or (full_st.turnover if full_st else None),
    )

    # BRAIN Submission Checks
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
    scorecard.is_submission_grade = passed == len(checks)

    return scorecard


class EvaluationEngine:
    """Persists and manages quantitative candidate evaluations."""

    def __init__(self, db: Database) -> None:
        self.db = db

    async def evaluate_candidate(
        self,
        candidate_id: int,
        dates: list[date],
        pnl_series: list[float],
        turnover_series: list[float],
        split_date: date | None = None,
        cost_bps: float = 5.0,
    ) -> EvaluationScorecard:
        """Evaluates an alpha candidate and updates SQLite records."""
        scorecard = evaluate_pnl_series(
            candidate_id=candidate_id,
            dates=dates,
            pnl_series=pnl_series,
            turnover_series=turnover_series,
            split_date=split_date,
            cost_bps=cost_bps,
        )

        async with self.db.session() as session:
            # 1. Upsert evaluation record
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
                    checks=scorecard.checks,
                    checks_passed=scorecard.checks_passed,
                    checks_total=scorecard.checks_total,
                )
                session.add(eval_row)
            else:
                existing_eval.is_sharpe = scorecard.is_sharpe
                existing_eval.is_fitness = scorecard.is_fitness
                existing_eval.is_turnover = scorecard.is_turnover
                existing_eval.is_returns = scorecard.is_returns
                existing_eval.is_drawdown = scorecard.is_drawdown
                existing_eval.is_margin = scorecard.is_margin
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
                existing_eval.checks = scorecard.checks
                existing_eval.checks_passed = scorecard.checks_passed
                existing_eval.checks_total = scorecard.checks_total
                existing_eval.evaluated_at = utcnow()

            # 2. Update candidate status
            new_status = (
                CandidateStatus.ACCEPTED
                if scorecard.is_submission_grade
                else (CandidateStatus.EVALUATED if (scorecard.full_sharpe or 0) > 0 else CandidateStatus.REJECTED)
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
            sharpe=scorecard.full_sharpe,
            fitness=scorecard.full_fitness,
            composite=scorecard.composite_score,
            submission_grade=scorecard.is_submission_grade,
        )
        return scorecard
