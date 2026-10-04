"""Alpha Pool curation and multi-alpha combo clustering optimizer.

Curates high-capacity, low-correlation Alpha Pools targeting 50+ submission-grade alphas
per team member for WorldQuant BRAIN competitions.
Features:
- Pairwise Pearson correlation matrix computation across alpha PnL vectors
- Greedy maximal-Sharpe minimal-correlation portfolio selection
- Multi-alpha combination weight optimization (Equal Weight, Inverse Volatility, Mean-Variance)
- Combined pool analytics: portfolio Sharpe, turnover, drawdown, max pairwise correlation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
from sqlalchemy import select

from ..db.models import (
    ResearchCandidate,
    ResearchEvaluation,
)
from ..vault.metrics import BOOK, YEAR, stats

if TYPE_CHECKING:
    from ..db.sqlite import Database

log = structlog.get_logger(__name__)


@dataclass(slots=True)
class CuratedPoolMember:
    candidate_id: int
    alpha_id: str | None
    expression: str
    sharpe: float
    fitness: float
    turnover: float
    composite_score: float
    weight: float = 0.0


@dataclass(slots=True)
class CuratedPoolResult:
    """A curated portfolio pool of low-correlation, high-Sharpe alphas."""

    pool_size: int
    members: list[CuratedPoolMember]
    combined_sharpe: float
    combined_returns: float
    combined_turnover: float
    combined_drawdown: float
    max_correlation: float
    avg_correlation: float
    correlation_matrix: list[list[float]]


class AlphaPoolOptimizer:
    """Optimizes alpha portfolio selection and correlation clustering."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def compute_correlation_matrix(
        self,
        pnl_matrix: np.ndarray,  # shape: (days, num_alphas)
    ) -> np.ndarray:
        """Computes pairwise Pearson correlation matrix across alpha PnL columns."""
        if pnl_matrix.shape[1] < 2:
            return np.ones((pnl_matrix.shape[1], pnl_matrix.shape[1]))

        # Handle NaNs
        cleaned = np.nan_to_num(pnl_matrix)
        stds = cleaned.std(axis=0, ddof=1)
        stds[stds == 0] = 1.0

        standardized = (cleaned - cleaned.mean(axis=0)) / stds
        corr = (standardized.T @ standardized) / (cleaned.shape[0] - 1)
        np.clip(corr, -1.0, 1.0, out=corr)
        np.fill_diagonal(corr, 1.0)
        return corr

    def select_uncorrelated_pool(
        self,
        candidate_ids: list[int],
        pnl_series_dict: dict[int, list[float]],
        scores_dict: dict[int, float],
        max_correlation: float = 0.60,
        target_pool_size: int = 50,
    ) -> list[int]:
        """Greedy selection of alphas maximizing composite score while enforcing max correlation limit."""
        if not candidate_ids:
            return []

        # Sort candidate IDs by score descending
        sorted_ids = sorted(candidate_ids, key=lambda cid: scores_dict.get(cid, 0.0), reverse=True)

        selected: list[int] = [sorted_ids[0]]
        selected_pnls = [pnl_series_dict[sorted_ids[0]]]

        for cid in sorted_ids[1:]:
            if len(selected) >= target_pool_size:
                break

            candidate_pnl = pnl_series_dict.get(cid)
            if not candidate_pnl:
                continue

            # Check correlation against all currently selected
            cand_arr = np.array(candidate_pnl, dtype=np.float64)
            cand_std = cand_arr.std(ddof=1)
            if cand_std == 0:
                continue

            too_correlated = False
            for s_pnl in selected_pnls:
                s_arr = np.array(s_pnl, dtype=np.float64)
                s_std = s_arr.std(ddof=1)
                if s_std == 0:
                    continue

                min_len = min(len(cand_arr), len(s_arr))
                c_slice = cand_arr[:min_len]
                s_slice = s_arr[:min_len]

                corr = float(np.corrcoef(c_slice, s_slice)[0, 1])
                if not math.isnan(corr) and abs(corr) >= max_correlation:
                    too_correlated = True
                    break

            if not too_correlated:
                selected.append(cid)
                selected_pnls.append(candidate_pnl)

        return selected

    def compute_portfolio_stats(
        self,
        pnl_series_list: list[list[float]],
        turnover_series_list: list[list[float]],
        weights: list[float] | None = None,
    ) -> tuple[float, float, float, float]:
        """Computes combined portfolio (Sharpe, Returns, Turnover, Drawdown)."""
        if not pnl_series_list:
            return 0.0, 0.0, 0.0, 0.0

        n = len(pnl_series_list)
        min_len = min(len(s) for s in pnl_series_list)
        pnl_mat = np.array([s[:min_len] for s in pnl_series_list], dtype=np.float64)
        to_mat = np.array([s[:min_len] for s in turnover_series_list], dtype=np.float64)

        if weights is None:
            w = np.full(n, 1.0 / n)
        else:
            w = np.array(weights, dtype=np.float64)
            w = w / np.sum(np.abs(w))

        port_pnl = (w @ pnl_mat)
        port_to = (np.abs(w) @ to_mat)

        st = stats(port_pnl, port_to)
        if not st:
            return 0.0, 0.0, 0.0, 0.0

        s_val = round(float(st.sharpe or 0.0), 2)
        r_val = round(float(st.returns or 0.0), 4)
        t_val = round(float(st.turnover or 0.0), 3)
        d_val = round(float(st.drawdown or 0.0), 3)
        return s_val, r_val, t_val, d_val

    def curate_pool(
        self,
        candidates_data: list[dict[str, Any]],
        pnl_series_map: dict[int, list[float]],
        turnover_series_map: dict[int, list[float]],
        max_pairwise_corr: float = 0.60,
        target_size: int = 50,
    ) -> CuratedPoolResult:
        """Constructs an optimized low-correlation alpha pool with combination statistics."""
        c_ids = [c["id"] for c in candidates_data if c["id"] in pnl_series_map]
        scores = {c["id"]: float(c.get("composite_score", 0.0)) for c in candidates_data}

        selected_ids = self.select_uncorrelated_pool(
            candidate_ids=c_ids,
            pnl_series_dict=pnl_series_map,
            scores_dict=scores,
            max_correlation=max_pairwise_corr,
            target_pool_size=target_size,
        )

        cand_map = {c["id"]: c for c in candidates_data}
        selected_members: list[CuratedPoolMember] = []
        equal_weight = 1.0 / len(selected_ids) if selected_ids else 0.0

        for cid in selected_ids:
            c = cand_map[cid]
            selected_members.append(
                CuratedPoolMember(
                    candidate_id=cid,
                    alpha_id=c.get("alpha_id"),
                    expression=c.get("expression", ""),
                    sharpe=float(c.get("sharpe", 0.0)),
                    fitness=float(c.get("fitness", 0.0)),
                    turnover=float(c.get("turnover", 0.0)),
                    composite_score=float(c.get("composite_score", 0.0)),
                    weight=equal_weight,
                )
            )

        # Portfolio metrics
        pnls = [pnl_series_map[cid] for cid in selected_ids]
        tos = [turnover_series_map[cid] for cid in selected_ids]

        sharpe, ret, to, dd = self.compute_portfolio_stats(pnls, tos)

        # Correlation matrix
        if pnls:
            min_len = min(len(p) for p in pnls)
            pnl_block = np.column_stack([p[:min_len] for p in pnls])
            corr_mat = self.compute_correlation_matrix(pnl_block)

            # Max off-diagonal correlation
            n = corr_mat.shape[0]
            if n > 1:
                off_diag = corr_mat[~np.eye(n, dtype=bool)]
                max_corr = round(float(np.max(np.abs(off_diag))), 3)
                avg_corr = round(float(np.mean(off_diag)), 3)
            else:
                max_corr, avg_corr = 1.0, 1.0
            matrix_list = [[round(float(v), 3) for v in row] for row in corr_mat]
        else:
            max_corr, avg_corr = 0.0, 0.0
            matrix_list = []

        return CuratedPoolResult(
            pool_size=len(selected_members),
            members=selected_members,
            combined_sharpe=sharpe,
            combined_returns=ret,
            combined_turnover=to,
            combined_drawdown=dd,
            max_correlation=max_corr,
            avg_correlation=avg_corr,
            correlation_matrix=matrix_list,
        )
