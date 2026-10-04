"""Long-term quantitative memory and feature attribution store.

Tracks statistical efficacy of Fast Expression operators, data fields, categories, and
prompt versions across generations, providing empirical evidence for the self-learning loop.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import select

from ..db.models import (
    ResearchCandidate,
    ResearchEvaluation,
    ResearchInsight,
    ResearchResearcher,
    ResearchSession,
    utcnow,
)

if TYPE_CHECKING:
    from ..db.sqlite import Database

log = structlog.get_logger(__name__)


@dataclass(slots=True)
class OperatorEfficacy:
    """Historical performance of a specific Fast Expression operator."""

    name: str
    appearances: int = 0
    winner_count: int = 0
    loser_count: int = 0
    avg_sharpe: float = 0.0
    sharpe_values: list[float] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return (self.winner_count / self.appearances) if self.appearances else 0.0


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

    def __init__(self, db: Database) -> None:
        self.db = db

    async def get_session_performance_split(
        self, session_id: int, generation: int | None = None
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Separates candidates into top performers (winners) and underperformers (losers).

        Returns: (winners, losers) list of dicts with expression, hypothesis, and metrics.
        """
        async with self.db.session() as session:
            stmt = (
                select(ResearchCandidate, ResearchEvaluation)
                .join(
                    ResearchEvaluation,
                    ResearchEvaluation.candidate_id == ResearchCandidate.id,
                )
                .where(ResearchCandidate.session_id == session_id)
            )
            if generation is not None:
                stmt = stmt.where(ResearchCandidate.generation == generation)

            res = await session.execute(stmt)
            pairs = res.all()

            if not pairs:
                return [], []

            records = []
            for cand, ev in pairs:
                records.append(
                    {
                        "candidate_id": cand.id,
                        "researcher_id": cand.researcher_id,
                        "generation": cand.generation,
                        "hypothesis": cand.hypothesis,
                        "expression": cand.expression,
                        "fields": cand.fields_used or [],
                        "operator_count": cand.operator_count,
                        "sharpe": ev.full_sharpe or 0.0,
                        "fitness": ev.full_fitness or 0.0,
                        "after_cost_sharpe": ev.after_cost_sharpe or 0.0,
                        "sharpe_decay": ev.sharpe_decay or 0.0,
                        "turnover": ev.is_turnover or 0.0,
                        "composite_score": ev.composite_score or 0.0,
                        "is_submission_grade": ev.checks_passed == ev.checks_total and ev.checks_total > 0,
                    }
                )

            # Sort by composite score descending
            records.sort(key=lambda r: float(r["composite_score"]), reverse=True)

            cutoff_top = max(1, len(records) // 4)
            winners = records[:cutoff_top]
            losers = [r for r in records[cutoff_top:] if r["sharpe"] < 0.8 or r["sharpe_decay"] > 0.4]

            return winners, losers

    async def compute_operator_stats(
        self, session_id: int
    ) -> dict[str, OperatorEfficacy]:
        """Aggregates operator-level Sharpe lift and win rates."""
        async with self.db.session() as session:
            stmt = (
                select(ResearchCandidate.expression, ResearchEvaluation.full_sharpe)
                .join(
                    ResearchEvaluation,
                    ResearchEvaluation.candidate_id == ResearchCandidate.id,
                )
                .where(ResearchCandidate.session_id == session_id)
            )
            res = await session.execute(stmt)
            rows = res.all()

            stats_map: dict[str, OperatorEfficacy] = {}
            for expr, sharpe in rows:
                if not expr:
                    continue
                s_val = float(sharpe or 0.0)

                # Simple token extraction for operator names
                import re
                ops = re.findall(r"([a-z_][a-z0-9_]*)\s*\(", expr.lower())
                for op in set(ops):
                    if op not in stats_map:
                        stats_map[op] = OperatorEfficacy(name=op)
                    item = stats_map[op]
                    item.appearances += 1
                    item.sharpe_values.append(s_val)
                    if s_val >= 1.25:
                        item.winner_count += 1
                    elif s_val < 0.5:
                        item.loser_count += 1

            for item in stats_map.values():
                if item.sharpe_values:
                    item.avg_sharpe = round(sum(item.sharpe_values) / len(item.sharpe_values), 3)

            return stats_map
