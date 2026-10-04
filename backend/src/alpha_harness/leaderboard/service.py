"""Multi-axis quantitative leaderboard and experimental research ranking service.

Treats the leaderboard as a live experimental instrument ranking research assets across
four dimensions:
1. Alpha Candidate Leaderboard (Composite Score, After-Cost Sharpe, OOS Stability)
2. LLM Researcher Leaderboard (Yield rate, top-decile alpha rate, avg Sharpe)
3. Prompt Strategy Leaderboard (Prompt version progression and Sharpe lift)
4. Historical Trend Archival (LeaderboardSnapshot persistence across generations)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import desc, func, select

from ..db.models import (
    LeaderboardSnapshot,
    PromptStrategy,
    ResearchCandidate,
    ResearchEvaluation,
    ResearchResearcher,
    ResearchSession,
    utcnow,
)

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


@dataclass(slots=True)
class ResearcherRankEntry:
    rank: int
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
            )

            if session_id is not None:
                stmt = stmt.where(ResearchCandidate.session_id == session_id)
            if min_sharpe is not None:
                stmt = stmt.where(ResearchEvaluation.full_sharpe >= min_sharpe)
            if submission_grade_only:
                stmt = stmt.where(
                    ResearchEvaluation.checks_passed == ResearchEvaluation.checks_total,
                    ResearchEvaluation.checks_total > 0,
                )

            stmt = stmt.order_by(desc(ResearchEvaluation.composite_score)).limit(limit)
            res = await session.execute(stmt)
            rows = res.all()

            entries: list[AlphaRankEntry] = []
            for idx, (cand, ev, r_label) in enumerate(rows, start=1):
                entries.append(
                    AlphaRankEntry(
                        rank=idx,
                        candidate_id=cand.id,
                        expression=cand.expression or "",
                        hypothesis=cand.hypothesis or "",
                        model_label=r_label or "Unknown Researcher",
                        generation=cand.generation or 0,
                        composite_score=round(float(ev.composite_score or 0.0), 3),
                        full_sharpe=round(float(ev.full_sharpe or 0.0), 2),
                        after_cost_sharpe=round(float(ev.after_cost_sharpe or 0.0), 2),
                        oos_sharpe=round(float(ev.oos_sharpe or 0.0), 2),
                        sharpe_decay=round(float(ev.sharpe_decay or 0.0), 2),
                        turnover=round(float(ev.is_turnover or 0.0), 3),
                        fitness=round(float(ev.full_fitness or 0.0), 2),
                        is_submission_grade=ev.checks_passed == ev.checks_total and ev.checks_total > 0,
                    )
                )

            return entries

    async def get_researcher_leaderboard(
        self, session_id: int | None = None
    ) -> list[ResearcherRankEntry]:
        """Ranks LLM researchers by collective research productivity and alpha quality."""
        async with self.db.session() as session:
            stmt = select(ResearchResearcher)
            if session_id is not None:
                stmt = stmt.where(ResearchResearcher.session_id == session_id)

            res = await session.execute(stmt)
            researchers = list(res.scalars().all())

            entries: list[ResearcherRankEntry] = []
            for r in researchers:
                tot_gen = r.total_generated or 0
                tot_val = r.total_valid or 0
                v_rate = round(tot_val / tot_gen, 2) if tot_gen else 0.0

                # Count submission grade alphas
                c_stmt = (
                    select(func.count(ResearchCandidate.id))
                    .join(
                        ResearchEvaluation,
                        ResearchEvaluation.candidate_id == ResearchCandidate.id,
                    )
                    .where(
                        ResearchCandidate.researcher_id == r.id,
                        ResearchEvaluation.checks_passed == ResearchEvaluation.checks_total,
                        ResearchEvaluation.checks_total > 0,
                    )
                )
                c_res = await session.execute(c_stmt)
                sub_count = c_res.scalar() or 0

                entries.append(
                    ResearcherRankEntry(
                        rank=0,
                        researcher_id=r.id,
                        label=r.label or "Researcher",
                        model_ref=r.model_ref or "",
                        strategy_focus=r.strategy_focus or "general",
                        total_generated=tot_gen,
                        total_valid=tot_val,
                        valid_rate=v_rate,
                        avg_sharpe=round(float(r.avg_sharpe or 0.0), 2),
                        best_sharpe=round(float(r.best_sharpe or 0.0), 2),
                        submission_grade_count=sub_count,
                        top_decile_rate=round(sub_count / tot_gen, 2) if tot_gen else 0.0,
                    )
                )

            # Rank by submission_grade_count desc, then avg_sharpe desc
            entries.sort(
                key=lambda e: (e.submission_grade_count, e.avg_sharpe), reverse=True
            )
            for idx, entry in enumerate(entries, start=1):
                entry.rank = idx

            return entries

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
                    "label": r.label,
                    "model_ref": r.model_ref,
                    "submission_count": r.submission_grade_count,
                    "avg_sharpe": r.avg_sharpe,
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
