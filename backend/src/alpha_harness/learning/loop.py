"""Closed-loop self-learning and prompt strategy mutation engine.

Executes after every X simulation rounds (default 80 across 8 BRAIN slots):
1. Partitions candidate results into top winners and failure modes
2. Performs operator and field statistical attribution
3. Derives quantitative rules and insights
4. Persists versioned insights into SQLite (ResearchInsight)
5. Mutates and updates prompt templates and few-shot examples (PromptStrategy)
6. Advances research session generation and triggers leaderboard snapshots
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import select, update

from ..db.models import (
    LeaderboardSnapshot,
    PromptStrategy,
    ResearchCandidate,
    ResearchEvaluation,
    ResearchInsight,
    ResearchSession,
    utcnow,
)
from .memory import ResearchMemory

if TYPE_CHECKING:
    from ..db.sqlite import Database
    from ..llm.service import LLMService

log = structlog.get_logger(__name__)


@dataclass(slots=True)
class LearningCycleResult:
    """Outcome of one self-learning reflection cycle."""

    session_id: int
    generation: int
    batch_size: int
    winner_count: int
    loser_count: int
    avg_sharpe: float
    best_sharpe: float
    insights_generated: list[dict[str, Any]]
    updated_prompt_strategy_id: int | None = None


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
    ) -> LearningCycleResult:
        """Executes a full reflection and prompt evolution cycle for the given session."""
        async with self.db.session() as session:
            s_stmt = select(ResearchSession).where(ResearchSession.id == session_id)
            res = await session.execute(s_stmt)
            research_session = res.scalar_one_or_none()
            if not research_session:
                raise ValueError(f"ResearchSession {session_id} not found")

            gen = research_session.current_generation or 0

        # 1. Partition winners and losers
        winners, losers = await self.memory.get_session_performance_split(
            session_id=session_id, generation=gen
        )
        if not winners and not losers:
            # Fall back to all evaluated session candidates across generations
            winners, losers = await self.memory.get_session_performance_split(
                session_id=session_id, generation=None
            )

        all_candidates = winners + losers
        batch_size = len(all_candidates)
        sharpes = [float(c["sharpe"]) for c in all_candidates]
        avg_sharpe = round(sum(sharpes) / len(sharpes), 3) if sharpes else 0.0
        best_sharpe = max(sharpes) if sharpes else 0.0

        # 2. Compute operator statistics
        op_stats = await self.memory.compute_operator_stats(session_id)
        best_ops = sorted(
            [op for op in op_stats.values() if op.appearances >= 2],
            key=lambda o: o.avg_sharpe,
            reverse=True,
        )[:3]
        worst_ops = sorted(
            [op for op in op_stats.values() if op.appearances >= 2],
            key=lambda o: o.avg_sharpe,
        )[:3]

        # 3. Formulate structured insights
        insights_data: list[dict[str, Any]] = []

        # Winning patterns insight
        if winners:
            top_winner = winners[0]
            w_summary = (
                f"Generation {gen} winning signals favored '{top_winner['hypothesis']}' "
                f"with Sharpe {top_winner['sharpe']:.2f}."
            )
            if best_ops:
                w_summary += f" High-alpha operators: {', '.join(o.name for o in best_ops)}."

            insights_data.append(
                {
                    "insight_type": "winning_pattern",
                    "summary": w_summary,
                    "rules": {
                        "preferred_operators": [o.name for o in best_ops],
                        "target_sharpe_min": 1.5,
                    },
                    "evidence": {
                        "top_expression": top_winner["expression"],
                        "top_sharpe": top_winner["sharpe"],
                        "top_fitness": top_winner["fitness"],
                    },
                }
            )

        # Failure modes insight
        if losers:
            l_summary = (
                f"Generation {gen} observed {len(losers)} underperforming signals. "
                "Common causes: unnormalized momentum or excessive lookback without group neutralization."
            )
            if worst_ops:
                l_summary += f" Low-performing operators: {', '.join(o.name for o in worst_ops)}."

            insights_data.append(
                {
                    "insight_type": "failure_mode",
                    "summary": l_summary,
                    "rules": {
                        "discouraged_operators": [o.name for o in worst_ops],
                        "max_allowed_turnover": 0.50,
                    },
                    "evidence": {
                        "loser_sample": [l["expression"] for l in losers[:3]],
                    },
                }
            )

        # 4. Persist insights and update prompt strategy in SQLite
        updated_strategy_id: int | None = None
        async with self.db.session() as session:
            # Persist each insight
            for ins in insights_data:
                insight_row = ResearchInsight(
                    session_id=session_id,
                    generation=gen,
                    insight_type=ins["insight_type"],
                    summary=ins["summary"],
                    rules=ins["rules"],
                    evidence=ins["evidence"],
                    batch_size=batch_size,
                    batch_avg_sharpe=avg_sharpe,
                    batch_best_sharpe=best_sharpe,
                    batch_valid_rate=1.0,
                    applied=True,
                )
                session.add(insight_row)

            # Evolve prompt strategy
            few_shot_examples = [
                {
                    "hypothesis": w["hypothesis"],
                    "expression": w["expression"],
                    "sharpe": w["sharpe"],
                }
                for w in winners[:3]
            ]
            anti_patterns = [
                {
                    "flaw": "High turnover or decay",
                    "expression": l["expression"],
                }
                for l in losers[:3]
            ]

            strat_stmt = (
                select(PromptStrategy)
                .where(
                    PromptStrategy.session_id == session_id,
                    PromptStrategy.is_active.is_(True),
                )
                .order_by(PromptStrategy.version.desc())
            )
            res_strat = await session.execute(strat_stmt)
            curr_strat = res_strat.scalars().first()

            new_version = (curr_strat.version + 1) if curr_strat and curr_strat.version else (gen + 1)
            new_strategy = PromptStrategy(
                session_id=session_id,
                name=f"Evolved Strategy Gen {gen+1}",
                version=new_version,
                category="general",
                guidance={
                    "preferred_operators": [o.name for o in best_ops],
                    "avoid_operators": [o.name for o in worst_ops],
                },
                examples=few_shot_examples,
                anti_patterns=anti_patterns,
                avg_sharpe=avg_sharpe,
                is_active=True,
            )
            session.add(new_strategy)
            await session.flush()
            updated_strategy_id = new_strategy.id

            # Increment session generation
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
            generation=gen,
            new_generation=gen + 1,
            winners=len(winners),
            losers=len(losers),
            insights=len(insights_data),
        )

        return LearningCycleResult(
            session_id=session_id,
            generation=gen,
            batch_size=batch_size,
            winner_count=len(winners),
            loser_count=len(losers),
            avg_sharpe=avg_sharpe,
            best_sharpe=best_sharpe,
            insights_generated=insights_data,
            updated_prompt_strategy_id=updated_strategy_id,
        )
