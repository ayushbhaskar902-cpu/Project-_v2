"""Application services for research-session workflows.

These services own multi-step orchestration and persistence decisions. FastAPI routes
remain responsible for transport validation, authorization dependencies, and mapping
domain exceptions to the established HTTP contract.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select

from ..db.models import (
    CandidateStatus,
    ResearchCandidate,
    ResearchEvaluation,
    ResearchSession,
    ResearchStatus,
    utcnow,
)
from ..experiment import ExperimentRunner
from ..orchestrator.researchers import DEFAULT_RESEARCHER_ROSTER, ResearcherConfig
from ..prompts.library import StrategyCategory
from ..realtime import TOPIC_SESSION


class ResearchSessionService:
    """Coordinates session creation, candidate dispatch, and experiment lifecycle."""

    def __init__(self, state: Any) -> None:
        self.state = state

    async def create(self, payload: Any) -> tuple[int, list[ResearcherConfig]]:
        async with self.state.db.session() as session:
            session_row = ResearchSession(
                name=payload.name,
                prompt=payload.prompt,
                instrument_type=payload.instrument_type,
                region=payload.region,
                universe=payload.universe,
                delay=payload.delay,
                simulation_budget=payload.simulation_budget,
                learning_frequency=payload.learning_frequency,
                model_refs=payload.model_refs,
                constraints=payload.constraints,
                status=ResearchStatus.IDLE,
            )
            session.add(session_row)
            await session.flush()
            session_id = session_row.id

        roster_lookup = {cfg.model_ref: cfg for cfg in DEFAULT_RESEARCHER_ROSTER}
        roster = [
            roster_lookup.get(
                ref,
                ResearcherConfig(
                    model_ref=ref,
                    label=f"Researcher {ref.split(':')[-1]}",
                    strategy_focus=StrategyCategory.GENERAL,
                ),
            )
            for ref in payload.model_refs
        ]
        await self.state.orchestrator.initialize_session_researchers(session_id, roster)
        return session_id, roster

    async def enqueue(self, session_id: int, payload: Any) -> list[int]:
        return await self.state.orchestrator.enqueue_candidates_for_simulation(
            session_id=session_id,
            candidate_ids=payload.candidate_ids,
            task=payload.task,
            batch_engine=getattr(self.state, "engine", None),
        )

    async def generate(self, session_id: int, total_candidates: int) -> list[int]:
        return await self.state.orchestrator.run_generation_round(
            session_id=session_id, total_target_candidates=total_candidates
        )

    def _runner(self, session_id: int) -> ExperimentRunner:
        runner = ExperimentRunner(
            session_id=session_id,
            db=self.state.db,
            orchestrator=self.state.orchestrator,
            evaluation=self.state.evaluation,
            learning=self.state.learning,
            leaderboard=self.state.leaderboard,
            engine=getattr(self.state, "engine", None),
            on_phase_change=lambda data: self.state.hub.broadcast(TOPIC_SESSION, data),
        )
        self.state.experiments.register(runner)
        return runner

    async def start(self, session_id: int) -> None:
        runner = self.state.experiments.get(session_id) or self._runner(session_id)
        await runner.start()

    async def pause(self, session_id: int) -> None:
        runner = self.state.experiments.get(session_id)
        if runner is None or not runner.is_running:
            raise RuntimeError(
                f"Session {session_id} does not have an actively running experiment loop to pause."
            )
        await runner.pause()

    async def resume(self, session_id: int) -> None:
        runner = self.state.experiments.get(session_id) or self._runner(session_id)
        await runner.resume()

    async def cancel(self, session_id: int) -> None:
        runner = self.state.experiments.get(session_id)
        if runner is not None:
            await runner.cancel()
            self.state.experiments.unregister(session_id)
            return
        async with self.state.db.session() as session:
            sess_row = await session.get(ResearchSession, session_id)
            if sess_row is None:
                raise ValueError(f"ResearchSession {session_id} not found")
            sess_row.status = ResearchStatus.FAILED
            sess_row.current_phase = ResearchStatus.FAILED
            sess_row.message = "Experiment cancelled by user."
            sess_row.finished_at = utcnow()
            await session.commit()

    async def status(self, session_id: int) -> dict[str, Any]:
        runner = self.state.experiments.get(session_id)
        if runner is not None:
            return await runner.status()
        async with self.state.db.session() as session:
            sess_row = await session.get(ResearchSession, session_id)
            if sess_row is None:
                raise ValueError(f"ResearchSession {session_id} not found")
            return {
                "sessionId": sess_row.id,
                "status": sess_row.status,
                "currentPhase": sess_row.current_phase,
                "currentGeneration": sess_row.current_generation,
                "simulationBudget": sess_row.simulation_budget,
                "simulationsUsed": sess_row.simulations_used,
                "errorCount": sess_row.error_count or 0,
                "message": sess_row.message,
                "isRunning": False,
                "lastCheckpointAt": (
                    sess_row.last_checkpoint_at.isoformat()
                    if sess_row.last_checkpoint_at
                    else None
                ),
            }


class ResearchWorkflowService:
    """Owns multi-step robustness, graduation, learning, and pool workflows."""

    def __init__(self, state: Any) -> None:
        self.state = state

    async def graduate(self, session_id: int, candidate_id: int) -> dict[str, Any]:
        async with self.state.db.session() as session:
            cand = await session.get(ResearchCandidate, candidate_id)
            if cand is None or cand.session_id != session_id:
                raise ValueError(f"Candidate {candidate_id} was not found in session {session_id}.")
        scorecard = await self.state.evaluation.graduate_candidate(candidate_id=candidate_id)
        return {
            "status": "GRADUATED",
            "candidateId": candidate_id,
            "holdoutSharpe": scorecard.holdout_sharpe,
            "holdoutFitness": scorecard.holdout_fitness,
            "holdoutTurnover": scorecard.holdout_turnover,
            "holdoutReturns": scorecard.holdout_returns,
            "holdoutDrawdown": scorecard.holdout_drawdown,
            "holdoutAfterCostSharpe": scorecard.holdout_after_cost_sharpe,
            "configHash": scorecard.config_hash,
            "canonicalSpec": scorecard.canonical_spec,
            "evaluatedAt": scorecard.holdout_evaluated_at.isoformat()
            if scorecard.holdout_evaluated_at
            else None,
        }

    async def learning_cycle(self, session_id: int, category: str | None) -> dict[str, Any]:
        result = await self.state.learning.execute_learning_cycle(
            session_id=session_id, category=category
        )
        await self.state.leaderboard.capture_snapshot(
            session_id=session_id, generation=result.generation
        )
        updated_ids = getattr(
            result,
            "updated_prompt_strategy_ids",
            [result.updated_prompt_strategy_id]
            if result.updated_prompt_strategy_id
            else [],
        )
        cats_evolved = getattr(
            result, "categories_evolved", [getattr(result, "category", "general")]
        )
        return {
            "status": "success",
            "sessionId": session_id,
            "generation": result.generation,
            "batchSize": result.batch_size,
            "winnerCount": result.winner_count,
            "loserCount": result.loser_count,
            "avgSharpe": result.avg_sharpe,
            "bestSharpe": result.best_sharpe,
            "insightsGenerated": len(result.insights_generated),
            "updatedPromptStrategyId": result.updated_prompt_strategy_id,
            "updatedPromptStrategyIds": updated_ids,
            "categoriesEvolved": cats_evolved,
        }

    async def robustness(self, session_id: int, payload: Any) -> dict[str, Any]:
        async with self.state.db.session() as session:
            stmt = (
                select(ResearchCandidate, ResearchEvaluation)
                .join(ResearchEvaluation, ResearchEvaluation.candidate_id == ResearchCandidate.id)
                .where(ResearchCandidate.session_id == session_id)
            )
            if payload.candidate_ids:
                stmt = stmt.where(ResearchCandidate.id.in_(payload.candidate_ids))
            else:
                stmt = stmt.where(ResearchEvaluation.full_sharpe > 0)
            rows = list((await session.execute(stmt)).all())

        all_results: list[dict[str, Any]] = []
        total_passed = 0
        for cand, ev in rows:
            suite = self.state.robustness.generate_test_suite(
                candidate_id=cand.id,
                base_neutralization=cand.sim_neutralization or "INDUSTRY",
                base_decay=cand.sim_decay or 4,
                base_universe=cand.sim_universe or "TOP3000",
            )
            suite_results = await self.state.robustness.run_candidate_suite(
                candidate_id=cand.id, suite=suite, baseline_eval_id=ev.id
            )
            for result in suite_results:
                total_passed += int(result.passed)
                all_results.append({
                    "candidateId": cand.id,
                    "testType": result.test_type,
                    "testValue": result.test_value,
                    "baselineSharpe": result.baseline_sharpe,
                    "perturbedSharpe": result.perturbed_sharpe,
                    "sharpeDelta": result.sharpe_delta,
                    "passed": result.passed,
                    "status": result.status,
                    "resultKind": result.result_kind,
                    "isSynthetic": result.is_synthetic,
                    "brainSimulationId": result.brain_simulation_id,
                    "perturbedAlphaId": result.perturbed_alpha_id,
                    "perturbationConfig": result.perturbation_config,
                    "errorDetails": result.error_details,
                    "valStartDate": result.val_start_date.isoformat() if result.val_start_date else None,
                    "valEndDate": result.val_end_date.isoformat() if result.val_end_date else None,
                    "message": result.message,
                })
        return {
            "status": "success",
            "sessionId": session_id,
            "candidatesTested": len(rows),
            "totalTests": len(all_results),
            "passedTests": total_passed,
            "results": all_results,
        }

    async def curate_pool(self, payload: Any) -> dict[str, Any]:
        async with self.state.db.session() as session:
            stmt = (
                select(ResearchCandidate, ResearchEvaluation)
                .join(ResearchEvaluation, ResearchEvaluation.candidate_id == ResearchCandidate.id)
                .where(
                    ResearchEvaluation.is_verified.is_(True),
                    ResearchCandidate.status.in_([
                        CandidateStatus.EVALUATED,
                        CandidateStatus.ACCEPTED,
                        CandidateStatus.GRADUATED,
                    ]),
                    func.coalesce(ResearchEvaluation.val_sharpe, ResearchEvaluation.full_sharpe) >= 1.0,
                )
            )
            if payload.session_id is not None:
                stmt = stmt.where(ResearchCandidate.session_id == payload.session_id)
            rows = list((await session.execute(stmt)).all())

        if not rows:
            return {"poolSize": 0, "members": [], "combinedSharpe": 0.0, "maxCorrelation": 0.0, "avgCorrelation": 0.0}

        alpha_ids = [cand.alpha_id for cand, _ in rows if cand.alpha_id]
        real_series = await self.state.alphas.series(alpha_ids) if alpha_ids else {}
        candidates_data: list[dict[str, Any]] = []
        pnl_map: dict[int, list[float]] = {}
        to_map: dict[int, list[float]] = {}
        for cand, ev in rows:
            if not cand.alpha_id or cand.alpha_id not in real_series:
                continue
            days_data = sorted(real_series[cand.alpha_id].items())
            if len(days_data) < 20:
                continue
            is_3way = ev.evaluation_scheme in ("THREE_WAY_CHRONOLOGICAL", "SPLIT_3WAY_FINAL")
            if is_3way and ev.is_end_date and ev.val_end_date:
                days_data = [(d, val) for d, val in days_data if ev.is_end_date < d <= ev.val_end_date]
                if len(days_data) < 20:
                    continue
            cid = cand.id
            candidates_data.append({
                "id": cid,
                "alpha_id": cand.alpha_id,
                "expression": cand.expression,
                "sharpe": ev.val_sharpe if (is_3way and ev.val_sharpe is not None) else (ev.full_sharpe or 0.0),
                "fitness": ev.val_fitness if (is_3way and ev.val_fitness is not None) else (ev.full_fitness or 0.0),
                "turnover": ev.val_turnover if (is_3way and ev.val_turnover is not None) else (ev.is_turnover or 0.15),
                "composite_score": ev.val_composite_score if (is_3way and ev.val_composite_score is not None) else (ev.composite_score or 0.0),
                "composite_score_spec": ev.composite_score_spec,
            })
            pnl_map[cid] = [p for _, (p, _) in days_data]
            to_map[cid] = [t for _, (_, t) in days_data]

        if not candidates_data:
            return {"poolSize": 0, "members": [], "combinedSharpe": 0.0, "maxCorrelation": 0.0, "avgCorrelation": 0.0}
        pool_result = self.state.pool_optimizer.curate_pool(
            candidates_data=candidates_data,
            pnl_series_map=pnl_map,
            turnover_series_map=to_map,
            max_pairwise_corr=payload.max_pairwise_correlation,
            target_size=payload.target_pool_size,
        )
        return {
            "poolSize": pool_result.pool_size,
            "combinedSharpe": pool_result.combined_sharpe,
            "combinedReturns": pool_result.combined_returns,
            "combinedTurnover": pool_result.combined_turnover,
            "combinedDrawdown": pool_result.combined_drawdown,
            "maxCorrelation": pool_result.max_correlation,
            "avgCorrelation": pool_result.avg_correlation,
            "correlationMatrix": pool_result.correlation_matrix,
            "members": [
                {
                    "candidateId": m.candidate_id,
                    "alphaId": m.alpha_id,
                    "expression": m.expression,
                    "sharpe": m.sharpe,
                    "fitness": m.fitness,
                    "turnover": m.turnover,
                    "compositeScore": m.composite_score,
                    "compositeScoreSpec": m.composite_score_spec,
                    "weight": m.weight,
                }
                for m in pool_result.members
            ],
        }
