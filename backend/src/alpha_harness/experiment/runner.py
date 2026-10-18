"""Autonomous experiment lifecycle runner (P0-05).

Drives a ``ResearchSession`` from start to completion through the cycle:

    generate → enqueue → wait for simulations → evaluate → learn → repeat

Every phase transition is durably checkpointed to SQLite, so a process restart
can resume where it left off. Budget enforcement, error isolation, and
pause/resume are handled at phase boundaries.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import select, update

from ..db.models import (
    CandidateStatus,
    ResearchCandidate,
    ResearchSession,
    ResearchStatus,
    SimStatus,
    SimulationRecord,
    utcnow,
)

if TYPE_CHECKING:
    from ..db.sqlite import Database
    from ..evaluation import EvaluationEngine
    from ..learning import SelfLearningEngine
    from ..leaderboard import LeaderboardService
    from ..orchestrator.researchers import MultiLLMOrchestrator

log = structlog.get_logger(__name__)

#: How often the runner checks whether all simulations for the current round have finished.
SIMULATION_POLL_SECONDS = 5.0

#: Maximum consecutive unrecoverable errors before the session transitions to FAILED.
MAX_CONSECUTIVE_ERRORS = 5

#: Active phases the runner can be in.
_ACTIVE_PHASES = frozenset({
    ResearchStatus.RUNNING,
    ResearchStatus.GENERATING,
    ResearchStatus.SIMULATING,
    ResearchStatus.EVALUATING,
    ResearchStatus.LEARNING,
})


class ExperimentRunner:
    """Owns and drives the autonomous loop for one research session.

    Lifecycle methods (``start``, ``pause``, ``resume``, ``cancel``) are safe to
    call from request handlers; the loop itself runs as a detached ``asyncio.Task``.
    """

    def __init__(
        self,
        session_id: int,
        db: Database,
        orchestrator: MultiLLMOrchestrator,
        evaluation: EvaluationEngine,
        learning: SelfLearningEngine,
        leaderboard: LeaderboardService,
        *,
        on_phase_change: Any | None = None,
        engine: Any | None = None,
    ) -> None:
        self.session_id = session_id
        self.db = db
        self.orchestrator = orchestrator
        self.evaluation = evaluation
        self.learning = learning
        self.leaderboard = leaderboard
        self._on_phase_change = on_phase_change
        self._engine = engine
        self._task: asyncio.Task[None] | None = None
        self._pause_requested = False

    # -- public lifecycle API ------------------------------------------------

    async def start(self) -> None:
        """Start the autonomous loop from IDLE. Raises if already running."""
        session = await self._load_session()
        if session is None:
            raise ValueError(f"ResearchSession {self.session_id} not found")
        if session.status in _ACTIVE_PHASES:
            raise RuntimeError(
                f"Session {self.session_id} is already in phase {session.status}"
            )
        if session.status == ResearchStatus.COMPLETE:
            raise RuntimeError(f"Session {self.session_id} is already COMPLETE")

        await self._checkpoint(ResearchStatus.RUNNING, "Autonomous experiment started.")
        self._spawn_loop()

    async def resume(self) -> None:
        """Resume a PAUSED or crashed-in-flight session."""
        session = await self._load_session()
        if session is None:
            raise ValueError(f"ResearchSession {self.session_id} not found")

        # Allow resume from PAUSED or from an active phase (crash recovery)
        if session.status == ResearchStatus.PAUSED:
            await self._checkpoint(ResearchStatus.RUNNING, "Resumed from PAUSED.")
        elif session.status in _ACTIVE_PHASES:
            log.info("experiment.resume_from_active", session_id=self.session_id,
                     phase=session.status)
        else:
            raise RuntimeError(
                f"Cannot resume session {self.session_id} from status {session.status}"
            )
        self._spawn_loop()

    async def pause(self) -> None:
        """Request pause at the next phase boundary."""
        self._pause_requested = True
        log.info("experiment.pause_requested", session_id=self.session_id)

    async def cancel(self) -> None:
        """Cancel the session immediately."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        await self._checkpoint(
            ResearchStatus.FAILED, "Experiment cancelled by user."
        )

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def status(self) -> dict[str, Any]:
        """Return current experiment status for the API."""
        session = await self._load_session()
        if session is None:
            return {"sessionId": self.session_id, "status": "NOT_FOUND"}
        return {
            "sessionId": self.session_id,
            "status": session.status,
            "currentPhase": session.current_phase,
            "currentGeneration": session.current_generation,
            "simulationBudget": session.simulation_budget,
            "simulationsUsed": session.simulations_used,
            "errorCount": session.error_count,
            "message": session.message,
            "isRunning": self.is_running,
            "lastCheckpointAt": (
                session.last_checkpoint_at.isoformat()
                if session.last_checkpoint_at
                else None
            ),
        }

    # -- internal loop -------------------------------------------------------

    def _spawn_loop(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._pause_requested = False
        self._task = asyncio.create_task(
            self._run(), name=f"experiment-{self.session_id}"
        )
        self._task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            log.info("experiment.task_cancelled", session_id=self.session_id)
        elif task.exception():
            log.error(
                "experiment.task_crashed",
                session_id=self.session_id,
                error=str(task.exception()),
            )

    async def _run(self) -> None:
        """The main generate-simulate-evaluate-learn loop."""
        log.info("experiment.loop_started", session_id=self.session_id)

        try:
            while True:
                # --- pause gate ---
                if self._pause_requested:
                    await self._checkpoint(
                        ResearchStatus.PAUSED, "Paused at phase boundary."
                    )
                    self._pause_requested = False
                    return

                # --- status & budget gate ---
                session = await self._load_session()
                if session is None or session.status in (
                    ResearchStatus.COMPLETE,
                    ResearchStatus.FAILED,
                ):
                    return
                budget = session.simulation_budget or 500
                used = session.simulations_used or 0
                if used >= budget:
                    await self._checkpoint(
                        ResearchStatus.COMPLETE,
                        f"Budget exhausted: {used}/{budget} simulations used.",
                    )
                    return

                # === PHASE 1: GENERATE ===
                await self._checkpoint(ResearchStatus.GENERATING, "Generating candidates...")
                candidate_ids = await self._phase_generate()
                if candidate_ids is None:
                    # Unrecoverable generation failure; session transitioned to FAILED
                    return
                if not candidate_ids:
                    if await self._should_fail():
                        return
                    continue

                # === PHASE 2: ENQUEUE & WAIT FOR SIMULATIONS ===
                await self._checkpoint(
                    ResearchStatus.SIMULATING,
                    f"Simulating {len(candidate_ids)} candidates...",
                    candidate_ids=candidate_ids,
                )
                await self._phase_simulate(candidate_ids)
                if await self._should_fail():
                    return

                # === PHASE 3: EVALUATE ===
                await self._checkpoint(
                    ResearchStatus.EVALUATING, "Evaluating completed simulations..."
                )
                await self._phase_evaluate()
                if await self._should_fail():
                    return

                # === PHASE 4: LEARN ===
                await self._checkpoint(
                    ResearchStatus.LEARNING, "Running self-learning cycle..."
                )
                await self._phase_learn()
                if await self._should_fail():
                    return

                # --- Sync used count and check budget again ---
                await self.orchestrator.sync_session_simulations(self.session_id)

        except asyncio.CancelledError:
            log.info("experiment.loop_cancelled", session_id=self.session_id)
            raise
        except Exception:
            log.exception("experiment.loop_fatal", session_id=self.session_id)
            await self._checkpoint(
                ResearchStatus.FAILED,
                "Experiment stopped due to an unexpected error. Check logs.",
            )

    # -- phase implementations -----------------------------------------------

    async def _phase_generate(self) -> list[int] | None:
        """Generate one round of candidates. Returns candidate IDs or None on failure."""
        try:
            valid_ids = await self.orchestrator.run_generation_round(
                session_id=self.session_id
            )
            if not valid_ids:
                await self._record_error("Generation produced no valid candidates.")
                if await self._should_fail():
                    return None
                # Still return empty — the loop will continue to the budget check
                return []
            await self._reset_errors()
            log.info(
                "experiment.generated",
                session_id=self.session_id,
                count=len(valid_ids),
            )
            return valid_ids
        except Exception as exc:
            log.warning(
                "experiment.generate_failed",
                session_id=self.session_id,
                error=str(exc),
            )
            await self._record_error(f"Generation failed: {exc}")
            if await self._should_fail():
                return None
            return []

    async def _phase_simulate(self, candidate_ids: list[int]) -> None:
        """Enqueue candidates and wait for all simulations to complete."""
        if not candidate_ids:
            return

        try:
            await self.orchestrator.enqueue_candidates_for_simulation(
                session_id=self.session_id,
                candidate_ids=candidate_ids,
                batch_engine=self._engine,
            )
        except Exception as exc:
            log.warning(
                "experiment.enqueue_failed",
                session_id=self.session_id,
                error=str(exc),
            )
            await self._record_error(f"Enqueue failed: {exc}")
            return

        start_time = asyncio.get_event_loop().time()
        # Poll until all simulation records for these candidates are terminal
        while True:
            if self._pause_requested:
                return

            await asyncio.sleep(SIMULATION_POLL_SECONDS)

            # Sync candidate states with simulation records
            await self.orchestrator.sync_session_simulations(self.session_id)

            # Check if all candidates in this round have resolved
            async with self.db.session() as session:
                stmt = (
                    select(ResearchCandidate)
                    .where(
                        ResearchCandidate.session_id == self.session_id,
                        ResearchCandidate.id.in_(candidate_ids),
                    )
                )
                result = await session.execute(stmt)
                candidates = list(result.scalars().all())

            # If all candidates have no simulation_record_id and no alpha_id, they cannot be synced
            if candidates and all(c.simulation_record_id is None and c.alpha_id is None for c in candidates):
                log.warning(
                    "experiment.simulations_untracked",
                    session_id=self.session_id,
                    count=len(candidates),
                )
                return

            still_pending = [
                c for c in candidates
                if not c.alpha_id
                and c.status in (
                    CandidateStatus.QUEUED,
                    CandidateStatus.SIMULATING,
                    CandidateStatus.VALID,
                )
            ]
            if not still_pending:
                log.info(
                    "experiment.simulations_complete",
                    session_id=self.session_id,
                    total=len(candidates),
                )
                await self._reset_errors()
                return

            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= 300.0:
                log.warning(
                    "experiment.simulation_timeout",
                    session_id=self.session_id,
                    pending=len(still_pending),
                    elapsed=elapsed,
                )
                async with self.db.session() as session:
                    for c in still_pending:
                        cand_row = await session.get(ResearchCandidate, c.id)
                        if cand_row:
                            cand_row.status = CandidateStatus.UNAVAILABLE
                            cand_row.validation_error = "Simulation timed out"
                    await session.commit()
                return

    async def _phase_evaluate(self) -> None:
        """Evaluate all candidates with completed simulations."""
        try:
            # Fetch candidates with alpha_ids that need evaluation
            async with self.db.session() as session:
                stmt = (
                    select(ResearchCandidate)
                    .where(
                        ResearchCandidate.session_id == self.session_id,
                        ResearchCandidate.is_valid.is_(True),
                        ResearchCandidate.alpha_id.is_not(None),
                        ResearchCandidate.status.in_([
                            CandidateStatus.SIMULATING,
                            CandidateStatus.QUEUED,
                            CandidateStatus.VALID,
                        ]),
                    )
                )
                result = await session.execute(stmt)
                candidates = list(result.scalars().all())

            evaluated = 0
            for cand in candidates:
                if cand.alpha_id is None:
                    continue
                try:
                    await self.evaluation.evaluate_candidate(
                        candidate_id=cand.id, cost_bps=5.0
                    )
                    evaluated += 1
                except Exception as exc:
                    log.warning(
                        "experiment.evaluate_candidate_failed",
                        candidate_id=cand.id,
                        error=str(exc),
                    )
                    async with self.db.session() as session:
                        c_row = await session.get(ResearchCandidate, cand.id)
                        if c_row and c_row.status in (
                            CandidateStatus.SIMULATING,
                            CandidateStatus.QUEUED,
                            CandidateStatus.VALID,
                        ):
                            c_row.status = CandidateStatus.UNAVAILABLE
                            c_row.validation_error = f"Evaluation failed: {exc}"
                        await session.commit()

            log.info(
                "experiment.evaluated",
                session_id=self.session_id,
                evaluated=evaluated,
                total=len(candidates),
            )
            await self._reset_errors()
        except Exception as exc:
            log.warning(
                "experiment.evaluate_failed",
                session_id=self.session_id,
                error=str(exc),
            )
            await self._record_error(f"Evaluation failed: {exc}")

    async def _phase_learn(self) -> None:
        """Run the self-learning reflection cycle and advance the generation."""
        try:
            result = await self.learning.execute_learning_cycle(
                session_id=self.session_id, category=None
            )
            await self.leaderboard.capture_snapshot(
                session_id=self.session_id,
                generation=result.generation,
            )
            updated_ids = getattr(
                result,
                "updated_prompt_strategy_ids",
                [result.updated_prompt_strategy_id] if result.updated_prompt_strategy_id else [],
            )
            cats_evolved = getattr(
                result,
                "categories_evolved",
                [getattr(result, "category", "general")],
            )
            log.info(
                "experiment.learned",
                session_id=self.session_id,
                generation=result.generation,
                categories_evolved=cats_evolved,
                updated_strategy_ids=updated_ids,
                winners=result.winner_count,
                insights=len(result.insights_generated),
            )
            await self._reset_errors()
        except Exception as exc:
            log.warning(
                "experiment.learn_failed",
                session_id=self.session_id,
                error=str(exc),
            )
            # Learning failure is non-fatal: log and continue to the next generation
            await self._record_error(f"Learning failed: {exc}")

    # -- helpers -------------------------------------------------------------

    async def _load_session(self) -> ResearchSession | None:
        async with self.db.session() as session:
            return await session.get(ResearchSession, self.session_id)

    async def _checkpoint(
        self,
        status: ResearchStatus,
        message: str,
        *,
        candidate_ids: list[int] | None = None,
    ) -> None:
        """Durably persist the current phase so a restart can resume."""
        values: dict[str, Any] = {
            "status": status,
            "current_phase": status,
            "message": message,
            "updated_at": utcnow(),
            "last_checkpoint_at": utcnow(),
        }
        if candidate_ids is not None:
            values["current_round_candidate_ids"] = candidate_ids
        if status == ResearchStatus.RUNNING:
            values["started_at"] = utcnow()
        if status in (ResearchStatus.COMPLETE, ResearchStatus.FAILED):
            values["finished_at"] = utcnow()
            values["current_round_candidate_ids"] = []

        async with self.db.session() as session:
            await session.execute(
                update(ResearchSession)
                .where(ResearchSession.id == self.session_id)
                .values(**values)
            )

        log.info(
            "experiment.checkpoint",
            session_id=self.session_id,
            status=status,
            message=message,
        )
        if self._on_phase_change:
            try:
                await self._on_phase_change(
                    {"sessionId": self.session_id, "status": status, "message": message}
                )
            except Exception:
                pass

    async def _record_error(self, message: str) -> None:
        """Increment the consecutive error counter."""
        async with self.db.session() as session:
            await session.execute(
                update(ResearchSession)
                .where(ResearchSession.id == self.session_id)
                .values(
                    error_count=ResearchSession.error_count + 1,
                    message=message,
                    updated_at=utcnow(),
                )
            )
        log.warning("experiment.error_recorded", session_id=self.session_id, message=message)

    async def _reset_errors(self) -> None:
        """Reset the consecutive error counter on success."""
        async with self.db.session() as session:
            await session.execute(
                update(ResearchSession)
                .where(ResearchSession.id == self.session_id)
                .values(error_count=0, updated_at=utcnow())
            )

    async def _should_fail(self) -> bool:
        """Check if the session should transition to FAILED due to too many errors."""
        session = await self._load_session()
        if session is None:
            return True
        errors = session.error_count or 0
        if errors >= MAX_CONSECUTIVE_ERRORS:
            await self._checkpoint(
                ResearchStatus.FAILED,
                f"Experiment stopped after {errors} consecutive errors.",
            )
            return True
        return False


class ExperimentRegistry:
    """Manages active ExperimentRunners, one per session.

    Owned by ``AppState``, this provides the coordination layer between
    API endpoints and background experiment loops.
    """

    def __init__(self) -> None:
        self._runners: dict[int, ExperimentRunner] = {}

    def get(self, session_id: int) -> ExperimentRunner | None:
        return self._runners.get(session_id)

    def register(self, runner: ExperimentRunner) -> None:
        self._runners[runner.session_id] = runner

    def unregister(self, session_id: int) -> None:
        self._runners.pop(session_id, None)

    def active_sessions(self) -> list[int]:
        return [sid for sid, r in self._runners.items() if r.is_running]

    async def stop_all(self) -> None:
        """Cancel all running experiments. Called during shutdown."""
        for runner in list(self._runners.values()):
            if runner.is_running:
                try:
                    await runner.cancel()
                except Exception:
                    log.exception(
                        "experiment.stop_failed", session_id=runner.session_id
                    )
        self._runners.clear()
