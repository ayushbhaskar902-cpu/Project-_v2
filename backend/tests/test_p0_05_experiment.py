"""Regression test suite for P0-05: Persistent Autonomous End-to-End Experiment API."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select

from alpha_harness.db.duck import Catalog
from alpha_harness.db.models import (
    CandidateStatus,
    ResearchCandidate,
    ResearchEvaluation,
    ResearchResearcher,
    ResearchSession,
    ResearchStatus,
    SimStatus,
    SimulationRecord,
    utcnow,
)
from alpha_harness.db.sqlite import Database
from alpha_harness.evaluation.metrics import (
    EvaluationEngine,
    EvaluationScorecard,
    InsufficientDataError,
)
from alpha_harness.experiment.runner import (
    MAX_CONSECUTIVE_ERRORS,
    ExperimentRegistry,
    ExperimentRunner,
)
from alpha_harness.leaderboard.service import LeaderboardService
from alpha_harness.learning.loop import LearningCycleResult, SelfLearningEngine
from alpha_harness.learning.memory import ResearchMemory
from alpha_harness.orchestrator.researchers import MultiLLMOrchestrator
from alpha_harness.vault.store import AlphaVault


async def setup_test_db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "test.db")
    await db.create_all()
    return db


async def create_test_session(
    db: Database,
    *,
    status: ResearchStatus = ResearchStatus.IDLE,
    budget: int = 100,
    used: int = 0,
    generation: int = 0,
) -> int:
    async with db.session() as session:
        sess = ResearchSession(
            name="Test Campaign",
            prompt="Find momentum alphas",
            instrument_type="EQUITY",
            region="USA",
            universe="TOP3000",
            delay=1,
            simulation_budget=budget,
            simulations_used=used,
            learning_frequency=80,
            current_generation=generation,
            status=status,
            current_phase=status,
            error_count=0,
        )
        session.add(sess)
        await session.commit()
        return sess.id


# ===========================================================================
# 1. ExperimentRunner Lifecycle & State Machine Tests
# ===========================================================================


def test_runner_start_from_idle(tmp_path: Path):
    """P0-05 Req 1 & 2: Start from IDLE transitions session to RUNNING and checkpoints started_at."""
    async def _run():
        db = await setup_test_db(tmp_path)
        try:
            sess_id = await create_test_session(db, status=ResearchStatus.IDLE)
            orchestrator = MagicMock(spec=MultiLLMOrchestrator)
            evaluation = MagicMock(spec=EvaluationEngine)
            learning = MagicMock(spec=SelfLearningEngine)
            leaderboard = MagicMock(spec=LeaderboardService)

            # Prevent background loop from running phases in this test
            runner = ExperimentRunner(
                session_id=sess_id,
                db=db,
                orchestrator=orchestrator,
                evaluation=evaluation,
                learning=learning,
                leaderboard=leaderboard,
            )
            # Patch _run so task finishes cleanly
            runner._run = AsyncMock()

            await runner.start()
            assert runner.is_running

            # Verify SQLite checkpoint
            async with db.session() as session:
                row = await session.get(ResearchSession, sess_id)
                assert row is not None
                assert row.status == ResearchStatus.RUNNING
                assert row.current_phase == ResearchStatus.RUNNING
                assert row.started_at is not None
                assert row.last_checkpoint_at is not None
        finally:
            await db.dispose()

    asyncio.run(_run())


def test_runner_start_rejects_already_running_or_complete(tmp_path: Path):
    """P0-05 Req 2: Calling start() on already active or COMPLETE sessions raises RuntimeError."""
    async def _run():
        db = await setup_test_db(tmp_path)
        try:
            running_id = await create_test_session(db, status=ResearchStatus.RUNNING)
            complete_id = await create_test_session(db, status=ResearchStatus.COMPLETE)

            runner_running = ExperimentRunner(
                session_id=running_id,
                db=db,
                orchestrator=MagicMock(),
                evaluation=MagicMock(),
                learning=MagicMock(),
                leaderboard=MagicMock(),
            )
            with pytest.raises(RuntimeError, match="already in phase"):
                await runner_running.start()

            runner_complete = ExperimentRunner(
                session_id=complete_id,
                db=db,
                orchestrator=MagicMock(),
                evaluation=MagicMock(),
                learning=MagicMock(),
                leaderboard=MagicMock(),
            )
            with pytest.raises(RuntimeError, match="already COMPLETE"):
                await runner_complete.start()
        finally:
            await db.dispose()

    asyncio.run(_run())


def test_runner_budget_enforcement(tmp_path: Path):
    """P0-05 Req 3: Loop stops when simulations_used >= simulation_budget and marks session COMPLETE."""
    async def _run():
        db = await setup_test_db(tmp_path)
        try:
            sess_id = await create_test_session(
                db, status=ResearchStatus.IDLE, budget=50, used=50
            )
            orchestrator = MagicMock(spec=MultiLLMOrchestrator)
            evaluation = MagicMock(spec=EvaluationEngine)
            learning = MagicMock(spec=SelfLearningEngine)
            leaderboard = MagicMock(spec=LeaderboardService)

            runner = ExperimentRunner(
                session_id=sess_id,
                db=db,
                orchestrator=orchestrator,
                evaluation=evaluation,
                learning=learning,
                leaderboard=leaderboard,
            )

            await runner.start()
            # Wait for the background loop to process the budget check
            await asyncio.sleep(0.1)

            async with db.session() as session:
                row = await session.get(ResearchSession, sess_id)
                assert row is not None
                assert row.status == ResearchStatus.COMPLETE
                assert row.finished_at is not None
                assert "Budget exhausted: 50/50" in (row.message or "")

            # No generation should have been invoked
            orchestrator.run_generation_round.assert_not_called()
        finally:
            await db.dispose()

    asyncio.run(_run())


def test_runner_pause_and_resume(tmp_path: Path):
    """P0-05 Req 5: Pause halts at phase boundary; resume continues from checkpoint."""
    async def _run():
        db = await setup_test_db(tmp_path)
        try:
            sess_id = await create_test_session(db, status=ResearchStatus.IDLE, budget=100, used=0)
            orchestrator = MagicMock(spec=MultiLLMOrchestrator)
            orchestrator.run_generation_round = AsyncMock(return_value=[1, 2])
            orchestrator.enqueue_candidates_for_simulation = AsyncMock(return_value=[1, 2])
            orchestrator.sync_session_simulations = AsyncMock(return_value={"completed": 0})

            runner = ExperimentRunner(
                session_id=sess_id,
                db=db,
                orchestrator=orchestrator,
                evaluation=MagicMock(),
                learning=MagicMock(),
                leaderboard=MagicMock(),
            )

            # Request pause before loop begins first phase
            await runner.pause()
            assert runner._pause_requested is True

            # Trigger loop
            await runner._run()

            # Should have paused at phase boundary
            async with db.session() as session:
                row = await session.get(ResearchSession, sess_id)
                assert row is not None
                assert row.status == ResearchStatus.PAUSED
                assert "Paused at phase boundary" in (row.message or "")

            # Now resume
            runner._run = AsyncMock()  # Mock to verify resume call
            await runner.resume()

            async with db.session() as session:
                row = await session.get(ResearchSession, sess_id)
                assert row is not None
                assert row.status == ResearchStatus.RUNNING
                assert "Resumed from PAUSED" in (row.message or "")
        finally:
            await db.dispose()

    asyncio.run(_run())


def test_runner_cancel(tmp_path: Path):
    """P0-05 Req 5 & 7: Cancel aborts loop and transitions session to FAILED."""
    async def _run():
        db = await setup_test_db(tmp_path)
        try:
            sess_id = await create_test_session(db, status=ResearchStatus.IDLE)
            runner = ExperimentRunner(
                session_id=sess_id,
                db=db,
                orchestrator=MagicMock(),
                evaluation=MagicMock(),
                learning=MagicMock(),
                leaderboard=MagicMock(),
            )

            await runner.start()
            assert runner.is_running

            await runner.cancel()
            assert not runner.is_running

            async with db.session() as session:
                row = await session.get(ResearchSession, sess_id)
                assert row is not None
                assert row.status == ResearchStatus.FAILED
                assert row.finished_at is not None
                assert "cancelled by user" in (row.message or "").lower()
        finally:
            await db.dispose()

    asyncio.run(_run())


def test_runner_error_threshold_transitions_to_failed(tmp_path: Path):
    """P0-05 Req 7: Consecutive unrecoverable errors transition session to FAILED."""
    async def _run():
        db = await setup_test_db(tmp_path)
        try:
            sess_id = await create_test_session(db, status=ResearchStatus.IDLE)
            orchestrator = MagicMock(spec=MultiLLMOrchestrator)
            # Generation always fails
            orchestrator.run_generation_round = AsyncMock(side_effect=RuntimeError("LLM provider unavailable"))

            learning = MagicMock(spec=SelfLearningEngine)
            learning.execute_learning_cycle = AsyncMock()
            leaderboard = MagicMock(spec=LeaderboardService)
            leaderboard.capture_snapshot = AsyncMock()
            evaluation = MagicMock(spec=EvaluationEngine)
            orchestrator.sync_session_simulations = AsyncMock(return_value={})

            runner = ExperimentRunner(
                session_id=sess_id,
                db=db,
                orchestrator=orchestrator,
                evaluation=evaluation,
                learning=learning,
                leaderboard=leaderboard,
            )

            await runner.start()
            # Give time for 5 consecutive failures
            for _ in range(20):
                await asyncio.sleep(0.05)
                async with db.session() as session:
                    row = await session.get(ResearchSession, sess_id)
                    if row and row.status == ResearchStatus.FAILED:
                        break

            async with db.session() as session:
                row = await session.get(ResearchSession, sess_id)
                assert row is not None
                assert row.status == ResearchStatus.FAILED
                assert row.error_count >= MAX_CONSECUTIVE_ERRORS
                assert "consecutive errors" in (row.message or "").lower()
        finally:
            await db.dispose()

    asyncio.run(_run())


def test_runner_evaluation_error_isolation(tmp_path: Path):
    """P0-05 Req 6: Individual candidate evaluation failure is isolated; others proceed."""
    async def _run():
        db = await setup_test_db(tmp_path)
        try:
            sess_id = await create_test_session(db, status=ResearchStatus.RUNNING)

            # Create two candidates in SQLite
            async with db.session() as session:
                c1 = ResearchCandidate(
                    session_id=sess_id,
                    expression="ts_decay_linear(returns, 5)",
                    is_valid=True,
                    status=CandidateStatus.SIMULATING,
                    alpha_id="alpha_fail_01",
                )
                c2 = ResearchCandidate(
                    session_id=sess_id,
                    expression="ts_rank(volume, 10)",
                    is_valid=True,
                    status=CandidateStatus.SIMULATING,
                    alpha_id="alpha_ok_02",
                )
                session.add_all([c1, c2])
                await session.commit()
                c1_id, c2_id = c1.id, c2.id

            evaluation = MagicMock(spec=EvaluationEngine)
            # Candidate 1 fails, Candidate 2 succeeds
            async def mock_eval(candidate_id, cost_bps=5.0):
                if candidate_id == c1_id:
                    raise InsufficientDataError("No series data in AlphaVault")
                return MagicMock(is_submission_grade=True)

            evaluation.evaluate_candidate = AsyncMock(side_effect=mock_eval)

            runner = ExperimentRunner(
                session_id=sess_id,
                db=db,
                orchestrator=MagicMock(),
                evaluation=evaluation,
                learning=MagicMock(),
                leaderboard=MagicMock(),
            )

            await runner._phase_evaluate()

            async with db.session() as session:
                cand1 = await session.get(ResearchCandidate, c1_id)
                assert cand1 is not None
                # Failing candidate marked UNAVAILABLE
                assert cand1.status == CandidateStatus.UNAVAILABLE
                assert "Evaluation failed" in (cand1.validation_error or "")

            # Evaluation was attempted for both candidates
            assert evaluation.evaluate_candidate.call_count == 2
        finally:
            await db.dispose()

    asyncio.run(_run())


def test_runner_websocket_phase_change_notification(tmp_path: Path):
    """P0-05 Req 9: on_phase_change callback receives checkpoint payloads."""
    async def _run():
        db = await setup_test_db(tmp_path)
        try:
            sess_id = await create_test_session(db, status=ResearchStatus.IDLE)
            notifications = []

            async def mock_broadcast(payload):
                notifications.append(payload)

            runner = ExperimentRunner(
                session_id=sess_id,
                db=db,
                orchestrator=MagicMock(),
                evaluation=MagicMock(),
                learning=MagicMock(),
                leaderboard=MagicMock(),
                on_phase_change=mock_broadcast,
            )
            runner._run = AsyncMock()

            await runner.start()
            assert len(notifications) >= 1
            assert notifications[0]["sessionId"] == sess_id
            assert notifications[0]["status"] == ResearchStatus.RUNNING
        finally:
            await db.dispose()

    asyncio.run(_run())


# ===========================================================================
# 2. ExperimentRegistry Tests
# ===========================================================================


def test_experiment_registry_lifecycle():
    """P0-05: ExperimentRegistry tracks, gets, unregisters, and stops all runners."""
    async def _run():
        registry = ExperimentRegistry()
        r1 = MagicMock(spec=ExperimentRunner)
        r1.session_id = 101
        r1.is_running = True
        r1.cancel = AsyncMock()

        r2 = MagicMock(spec=ExperimentRunner)
        r2.session_id = 102
        r2.is_running = False
        r2.cancel = AsyncMock()

        registry.register(r1)
        registry.register(r2)

        assert registry.get(101) is r1
        assert registry.get(102) is r2
        assert registry.active_sessions() == [101]

        registry.unregister(102)
        assert registry.get(102) is None

        await registry.stop_all()
        r1.cancel.assert_awaited_once()
        assert registry.active_sessions() == []

    asyncio.run(_run())


# ===========================================================================
# 3. Startup Crash Recovery Tests
# ===========================================================================


def test_startup_resumes_active_sessions(tmp_path: Path):
    """P0-05 Req 4: AppState._resume_experiments identifies interrupted sessions and resumes them."""
    async def _run():
        db = await setup_test_db(tmp_path)
        try:
            # Seed sessions in various states
            s_idle = await create_test_session(db, status=ResearchStatus.IDLE)
            s_complete = await create_test_session(db, status=ResearchStatus.COMPLETE)
            s_failed = await create_test_session(db, status=ResearchStatus.FAILED)
            s_running = await create_test_session(db, status=ResearchStatus.RUNNING)
            s_generating = await create_test_session(db, status=ResearchStatus.GENERATING)
            s_evaluating = await create_test_session(db, status=ResearchStatus.EVALUATING)

            # Create minimal Mock AppState
            app_state = MagicMock()
            app_state.db = db
            app_state.orchestrator = MagicMock()
            app_state.evaluation = MagicMock()
            app_state.learning = MagicMock()
            app_state.leaderboard = MagicMock()
            app_state.engine = None
            app_state.hub = MagicMock()
            app_state.experiments = ExperimentRegistry()

            # Import the method directly from state.py
            from alpha_harness.state import AppState
            # Bind _resume_experiments to app_state
            await AppState._resume_experiments(app_state)

            # Only s_running, s_generating, s_evaluating should be registered
            active = app_state.experiments.active_sessions()
            registered = [s_running, s_generating, s_evaluating]
            for sid in registered:
                assert app_state.experiments.get(sid) is not None

            # Idle, Complete, Failed must NOT be registered
            assert app_state.experiments.get(s_idle) is None
            assert app_state.experiments.get(s_complete) is None
            assert app_state.experiments.get(s_failed) is None

            # Cleanup
            await app_state.experiments.stop_all()
        finally:
            await db.dispose()

    asyncio.run(_run())


# ===========================================================================
# 4. REST API Endpoint Tests
# ===========================================================================


def test_api_endpoints_run_pause_resume_cancel_status(tmp_path: Path):
    """P0-05 Req 1 & 5: REST API endpoints execute lifecycle actions correctly."""
    async def _run():
        db = await setup_test_db(tmp_path)
        try:
            sess_id = await create_test_session(db, status=ResearchStatus.IDLE)

            state = MagicMock()
            state.db = db
            state.orchestrator = MagicMock()
            state.orchestrator.run_generation_round = AsyncMock(return_value=[])
            state.evaluation = MagicMock()
            state.learning = MagicMock()
            state.leaderboard = MagicMock()
            state.engine = None
            state.hub = MagicMock()
            state.hub.broadcast = AsyncMock()
            state.experiments = ExperimentRegistry()

            from alpha_harness.api.research import (
                cancel_experiment_session,
                get_experiment_session_status,
                pause_experiment_session,
                resume_experiment_session,
                run_experiment_session,
            )

            # 1. GET status before running
            st = await get_experiment_session_status(sess_id, state)
            assert st["sessionId"] == sess_id
            assert st["status"] == ResearchStatus.IDLE
            assert st["isRunning"] is False

            # 2. POST /run
            run_res = await run_experiment_session(sess_id, state)
            assert run_res["status"] == "started"
            assert state.experiments.get(sess_id) is not None

            # Check status while running
            st = await get_experiment_session_status(sess_id, state)
            assert st["isRunning"] is True

            # 3. POST /pause
            pause_res = await pause_experiment_session(sess_id, state)
            assert pause_res["status"] == "pause_requested"

            # 4. POST /cancel
            cancel_res = await cancel_experiment_session(sess_id, state)
            assert cancel_res["status"] == "cancelled"
            assert state.experiments.get(sess_id) is None

            # 5. Check status after cancel
            st = await get_experiment_session_status(sess_id, state)
            assert st["status"] == ResearchStatus.FAILED
            assert st["isRunning"] is False
        finally:
            await db.dispose()

    asyncio.run(_run())
