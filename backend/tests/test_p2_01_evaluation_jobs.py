"""P2-01 durable evaluation lifecycle and evidence-gated readiness tests."""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from alpha_harness.brain.errors import BrainDailyLimitReached, BrainRateLimited, BrainServiceUnavailable
from alpha_harness.db.models import (
    BackfillStatus,
    EvaluationJob,
    EvaluationJobStatus,
    ResearchCandidate,
    ResearchEvaluation,
    ResearchSession,
    SimStatus,
    SimulationRecord,
)
from alpha_harness.db.sqlite import Database
from alpha_harness.evaluation.jobs import (
    EvaluationJobService,
    _classify_backfill_error,
    _safe_error,
)


def _candidate(
    *, session_id: int, simulation_record_id: int | None = None, alpha_id: str | None = None
) -> ResearchCandidate:
    return ResearchCandidate(
        session_id=session_id,
        is_valid=True,
        status="VALID",
        expression="rank(close)",
        alpha_id=alpha_id,
        simulation_record_id=simulation_record_id,
        settings_hash="settings-hash",
        candidate_provenance_hash="provenance-hash",
    )


def _simulation(*, status: SimStatus, alpha_id: str = "alpha-1") -> SimulationRecord:
    return SimulationRecord(
        platform_id="brain-job-1",
        alpha_id=alpha_id,
        request_hash="request-hash",
        payload={"expression": "rank(close)"},
        expression="rank(close)",
        region="USA",
        delay=1,
        universe="TOP3000",
        status=status,
    )


def test_job_transitions_are_monotonic_and_timestamped() -> None:
    job = EvaluationJob(candidate_id=1, session_id=1, status=EvaluationJobStatus.QUEUED)
    job.transition(EvaluationJobStatus.SUBMITTED)
    job.transition(EvaluationJobStatus.SIMULATING)
    job.transition(EvaluationJobStatus.RESULT_BACKFILL)
    job.transition(EvaluationJobStatus.EVALUABLE)
    job.transition(EvaluationJobStatus.EVALUATED)
    assert job.status == EvaluationJobStatus.EVALUATED
    assert all((job.submitted_at, job.simulating_at, job.backfill_started_at, job.evaluable_at, job.evaluated_at))
    with pytest.raises(ValueError):
        job.transition(EvaluationJobStatus.QUEUED)


def test_p2_04_backfill_error_classification_and_sanitization() -> None:
    classification, retryable, retry_after = _classify_backfill_error(
        BrainRateLimited("rate limited", retry_after=42)
    )
    assert (classification, retryable, retry_after) == ("RATE_LIMIT", True, 42)

    classification, retryable, _ = _classify_backfill_error(BrainDailyLimitReached("quota"))
    assert (classification, retryable) == ("DAILY_LIMIT", False)

    classification, retryable, _ = _classify_backfill_error(BrainServiceUnavailable("offline"))
    assert (classification, retryable) == ("BRAINSERVICEUNAVAILABLE", True)

    assert "token=<redacted>" in _safe_error(RuntimeError("token=secret"))


def test_jobs_are_idempotent_per_candidate_attempt_and_distinct_attempts_are_preserved(tmp_path) -> None:
    async def run() -> None:
        db = Database(tmp_path / "p2_01_idempotency.db")
        await db.create_all()
        try:
            async with db.session() as session:
                research_session = ResearchSession(name="P2-01", prompt="test")
                session.add(research_session)
                await session.flush()
                candidate = _candidate(session_id=research_session.id)
                session.add(candidate)
                await session.flush()
                candidate_id = candidate.id

            service = EvaluationJobService(db, SimpleNamespace())
            first, same = await asyncio.gather(
                service.ensure_job(candidate_id, simulation_attempt=1),
                service.ensure_job(candidate_id, simulation_attempt=1),
            )
            second = await service.ensure_job(candidate_id, simulation_attempt=2)
            assert first.id == same.id
            assert second.id != first.id
            async with db.session() as session:
                jobs = (await session.execute(
                    select(EvaluationJob).where(EvaluationJob.candidate_id == candidate_id)
                )).scalars().all()
                assert len(jobs) == 2
        finally:
            await db.dispose()

    asyncio.run(run())


def test_processing_requires_platform_and_verified_vault_evidence(tmp_path) -> None:
    async def run() -> None:
        db = Database(tmp_path / "p2_01_readiness.db")
        await db.create_all()
        try:
            async with db.session() as session:
                research_session = ResearchSession(name="P2-01", prompt="test")
                session.add(research_session)
                await session.flush()
                queued_candidate = _candidate(session_id=research_session.id)
                session.add(queued_candidate)
                await session.flush()
                queued_id = queued_candidate.id

                active_sim = _simulation(status=SimStatus.RUNNING)
                session.add(active_sim)
                await session.flush()
                active_candidate = _candidate(session_id=research_session.id, simulation_record_id=active_sim.id, alpha_id="alpha-1")
                session.add(active_candidate)
                await session.flush()
                active_id = active_candidate.id

                complete_sim = _simulation(status=SimStatus.COMPLETE, alpha_id="alpha-2")
                session.add(complete_sim)
                await session.flush()
                complete_candidate = _candidate(session_id=research_session.id, simulation_record_id=complete_sim.id, alpha_id="alpha-2")
                session.add(complete_candidate)
                await session.flush()
                complete_id = complete_candidate.id

            class EmptyVault:
                async def series_length(self, _alpha_id: str) -> int:
                    return 0

                async def load_verified_series(self, _alpha_id: str) -> None:
                    return None

            service = EvaluationJobService(db, SimpleNamespace(vault=EmptyVault()))
            queued = await service.ensure_job(queued_id)
            active = await service.ensure_job(active_id)
            complete = await service.ensure_job(complete_id)
            assert (await service.process_job(queued.id))["status"] == EvaluationJobStatus.QUEUED
            assert (await service.process_job(active.id))["status"] == EvaluationJobStatus.SIMULATING
            assert (await service.process_job(complete.id))["status"] == EvaluationJobStatus.RESULT_BACKFILL
        finally:
            await db.dispose()

    asyncio.run(run())


def test_backfill_failures_retry_then_become_terminal(tmp_path) -> None:
    async def run() -> None:
        db_path = tmp_path / "p2_01_retry.db"
        db = Database(db_path)
        await db.create_all()
        try:
            async with db.session() as session:
                research_session = ResearchSession(name="P2-01", prompt="test")
                session.add(research_session)
                await session.flush()
                simulation = _simulation(status=SimStatus.COMPLETE)
                session.add(simulation)
                await session.flush()
                candidate = _candidate(session_id=research_session.id, simulation_record_id=simulation.id, alpha_id="alpha-1")
                session.add(candidate)
                await session.flush()
                candidate_id = candidate.id

            class FailingVault:
                async def series_length(self, _alpha_id: str) -> int:
                    return 1

                async def load_verified_series(self, _alpha_id: str) -> None:
                    raise RuntimeError("vault temporarily unavailable")

            now = [datetime(2026, 1, 1, tzinfo=UTC)]
            results = []
            for attempt, delay in enumerate((0, 30, 60)):
                if attempt:
                    # A new service and database engine model a process restart between retries.
                    await db.dispose()
                    db = Database(db_path)
                    await db.create_all()
                service = EvaluationJobService(
                    db, SimpleNamespace(vault=FailingVault()), clock=lambda: now[0]
                )
                job = await service.ensure_job(candidate_id)
                now[0] += timedelta(seconds=delay)
                results.append(await service.process_job(job.id))
            assert [result["retryCount"] for result in results] == [1, 2, 3]
            assert results[-1]["status"] == EvaluationJobStatus.FAILED
            assert results[0]["status"] == EvaluationJobStatus.RESULT_BACKFILL
            assert results[0]["nextRetryAt"] is not None
            assert results[-1]["backfillStatus"] == "REVIEW_REQUIRED"
        finally:
            await db.dispose()

    asyncio.run(run())


def test_p2_04_manual_requeue_preserves_persisted_failure_history(tmp_path) -> None:
    async def run() -> None:
        db = Database(tmp_path / "p2_04_manual_requeue.db")
        await db.create_all()
        try:
            async with db.session() as session:
                research_session = ResearchSession(name="P2-04", prompt="requeue")
                session.add(research_session)
                await session.flush()
                candidate = _candidate(session_id=research_session.id, alpha_id="alpha-1")
                session.add(candidate)
                await session.flush()
                job = EvaluationJob(
                    candidate_id=candidate.id,
                    session_id=research_session.id,
                    alpha_id="alpha-1",
                    status=EvaluationJobStatus.FAILED,
                    backfill_status=BackfillStatus.REVIEW_REQUIRED,
                    retry_count=3,
                    backfill_attempt_count=3,
                    backfill_error_class="TRANSIENT_ERROR",
                    attempt_history=[{"attempt": 3, "classification": "TRANSIENT_ERROR"}],
                )
                session.add(job)
                await session.flush()
                job_id = job.id

            service = EvaluationJobService(db, SimpleNamespace())
            result = await service.manual_requeue(job_id, reason="operator reviewed")
            assert result["status"] == EvaluationJobStatus.QUEUED
            assert result["retryCount"] == 3
            assert result["backfillAttemptCount"] == 3
            assert result["manualRequeueCount"] == 1
            assert result["attemptHistory"][0]["attempt"] == 3
            assert result["attemptHistory"][1]["event"] == "manual_requeue"
            async with db.session() as session:
                persisted = await session.get(EvaluationJob, job_id)
                assert persisted is not None
                assert persisted.backfill_status == BackfillStatus.PENDING
                assert len(persisted.attempt_history) == 2
        finally:
            await db.dispose()

    asyncio.run(run())


def test_p2_04_expired_worker_cannot_overwrite_newer_worker_outcome(tmp_path) -> None:
    async def run() -> None:
        db = Database(tmp_path / "p2_04_stale_worker.db")
        await db.create_all()
        try:
            async with db.session() as session:
                research_session = ResearchSession(name="P2-04", prompt="stale worker")
                session.add(research_session)
                await session.flush()
                simulation = _simulation(status=SimStatus.COMPLETE)
                session.add(simulation)
                await session.flush()
                candidate = _candidate(
                    session_id=research_session.id,
                    simulation_record_id=simulation.id,
                    alpha_id="alpha-1",
                )
                session.add(candidate)
                await session.flush()
                candidate_id = candidate.id

            now = [datetime(2026, 1, 1, tzinfo=UTC)]
            first_worker_started = asyncio.Event()
            release_first_worker = asyncio.Event()

            class GatedVault:
                def __init__(self) -> None:
                    self.calls = 0

                async def load_verified_series(self, _alpha_id: str):
                    self.calls += 1
                    if self.calls == 1:
                        first_worker_started.set()
                        await release_first_worker.wait()
                    return (["2026-01-01"] * 60, [0.01] * 60, [0.1] * 60)

            class PersistingEvaluation:
                def __init__(self, vault: GatedVault) -> None:
                    self.vault = vault
                    self.calls = 0

                async def evaluate_candidate(self, *, candidate_id: int, cost_bps: float) -> None:
                    del cost_bps
                    self.calls += 1
                    async with db.session() as session:
                        session.add(ResearchEvaluation(
                            candidate_id=candidate_id,
                            is_verified=True,
                            provenance="BRAIN_VAULT",
                        ))

            vault = GatedVault()
            evaluation = PersistingEvaluation(vault)
            first_service = EvaluationJobService(db, evaluation, clock=lambda: now[0])
            second_service = EvaluationJobService(db, evaluation, clock=lambda: now[0])
            job = await first_service.ensure_job(candidate_id)
            first_task = asyncio.create_task(first_service.process_job(job.id))
            await first_worker_started.wait()

            now[0] += timedelta(minutes=6)
            newer_result = await second_service.process_job(job.id)
            release_first_worker.set()
            stale_result = await first_task

            assert newer_result["status"] == EvaluationJobStatus.EVALUATED
            assert stale_result["status"] == EvaluationJobStatus.EVALUATED
            assert vault.calls == 2
            assert evaluation.calls == 1
            async with db.session() as session:
                persisted = await session.get(EvaluationJob, job.id)
                assert persisted is not None
                assert persisted.backfill_status == BackfillStatus.COMPLETE
                assert persisted.backfill_attempt_count == 2
        finally:
            await db.dispose()

    asyncio.run(run())


def test_job_is_evaluable_only_after_verified_result_is_persisted(tmp_path) -> None:
    async def run() -> None:
        db = Database(tmp_path / "p2_01_persisted_result.db")
        await db.create_all()
        try:
            async with db.session() as session:
                research_session = ResearchSession(name="P2-01", prompt="test")
                session.add(research_session)
                await session.flush()
                simulation = _simulation(status=SimStatus.COMPLETE)
                session.add(simulation)
                await session.flush()
                candidate = _candidate(
                    session_id=research_session.id,
                    simulation_record_id=simulation.id,
                    alpha_id="alpha-1",
                )
                session.add(candidate)
                await session.flush()
                candidate_id = candidate.id

            class VerifiedVault:
                async def series_length(self, _alpha_id: str) -> int:
                    return 60

                async def load_verified_series(self, _alpha_id: str) -> tuple[list[str], list[float], list[float]]:
                    return (["2026-01-01"] * 60, [0.01] * 60, [0.1] * 60)

            class PersistingEvaluation:
                def __init__(self) -> None:
                    self.vault = VerifiedVault()

                async def evaluate_candidate(self, *, candidate_id: int, cost_bps: float) -> None:
                    del cost_bps
                    async with db.session() as session:
                        session.add(ResearchEvaluation(
                            candidate_id=candidate_id,
                            is_verified=True,
                            provenance="BRAIN_VAULT",
                        ))

            service = EvaluationJobService(db, PersistingEvaluation())
            job = await service.ensure_job(candidate_id)
            result = await service.process_job(job.id)
            assert result["status"] == EvaluationJobStatus.EVALUATED
            assert result["evaluationId"] is not None
            async with db.session() as session:
                persisted = await session.get(ResearchEvaluation, result["evaluationId"])
                assert persisted is not None and persisted.is_verified is True
        finally:
            await db.dispose()

    asyncio.run(run())
