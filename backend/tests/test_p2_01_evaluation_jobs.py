"""P2-01 durable evaluation lifecycle and evidence-gated readiness tests."""

import asyncio
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from alpha_harness.db.models import (
    EvaluationJob,
    EvaluationJobStatus,
    ResearchCandidate,
    ResearchEvaluation,
    ResearchSession,
    SimStatus,
    SimulationRecord,
)
from alpha_harness.db.sqlite import Database
from alpha_harness.evaluation.jobs import EvaluationJobService


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
        db = Database(tmp_path / "p2_01_retry.db")
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

            service = EvaluationJobService(db, SimpleNamespace(vault=FailingVault()))
            job = await service.ensure_job(candidate_id)
            results = [await service.process_job(job.id) for _ in range(3)]
            assert [result["retryCount"] for result in results] == [1, 2, 3]
            assert results[-1]["status"] == EvaluationJobStatus.FAILED
            assert results[-1]["lastError"].startswith("permanent after 3 attempts")
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
