"""Regression coverage for durable evaluation context in background vault backfills."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from alpha_harness.db.models import (
    BackfillStatus,
    EvaluationJob,
    EvaluationJobStatus,
    ResearchCandidate,
    ResearchSession,
    SimStatus,
    SimulationRecord,
)
from alpha_harness.db.sqlite import Database
from alpha_harness.evaluation.jobs import EvaluationJobService
from alpha_harness.vault.backfill import Backfill


def test_fetch_each_reports_the_matching_evaluation_job_and_continues_siblings() -> None:
    async def run() -> None:
        backfill = object.__new__(Backfill)
        backfill._evaluation_jobs = {}
        fetched: list[str] = []
        reported: list[tuple[int, int, str, int, str, BaseException]] = []

        async def fetch_returns(alpha_id: str) -> None:
            fetched.append(alpha_id)
            if alpha_id == "alpha-fails":
                raise RuntimeError("temporary vault outage")

        async def on_failure(
            job_id: int,
            simulation_attempt: int,
            alpha_id: str,
            backfill_attempt: int,
            worker_token: str,
            exc: BaseException,
        ) -> None:
            reported.append((job_id, simulation_attempt, alpha_id, backfill_attempt, worker_token, exc))

        backfill.fetch_returns = fetch_returns
        backfill.register_evaluation_job("alpha-fails", 73, 4, 2, "worker-old", on_failure)

        await backfill._fetch_each(["alpha-fails", "alpha-succeeds"])

        assert fetched == ["alpha-fails", "alpha-succeeds"]
        assert len(reported) == 1
        job_id, simulation_attempt, alpha_id, attempt, worker_token, error = reported[0]
        assert (job_id, simulation_attempt, alpha_id, attempt, worker_token) == (
            73,
            4,
            "alpha-fails",
            2,
            "worker-old",
        )
        assert str(error) == "temporary vault outage"
        assert "alpha-fails" not in backfill._evaluation_jobs

    asyncio.run(run())


def test_old_fetch_cannot_report_as_or_unregister_a_newer_worker() -> None:
    async def run() -> None:
        backfill = object.__new__(Backfill)
        backfill._evaluation_jobs = {}
        fetch_started = asyncio.Event()
        release_fetch = asyncio.Event()
        reported: list[tuple[int, str]] = []

        async def fetch_returns(_alpha_id: str) -> None:
            fetch_started.set()
            await release_fetch.wait()
            raise RuntimeError("late failure from old request")

        async def old_reporter(
            _job_id: int,
            _simulation_attempt: int,
            _alpha_id: str,
            _backfill_attempt: int,
            _worker_token: str,
            _exc: BaseException,
        ) -> None:
            reported.append((1, "old"))

        async def new_reporter(
            _job_id: int,
            _simulation_attempt: int,
            _alpha_id: str,
            _backfill_attempt: int,
            _worker_token: str,
            _exc: BaseException,
        ) -> None:
            reported.append((2, "new"))

        backfill.fetch_returns = fetch_returns
        backfill.register_evaluation_job("alpha", 73, 4, 1, "old-token", old_reporter)
        old_task = asyncio.create_task(backfill._fetch_each(["alpha"]))
        await fetch_started.wait()

        backfill.register_evaluation_job("alpha", 73, 4, 2, "new-token", new_reporter)
        backfill.unregister_evaluation_job("alpha", "old-token")
        release_fetch.set()
        await old_task

        assert reported == [(1, "old")]
        context = backfill._evaluation_jobs["alpha"]
        assert context[2:4] == (2, "new-token")

    asyncio.run(run())


def test_fetch_each_persists_failure_to_the_matching_evaluation_job(tmp_path: Path) -> None:
    async def run() -> None:
        db_path = tmp_path / "p2_04_background_failure.db"
        db = Database(db_path)
        await db.create_all()
        try:
            async with db.session() as session:
                research_session = ResearchSession(name="P2-04", prompt="background association")
                session.add(research_session)
                await session.flush()
                simulation = SimulationRecord(
                    platform_id="brain-job-73",
                    alpha_id="alpha-fails",
                    request_hash="request-hash",
                    payload={"expression": "rank(close)"},
                    expression="rank(close)",
                    region="USA",
                    delay=1,
                    universe="TOP3000",
                    status=SimStatus.COMPLETE,
                )
                session.add(simulation)
                await session.flush()
                candidate = ResearchCandidate(
                    session_id=research_session.id,
                    simulation_record_id=simulation.id,
                    alpha_id="alpha-fails",
                    is_valid=True,
                    status="VALID",
                    expression="rank(close)",
                )
                session.add(candidate)
                await session.flush()
                job = EvaluationJob(
                    candidate_id=candidate.id,
                    session_id=research_session.id,
                    simulation_record_id=simulation.id,
                    simulation_attempt=4,
                    alpha_id="alpha-fails",
                    status=EvaluationJobStatus.RESULT_BACKFILL,
                )
                session.add(job)
                await session.flush()
                job_id = job.id

            now = datetime(2026, 1, 1, tzinfo=UTC)
            coordinator = EvaluationJobService(db, SimpleNamespace(), clock=lambda: now)
            backfill = object.__new__(Backfill)
            backfill._evaluation_jobs = {}

            async def fail_fetch(alpha_id: str) -> None:
                if alpha_id == "alpha-fails":
                    raise RuntimeError("temporary vault outage")

            backfill.fetch_returns = fail_fetch
            backfill.register_evaluation_job(
                "alpha-fails",
                job_id,
                4,
                0,
                "worker-current",
                coordinator._record_background_backfill_failure,
            )
            await backfill._fetch_each(["alpha-fails"])

            await db.dispose()
            db = Database(db_path)
            await db.create_all()
            async with db.session() as session:
                persisted = await session.get(EvaluationJob, job_id)
                assert persisted is not None
                assert persisted.backfill_status == BackfillStatus.RETRY_SCHEDULED
                assert persisted.backfill_attempt_count == 1
                assert persisted.retry_count == 1
                assert persisted.backfill_error_class == "TRANSIENT_ERROR"
                assert persisted.next_retry_at is not None
                assert persisted.attempt_history[0]["source"] == "background"
                assert persisted.status == EvaluationJobStatus.RESULT_BACKFILL

                persisted.backfill_attempt_count = 3
                persisted.backfill_worker_token = "new-worker"
                persisted.backfill_status = BackfillStatus.IN_PROGRESS

            await coordinator._record_background_backfill_failure(
                job_id, 4, "alpha-fails", 2, "stale-worker", RuntimeError("late failure")
            )

            async with db.session() as session:
                persisted = await session.get(EvaluationJob, job_id)
                assert persisted is not None
                assert persisted.backfill_attempt_count == 3
                assert persisted.retry_count == 1
                assert persisted.backfill_worker_token == "new-worker"
                assert persisted.backfill_status == BackfillStatus.IN_PROGRESS
        finally:
            await db.dispose()

    asyncio.run(run())
