import asyncio
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import select

from alpha_harness.db.models import (
    GenerationAttempt,
    GenerationAttemptStatus,
    ResearchCandidate,
    ResearchResearcher,
    ResearchSession,
    ResearchStatus,
    utcnow,
)
from alpha_harness.db.sqlite import Database
from alpha_harness.generator.pipeline import (
    CandidateValidationResult,
    GeneratedCandidate,
    GenerationBatch,
)
from alpha_harness.orchestrator.researchers import MultiLLMOrchestrator


class _TransientError(RuntimeError):
    status = 503


class _PermanentError(RuntimeError):
    pass


def _candidate(valid: bool) -> SimpleNamespace:
    return SimpleNamespace(
        validation=SimpleNamespace(
            is_valid=valid,
            repair_history=[],
            fields_used=[],
            operators_used=[],
            operator_count=0,
            field_count=0,
            error=None,
            retry_count=0,
        ),
        expression="rank(close)",
        hypothesis="test",
        raw_response={},
        target_dataset="US_TOP3000",
        simulation_settings_overrides=None,
    )


def _batch(*candidates: SimpleNamespace) -> GenerationBatch:
    return GenerationBatch(
        summary="test",
        candidates=list(candidates),
        model_ref="test:model",
        total_tokens=12,
        duration_seconds=0.1,
    )


def _valid_batch(model_ref: str) -> GenerationBatch:
    validation = CandidateValidationResult(
        is_valid=True,
        expression="rank(close)",
        operator_count=1,
        field_count=1,
        fields_used=["close"],
    )
    return GenerationBatch(
        summary="test",
        candidates=[
            GeneratedCandidate(
                hypothesis="test hypothesis",
                expression="rank(close)",
                category="general",
                target_dataset=None,
                primary_fields=["close"],
                validation=validation,
                raw_response={"expression": "rank(close)"},
            )
        ],
        model_ref=model_ref,
        total_tokens=12,
        duration_seconds=0.1,
    )


def _dispatch(orchestrator: MultiLLMOrchestrator, attempt_id: int = 7):
    return orchestrator._dispatch_researcher(
        researcher_id=3,
        model_ref="test:model",
        category="GENERAL",
        count=1,
        region="USA",
        universe="TOP3000",
        delay=1,
        generation=1,
        temperature=0.7,
        attempt_id=attempt_id,
    )


def test_p2_03_provider_failure_is_durable_and_sanitized_without_fallback():
    async def run():
        class Pipeline:
            async def generate_batch(self, **kwargs):
                raise _PermanentError("api_key=super-secret credential failure")

        orchestrator = MultiLLMOrchestrator.__new__(MultiLLMOrchestrator)
        orchestrator.pipeline = Pipeline()
        outcome = await _dispatch(orchestrator)
        assert outcome.status == GenerationAttemptStatus.FAILED
        assert outcome.validation_outcome == "PROVIDER_FAILURE"
        assert outcome.batch is None
        assert "super-secret" not in (outcome.error_message or "")
        assert outcome.retry_count == 0

    asyncio.run(run())


def test_p2_03_transient_retry_is_bounded_and_does_not_duplicate_candidates():
    async def run():
        class Pipeline:
            def __init__(self):
                self.calls = 0

            async def generate_batch(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise _TransientError("temporary provider outage")
                return _batch(_candidate(True))

        pipeline = Pipeline()
        orchestrator = MultiLLMOrchestrator.__new__(MultiLLMOrchestrator)
        orchestrator.pipeline = pipeline
        outcome = await _dispatch(orchestrator)
        assert pipeline.calls == 2
        assert outcome.status == GenerationAttemptStatus.SUCCEEDED
        assert outcome.retry_count == 1
        assert len(outcome.batch.candidates) == 1

    asyncio.run(run())


def test_p2_03_empty_and_invalid_outputs_are_distinguished():
    async def run():
        class EmptyPipeline:
            async def generate_batch(self, **kwargs):
                return _batch()

        class InvalidPipeline:
            async def generate_batch(self, **kwargs):
                return _batch(_candidate(False))

        empty_orchestrator = MultiLLMOrchestrator.__new__(MultiLLMOrchestrator)
        empty_orchestrator.pipeline = EmptyPipeline()
        invalid_orchestrator = MultiLLMOrchestrator.__new__(MultiLLMOrchestrator)
        invalid_orchestrator.pipeline = InvalidPipeline()
        empty = await _dispatch(empty_orchestrator, attempt_id=8)
        invalid = await _dispatch(invalid_orchestrator, attempt_id=9)
        assert empty.status == GenerationAttemptStatus.SUCCEEDED
        assert empty.validation_outcome == "EMPTY_OUTPUT"
        assert invalid.status == GenerationAttemptStatus.FAILED
        assert invalid.validation_outcome == "OUTPUT_VALIDATION_FAILED"

    asyncio.run(run())


def test_p2_03_stale_running_attempts_are_recovered(tmp_path: Path):
    async def run():
        db = Database(tmp_path / "p2_03.db")
        await db.create_all()
        async with db.session() as session:
            research_session = ResearchSession(name="test", prompt="test")
            session.add(research_session)
            await session.flush()
            researcher = ResearchResearcher(
                session_id=research_session.id,
                model_ref="test:model",
                label="test",
            )
            session.add(researcher)
            await session.flush()
            attempt = GenerationAttempt(
                session_id=research_session.id,
                researcher_id=researcher.id,
                model_ref="test:model",
                status=GenerationAttemptStatus.RUNNING,
                started_at=utcnow() - timedelta(hours=1),
            )
            session.add(attempt)
        orchestrator = MultiLLMOrchestrator.__new__(MultiLLMOrchestrator)
        orchestrator.db = db
        assert await orchestrator.reconcile_generation_attempts(stale_after_seconds=60) == 1
        async with db.session() as session:
            recovered = await session.get(GenerationAttempt, attempt.id)
            assert recovered.status == GenerationAttemptStatus.FAILED
            assert recovered.error_class == "StaleAttemptRecovery"
            assert recovered.validation_outcome == "STALE_RECOVERED"
        await db.dispose()

    asyncio.run(run())


def test_p2_03_persists_before_provider_and_keeps_sibling_success(tmp_path: Path):
    async def run():
        db = Database(tmp_path / "partial.db")
        await db.create_all()
        observed_running: list[bool] = []

        class Pipeline:
            async def generate_batch(self, **kwargs):
                async with db.session() as session:
                    running = await session.scalar(
                        select(GenerationAttempt.id).where(
                            GenerationAttempt.model_ref == kwargs["model_ref"],
                            GenerationAttempt.status == GenerationAttemptStatus.RUNNING,
                        )
                    )
                observed_running.append(running is not None)
                if kwargs["model_ref"] == "test:failed":
                    raise _PermanentError("provider rejected request")
                return _valid_batch(kwargs["model_ref"])

        orchestrator = MultiLLMOrchestrator(db=db, pipeline=Pipeline())

        async with db.session() as session:
            research_session = ResearchSession(name="partial", prompt="test", status=ResearchStatus.IDLE)
            session.add(research_session)
            await session.flush()
            session.add_all(
                [
                    ResearchResearcher(
                        session_id=research_session.id,
                        model_ref="test:failed",
                        label="failed",
                        enabled=True,
                    ),
                    ResearchResearcher(
                        session_id=research_session.id,
                        model_ref="test:success",
                        label="success",
                        enabled=True,
                    ),
                ]
            )
            await session.flush()
            session_id = research_session.id

        result = await orchestrator.run_generation_round(session_id, total_target_candidates=2)
        assert len(result) == 1
        assert result.summary["partialSuccess"] is True
        assert result.summary["failedResearchers"] == 1
        assert all(observed_running)

        async with db.session() as session:
            attempts = list(
                (
                    await session.scalars(
                        select(GenerationAttempt).where(GenerationAttempt.session_id == session_id)
                    )
                ).all()
            )
            candidates = list(
                (
                    await session.scalars(
                        select(ResearchCandidate).where(ResearchCandidate.session_id == session_id)
                    )
                ).all()
            )
            assert len(attempts) == 2
            assert {attempt.status for attempt in attempts} == {
                GenerationAttemptStatus.FAILED,
                GenerationAttemptStatus.SUCCEEDED,
            }
            assert len(candidates) == 1
            assert candidates[0].generation_attempt_id == next(
                attempt.id for attempt in attempts if attempt.status == GenerationAttemptStatus.SUCCEEDED
            )
        await db.dispose()

    asyncio.run(run())


def test_p2_03_retry_exhaustion_and_cancellation_are_terminal():
    async def run():
        class AlwaysTransient:
            def __init__(self):
                self.calls = 0

            async def generate_batch(self, **kwargs):
                self.calls += 1
                raise _TransientError("still unavailable")

        transient = AlwaysTransient()
        orchestrator = MultiLLMOrchestrator.__new__(MultiLLMOrchestrator)
        orchestrator.pipeline = transient
        exhausted = await _dispatch(orchestrator)
        assert transient.calls == 3
        assert exhausted.status == GenerationAttemptStatus.FAILED
        assert exhausted.retry_count == 2

        class Cancelled:
            async def generate_batch(self, **kwargs):
                raise asyncio.CancelledError()

        cancelled_orchestrator = MultiLLMOrchestrator.__new__(MultiLLMOrchestrator)
        cancelled_orchestrator.pipeline = Cancelled()
        cancelled = await _dispatch(cancelled_orchestrator, attempt_id=10)
        assert cancelled.status == GenerationAttemptStatus.CANCELLED
        assert cancelled.validation_outcome == "CANCELLED"

    asyncio.run(run())


def test_p2_03_unexpected_gather_exception_is_persisted(tmp_path: Path):
    async def run():
        db = Database(tmp_path / "gather.db")
        await db.create_all()
        try:
            async with db.session() as session:
                research_session = ResearchSession(name="gather", prompt="test")
                session.add(research_session)
                await session.flush()
                session.add(
                    ResearchResearcher(
                        session_id=research_session.id,
                        model_ref="test:model",
                        label="test",
                        enabled=True,
                    )
                )
                await session.flush()
                session_id = research_session.id

            orchestrator = MultiLLMOrchestrator(db=db, pipeline=SimpleNamespace())

            async def unexpected(**kwargs):
                raise RuntimeError("unexpected orchestration failure")

            orchestrator._dispatch_researcher = unexpected
            result = await orchestrator.run_generation_round(session_id, total_target_candidates=1)
            assert result == []
            assert result.summary["failedResearchers"] == 1
            assert result.summary["failureDetails"][0]["validationOutcome"] == "ORCHESTRATION_EXCEPTION"
            async with db.session() as session:
                attempt = await session.scalar(
                    select(GenerationAttempt).where(GenerationAttempt.session_id == session_id)
                )
                assert attempt is not None
                assert attempt.status == GenerationAttemptStatus.FAILED
                assert attempt.error_class == "RuntimeError"
        finally:
            await db.dispose()

    asyncio.run(run())
