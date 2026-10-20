import asyncio
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

from alpha_harness.db.models import (
    GenerationAttempt,
    GenerationAttemptStatus,
    ResearchResearcher,
    ResearchSession,
    utcnow,
)
from alpha_harness.db.sqlite import Database
from alpha_harness.generator.pipeline import GenerationBatch
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
