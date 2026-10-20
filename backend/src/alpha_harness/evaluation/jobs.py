"""Durable coordination for research-candidate evaluation."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from ..db.models import (
    EvaluationJob,
    EvaluationJobStatus,
    ResearchCandidate,
    ResearchEvaluation,
    SimulationRecord,
    SimStatus,
)
from .metrics import EvaluationEngine, InsufficientDataError


class EvaluationJobService:
    """Creates, reconciles, and processes idempotent evaluation attempts."""

    def __init__(self, db: Any, evaluation: EvaluationEngine, *, backfill: Any = None, auth: Any = None) -> None:
        self.db = db
        self.evaluation = evaluation
        self.backfill = backfill
        self.auth = auth

    async def ensure_job(self, candidate_id: int, simulation_attempt: int = 1) -> EvaluationJob:
        async with self.db.session() as session:
            job = (await session.execute(
                select(EvaluationJob).where(
                    EvaluationJob.candidate_id == candidate_id,
                    EvaluationJob.simulation_attempt == simulation_attempt,
                )
            )).scalar_one_or_none()
            candidate = await session.get(ResearchCandidate, candidate_id)
            if candidate is None:
                raise ValueError(f"Candidate {candidate_id} was not found")
            if job is None:
                job = EvaluationJob(
                    candidate_id=candidate.id,
                    session_id=candidate.session_id,
                    simulation_record_id=candidate.simulation_record_id,
                    simulation_attempt=simulation_attempt,
                    alpha_id=candidate.alpha_id,
                    settings_hash=candidate.settings_hash,
                    provenance_hash=candidate.candidate_provenance_hash,
                )
                try:
                    session.add(job)
                    await session.flush()
                except IntegrityError:
                    # The unique candidate/attempt key is the concurrency boundary:
                    # another request may have created the durable row between our
                    # read and insert. Reuse that committed row instead of creating a
                    # second evaluation attempt or returning a transient 500.
                    await session.rollback()
                    job = (await session.execute(
                        select(EvaluationJob).where(
                            EvaluationJob.candidate_id == candidate_id,
                            EvaluationJob.simulation_attempt == simulation_attempt,
                        )
                    )).scalar_one()
            elif job.status == EvaluationJobStatus.FAILED and job.retry_count < job.max_retries:
                job.transition(EvaluationJobStatus.QUEUED)
            return job

    async def start_session_jobs(
        self,
        session_id: int,
        candidate_ids: list[int] | None = None,
        simulation_attempt: int = 1,
        cost_bps: float = 5.0,
    ) -> list[dict[str, Any]]:
        """Reconcile existing evidence and process jobs; never infer completion from time."""
        stmt = select(ResearchCandidate).where(
            ResearchCandidate.session_id == session_id,
            ResearchCandidate.is_valid.is_(True),
        )
        if candidate_ids:
            stmt = stmt.where(ResearchCandidate.id.in_(candidate_ids))
        async with self.db.session() as session:
            candidates = list((await session.execute(stmt)).scalars().all())

        jobs: list[dict[str, Any]] = []
        for candidate in candidates:
            job = await self.ensure_job(candidate.id, simulation_attempt)
            jobs.append(await self.process_job(job.id, cost_bps=cost_bps))
        return jobs

    async def process_job(self, job_id: int, *, cost_bps: float = 5.0) -> dict[str, Any]:
        async with self.db.session() as session:
            job = await session.get(EvaluationJob, job_id, with_for_update=True)
            if job is None:
                raise ValueError(f"Evaluation job {job_id} was not found")
            candidate = await session.get(ResearchCandidate, job.candidate_id)
            simulation = (
                await session.get(SimulationRecord, candidate.simulation_record_id)
                if candidate and candidate.simulation_record_id
                else None
            )
            if candidate is None:
                job.last_error = "permanent: candidate deleted"
                if job.status != EvaluationJobStatus.FAILED:
                    job.transition(EvaluationJobStatus.FAILED)
                return self._view(job)

            job.simulation_record_id = candidate.simulation_record_id
            job.alpha_id = candidate.alpha_id
            job.settings_hash = candidate.settings_hash
            job.provenance_hash = candidate.candidate_provenance_hash

            if job.status == EvaluationJobStatus.EVALUATED:
                return self._view(job)
            if simulation is None or not candidate.alpha_id:
                return self._view(job)

            job.platform_job_id = simulation.platform_id
            if simulation.status in (SimStatus.QUEUED, SimStatus.PENDING, SimStatus.ORPHANED, SimStatus.RUNNING):
                if simulation.platform_id and job.status == EvaluationJobStatus.QUEUED:
                    job.transition(EvaluationJobStatus.SUBMITTED)
                if simulation.platform_id and job.status == EvaluationJobStatus.SUBMITTED:
                    job.transition(EvaluationJobStatus.SIMULATING)
                return self._view(job)
            if simulation.status in (SimStatus.ERROR, SimStatus.FAILED, SimStatus.CANCELLED, SimStatus.TIMEOUT, SimStatus.REJECTED):
                job.retry_count += 1
                job.last_error = f"permanent: platform simulation {simulation.status}: {simulation.message or 'failed'}"
                if job.status != EvaluationJobStatus.FAILED:
                    job.transition(EvaluationJobStatus.FAILED)
                return self._view(job)

            if simulation.status not in (SimStatus.COMPLETE, SimStatus.WARNING):
                return self._view(job)
            if job.status == EvaluationJobStatus.QUEUED:
                if not simulation.platform_id:
                    return self._view(job)
                job.transition(EvaluationJobStatus.SUBMITTED)
            if job.status == EvaluationJobStatus.SUBMITTED:
                job.transition(EvaluationJobStatus.SIMULATING)
            if job.status == EvaluationJobStatus.SIMULATING:
                job.transition(EvaluationJobStatus.RESULT_BACKFILL)
            alpha_id = candidate.alpha_id

        # Backfill and vault verification happen outside the transaction. No state is
        # advanced until the genuine verified series is available.
        try:
            if self.backfill is not None and self.auth is not None and getattr(self.auth.session, "authenticated", False):
                if await self.evaluation.vault.series_length(alpha_id) == 0:
                    await self.backfill.fetch_returns(alpha_id)
            verified = await self.evaluation.vault.load_verified_series(alpha_id)
            if verified is None:
                return await self._view_by_id(job_id)
        except InsufficientDataError:
            return await self._view_by_id(job_id)
        except Exception as exc:
            return await self._retry_or_fail(job_id, str(exc))

        async with self.db.session() as session:
            job = await session.get(EvaluationJob, job_id, with_for_update=True)
            if job is None:
                raise ValueError(f"Evaluation job {job_id} was not found")
            if job.status == EvaluationJobStatus.RESULT_BACKFILL:
                job.transition(EvaluationJobStatus.EVALUABLE)

        try:
            await self.evaluation.evaluate_candidate(candidate_id=job.candidate_id, cost_bps=cost_bps)
        except InsufficientDataError:
            return await self._view_by_id(job_id)
        except Exception as exc:
            return await self._retry_or_fail(job_id, str(exc))

        async with self.db.session() as session:
            job = await session.get(EvaluationJob, job_id, with_for_update=True)
            evaluation = (await session.execute(
                select(ResearchEvaluation).where(ResearchEvaluation.candidate_id == job.candidate_id)
            )).scalar_one_or_none()
            if evaluation is None or not evaluation.is_verified:
                job.last_error = "retryable: evaluation result was not verified and persisted"
                return self._view(job)
            job.result_evaluation_id = evaluation.id
            if job.status == EvaluationJobStatus.EVALUABLE:
                job.transition(EvaluationJobStatus.EVALUATED)
            return self._view(job)

    async def _retry_or_fail(self, job_id: int, error: str) -> dict[str, Any]:
        async with self.db.session() as session:
            job = await session.get(EvaluationJob, job_id, with_for_update=True)
            job.retry_count += 1
            job.last_error = f"retryable: {error}"
            if job.retry_count >= job.max_retries:
                job.last_error = f"permanent after {job.retry_count} attempts: {error}"
                if job.status != EvaluationJobStatus.FAILED:
                    job.transition(EvaluationJobStatus.FAILED)
            return self._view(job)

    async def _view_by_id(self, job_id: int) -> dict[str, Any]:
        async with self.db.session() as session:
            job = await session.get(EvaluationJob, job_id)
            return self._view(job)

    @staticmethod
    def _view(job: EvaluationJob) -> dict[str, Any]:
        return {
            "jobId": job.id,
            "candidateId": job.candidate_id,
            "sessionId": job.session_id,
            "simulationRecordId": job.simulation_record_id,
            "simulationAttempt": job.simulation_attempt,
            "platformJobId": job.platform_job_id,
            "alphaId": job.alpha_id,
            "status": job.status,
            "settingsHash": job.settings_hash,
            "provenanceHash": job.provenance_hash,
            "retryCount": job.retry_count,
            "maxRetries": job.max_retries,
            "lastError": job.last_error,
            "evaluationId": job.result_evaluation_id,
            "queuedAt": job.queued_at.isoformat() if job.queued_at else None,
            "evaluatedAt": job.evaluated_at.isoformat() if job.evaluated_at else None,
            "failedAt": job.failed_at.isoformat() if job.failed_at else None,
        }
