"""Durable coordination for research-candidate evaluation."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
import re
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from ..db.models import (
    EvaluationJob,
    EvaluationJobStatus,
    BackfillStatus,
    ResearchCandidate,
    ResearchEvaluation,
    SimulationRecord,
    SimStatus,
)
from .metrics import EvaluationEngine, InsufficientDataError


def _now() -> datetime:
    return datetime.now(UTC)


def _classify_backfill_error(exc: BaseException) -> tuple[str, bool, float | None]:
    """Return (classification, retryable, provider_retry_after) without leaking secrets."""
    from ..brain.errors import (
        BrainDailyLimitReached,
        BrainRateLimited,
        BrainServiceUnavailable,
        BrainTransportError,
    )

    if isinstance(exc, BrainDailyLimitReached):
        return "DAILY_LIMIT", False, None
    if isinstance(exc, BrainRateLimited):
        return "RATE_LIMIT", True, getattr(exc, "retry_after", None)
    if isinstance(exc, (BrainServiceUnavailable, BrainTransportError)):
        return type(exc).__name__.upper(), True, None
    status = getattr(exc, "status", None)
    if isinstance(status, int) and (status == 429 or status >= 500):
        return f"HTTP_{status}", True, getattr(exc, "retry_after", None)
    text = str(exc).lower()
    transient = any(
        word in text
        for word in ("timeout", "temporar", "unavailable", "connection", "rate limit", "429")
    )
    return ("TRANSIENT_ERROR" if transient else "PERMANENT_ERROR"), transient, None


def _safe_error(exc: BaseException) -> str:
    value = re.sub(
        r"(authorization|cookie|token|password|api[_ -]?key)\s*[:=]\s*[^,; ]+",
        r"\1=<redacted>",
        str(exc),
        flags=re.I,
    )
    return value[:500]


class EvaluationJobService:
    """Creates, reconciles, and processes idempotent evaluation attempts."""

    def __init__(
        self,
        db: Any,
        evaluation: EvaluationEngine,
        *,
        backfill: Any = None,
        auth: Any = None,
        clock: Any = _now,
    ) -> None:
        self.db = db
        self.evaluation = evaluation
        self.backfill = backfill
        self.auth = auth
        self.clock = clock
        self._ensure_lock = asyncio.Lock()
        self._process_locks: dict[int, asyncio.Lock] = {}

    async def ensure_job(self, candidate_id: int, simulation_attempt: int = 1) -> EvaluationJob:
        """Get or create a job without concurrent same-process insert races."""
        async with self._ensure_lock:
            return await self._ensure_job(candidate_id, simulation_attempt)

    async def _ensure_job(self, candidate_id: int, simulation_attempt: int = 1) -> EvaluationJob:
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
            elif (
                job.status == EvaluationJobStatus.FAILED
                and job.backfill_status == BackfillStatus.RETRY_SCHEDULED
                and job.retry_count < job.max_retries
                and (job.next_retry_at is None or job.next_retry_at <= self.clock())
            ):
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
        lock = self._process_locks.setdefault(job_id, asyncio.Lock())
        async with lock:
            return await self._process_job(job_id, cost_bps=cost_bps)

    async def _process_job(self, job_id: int, *, cost_bps: float = 5.0) -> dict[str, Any]:
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
            if simulation.status in (
                SimStatus.ERROR,
                SimStatus.FAILED,
                SimStatus.CANCELLED,
                SimStatus.TIMEOUT,
                SimStatus.REJECTED,
            ):
                job.retry_count += 1
                job.last_error = (
                    f"permanent: platform simulation {simulation.status}: "
                    f"{simulation.message or 'failed'}"
                )
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
            if (
                job.status == EvaluationJobStatus.RESULT_BACKFILL
                and job.next_retry_at
                and job.next_retry_at > self.clock()
            ):
                return self._view(job)
            if (
                job.status == EvaluationJobStatus.RESULT_BACKFILL
                and job.backfill_lease_until
                and job.backfill_lease_until > self.clock()
                and job.backfill_worker_token
            ):
                return self._view(job)
            worker_token = uuid4().hex
            job.backfill_status = BackfillStatus.IN_PROGRESS
            job.backfill_worker_token = worker_token
            job.backfill_lease_until = self.clock() + timedelta(minutes=5)
            job.backfill_attempt_count += 1
            job.next_retry_at = None
            job.required_artifacts = {"pnl": False, "turnover": False, "verified_series": False}
            alpha_id = candidate.alpha_id

        # Backfill and vault verification happen outside the transaction. No state is
        # advanced until the genuine verified series is available.
        try:
            if self.backfill is not None and hasattr(self.backfill, "register_evaluation_job"):
                self.backfill.register_evaluation_job(
                    alpha_id,
                    job_id,
                    job.simulation_attempt,
                    self._record_background_backfill_failure,
                )
            if (
                self.backfill is not None
                and self.auth is not None
                and getattr(self.auth.session, "authenticated", False)
            ):
                if await self.evaluation.vault.series_length(alpha_id) == 0:
                    await self.backfill.fetch_returns(alpha_id)
            verified = await self.evaluation.vault.load_verified_series(alpha_id)
            if verified is None:
                await self._mark_backfill_pending(
                    job_id, worker_token, "verified result artifacts are missing or incomplete"
                )
                return await self._view_by_id(job_id)
        except InsufficientDataError:
            await self._mark_backfill_pending(
                job_id, worker_token, "verified result artifacts are insufficient"
            )
            return await self._view_by_id(job_id)
        except Exception as exc:
            return await self._record_backfill_failure(job_id, worker_token, exc)
        finally:
            if self.backfill is not None and hasattr(self.backfill, "unregister_evaluation_job"):
                self.backfill.unregister_evaluation_job(alpha_id)

        async with self.db.session() as session:
            job = await session.get(EvaluationJob, job_id, with_for_update=True)
            if job is None:
                raise ValueError(f"Evaluation job {job_id} was not found")
            if (
                job.status == EvaluationJobStatus.RESULT_BACKFILL
                and job.backfill_worker_token == worker_token
                and job.backfill_lease_until
                and job.backfill_lease_until >= self.clock()
            ):
                job.backfill_status = BackfillStatus.COMPLETE
                job.required_artifacts = {"pnl": True, "turnover": True, "verified_series": True}
                job.verified_artifacts = {"source": "BRAIN_VAULT", "verified": True}
                job.next_retry_at = None
                job.backfill_worker_token = None
                job.backfill_lease_until = None
                job.transition(EvaluationJobStatus.EVALUABLE)
            if job.status != EvaluationJobStatus.EVALUABLE:
                return self._view(job)

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

    async def _mark_backfill_pending(self, job_id: int, worker_token: str, detail: str) -> None:
        async with self.db.session() as session:
            job = await session.get(EvaluationJob, job_id, with_for_update=True)
            if job is None or job.backfill_worker_token != worker_token:
                return
            job.backfill_status = BackfillStatus.PENDING
            job.backfill_error_class = "INCOMPLETE_ARTIFACTS"
            job.backfill_error_detail = detail[:500]
            job.last_error = f"pending: {detail[:500]}"
            job.required_artifacts = {"pnl": False, "turnover": False, "verified_series": False}
            job.backfill_worker_token = None
            job.backfill_lease_until = None

    async def _record_background_backfill_failure(
        self, job_id: int, simulation_attempt: int, alpha_id: str, exc: BaseException
    ) -> None:
        """Persist a failure reported by Backfill._fetch_each when job context exists."""
        async with self.db.session() as session:
            job = await session.get(EvaluationJob, job_id, with_for_update=True)
            if (
                job is None
                or job.simulation_attempt != simulation_attempt
                or job.alpha_id != alpha_id
                or job.status in (EvaluationJobStatus.EVALUATED, EvaluationJobStatus.FAILED)
            ):
                return
            classification, retryable, provider_after = _classify_backfill_error(exc)
            detail = _safe_error(exc)
            job.backfill_attempt_count += 1
            job.retry_count += 1
            job.backfill_error_class = classification
            job.backfill_error_detail = detail
            job.last_error = f"{classification.lower()}: {detail}"
            job.backfill_status = (
                BackfillStatus.RETRY_SCHEDULED if retryable and job.retry_count < job.max_retries
                else (BackfillStatus.REVIEW_REQUIRED if retryable else BackfillStatus.TERMINAL_FAILURE)
            )
            job.next_retry_at = (
                self.clock()
                + timedelta(
                    seconds=max(
                        1.0,
                        provider_after
                        or min(3600.0, 30.0 * (2 ** (job.retry_count - 1))),
                    )
                )
                if job.backfill_status == BackfillStatus.RETRY_SCHEDULED
                else None
            )
            history = list(job.attempt_history or [])
            history.append(
                {
                    "attempt": job.backfill_attempt_count,
                    "source": "background",
                    "at": self.clock().isoformat(),
                    "classification": classification,
                    "detail": detail,
                }
            )
            job.attempt_history = history[-20:]
            if job.backfill_status in (BackfillStatus.REVIEW_REQUIRED, BackfillStatus.TERMINAL_FAILURE):
                if job.status != EvaluationJobStatus.FAILED:
                    job.transition(EvaluationJobStatus.FAILED)

    async def _record_backfill_failure(
        self, job_id: int, worker_token: str, exc: BaseException
    ) -> dict[str, Any]:
        classification, retryable, provider_after = _classify_backfill_error(exc)
        detail = _safe_error(exc)
        async with self.db.session() as session:
            job = await session.get(EvaluationJob, job_id, with_for_update=True)
            if job is None:
                raise ValueError(f"Evaluation job {job_id} was not found")
            if job.backfill_worker_token != worker_token:
                return self._view(job)
            job.backfill_error_class = classification
            job.backfill_error_detail = detail
            job.last_error = f"{classification.lower()}: {detail}"
            job.backfill_worker_token = None
            job.backfill_lease_until = None
            history = list(job.attempt_history or [])
            history.append(
                {
                    "attempt": job.backfill_attempt_count,
                    "at": self.clock().isoformat(),
                    "classification": classification,
                    "detail": detail,
                }
            )
            job.attempt_history = history[-20:]
            job.retry_count += 1
            if retryable and job.retry_count < job.max_retries:
                delay = (
                    provider_after
                    if provider_after is not None
                    else min(3600.0, 30.0 * (2 ** (job.backfill_attempt_count - 1)))
                )
                job.next_retry_at = self.clock() + timedelta(seconds=max(1.0, delay))
                job.backfill_status = BackfillStatus.RETRY_SCHEDULED
                # Keep the job visibly blocked; FAILED is reserved for a terminal outcome.
                return self._view(job)
            job.next_retry_at = None
            job.backfill_status = BackfillStatus.REVIEW_REQUIRED if retryable else BackfillStatus.TERMINAL_FAILURE
            if job.status != EvaluationJobStatus.FAILED:
                job.transition(EvaluationJobStatus.FAILED)
            return self._view(job)

    async def manual_requeue(self, job_id: int, *, reason: str) -> dict[str, Any]:
        """Explicitly reopen a terminal/review job; never performed by polling."""
        reason = _safe_error(ValueError(reason))
        async with self.db.session() as session:
            job = await session.get(EvaluationJob, job_id, with_for_update=True)
            if job is None:
                raise ValueError(f"Evaluation job {job_id} was not found")
            if job.status != EvaluationJobStatus.FAILED:
                return self._view(job)
            job.transition(EvaluationJobStatus.QUEUED)
            job.backfill_status = BackfillStatus.PENDING
            job.next_retry_at = None
            job.backfill_worker_token = None
            job.backfill_lease_until = None
            job.manual_requeue_count += 1
            job.last_error = f"manual requeue: {reason}"
            history = list(job.attempt_history or [])
            history.append({"event": "manual_requeue", "at": self.clock().isoformat(), "reason": reason})
            job.attempt_history = history[-20:]
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
            "backfillStatus": job.backfill_status,
            "backfillAttemptCount": job.backfill_attempt_count,
            "backfillErrorClass": job.backfill_error_class,
            "backfillErrorDetail": job.backfill_error_detail,
            "nextRetryAt": job.next_retry_at.isoformat() if job.next_retry_at else None,
            "requiredArtifacts": job.required_artifacts,
            "verifiedArtifacts": job.verified_artifacts,
            "attemptHistory": job.attempt_history or [],
            "backfillLeaseUntil": job.backfill_lease_until.isoformat() if job.backfill_lease_until else None,
            "manualRequeueCount": job.manual_requeue_count,
            "evaluationId": job.result_evaluation_id,
            "queuedAt": job.queued_at.isoformat() if job.queued_at else None,
            "evaluatedAt": job.evaluated_at.isoformat() if job.evaluated_at else None,
            "failedAt": job.failed_at.isoformat() if job.failed_at else None,
        }
