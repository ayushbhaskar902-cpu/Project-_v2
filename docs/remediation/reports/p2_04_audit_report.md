# P2-04 Audit Report — BRAIN Result Backfill Failure Is Not a Terminally Modeled State

**Status:** AUDITED — PENDING APPROVAL  
**Implementation:** Not started.

## Canonical issue

**Canonical title:** BRAIN result backfill failure is not a terminally modeled state  
**Area:** evaluation API/backfill integration  
**Problem statement:** Backfill failures are logged, but candidates can proceed into fallback behavior instead of remaining explicitly blocked on real data.  
**Canonical fix:** Track backfill status, attempts, last error, next retry and required artifact availability. Backoff on temporary errors; stop/review after a defined retry budget; never replace missing data with fabricated returns.  
**Canonical acceptance tests:** Backfill failure leaves candidate unevaluated; a later successful backfill resumes evaluation; terminal failure is visible and auditable.

## Verified current behavior and defects

1. `backend/src/alpha_harness/evaluation/jobs.py`, `EvaluationJobService.process_job()` transitions a completed simulation to `RESULT_BACKFILL` at lines 136–145, then calls `state.backfill.fetch_returns()` and `vault.load_verified_series()` at lines 149–153. The durable `EvaluationJob` row has generic `retry_count`, `max_retries`, `last_error`, and timestamps, but no backfill-specific attempt count, error category, required-artifact state, or next-retry timestamp (`backend/src/alpha_harness/db/models.py:1236-1277`).

2. A backfill/provider exception is caught at `process_job()` lines 155–160 and sent to `_retry_or_fail()` at lines 187–198. `_retry_or_fail()` labels every exception `retryable`, increments the generic retry count, and only marks the job `FAILED` after the count reaches `max_retries`. There is no time-based backoff or persisted `next_retry_at`; a later API call can retry immediately.

3. `InsufficientDataError` is caught separately at `process_job()` lines 155–156 and returns the job view without updating `last_error`, retry count, or a terminal/blocked reason. Likewise, a `None` result from `load_verified_series()` at line 153 returns at line 157 while the job remains `RESULT_BACKFILL`. This is an explicitly confirmed indefinite-pending state when required verified data does not arrive.

4. `backend/src/alpha_harness/vault/backfill.py`, `Backfill._fetch_each()` lines 398–406 catches exceptions from background `fetch_returns()` work, logs a truncated message, and drops the failure. That background path has no durable association to an `EvaluationJob`, retry schedule, terminal state, or required-artifact record.

5. `backend/src/alpha_harness/vault/backfill.py`, `Backfill.fetch_returns()` lines 359–377 performs multiple provider recordset requests and vault writes, but exposes only an integer count or an exception. It does not return a structured result identifying which PnL, turnover, or yearly-stat artifact is missing or which error is transient/permanent.

## Evidence versus hypotheses

### Confirmed

- P2-01’s `EvaluationJobService` does not call `EvaluationEngine.evaluate_candidate()` until `load_verified_series()` succeeds; the transition to `EVALUABLE` occurs only at `process_job()` lines 161–166, and evaluation errors do not create a verified scorecard automatically.
- `backend/src/alpha_harness/evaluation/metrics.py:453-482` requires a verified vault series and raises `InsufficientDataError` when it is absent. The inspected evaluation path therefore does not fabricate PnL to complete evaluation.
- Existing tests confirm the current behavior: `backend/tests/test_p2_01_evaluation_jobs.py:test_processing_requires_platform_and_verified_vault_evidence` leaves a job in `RESULT_BACKFILL`, and `test_backfill_failures_retry_then_become_terminal` reaches `FAILED` only after three manually repeated calls.
- Existing `Backfill._fetch_each()` logging is process-local and is not an auditable evaluation-job state transition.

### Not confirmed by this audit

- No reproduction established that the active evaluation-job path substitutes fabricated returns after a backfill failure. The separate robustness endpoint contains an explicitly documented synthetic-result path at `backend/src/alpha_harness/api/research.py:850-862`, but that is not the evaluation backfill path and should not be conflated with this issue.
- Provider-specific retryability, HTTP status semantics, and exact required-artifact rules vary by endpoint and were not inferred without provider-contract tests.
- Partial-vault-write behavior across the PnL/turnover/yearly-stat sequence was not reproduced; it requires a fault-injection test around `AlphaVault` writes.

## Minimal proposed remediation

Extend `EvaluationJob` additively with a backfill sub-state and telemetry: backfill attempt count, last backfill error/class, next retry time, and required-artifact availability. Keep `RESULT_BACKFILL` as the blocked state and add a clearly visible terminal `FAILED` reason after a bounded budget. Classify temporary transport/rate-limit failures separately from permanent authorization, invalid-alpha, and unavailable-artifact failures. Persist exponential backoff before allowing another automatic attempt. On each retry, re-read the durable job and verified vault evidence; transition to `EVALUABLE` only after the required real artifacts are present. Never create or accept synthetic returns in this path.

Keep the existing `EvaluationJobService` orchestration and route contracts; avoid a new queue framework unless tests demonstrate that the existing service cannot safely own the state machine. Background `Backfill._fetch_each()` should report failures to the durable coordinator or remain explicitly best-effort and separate from evaluation readiness.

## Acceptance and regression tests

- A provider/backfill exception leaves the candidate without a `ResearchEvaluation`, keeps the job blocked or marks it terminally failed according to the persisted retry policy, and records sanitized error class/message, attempt count, and next retry/terminal timestamp.
- A missing or incomplete required artifact leaves the job in `RESULT_BACKFILL` (or an explicit blocked equivalent), never `EVALUABLE`/`EVALUATED`; no scorecard or fabricated returns are created.
- A later successful backfill makes verified PnL/turnover available, resumes the same durable job exactly once, and produces a verified `BRAIN_VAULT` evaluation.
- Temporary failures use bounded exponential backoff; permanent authorization/invalid-request failures do not consume transient retries indefinitely.
- Retry exhaustion produces visible API/UI terminal status with an auditable last error and does not silently disappear from the session.
- Concurrent processing of the same candidate/attempt remains idempotent and cannot create duplicate `EvaluationJob` or `ResearchEvaluation` rows.
- Fault injection during PnL/turnover/yearly-stat retrieval or vault persistence verifies transaction behavior and required-artifact reporting.
- Existing P2-01 tests continue to pass, including evidence-gated readiness, job idempotency, monotonic transitions, and verified-result persistence.

## Compatibility and safety

- Preserve the existing `POST /api/research/sessions/{session_id}/evaluate` response shape and the read-only evaluation-job endpoint; add fields rather than renaming `status`, `retryCount`, or `lastError`.
- Existing jobs and legacy candidates need nullable/default migration values. A migration must not mark an old `RESULT_BACKFILL` job evaluated or invent artifact availability.
- SQLite transaction boundaries must keep state transitions and telemetry durable while provider calls remain outside long write transactions. Concurrency must continue to use the candidate/attempt uniqueness boundary and re-read state before committing evaluation.
- P0-01/P0-02 provenance rules require real verified vault evidence; P0-03 holdout sealing must remain untouched because no evaluation or retry path may access holdout data.
- P0-05/P2-02 workflow and API behavior must remain backward-compatible. P2-01’s readiness gate is a prerequisite and should be extended, not bypassed. P2-02’s service extraction should remain the integration point. P2-03’s researcher-attempt accounting is upstream and independent; backfill failures must not be misclassified as researcher failures.
- P1-10/P1-11/P1-13 statistical selection and researcher learning must not consume incomplete or unverified evaluations. P1-15 settings/provenance hashes must be preserved when a job retries.

## Dependencies, assumptions, and deferred work

P2-04 can be implemented independently of unfinished issues, but it depends on the completed P2-01 `EvaluationJob` lifecycle and verified-vault gate, and should use the P2-02 application-service boundary. It does not require P2-05 and must not implement P2-05’s reproducibility work.

Open assumptions are the provider-specific transient/permanent error taxonomy, whether yearly statistics are required for every evaluation or only for selected views, and whether background vault backfill should be coupled to evaluation jobs or remain a separate best-effort importer. Migration design, provider-specific retry policy, dead-letter/review UI, and richer artifact-level telemetry are deferred until implementation approval. All existing deferred issues, including `DEFERRED-TEST-01`, statistical, memory, benchmarking, and prompt-evolution items, remain preserved.
