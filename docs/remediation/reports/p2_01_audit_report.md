# P2-01 Audit Report — Frontend Can Evaluate Before BRAIN Simulations Finish

**Canonical issue title:** Frontend can evaluate before BRAIN simulations finish
**Canonical problem statement:** Workflow can enqueue simulations and immediately call evaluate; asynchronous BRAIN completion/backfill may not be ready.
**Status:** AUDITED — PENDING APPROVAL
**Implementation:** Not started

## Findings

### Confirmed behavior

1. `frontend/src/screens/research/index.tsx`, `handleSimulateCandidates()` first POSTs `/sessions/{id}/enqueue` and immediately POSTs `/sessions/{id}/evaluate` (around lines 366–378). This is the confirmed enqueue-then-evaluate race window.
2. `frontend/src/screens/research/index.tsx`, the `useEffect` beginning around line 418 uses a 10-second `setInterval`, but each poll POSTs the same `/evaluate` endpoint (around lines 422–431). It does not observe a persisted evaluation-job state or distinguish backfill phases.
3. `backend/src/alpha_harness/api/research.py`, `evaluate_session()` (around lines 707–840), selects candidates with `alpha_id` and invokes `state.evaluation.evaluate_candidate()` in the request path. It returns `pending_simulations` when candidate IDs are not ready, but does not create an evaluation job or persist a retryable evaluation state.
4. `backend/src/alpha_harness/experiment/runner.py`, `_phase_simulate()` and `_phase_evaluate()` (around lines 349–444), do persist the broader `ResearchSession` phase and can resume the experiment loop. This is useful existing infrastructure, but it is not a per-candidate evaluation/backfill state machine and does not make the frontend evaluation request idempotently resumable.
5. `backend/src/alpha_harness/evaluation/metrics.py`, `EvaluationEngine.evaluate_candidate()`, requires verified vault data and raises when it is unavailable. This prevents fabricated metrics, but the caller records a generic unavailable/error path rather than a durable queued/backfill/evaluable state.

### Not established by this audit

- A specific BRAIN API response timing failure was not reproduced against live BRAIN data.
- The audit does not claim that every enqueue currently races in production; it identifies the confirmed request ordering and absence of durable evaluation-job coordination that permit the race.
- Existing session checkpointing and frontend polling are not being removed; they are the compatibility baseline for the smallest fix.

Relevant paths:

- `backend/src/alpha_harness/api/research.py`: evaluation endpoint and candidate readiness selection.
- `backend/src/alpha_harness/evaluation/metrics.py`: synchronous evaluation orchestration.
- `backend/src/alpha_harness/db/models.py`: candidate and simulation lifecycle state, but no evaluation-job state machine.
- `backend/src/alpha_harness/experiment/runner.py`: round orchestration and simulation completion handling.
- `frontend/src/screens/research/index.tsx`: evaluation trigger, polling/refresh behavior, and status messaging.

## Proposed smallest fix

Add a persisted evaluation-job record (or an additive job-state column on the existing candidate/evaluation boundary) with explicit states:

`QUEUED → SUBMITTED → SIMULATING → RESULT_BACKFILL → EVALUABLE → EVALUATED`, with `FAILED` and retry metadata as terminal/recoverable branches.

The backend should enqueue or reconcile jobs idempotently, advance state only when required BRAIN artifacts exist, and make evaluation a no-op until `EVALUABLE`. The frontend should subscribe to the existing realtime topic and fall back to bounded polling of the persisted job state; refreshes must resume from the server state rather than repeatedly POSTing evaluation.

## Acceptance criteria

1. A delayed BRAIN response transitions to `EVALUABLE` and then evaluates without a manual re-click.
2. A transient backfill or polling failure records a retryable failure and resumes within bounded retry policy.
3. Missing PnL or incomplete platform results never trigger evaluation or fabricated metrics.
4. State transitions are monotonic/idempotent and survive process/browser restart.
5. UI labels distinguish queued, simulating, backfill, evaluable, evaluated, and failed states.
6. Existing P0/P1 safeguards and P1-16 eligibility/submission separation remain unchanged.

## Required tests

- Backend state-machine transition and invalid-transition tests.
- Delayed simulation/backfill integration test with automatic completion.
- Retry and terminal-failure recovery tests.
- Idempotent duplicate-enqueue test.
- Frontend/realtime or API polling test proving refresh/resume without manual re-click.
- Full backend regression suite and relevant frontend type/lint/build checks.
