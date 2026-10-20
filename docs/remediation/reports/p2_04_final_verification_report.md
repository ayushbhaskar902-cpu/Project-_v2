# P2-04 Final Verification Report — BRAIN Result Backfill Failure Lifecycle

**Canonical issue:** BRAIN result backfill failure is not a terminally modeled state
**Status:** IMPLEMENTED — UNVERIFIED
**Implementation date:** 2026-10-20

## Implemented safeguards

- Added durable backfill state, retry count, bounded `next_retry_at`, sanitized error classification/details, required and verified artifact maps, attempt history, lease expiry, and manual-requeue accounting to `EvaluationJob`.
- Backfill failures are classified as transient or permanent. Transient failures use bounded exponential backoff and rate-limit guidance; permanent failures become terminal without indefinite retry.
- Missing, incomplete, or malformed verified-vault artifacts remain explicitly blocked and cannot transition to `EVALUABLE`.
- `EVALUABLE` is reached only after `load_verified_series()` returns genuine verified data; the job records `BRAIN_VAULT` artifact provenance before evaluation.
- API job views expose additive backfill readiness and failure fields. Manual requeue is explicit and recorded in attempt history.
- Backfill leases now carry a worker token; stale workers and same-process duplicate processors cannot commit newer outcomes over a current worker.
- Background `Backfill._fetch_each()` failures now report the registered job ID, simulation attempt, alpha ID, and sanitized exception to the durable evaluation coordinator; reporting failures do not abort sibling backfills.
- The research UI now renders persisted waiting, in-progress, retry-scheduled, incomplete-artifact, terminal-failure, review-required, and verified-ready states without treating them as evaluated.

## Verification status

The compatible installed Python 3.14.7 interpreter was found and project dependencies plus pytest were installed into it without changing `requires-python` or deleting the repository environment. The default UV interpreter discovery still fails because its sandboxed cache cannot be written. Verification remains incomplete because the managed Windows Python process cannot initialize an asyncio event loop in this environment: even `asyncio.run(asyncio.sleep(0))` stalls before project code executes. The Python 3.13 installation is not compatible with the declared `requires-python >=3.14` and has no project dependencies installed.

### Isolated runtime evidence

The direct interpreter was `C:\Users\LENOVO\AppData\Local\Programs\Python\Python314\python.exe` and reported Python 3.14.7. A plain flushed `print()` process exited 0 in approximately 92 ms. A flushed `asyncio.run(asyncio.sleep(0))` process printed `before` and was externally terminated after approximately 5,043 ms. An explicitly created event loop printed `before` and was externally terminated after approximately 5,044 ms. A lower-level `socket.socketpair()` probe behaved identically, printing `before` and timing out after approximately 5,043 ms. All emitted only the interpreter warning `Failed to find real location ...python.exe`; none reached project imports or database code. This isolates the failure below asyncio, in the managed Windows process/sandbox environment, rather than an application deadlock.

| Run | Result |
|---|---:|
| `python 3.14.7 -m pytest ...::test_p2_04_backfill_error_classification_and_sanitization ...::test_job_transitions_are_monotonic_and_timestamped -q -p no:anyio -p no:timeout --basetemp=<workspace>/backend/pytest-tmp-p204` | **2 passed, exit 0, 8.44s** |
| `python 3.14.7 -m pytest ...::test_backfill_failures_retry_then_become_terminal -s -vv -p no:anyio -p no:timeout --basetemp=<workspace>/backend/pytest-tmp-p204` | **blocked in `asyncio.run()`; terminated after 30s** |
| `python 3.14.7 -c "import asyncio; asyncio.run(asyncio.sleep(0))"` | **confirmed environment blocker: stalls before the coroutine runs** |
| direct explicit-loop probe (`new_event_loop(); run_until_complete(asyncio.sleep(0))`) | **confirmed environment blocker: printed `before`, externally killed after ~5.04s** |
| direct `socket.socketpair()` probe | **confirmed lower-level environment blocker: printed `before`, externally killed after ~5.04s** |
| `python 3.14.7 -m pytest ...::test_jobs_are_idempotent_per_candidate_attempt_and_distinct_attempts_are_preserved -q` | **not completed because the same asyncio initialization blocker applies** |
| `pytest-timeout` diagnostic | **inconclusive: plugin attempted POSIX `SIGALRM` on Windows, exit 3** |
| `python -m compileall -q backend/src` | **passed, exit 0** |
| `git diff --check` | **passed, exit 0** |

The concurrent/stale-worker/database-failure and frontend-rendering acceptance tests still require a completed runtime test run. P2-04 remains **UNVERIFIED**. The frontend TypeScript build check also remains blocked by the existing project baseline: direct `tsc -b` returned exit 2 with pre-existing React Query export/type and unused-symbol errors; no frontend pass is claimed.

No full backend suite result is claimed. The targeted and complete backend commands remain:

```text
uv run pytest tests/test_p2_01_evaluation_jobs.py -q
uv run pytest --tb=short -q
```

The report must be amended with actual exit codes and counts when the repository test runtime is available.
