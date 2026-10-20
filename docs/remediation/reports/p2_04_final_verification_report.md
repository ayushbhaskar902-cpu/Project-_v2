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

The local Windows runtime remains blocked: UV cannot write its cache, and direct Python 3.14 probes stall in `socket.socketpair()` before application code. Independent verification now runs on a clean GitHub-hosted Ubuntu runner with Python 3.14. The environment diagnostic and frozen dependency installation passed. P2-04 remains **UNVERIFIED** because the full backend suite passes but frontend `check` and `build` fail on diagnostics also present at the base commit; final targeted database-failure regression additions are pending a new run.

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

### Independent GitHub Actions run

Run [39436941450](https://github.com/ayushbhaskar902-cpu/Project-_v2/actions/runs/39436941450) at checkpoint `82d6649feb45b83af66c061b1f0b0d665820e2f2` completed on `ubuntu-latest` in approximately 1m35s. Python 3.14 diagnostic, `uv sync --frozen`, and frontend frozen installation passed. Pytest `8.4.2` was added ephemerally with `uv run --with`; it was not added to project dependencies or the lockfile.

| Suite | Result | Pytest time |
|---|---:|---:|
| Targeted P2-04 regressions | **8 passed; 0 failed/errors/skipped** | 2.54s |
| P2-01 evaluation-job file | **9 passed; 0 failed/errors/skipped** | 0.96s |
| P0 regression files | **60 passed; 0 failed/errors/skipped; 12 warnings** | 7.09s |
| P1 regression files | **269 passed; 0 failed/errors/skipped; 33 warnings** | 11.72s |
| P2-02 regression | **5 passed; 0 failed/errors/skipped** | 0.55s |
| P2-03 regression | **7 passed; 0 failed/errors/skipped** | 1.84s |
| Full backend suite | **352 passed; 0 failed/errors/skipped; 31 warnings** | 16.16s |

The backend warnings include an `Event loop is closed` teardown diagnostic, consistent with the preserved `DEFERRED-TEST-01` async worker-thread cleanup issue. No test was skipped. The first independent run exposed an undeclared `httpx` import in `learning/loop.py`, although the frozen project supplies `httpx2`; the import and corresponding timeout test now use the locked `httpx2` package. The passing backend figures above include that compatibility fix.

The frontend check and build both failed on the working branch. A comparison run at base commit `2be883eea92d92e946c68352fca5cb40651a1a83` failed both checks with the same existing unused imports, variables, and constants in `frontend/src/screens/research/index.tsx`. The working-tree diagnostics refer to those unchanged declarations; none point into the added P2-04 lifecycle rendering block. Thus these failures are confirmed baseline debt, while a successful frontend build is not established.

The passing targeted tests cover durable `_fetch_each()` job association after database restart, retry classification/scheduling across service restarts, missing-artifact gating, explicit manual requeue history, expired-worker fencing, and a simulated commit failure that leaves the job blocked in `RESULT_BACKFILL`. P0/P1/P2-02/P2-03 and the full backend suite all passed. The overall workflow exited **1** because frontend `pnpm check` and `pnpm build` failed. The comparison workflow at the base commit also failed both commands with the same unused imports, variables, and constants in unchanged lines of `research/index.tsx`; the lifecycle-rendering changes received no diagnostics. The current UI still lacks a component-level state-rendering test. P2-04 remains **UNVERIFIED** because required frontend checks did not pass, even though the failures are confirmed at baseline.
