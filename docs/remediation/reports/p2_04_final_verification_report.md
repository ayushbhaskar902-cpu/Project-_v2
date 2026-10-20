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

Run [39436169099](https://github.com/ayushbhaskar902-cpu/Project-_v2/actions/runs/39436169099) at checkpoint `44d3df32e2f82712b29358f5369f51d03403897f` completed on `ubuntu-latest` in approximately 1m34s. Python 3.14 diagnostic, `uv sync --frozen`, frontend frozen install, and build workflow setup completed. Pytest `8.4.2` was added ephemerally with `uv run --with`; it was not added to project dependencies or the lockfile.

| Suite | Result | Pytest time |
|---|---:|---:|
| Targeted P2-04 regressions | **6 passed; 0 failed/errors/skipped** | 1.74s |
| P2-01 evaluation-job file | **8 passed; 0 failed/errors/skipped** | 0.85s |
| P0 regression files | **60 passed; 0 failed/errors/skipped; 12 warnings** | 6.25s |
| P1 regression files | **269 passed; 0 failed/errors/skipped; 22 warnings** | 11.56s |
| P2-02 regression | **5 passed; 0 failed/errors/skipped** | 0.51s |
| P2-03 regression | **7 passed; 0 failed/errors/skipped** | 1.81s |
| Full backend suite | **350 passed; 0 failed/errors/skipped; 31 warnings** | 16.56s |

The backend warnings include an `Event loop is closed` teardown diagnostic, consistent with the preserved `DEFERRED-TEST-01` async worker-thread cleanup issue. No test was skipped. The first independent run exposed an undeclared `httpx` import in `learning/loop.py`, although the frozen project supplies `httpx2`; the import and corresponding timeout test now use the locked `httpx2` package. The passing backend figures above include that compatibility fix.

The frontend check and build both failed on the working branch. A comparison run at base commit `2be883eea92d92e946c68352fca5cb40651a1a83` failed both checks with the same existing unused imports, variables, and constants in `frontend/src/screens/research/index.tsx`. The working-tree diagnostics refer to those unchanged declarations; none point into the added P2-04 lifecycle rendering block. Thus these failures are confirmed baseline debt, while a successful frontend build is not established.

The rerun now covers durable `_fetch_each()` association, retry classification and schedules across service/database restarts, artifact gating, manual requeue history, and an expired stale worker. A focused database-commit-failure regression and durable callback persistence regression have since been added and must pass on the next runner. Frontend state rendering has no component-level regression test, and frontend check/build remain failing on confirmed baseline unused-symbol diagnostics. P2-04 remains **UNVERIFIED**.

No full backend suite result is claimed. The targeted and complete backend commands remain:

```text
uv run pytest tests/test_p2_01_evaluation_jobs.py -q
uv run pytest --tb=short -q
```

The report must be amended with actual exit codes and counts when the repository test runtime is available.
