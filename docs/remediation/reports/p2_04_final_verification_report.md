# P2-04 Final Verification Report — BRAIN Result Backfill Failure Lifecycle

**Canonical issue:** BRAIN result backfill failure is not a terminally modeled state
**Status:** VERIFIED COMPLETE (independent runner)
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

The local Windows Python 3.14.7 runtime remains blocked before project code: asyncio and `socket.socketpair()` stall, and UV cannot write its sandbox cache. Independent verification on GitHub-hosted `ubuntu-latest` passed the Python/asyncio/socket diagnostic and frozen installs. P2-04 is **VERIFIED COMPLETE** on source commit `7fd19e3d6343ed74a48d302b3bb918422b718c79`: backend and frontend tests passed, and frontend check/build both exited 0. The clean-base comparison confirms both frontend commands failed at baseline due unused symbols in `research/index.tsx`, which were safely removed. Existing async teardown warnings remain deferred as `DEFERRED-TEST-01`.

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

Run [39440343617](https://github.com/ayushbhaskar902-cpu/Project-_v2/actions/runs/39440343617) completed on `ubuntu-latest` in 2m07s (14:33:41–14:35:48 UTC), at source commit `7fd19e3d6343ed74a48d302b3bb918422b718c79`. Python 3.14 runtime diagnostics, frozen backend/frontend dependency installs, all test suites, component tests, frontend check, and frontend build passed. Pytest `8.4.2` was provided ephemerally through `uv run --with`; it was not added to project dependencies or the lockfile.

| Suite/check | Result | Time |
|---|---:|---:|
| Targeted P2-04 backend regressions | **9 passed; 0 failed/errors/skipped** | 3.47s |
| P2-01 evaluation-job suite | **9 passed; 0 failed/errors/skipped** | 1.65s |
| P0 regression files | **60 passed; 0 failed/errors/skipped; 12 warnings** | 11.17s |
| P1 regression files | **269 passed; 0 failed/errors/skipped; 16 warnings** | 21.66s |
| P2-02 regression | **5 passed; 0 failed/errors/skipped** | 0.95s |
| P2-03 regression | **7 passed; 0 failed/errors/skipped** | 2.49s |
| Full backend suite | **353 passed; 0 failed/errors/skipped; 39 warnings** | 30.03s |
| Frontend component tests | **9 passed; 0 failed/skipped; exit 0** | 0.33s local run; runner step passed |
| `pnpm check` | **exit 0** | Runner step passed |
| `pnpm build` | **exit 0** | Runner step passed |

The full-suite warnings include `Event loop is closed` async teardown diagnostics, preserved under `DEFERRED-TEST-01`. No test was skipped. The earlier run at `40a3c2c` identified a UI defect: the “Refresh Status” control called the simulation/evaluation POST flow. It now calls the read-only evaluation-job GET endpoint. A second concurrency review found delayed `_fetch_each()` failures could be misattributed to a newer worker or unregister its callback; worker token and attempt fencing now prevent both, with a dedicated regression. The P2-04 display tests cover waiting, in-progress, retry time, incomplete artifacts, terminal failure, human review, verified-ready, and evaluated states, and assert polling/refresh uses GET.

The frontend baseline job at clean commit `2be883eea92d92e946c68352fca5cb40651a1a83` still returns exit 1 for both `pnpm check` and `pnpm build`, due to unused imports/state/constants in `frontend/src/screens/research/index.tsx`. The same checks on the WIP source now return 0 after removing those demonstrably unused declarations and the unreachable, nonfunctional New Campaign control/handler. Thus the previous failures were confirmed baseline issues and are resolved on the WIP branch without suppressing checks.

The tested implementation now satisfies the P2-04 acceptance criteria: durable error association and retry history; transient/permanent classification and bounded schedules; missing/incomplete artifact gating; provenance-verified persisted artifacts before `EVALUABLE`; duplicate/concurrent processing and stale-worker fencing, including delayed background failures; restart and database-failure safety; auditable manual requeue; accurate UI status rendering; and preserved P0 provenance/holdout and P2-01 evaluation safeguards. The local Windows interpreter remains an environment-specific limitation; it does not affect the independent-runner result. P2-04 is **VERIFIED COMPLETE**. The dedicated branch is ready for review; no merge to `main` is included in this checkpoint.
