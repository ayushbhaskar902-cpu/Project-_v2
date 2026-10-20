# P2-03 Final Verification Report — Researcher Failure Accounting

**Canonical issue:** Researcher failures may be converted into missing output  
**Status:** VERIFIED COMPLETE  
**Implementation date:** 2026-10-20

## Implemented safeguards

- `GenerationAttempt` now records additive lifecycle, timing, retry, bounded error, output-validation, token, cost, and provider-request metadata. Existing rows remain readable with nullable fields.
- `run_generation_round()` creates one `RUNNING` attempt per logical researcher before starting provider calls. Provider failures, cancellations, unexpected orchestration exceptions, empty output, and validation failures are recorded distinctly.
- `_dispatch_researcher()` disables synthetic fallback for orchestrated calls and retries only explicitly transient/idempotent failures, with a maximum of two retries and bounded backoff.
- Successful siblings and valid candidates continue through the existing provenance, dataset/catalog, simulation-settings, fingerprint, and diversity paths. Candidate persistence remains one row per returned candidate and retries do not create rows.
- The generation API preserves existing response fields and adds additive partial-round accounting. The frontend displays failed/cancelled researcher outcomes and retained valid candidates.
- Stale `RUNNING` attempts older than 15 minutes are recovered as terminal failures at the next generation-round reconciliation.

## Verification

| Run | Result |
|---|---:|
| `pytest tests/test_p2_03_researcher_attempts.py -q` | **4 passed** |
| Relevant P0/P1/P2 regression selection | **146 passed, 1 existing async warning** |
| `pytest --tb=short -q` | **343 passed, 21 warnings, exit code 0** |

The warnings are existing SQLite/SQLAlchemy asynchronous connection teardown warnings tracked by `DEFERRED-TEST-01`; no new P2-03-specific failure was observed. Frontend source was updated for additive status display; frontend build verification remains subject to the repository's pre-existing TypeScript unused-symbol baseline if invoked.

## Deferred work preserved

The complete `deferred_issues_log.md` remains preserved, including `DEFERRED-TEST-01` and all prior statistical, benchmarking, memory, prompt, and settings items. Provider-specific cost pricing, richer request-ID extraction, and percentile latency dashboards remain non-blocking follow-up work because their provider contracts are not uniform.
