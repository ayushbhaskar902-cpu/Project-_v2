# P2-02 Final Verification Report — API Module Contains Too Much Business/Research Logic

**Canonical issue:** API module contains too much business/research logic  
**Status:** Implementation complete; verification evidence recorded 2026-10-20.

## Implementation

The extraction is incremental and preserves the public API surface:

- `backend/src/alpha_harness/application/research.py` now owns `ResearchSessionService` for session creation, candidate dispatch, generation, and experiment lifecycle, plus `ResearchWorkflowService` for graduation, robustness, learning, and pool curation.
- `backend/src/alpha_harness/api/research.py` remains the transport adapter: it validates request models, calls services, maps domain failures to the existing HTTP responses, and retains read-only query/serialization handlers.
- `backend/src/alpha_harness/application/__init__.py` establishes the application-service package.
- `backend/tests/test_p2_02_research_services.py` covers service delegation, learning response/snapshot behavior, and empty robustness/curation contracts without FastAPI routing.

No database schema or migration was added. Existing route paths, response keys, status messages, realtime topic (`TOPIC_SESSION`), explicit cancellation commit, and holdout exception mappings were retained.

## Compatibility and safeguards

The extraction does not alter P0/P1 safeguards. Provenance and generation-attempt persistence remain in the orchestrator; P2-01 durable evaluation jobs remain in `EvaluationJobService`; holdout chronology, immutability, overlap, and indeterminacy checks remain in the evaluation service; statistical selection and prompt/leaderboard safeguards remain in their existing modules. The extracted service calls preserve the prior transaction scopes: session creation and workflow selection queries use the existing database context, and cancellation explicitly commits the same state update as the route did previously.

## Verification evidence

- Static route/service inspection confirms the extracted handlers delegate to services and retain their prior response contracts.
- `git diff --check`: passed.
- Ruff parsing completed; the repository still contains pre-existing lint findings in the large API module, and the new service introduces no import-resolution finding beyond those existing file-wide findings.
- P2-02 service tests plus the relevant P0/P1/P2 regression groups: **35 passed, 2 warnings**.
- Complete backend suite: **339 passed, 32 warnings, exit code 0**.
- The warnings are existing async SQLite/SQLAlchemy teardown warnings; no failure or new P2-02-specific warning was reported.

## Remaining debt and deferred work

Read-only query/response shaping remains in `research.py` by design; a broader decomposition would be a separate change. P2-03 is intentionally not implemented. Existing deferred statistical, benchmarking, memory, prompt, settings, and async teardown issues remain unchanged.
