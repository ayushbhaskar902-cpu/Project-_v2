# P2-01 Final Verification Report — Frontend Can Evaluate Before BRAIN Simulations Finish

**Status:** VERIFIED COMPLETE
**Verified:** 2026-10-20
**Canonical issue:** Frontend can evaluate before BRAIN simulations finish

## Implemented behavior

- Added durable `EvaluationJob` records with candidate/session/simulation/attempt identity, platform and alpha IDs, settings/provenance fingerprints, retry/error metadata, timestamps, result reference, and a guarded lifecycle: `QUEUED`, `SUBMITTED`, `SIMULATING`, `RESULT_BACKFILL`, `EVALUABLE`, `EVALUATED`, and `FAILED`.
- Added a unique `(candidate_id, simulation_attempt)` key and concurrent-create recovery so duplicate requests reuse one job while distinct attempts remain distinct.
- The backend now reconciles actual `SimulationRecord` state and only advances to `EVALUABLE` after verified vault data is available. `EVALUATED` is written only after a verified `ResearchEvaluation` is persisted.
- `POST /sessions/{session_id}/evaluate` creates/reuses and processes durable jobs. `GET /sessions/{session_id}/evaluation-jobs` is the read-only progress endpoint.
- The frontend submits the workflow once and polls only the GET progress endpoint. Refresh and session changes reload durable jobs; no polling path repeats `POST /evaluate`.
- The legacy `pending_simulations` response status and existing response fields remain available for compatibility; detailed job phases are additive.
- The existing `ResearchEvaluation.evaluated_at` field remains on its original model after adding the new job model.

## Changed artifacts

- `backend/src/alpha_harness/db/models.py`
- `backend/src/alpha_harness/evaluation/jobs.py`
- `backend/src/alpha_harness/api/research.py`
- `backend/tests/test_p2_01_evaluation_jobs.py`
- `frontend/src/screens/research/index.tsx`
- `docs/remediation/reports/p2_01_audit_report.md`
- `docs/remediation/reports/p2_01_final_verification_report.md`
- `docs/remediation/reports/p2_02_audit_report.md`
- `Project_v2_Remediation_Backlog_and_LLM_Prompt.md`
- `AI_HANDOFF.md`

## Verification results

| Check | Result |
|---|---|
| P2-01 targeted lifecycle tests | **5 passed**, exit code 0 |
| Closely coupled P0/P1 regressions | **61 passed**, **21 warnings**, exit code 0 |
| Full backend suite (`uv run pytest -q --tb=short`) | **334 passed**, **36 warnings**, exit code 0 |
| Python compile check | Passed, exit code 0 |
| Frontend TypeScript check | Existing unrelated unused-symbol errors in `frontend/src/screens/research/index.tsx`; no P2-01 type errors reported |
| Frontend formatter/lint check | Existing repository CRLF/format baseline plus the same pre-existing unused symbols; no new P2-01-specific type failure |

The remaining warnings are primarily `aiosqlite`/SQLAlchemy connection cleanup warnings from existing async test teardown paths, plus one event-loop worker-thread warning. They do not change the passing exit code. No secrets, `.env` files, credentials, local databases, generated runtime data, or dependency directories are included in the implementation change set; such runtime artifacts remain ignored.

## Acceptance coverage assessment

The five new tests directly cover guarded transitions and invalid transitions, concurrent duplicate reuse and distinct attempts, platform/readiness gating, retry-to-terminal failure, and persistence-before-`EVALUATED`. Existing P0/P1 suites cover provenance, statistical/holdout isolation, settings fingerprints, routing, and P1-16 eligibility separation. The frontend check found no P2-01 type error, but the repository’s existing screen-level unused-state errors prevent a clean frontend check; a browser-level test harness is not configured in this repository.

## Safeguards and known limitations

P0 provenance and verified-vault checks remain authoritative; P0-03 holdout metrics remain outside routine evaluation; P1-15 settings fingerprints and P1-16 quality/eligibility separation are carried into jobs and are not treated as submission evidence. Existing session checkpointing remains in place.

The implementation uses the existing additive schema migrator to create the new table. Platform terminal states are surfaced as failed job state; backfill/evaluation exceptions use bounded retry metadata. Scheduling/backoff policy beyond the bounded retry count, realtime push notifications, and frontend browser automation remain non-blocking technical debt and are not silently treated as complete.
