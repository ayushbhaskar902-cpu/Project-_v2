# P2-02 Audit Report — API Module Contains Too Much Business/Research Logic

**Canonical issue title:** API module contains too much business/research logic
**Canonical problem statement:** HTTP routing, BRAIN synchronization, synthetic fallback, evaluation and robustness logic are coupled in a large module.
**Status:** VERIFIED COMPLETE
**Implementation:** Incremental application-service extraction completed; P2-03 remains audit-only.

## Scope and confirmed evidence

The audited module is `backend/src/alpha_harness/api/research.py` (1,474 lines, 21 route handlers and multiple request/response models). The following are confirmed from the current source:

1. `enqueue_session_candidates()` directly coordinates request validation, candidate selection and `state.orchestrator.enqueue_candidates_for_simulation()`.
2. `trigger_generation_round()` directly constructs and starts `ExperimentRunner`, then packages orchestration results as an HTTP response.
3. `run_experiment_session()`, `pause_experiment_session()`, `resume_experiment_session()`, `cancel_experiment_session()`, and `get_experiment_session_status()` directly own experiment lifecycle control and error-to-HTTP mapping.
4. `evaluate_session_candidates()` now coordinates simulation synchronization and `EvaluationJobService`; its route-level response aggregation remains in the API module.
5. `run_robustness_suite()` directly validates a request, loads a candidate, constructs robustness execution inputs, invokes robustness processing, and serializes the response.
6. `graduate_candidate_endpoint()` directly invokes evaluation graduation logic and maps domain failures to HTTP responses.
7. `trigger_learning_cycle()` directly invokes the learning loop and returns orchestration output.
8. `curate_alpha_pool()` directly loads candidate/evaluation rows, applies selection inputs, invokes pool curation, and builds the API response.
9. The same module also owns candidate/session queries, leaderboards, prompt-strategy views, snapshots, and response shaping. This is confirmed coupling by source structure; it is not a claim that every operation is incorrect.

## What is confirmed versus not established

Confirmed: the route module contains transport concerns alongside orchestration calls, domain-error translation, database query construction, response shaping, and workflow-specific business decisions.

Not established: no evidence was found in this audit that extraction is required to fix a currently failing behavior, nor that every handler needs a separate service. Existing route and regression tests are the compatibility baseline. No performance or concurrency regression is claimed solely from module size.

## Proposed minimal remediation

After approval, extract one cohesive service boundary at a time, beginning with the highest-risk workflow already protected by tests:

1. Keep route signatures, authentication dependencies, status codes, and response schemas unchanged.
2. Move orchestration/evaluation response assembly into a focused application service, reusing `EvaluationJobService` for P2-01 rather than duplicating lifecycle logic.
3. Move robustness execution and learning-cycle coordination next only where service-level tests can be written first.
4. Leave read-only query/serialization helpers together until their behavior is covered; avoid a big-bang package rewrite.
5. Delete no deferred issue and preserve the current route paths for backward compatibility.

## Acceptance criteria and required regression tests

- Route handlers remain thin adapters: parse/authorize, call a service, map a domain result, and return the existing schema.
- Service tests exercise evaluation, robustness, learning, and curation without constructing FastAPI routes.
- Route tests preserve current paths, response fields, validation behavior, and error status codes.
- P0/P1 provenance, holdout isolation, statistical correctness, settings fingerprints, and P1-16 eligibility/submission separation remain unchanged.
- Full backend suite remains green; frontend behavior remains unchanged except for already verified P2-01 polling behavior.

Required tests: service-level success/failure/partial-failure cases for each extracted workflow; route contract tests; authorization and invalid-input tests; persistence/rollback tests; and the complete regression suite.

## Risks, migration, and deferred technical debt

- No database migration is expected for a pure service extraction; any new persistence boundary must use the existing additive migrator and be separately reviewed.
- The main regression risk is changing transaction scope, error mapping, or response ordering while moving code out of the route module.
- Existing callers may depend on undocumented response details; route contract tests must capture observed behavior before extraction.
- `research.py` will remain a valid read/query surface after incremental extraction; full decomposition is non-blocking technical debt.
- Realtime notification integration, broader async teardown cleanup, and a browser-level frontend harness remain outside P2-02 scope.
