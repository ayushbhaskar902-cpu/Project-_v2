# P2-03 Audit Report — Researcher Failures May Be Converted into Missing Output

**Canonical issue title:** Researcher failures may be converted into missing output  
**Canonical problem statement:** Concurrent dispatch uses `return_exceptions=True`; exception paths are logged but need durable per-attempt accounting so a failed model does not simply disappear.  
**Status:** AUDITED — PENDING APPROVAL  
**Implementation:** Not started.

## Confirmed source findings

1. `backend/src/alpha_harness/orchestrator/researchers.py`, `run_generation_round()` uses `asyncio.gather(*tasks, return_exceptions=True)` at the concurrent dispatch boundary. The result loop recognizes an `Exception` item only by logging `orchestrator.unexpected_dispatch_exception` and does not persist a failure record for that item.
2. The same module’s `_dispatch_researcher()` catches every `Exception`, logs `orchestrator.researcher_dispatch_failed`, and returns a synthetic empty `GenerationBatch` whose summary starts with `Generation failed:`. This means the normal persistence path can continue with zero candidates, without an explicit durable failed-attempt state.
3. `backend/src/alpha_harness/db/models.py`, `GenerationAttempt` persists prompt/model/dataset lineage and a creation timestamp, but has no status, error class/message, retry count, latency outcome, token/cost outcome, or terminal-failure field. The current model therefore records successful-shaped dispatch context, not a complete attempt outcome.
4. `run_generation_round()` persists a `GenerationAttempt` after dispatch results are normalized and then persists candidates from each returned batch. The current schema and flow make a provider exception indistinguishable from a valid empty batch to downstream candidate accounting unless logs are inspected.

## Evidence versus hypothesis

Confirmed behavior: the broad dispatch catch returns an empty batch; concurrent exception items are logged without a durable attempt record; the current `GenerationAttempt` model lacks outcome fields.  
Not established by this audit: the exact retry policy required by each provider, whether a provider SDK exposes reliable token/cost metadata for every failure, and whether all callers can migrate atomically to a new attempt-status vocabulary. Those require implementation-time inspection and tests.

## Minimal remediation proposed

Add an additive, durable outcome representation for each researcher dispatch—preferably extending `GenerationAttempt` if the existing lineage row can remain backward-compatible, otherwise adding a narrowly related attempt-outcome table. Record `RUNNING`, `SUCCEEDED`, `FAILED`, and (if needed) `CANCELLED` outcomes with provider/model, normalized error class/message, retry count, elapsed time, token/cost metadata when available, and candidate/output counts. Preserve existing provenance fields. Classify retryable failures explicitly, retry only transient idempotent failures with bounded backoff, and expose partial success in generation-round summaries. Do not treat an empty successful response and a provider failure as the same outcome.

## Acceptance criteria and required regression tests

- Every researcher dispatch creates one durable attempt outcome, including provider exceptions and unexpected concurrent task exceptions.
- A provider failure is visible as failed, never silently represented only by an empty candidate batch.
- Partial success preserves successful candidates while reporting failed researchers/attempts.
- Retry behavior is bounded, idempotent, and limited to explicitly transient failures; permanent failures are not retried.
- Existing provenance, prompt-version, dataset-selection, candidate lineage, session generation, and response behavior remain backward-compatible.
- Tests cover provider failure, unexpected gather exception, permanent failure without retry, transient retry success, partial-success accounting, persistence rollback/commit behavior, and legacy rows without new outcome fields.
- Full backend regression suite passes with no new warnings attributable to P2-03.

## Interactions, risks, and deferred concerns

The change must preserve P0-01 generation provenance and P1-03/P1-12 prompt and researcher attribution. It must not bypass P1-04/P1-05 dataset/catalog lineage, P1-14 diversity accounting, P1-15 settings fingerprints, or P2-01 evaluation-job readiness. The principal risks are duplicate provider calls, incorrect retry classification, transaction rollback leaving an attempt without a terminal state, and changing generation counts when failures are surfaced. No migration is implemented by this audit; an additive migration is expected if persistent columns or a new table are approved. Cost/telemetry enrichment and broader provider-specific retry policy remain non-blocking technical debt until their data contracts are verified.
