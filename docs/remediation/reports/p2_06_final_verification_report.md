# P2-06 Verification Report — Composite Score Versioning

**Canonical issue:** Composite score is hard-coded without calibrated justification.

**Status:** **VERIFIED COMPLETE — PUSHED AND REMOTE CONFIRMED.** The implementation checkpoint is `d6a18ced8dd9497b365d024b30686197ce803cdb`; the report/tracking synchronization checkpoint `d1b417fdf288dd4e568c898cbb9bce2900e1ba7e` is present on `origin/main`.

**Approval:** Approved for implementation on 2026-10-20. No approval to recalibrate weights was given or inferred.

**Repository base:** `01a12ba16c4e82a5017027d53ab85b99077f435d`. Implementation checkpoint: `d6a18ced8dd9497b365d024b30686197ce803cdb`. The reporting checkpoint `d1b417fdf288dd4e568c898cbb9bce2900e1ba7e` was pushed normally (no force) and `git ls-remote` confirmed it exactly matches `origin/main`.

**Latest tested source/test snapshot:** base `01a12ba16c4e82a5017027d53ab85b99077f435d`, with local modifications. SHA-256 of the sorted path-and-file-hash manifest over every modified/untracked backend source, backend test, and frontend source file: `50d85f47b06f3844778d5108c7a6c5fe3da413f41192e573b15df58ac848fd36` (12 files). The source snapshot includes the final lint-only line wrap in researcher memory. `origin/main` was last refreshed at `01a12ba16c4e82a5017027d53ab85b99077f435d`; a final refresh is required before checkpointing.

## Changes implemented locally

- Added stable formula IDs `validation-composite-v1`, `legacy-composite-v1` (compatibility helper), and `legacy-composite-v2` (new legacy evaluations). Coefficients were not changed. New legacy evaluations use v2 so observed `0.0` turnover is no longer confused with missing turnover; historical scores are not recalculated.
- Added persisted JSON `ResearchEvaluation.composite_score_spec` with formula ID, inputs (including `null` for missing), coefficients, defaults, turnover policy, and exact rounding rule. The existing additive SQLite migrator adds the nullable field without touching historical scalar scores; rows without reliable metadata remain `NULL`/unknown.
- Added finite-value and negative-turnover checks at the metric parser/statistics and composite-score boundaries. Non-finite legacy scores are kept out of the API composite fields, leaderboard, memory, and portfolio selection rather than being turned into plausible scores. API responses include formula metadata, memory retains it, pool/leaderboard surfaces carry it, and equal-score consumer ordering now has a candidate-ID tiebreak.
- Added formula identity display to the Research Studio leaderboard. Holdout remains excluded from score components.
- Added P2-06 tests for golden scores, versioned zero behavior, missing values, negative/above-one turnover, non-finite inputs, exact recomputation, API compatibility, pool ordering, and SQLite historical-score preservation. Extended an existing EvaluationEngine persistence test to assert stored metadata matches the scorecard.

## Verification evidence

The managed UV environment could not be queried: `.venv/Scripts/python.exe` returned `Access is denied` to UV. I did not change `requires-python`, dependencies, or the existing environment. Tests were run using the installed Python **3.14.7** directly; that interpreter reports **pytest 8.4.2**, **NumPy 2.5.4**, and **SQLAlchemy 2.1.5**. Windows sandboxed execution initially blocked asyncio's socketpair setup; using the approved test invocation outside that restricted boundary allowed DB-backed tests to run.

| Command / scope | Result | Exit |
|---|---:|---:|
| `pytest tests/test_p2_06_composite_scores.py` | 14 passed, 0 failed | 0 |
| P2-06 suite + EvaluationEngine persisted-spec regression | 15 passed, 0 failed | 0 |
| P2-06 + P2-01–P2-05 + affected P0/P1 test files | 111 passed, 0 failed, 17 warnings | 0 |
| P2-06 + P0-03 holdout suite | 30 passed, 0 failed, 10 warnings | 0 |
| Full backend `python -m pytest --tb=short -q` (before diagnosing subprocess resource limit) | 375 passed, 1 failed, 54 warnings | 1 |
| Full backend `python -m pytest --tb=short -q` (diagnostic assertion, before BLAS cap) | 375 passed, 1 failed, 42 warnings | 1 |
| Full backend `python -m pytest --tb=short -q` (latest formatted source/test snapshot, BLAS subprocess cap in place) | **376 passed, 0 failed, 39 warnings, 55.77s** | **0** |
| Focused P2-06 suite + EvaluationEngine persisted-spec regression on latest implementation | **15 passed, 0 failed, 3.86s** | **0** |
| Ruff 0.16.8 on changed backend source lines, compared against HEAD | **0 new diagnostics** (legacy diagnostics remain in untouched lines) | **0** |
| `git diff --check` | passed; only Git's LF/CRLF checkout notices | **0** |
| Isolated `test_generated_preview_is_stable_across_process_hash_seeds` rerun | 1 passed | 0 |
| Frontend `pnpm test` | 9 passed, 0 failed/skipped | 0 |
| Biome check on changed Research Studio screen | passed | 0 |
| TypeScript `tsc -b` | passed | 0 |
| Frontend `pnpm build` | passed | 0 |
| Repository-wide `pnpm check` | failed in Biome because the Windows checkout has CRLF working files while Biome's default is LF; changed screen was formatted and its focused Biome check passes. `tsc -b` was run separately and passed. | 1 |
| Ruff over changed backend files | 396 diagnostics across complete touched files; no baseline comparison was performed, so these cannot be attributed to this change. New P2-06 test module separately passes format and Ruff checks with `S101` ignored for normal pytest assertions. | 1 |

The full-suite-only failure was `tests/test_p2_05_reproducibility.py::test_generated_preview_is_stable_across_process_hash_seeds`. The improved failure assertion exposed the child stderr: `OpenBLAS error: Memory allocation still failed after 10 retries, giving up.` The test imports NumPy in each child and inherited unrestricted BLAS worker settings. The test now caps OpenBLAS/OMP/MKL workers at one for those child processes; it does not change generated values, seed handling, or assertions. The latest complete suite passed (376 passed, 39 warnings, exit 0). This is a test-process resource isolation fix, not evidence that the remaining `aiosqlite` event-loop/connection cleanup warnings are resolved; those remain tracked under `DEFERRED-TEST-01`.

The available Python was 3.14.7, but this was not the lockfile-matched environment: installed NumPy 2.5.4 differs from `uv.lock` 2.5.3, and installed SQLAlchemy 2.1.5 differs from locked 2.0.54. UV could not access/persist its managed interpreter metadata (`Access is denied`), and `.venv\Scripts\python.exe` was inaccessible. No dependency, Python requirement, or environment was changed. Therefore the local passing run is useful evidence but is not authoritative lockfile-environment verification.

P2-06 remains **NOT VERIFIED COMPLETE**. The backend suite is green on the available interpreter, and changed-line Ruff has no new findings, but the acceptance gate is still open: focused API response-variant and historical non-finite-record tests, persisted score/spec consistency beyond the existing EvaluationEngine case, real database leaderboard missing/tie ordering, researcher-memory metadata propagation, portfolio consumer coverage, and holdout-mutation invariance across affected consumers remain incomplete or need explicit confirmation. Repository-wide frontend verification in an LF checkout or authoritative CI and independent review of the entire final diff are also outstanding. A clean full backend exit does not by itself certify P2-06.

## Remaining verification gaps

- Run the full backend suite in a lockfile-matched environment or obtain authoritative CI evidence for this exact source/test snapshot.
- Run the repository-wide frontend check in a non-CRLF checkout or authoritative CI; the changed screen's focused Biome check, TypeScript project check, frontend tests, and build passed locally.
- Record the changed-line baseline-aware Ruff result: zero new backend source diagnostics (Ruff 0.16.8); historical diagnostics outside changed lines remain.
- Add/confirm focused tests for all API response shapes and historical invalid metric records, persisted score/spec consistency, leaderboard order on real stored ties/missing values, memory metadata propagation, portfolio consumption, and holdout mutation invariance across every affected consumer.
- Perform an independent source review and inspect the final complete diff before checkpointing.

## Deferred work preserved

No coefficients were empirically calibrated. Calibration/ablation, sensitivity analysis, historical formula-version inventory, and cross-cohort robustness remain deferred. `DEFERRED-TEST-01` and every other entry in `deferred_issues_log.md` remain open and unchanged. No P2-07 implementation was started.

**Decision:** P2-06 remains **APPROVED — IMPLEMENTED LOCALLY, VERIFICATION INCOMPLETE**. Do not commit/push a completion claim until the remaining verification is satisfied.

---

## Final verification addendum — 2026-10-20 (authoritative for the current snapshot)

The earlier status and “remaining gaps” above describe an earlier verification stage and are superseded by this addendum. The final tree was reviewed and tested against the snapshot manifest listed at the top of this report.

### Independent review and acceptance matrix

| Requirement | Relevant implementation | Test or independent-review evidence | Result | Remaining issue |
|---|---|---|---|---|
| Preserve existing formula weights; distinguish formula versions and zero from missing | `backend/src/alpha_harness/evaluation/metrics.py` | `test_formula_golden_vectors_and_zero_turnover_compatibility`, formula-boundary and missing/default tests; source review confirmed coefficients unchanged | PASS | Empirical calibration/ablation remains deferred; no coefficient retuning was authorized |
| Never silently recompute historical scores; unknown provenance stays unknown | `backend/src/alpha_harness/db/models.py`, additive SQLite migration, evaluation/API serialization | `test_sqlite_migration_preserves_historical_score_as_unknown`, `test_formula_specs_and_historical_scores_survive_database_reopen`; actual persisted legacy row reopened with scalar unchanged and spec `NULL` | PASS | Historical formula inventory remains deferred |
| Persisted score exactly matches its stored formula specification, including defaults and missing values | `backend/src/alpha_harness/evaluation/metrics.py`, `backend/src/alpha_harness/application/research.py` | `test_persisted_spec_recomputes_exactly`, `test_formula_specs_and_historical_scores_survive_database_reopen`, EvaluationEngine persisted-provenance test | PASS | None identified |
| Invalid negative and non-finite metric inputs cannot become plausible persisted scores | `backend/src/alpha_harness/vault/metrics.py`, `evaluation/metrics.py` | `test_formula_boundaries_reject_non_finite_values`, `test_negative_turnover_is_rejected`, `test_evaluation_input_series_reject_non_finite_and_negative_turnover` | PASS | None identified |
| API response compatibility and historic records without score metadata | `backend/src/alpha_harness/api/research.py` | `test_api_keeps_existing_score_field_and_exposes_formula_provenance`, `test_persisted_api_leaderboard_and_memory_quarantine_invalid_metrics`; exercised actual candidate, evaluation, and leaderboard endpoint functions; JSON strict serialization checked | PASS | None identified |
| Invalid historical metrics/scores are excluded from ranking and memory consumption; missing scores remain explicit | `backend/src/alpha_harness/leaderboard/service.py`, `learning/memory.py` | `test_persisted_api_leaderboard_and_memory_quarantine_invalid_metrics` uses persisted tied, missing, unknown-provenance, infinite-metric, and infinite-score rows and asserts downstream order/propagation | PASS | None identified |
| Deterministic tie-breaking preserves consumer ordering otherwise | leaderboard and memory ordering code | Persisted equal-score candidate rows in `test_persisted_api_leaderboard_and_memory_quarantine_invalid_metrics`; stable pool tie test `test_portfolio_ties_are_candidate_id_stable_and_non_finite_scores_rejected` | PASS | None identified |
| Portfolio/pool rejects invalid score and metric inputs | `backend/src/alpha_harness/portfolio/pool.py` | `test_portfolio_rejects_invalid_scores_and_metric_inputs` parameterizes non-finite composite/sharpe/PnL/turnover and negative turnover; separate stable-tie test | PASS | None identified |
| Holdout cannot affect score, formula provenance, API, ranking, memory, or pool selection | validation score path plus affected consumers above | `test_holdout_mutation_does_not_change_p2_06_consumers` mutates holdout values in the persisted rows and compares API/leaderboard/memory/pool snapshots; direct score/spec comparison also included | PASS | None identified |
| Dependency-matched complete backend verification on final tree | Python 3.14.7; lock-matched NumPy 2.5.3, SQLAlchemy 2.0.54, Ruff 0.16.8; pytest 8.4.2 added only as a verification tool to disposable `.venv-p2-06-verify` | Focused 23 passed; affected P0/P1/P2 regression set 127 passed, 17 warnings; full backend 384 passed, 53 warnings, 37.24s, exit 0. Full run command: `python -m pytest --tb=short -q --basetemp <workspace-temp-dir>` | PASS | Existing async DB teardown/thread warnings remain tracked in `DEFERRED-TEST-01` |
| Frontend repository-wide formatting/type check, tests, and production build | `frontend/src/screens/research/index.tsx` formula label | Clean LF disposable checkout with the exact frontend source diff: `pnpm check` exit 0 (Biome checked 132 files; `tsc -b` passed); `pnpm test` 9 passed, 0 failed/skipped; `pnpm build` exit 0 | PASS | Windows working-tree LF/CRLF notices remain environmental; project formatting config was not changed |
| Baseline-aware Ruff and final whitespace check | Changed backend source lines | Ruff 0.16.8: 0 diagnostics on added/changed source lines across 8 modified files; 278 whole-file diagnostics exist, not on changed lines. `git diff --check` exit 0 (Git emitted only LF/CRLF notices) | PASS | Existing whole-file diagnostics are not newly introduced |

The full backend suite was attempted once after a lint-only line-wrap correction and hit `OSError: [Errno 28] No space left on device` while pytest wrote temporary/cache data (exit 1). Only the exact temporary directories created for these runs were removed; the disposable environment and repository were preserved. The final retry on the same snapshot completed with 384 passed, 53 warnings, exit 0. This infrastructure retry failure is not hidden or counted as a passing run.

### Review conclusion and checkpoint status

Independent source/test review found and corrected two Ruff E501 findings on changed lines; no behavioral defect remains identified within P2-06. The behavioral tests exercise persistence reopen/migration, actual API endpoint payloads, downstream leaderboard and memory consumption, portfolio selection, and holdout mutation rather than relying only on formula helpers. Existing formula coefficients remain untouched. Deferred calibration, ablation, sensitivity analysis, historical-version inventory, and cross-cohort robustness remain open; `DEFERRED-TEST-01` and all other prior entries are preserved.

**Verification decision:** P2-06 is **VERIFIED COMPLETE** on the exact manifest above. The implementation checkpoint `d6a18ced8dd9497b365d024b30686197ce803cdb` and reporting checkpoint `d1b417fdf288dd4e568c898cbb9bce2900e1ba7e` were pushed with a normal fast-forward update; immediately afterward local `HEAD` and `git ls-remote origin refs/heads/main` both returned `d1b417fdf288dd4e568c898cbb9bce2900e1ba7e`. The staged-file list contained only 17 intended P2-06 implementation, test, and tracking files; the pre-existing untracked `p2_07_audit_report.md` remains excluded. P2-07 implementation remains gated on this verified checkpoint and was not started.
