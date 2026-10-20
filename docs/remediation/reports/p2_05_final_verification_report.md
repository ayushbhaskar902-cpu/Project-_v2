# P2-05 Final Verification Report — Reproducible Research Seeds

**Canonical issue:** Reproducibility is compromised by randomized synthetic seeds

**Status:** VERIFIED COMPLETE on the independent runner

**Source commit:** `0f9dc7f955eb832a34d1ac7a47da6a4163e2c463`

**Runner:** GitHub Actions `ubuntu-latest`, Python 3.14

**Run:** [39448158145](https://github.com/ayushbhaskar902-cpu/Project-_v2/actions/runs/39448158145)

## Verification outcome

The independent workflow diagnosed and exercised Python 3.14 successfully (`asyncio.run` and `socket.socketpair`), installed backend dependencies with `uv sync --frozen`, and completed every required regression and full-suite step. The aggregate job and both workflow jobs concluded successfully for the exact source commit above.

| Suite | Result | Failures / errors / skipped |
|---|---:|---:|
| P2-05 reproducibility | 9 passed | 0 / 0 / 0 |
| P2-04 backfill regressions | 9 passed | 0 / 0 / 0 |
| P2-01 evaluation jobs | 9 passed | 0 / 0 / 0 |
| P0 regressions | 60 passed | 0 / 0 / 0 |
| P1 regressions | 269 passed | 0 / 0 / 0 |
| P2-02 regression | 5 passed | 0 / 0 / 0 |
| P2-03 regression | 7 passed | 0 / 0 / 0 |
| Complete backend suite | 362 passed | 0 / 0 / 0 |

P0 reported 12 warnings, P1 reported 28, and the complete suite reported 40. The warnings are existing SQLite/`aiosqlite` asynchronous teardown diagnostics (`Event loop is closed`) tracked as `DEFERRED-TEST-01`; no test failed, errored, or was skipped. The targeted P2 suites reported no warnings in their published summaries.

Frontend component tests, `pnpm check`, and `pnpm build` all exited successfully. The workflow's public annotations record success and exit code 0 for checks/build; they do not publish a frontend test-case count. No frontend source changed in P2-05.

The local Windows Python/asyncio startup problem was not repaired by this run. It was bypassed for verification by the independent Ubuntu runner, whose Python 3.14 diagnostic passed. This is verified evidence for the code, not a claim that the local Windows runtime issue is resolved.

## Acceptance evidence

- Cross-process seed determinism: `test_generated_preview_is_stable_across_process_hash_seeds` compares the complete generated preview under different `PYTHONHASHSEED` values and a distinct experiment identity.
- Stable, distinct derivation and input validation: `test_seed_derivation_is_canonical_and_configuration_sensitive` and `test_seed_derivation_rejects_missing_identity_and_negative_seed`; implementation uses versioned SHA-256 over canonical JSON in `backend/src/alpha_harness/labs/seeding.py`.
- Persistence and restart: `test_add_study_persists_effective_seed_and_configuration` closes the original database engine, opens a new engine against the same SQLite file, then checks task identity, seed, and derivation metadata.
- Legacy provenance: `test_seed_metadata_records_unknown_legacy_as_unknown`; legacy Power Pool tasks without a recorded seed are explicitly refused rather than assigned an untraceable one.
- Local RNG coverage: `test_all_random_lab_draws_accept_repeatable_local_rngs` exercises preview, GA mutation, and Power Pool draws; settings batch order is separately covered by `test_settings_batch_shuffle_replays_and_new_seed_changes_order`.
- Power Pool retry payload: `test_power_pool_pending_response_survives_param_round_trip` confirms pending provider output and seed metadata survive serialization for replay. A provider response lost before its first durable write remains an external-call crash window without provider idempotency; this is documented, not claimed solved.
- Genuine-evidence boundary: source review confirms robustness completion still requires verified BRAIN evidence and synthetic/incomplete robustness rows remain excluded from production summaries. P0/P1 and the full backend regression suites passed. P2-05 did not alter holdout evaluation or the P2-04 lifecycle.

## Remaining limitations and deferred work

No P2-05 acceptance criterion remains unverified by the implementation review and executed backend tests. The provider-response-to-first-persistence crash window remains an explicit limitation. The 40 full-suite asynchronous teardown warnings remain open under `DEFERRED-TEST-01`; all deferred entries were preserved. P2-06 remains unaudited here and was not implemented as part of P2-05.

## Checkpoint

The verified implementation and workflow changes are on `wip/p2-05-reproducible-seeds` at the source commit above. This report and canonical status documents are updated after the successful run; the main-branch merge/checkpoint is recorded separately once completed.
