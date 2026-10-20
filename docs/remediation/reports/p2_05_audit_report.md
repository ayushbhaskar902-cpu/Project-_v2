# P2-05 Audit Report — Reproducibility Is Compromised by Randomized Synthetic Seeds

**Status:** VERIFIED COMPLETE — independent runner and source audit passed
**Canonical issue:** P2-05 — Reproducibility is compromised by randomized synthetic seeds.
**Implementation branch:** `wip/p2-05-reproducible-seeds`.

## Canonical issue

Output-affecting research randomness must be explicitly seeded and reproducible across fresh processes. Synthetic fixtures must remain explicitly identified and separated from genuine BRAIN evidence.

## Verified current behavior

1. The robustness route in `api/research.py` contained a stale docstring claiming synthetic degradation was generated. The actual `ResearchWorkflowService.robustness()` path calls `RobustnessEngine.run_candidate_suite()`, which calls `execute_perturbation()` and requires a real BRAIN simulation, result-recordset ingestion, and `AlphaVault.load_verified_series()` before producing a completed result. When BRAIN is unavailable or unauthenticated it records `NOT_RUN` with `is_synthetic=False` and an explicit prohibition message. The stale docstring is corrected in the implementation branch.
2. `evaluation/robustness.py:350-785` consistently records unsupported, failed, unavailable, and completed outcomes as non-synthetic; completed metrics are calculated from the verified perturbed vault series. `get_candidate_robustness_summary()` excludes synthetic and incomplete rows from production pass-rate and `all_passed` calculations. No path was found that feeds robustness metrics into ordinary `ResearchEvaluation`, researcher memory, leaderboard ranking, graduation, platform eligibility, or submission.
3. Before implementation, `labs/ga.py`, `labs/launch.py`, and `labs/power_pool.py` created unseeded `random.Random()` instances; `tools/settings_sampler.py` used module-global shuffles. These affect child generation, preview draws, Power Pool settings, and persisted Settings Sampler trial order. Their seeded local-RNG replacements are in the implementation branch.
4. No process-randomized built-in Python `hash(...)` seed was found in `backend/src/alpha_harness`. `vault/yields.py` calls `hash(checks)` inside DuckDB SQL; that is a database function, not Python's randomized `hash()` and not seed derivation. Existing seeded paths include `labs/template.py` and `orchestrator/diversity.py`. `brain/client.py` uses random jitter only for operational retry timing, not experiment selection or persisted output.

## P0-01/P0-04 integrity conclusion

The suspected integrity violation is **not substantiated in the inspected production paths**. The confirmed findings are a stale robustness-route docstring and unseeded lab-generation paths, not fabricated BRAIN evidence entering normal evaluation. Unsupported robustness scenarios are disclosed as `UNSUPPORTED` or `NOT_RUN`; legacy synthetic database rows are marked and excluded by the migration and summary filters. No P0 production-logic change was warranted; the branch corrects misleading documentation and adds seed controls.

## Minimal proposed remediation

- Preserve real-BRAIN robustness behavior and correct the stale route documentation without redesigning the workflow.
- Derive a stable seed using SHA-256 over versioned canonical JSON containing a unique task identity, canonical parameters, and optional user seed.
- Persist the effective seed and derivation/configuration metadata with each new Study; expose whether older rows have known provenance.
- Use seeded local RNGs for generators; stop legacy randomized tasks that lack a recorded seed, and persist pending Power Pool provider responses before trial materialization.
- Add cross-process/hash-seed, generator, persistence, legacy-metadata, retry-payload, and existing provenance-boundary regression tests.

## Acceptance tests

- Same experiment identity/configuration/seed derives the same seed across separate processes and `PYTHONHASHSEED` values.
- Distinct run identities and relevant configurations alter the derived seed.
- Effective seed and derivation version/configuration are persisted and exposed; legacy rows without matching metadata report reproducibility as unknown.
- Seeded preview, GA mutation, Power Pool settings selection, Settings Sampler ordering, and TPE setup use explicit local seed inputs.
- A persisted pending Power Pool response is reused after a processing failure rather than re-requested; the provider-response-to-first-persistence crash window remains an external-call limitation absent provider idempotency.
- Production evaluation cannot consume synthetic metrics or synthetic robustness rows.
- Existing holdout isolation, P0/P1 safeguards, and P2-01–P2-04 lifecycle behavior remain unchanged.

## Scope boundary

P2-05 is **VERIFIED COMPLETE** on source commit `0f9dc7f955eb832a34d1ac7a47da6a4163e2c463`; see [`p2_05_final_verification_report.md`](p2_05_final_verification_report.md) for the run URL, suite counts, warnings, and acceptance evidence. This audit does not authorize implementation of P2-06. P2-05 does not change the P2-04 backfill lifecycle, statistical methods, or holdout policy.
