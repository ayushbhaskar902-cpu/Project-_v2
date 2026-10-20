# P2-06 Audit Report — Composite Score Is Hard-Coded Without Calibrated Justification

**Canonical issue:** Composite score is hard-coded without calibrated justification.

**Canonical problem statement:** “A weighted formula ranks Sharpe, Fitness, decay and turnover using fixed weights. It may be a useful heuristic, but no calibration/ablation evidence is demonstrated.”

**Audit status:** AUDITED — APPROVED FOR IMPLEMENTATION. Implementation verification is recorded separately in `p2_06_final_verification_report.md`; the verified implementation and reporting checkpoints are present in the local `main` history at `b257246b1663ce7458b084d2cd85d73efb81742c`.

**Scope:** Source and backlog audit only. No P2-06 code, weights, schema, or runtime behavior were changed.

## Audit basis and limitation

P2-05 was integrated into `main` at `4e3fce62c3d57f0fb49d093203e177ae610f0d6d` before this audit. The referenced `docs/remediation/reports/p2_06_audit_report.md` was absent from the merged tree, so this is a fresh source-backed audit, not a comparison against or endorsement of that unavailable artifact. The canonical issue text was verified in `Project_v2_Remediation_Backlog_and_LLM_Prompt.md` and its status in `AI_HANDOFF.md`.

## Findings

### Confirmed from source

1. **There are two fixed formulas, selected by evaluation mode.** In `backend/src/alpha_harness/evaluation/metrics.py`, `compute_validation_composite_score()` computes `0.50 * validation_after_cost_sharpe + 0.35 * validation_fitness - 0.15 * min(validation_turnover, 1.0)`. It is used for three-way validation at `evaluate_pnl_series()` and assigned to both `val_composite_score` and the compatibility `composite_score` field. `compute_composite_score()` retains a separate legacy formula: `0.40 * after_cost_sharpe + 0.30 * fitness + 0.20 * (1 - clamped_decay) - 0.10 * min(turnover, 1.0)`.

2. **Missing-value defaults are implicit and differ by component/formula.** Both functions map missing Sharpe and Fitness to `0.0`. The validation formula maps only `None` turnover to `0.20`. The legacy formula uses `turnover_val or 0.20`, so a real `0.0` turnover is treated as missing and receives the `0.20` penalty. Given the legacy turnover coefficient, this lowers a zero-turnover score by `0.02` compared with a literal zero penalty. Both formulas cap turnover only at the upper bound of 1.0; neither clamps or rejects negative turnover, which would increase the score through the negative-turnover term if such invalid input reached the helper. These are confirmed code behaviors; production ranking impact is not established.

3. **Neither formula explicitly rejects non-finite inputs.** The score helpers perform arithmetic and `round()` without `math.isfinite` checks. `backend/src/alpha_harness/vault/metrics.py`'s `_number()` accepts Python floats without a finiteness check and `stats()` performs NumPy calculations without an explicit finite-input guard. Thus NaN or infinities can reach arithmetic from these paths in principle. Whether persisted production BRAIN records currently contain such values, or whether a downstream layer rejects them, was not established. Treat this as a confirmed validation gap and a downstream-impact hypothesis, not evidence of observed corrupt rows.

4. **Scores are persisted and consumed in ranking/selection paths.** `EvaluationEngine.evaluate_candidate()` stores `composite_score` and `val_composite_score` in `ResearchEvaluation`; the model has these numeric columns but no dedicated score-formula-version column. `leaderboard/service.py` orders by `coalesce(val_composite_score, composite_score)`. `learning/memory.py` carries the validation score into researcher records and sorts those records by it. `portfolio/pool.py` sorts candidate IDs by the supplied score before applying its correlation constraint. The research API exposes the composite and several score components.

5. **A general methodology version exists, but it does not version the formula itself.** `_canonical_spec()` in `evaluation/metrics.py` stores `methodology_version: "P0-03-V1"` alongside evaluation dates, scheme, settings, and candidate identity. The stored score has no separate formula identifier or weights snapshot, and that spec does not record either formula's coefficients/default policy. Historical rows therefore do not self-describe which exact formula produced the persisted score. The repository does not establish that historical rows are actually mixed across formula revisions; that is a compatibility risk to inventory, not a confirmed incident.

6. **Validation/holdout isolation is already guarded in the three-way path.** The validation formula receives only validation after-cost Sharpe, validation Fitness, and validation turnover. `test_holdout_mutation_invariance_in_development` asserts changing holdout PnL leaves the validation composite unchanged; `test_is_mutation_invariance_of_validation_score` similarly checks IS mutation; the leaderboard test checks rank ordering by validation score and masks holdout metrics before graduation. This is useful existing protection, but it does not cover future score-version metadata or every downstream consumer under arbitrary holdout mutation.

### Not established (hypotheses requiring evidence)

- No historical benchmark, calibration study, ablation, or objective-based comparison supporting the fixed coefficients was found in the inspected source/tests/backlog. This confirms absence of demonstrated evidence in the repository, not that the coefficients are statistically invalid.
- No evidence establishes that one coefficient dominates ranking, that the current formula worsens portfolio outcomes, or that changing weights would improve results.
- No production occurrence of NaN/Infinity persistence or zero-turnover ranking distortion was found.
- The historical extent of rows computed under each formula/version is unknown; `P0-03-V1` may identify an evaluation methodology generation but is not proof of the formula used for every persisted row.

## Impact and evidence

The confirmed risk is auditability and reproducibility: a scalar score directly affects leaderboard ordering, learning-memory ordering, and greedy portfolio-pool candidate priority, while the coefficient/default policy is embedded in implementation and is not stored with each row. The legacy zero-turnover fallback is a concrete score discrepancy; non-finite values have no explicit guard. The impact of either issue on real rankings remains unquantified. Scores are heuristics and do not replace internal threshold checks, statistical safeguards, verified BRAIN provenance, or the sealed holdout gate.

## Minimal proposed remediation (for approval only)

- Preserve both existing formulas and their historical semantics initially; do not retune weights by intuition.
- Give each formula an explicit stable identifier and persist that identifier plus the exact effective coefficients/default policy with each new evaluation. Mark historical rows whose formula cannot be proven as legacy/unknown; do not silently recompute or relabel them.
- Define explicit missing-value behavior per input and distinguish `None` from valid zero. Reject or quarantine non-finite inputs before score persistence and downstream ranking.
- Continue exposing component metrics and expose the score identifier with the scalar through the API/UI surfaces that show it.
- Before changing weights or making the score configurable, run a predeclared calibration/ablation study against declared portfolio objectives using development data only. Freeze the candidate version before any final holdout evaluation; never use holdout outcomes to tune coefficients.

## Proposed acceptance criteria and regression tests

1. **Numerical consistency:** Golden-vector tests cover both formulas, each component independently, coefficient signs, rounding, clamping, and boundary values; recomputing from persisted components plus the stored formula identifier reproduces the persisted scalar exactly within the defined rounding policy.
2. **Missing and zero inputs:** Tests specify behavior for each missing component and prove `turnover=0.0` is not confused with missing. The legacy compatibility policy must be explicit; any intentional correction uses a new formula version rather than silently changing old semantics. Define and test the valid turnover domain, including negative and above-one values.
3. **Non-finite inputs:** NaN, positive infinity, and negative infinity are rejected or deterministically quarantined before database writes, API serialization, ranking, memory, or pool selection. Include direct helper and full evaluation-path cases.
4. **Historical compatibility:** Tests load representative legacy and three-way rows, prove their version/unknown status is retained, ensure APIs and consumers select the correct historical score, and verify no implicit recalculation occurs during read or migration.
5. **Ranking behavior:** Integration tests cover ties, missing score, legacy-vs-validation fallback, zero turnover, and exact ordering in leaderboard, learning-memory ordering, and portfolio pool selection. Document rounding points so API display rounding cannot alter server ranking.
6. **Holdout isolation:** Mutating only holdout PnL/turnover must leave the validation score, its formula metadata, pre-graduation API output, leaderboard ordering, memory ordering, and portfolio candidate priority unchanged. Holdout remains unavailable for calibration/ablation; test the graduation boundary separately.
7. **Calibration evidence:** Before proposing new weights, document a preregistered objective, dataset/cohort split, comparison baselines, ablation plan, uncertainty, and limitations. Use no final holdout data for selection; report results as evidence, not proof of universal optimality.
8. **Weight configuration (if approved for implementation):** A validated configuration schema accepts only finite coefficients and enforces explicitly documented domain/normalization constraints; invalid values fail before evaluation. Do not impose a sum-to-one rule without an explicit score interpretation that requires it.

## Compatibility, dependencies, regressions, and deferred work

- **Compatibility/dependencies:** P2-06 touches the P0-03 three-way/legacy evaluation contract and persisted `ResearchEvaluation` rows; consumers include P1-10 selection safeguards, P1-11 leaderboard, P1-12 memory, P1-14 portfolio diversity/pool curation, API response models, and the frontend. It is independent of P2-05 seed derivation and must preserve P2-04 evaluation-job lifecycle and verified-vault provenance.
- **Potential regressions:** Changing weights/defaults can reorder candidates, alter researcher learning and curated pools, and cause historical before/after scores to be incomparable. Adding fields requires SQLite migration and API/client compatibility work. Strict finite validation may surface previously hidden dirty records. Correcting zero-turnover behavior changes legacy results; version it rather than silently mutating history.
- **Deferred technical debt:** Empirical calibration/ablation, cross-cohort robustness, sensitivity analysis, historical-version inventory, and non-finite data hygiene require separate evidence and may coordinate with `DEFERRED-STAT-01`, `DEFERRED-STAT-02`, `DEFERRED-BENCH-01`, and P2-10. No deferred item is closed by this audit.

## Decision requested

P2-06 was approved for implementation by the user on 2026-10-20. The implementation is now verified locally; final evidence and the source snapshot are in [`p2_06_final_verification_report.md`](p2_06_final_verification_report.md). This audit does not authorize weight recalibration. P2-07 remains gated on confirmation of the P2-06 checkpoint.
