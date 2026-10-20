# Issue P1-09 Final Verification Report: Operator Attribution & Confounder Control

**Issue**: P1-09 — Operator attribution uses naive mean Sharpe / confounders  
**Date**: 2026-10-16  
**Status**: VERIFIED & COMPLETE  
**Repository Working Tree**: `c:\Users\LENOVO\Desktop\alpha-harness`  
**Test Suite**: 224 / 224 passed (100% pass rate across entire repository)  

---

## 1. Executive Summary

Prior to Issue P1-09, the learning system's operator attribution relied on fragile regular expressions (`re.findall(r"([a-z_][a-z0-9_]*)\s*\(")`), attributed 100% of candidate Sharpe uniformly to all parsed operators without complexity discounting, ignored data field quality baselines, and computed raw sample means with a loose $N \ge 2$ threshold. This allowed lucky operators appearing in only 1–2 high-variance alphas to dominate `preferred_operators`, distorting downstream prompt evolution and LLM reflection loops.

Under P1-09:
1. **AST-Based Operator Extraction**: Replaced regex with AST parsing (`fastexpr.extract_ast_operators`) validated against `OperatorCatalog` (P1-06), rejecting non-call identifiers, constants, and unknown operators, while deduplicating calls per candidate formula (1 sample per candidate).
2. **Controlled Marginal Attribution**:
   - Subtracted cohort baseline performance ($\text{Sharpe}_i - \overline{\text{Sharpe}}_{\text{cohort}}$).
   - Discounted multi-operator credit by formula complexity ($1 / \sqrt{M_i}$), penalizing bloated expressions.
   - Controlled for data field covariates by attenuating field-level excess Sharpe.
3. **Empirical Bayes / James-Stein Shrinkage**:
   - Shrunk estimated marginal lifts toward zero-prior: $\text{Shrunk Lift}_j = \frac{N_j}{N_j + k} \cdot \overline{\text{Lift}}_j$ ($k = 3.0$).
   - Single-trial anomalies are aggressively shrunk, preventing noise from outranking proven operators.
4. **Uncertainty Quantification & Small-Sample Gating**:
   - Computes standard error ($\text{SE}_j$), t-statistic, lower confidence bound ($\text{LCB}_j$), and upper confidence bound ($\text{UCB}_j$).
   - Mandated sample size gate ($N \ge 3$) for prompt strategy inclusion.
   - Categorizes evidence into `POSITIVE_ASSOCIATION`, `NEGATIVE_ASSOCIATION`, `INCONCLUSIVE`, and `INSUFFICIENT_EVIDENCE`.
   - Categorizes confidence into `HIGH` ($N \ge 10$), `MEDIUM` ($5 \le N < 10$), `LOW` ($3 \le N < 5$), and `INSUFFICIENT_SAMPLE` ($N < 3$).
5. **Downstream Evolution & Observational Disclaimers**:
   - Updated `SelfLearningEngine` (`loop.py`) to select `preferred_operators` ranked conservatively by Lower Confidence Bound (LCB) and `avoid_operators` by Upper Confidence Bound (UCB).
   - Added explicit observational association disclaimers in reflection prompt formatting (`prompts/library.py`), explicitly stating that historical in-sample correlations do not prove causation.
6. **Strict Holdout Isolation**: Queries only validation/in-sample Sharpe (`val_sharpe` / `full_sharpe`) and completely isolates final holdout performance.

---

## 2. Verification Evidence

### Targeted P1-09 Test Suite
All 11 tests in `backend/tests/test_p1_09_operator_attribution.py` passed cleanly:

| Test Name | Verification Focus | Result |
|---|---|---|
| `test_ast_operator_extraction_vs_regex` | AST parsing, operator extraction, and formula deduplication | **PASSED** |
| `test_ast_operator_extraction_rejects_unknown_operators` | OperatorCatalog validation and rejection of fake operators | **PASSED** |
| `test_ast_operator_extraction_handles_empty_and_malformed` | Empty, whitespace, and unparseable expression resilience | **PASSED** |
| `test_operator_attribution_deconfounding` | Complexity discounting ($1/\sqrt{M}$) and cohort baseline adjustment | **PASSED** |
| `test_empirical_bayes_shrinkage_penalizes_small_n` | $N=1$ lucky trials shrunk toward zero, outranked by $N=10$ | **PASSED** |
| `test_minimum_sample_gate_and_evidence_levels` | $N < 3$ produces `INSUFFICIENT_SAMPLE` and `INSUFFICIENT_EVIDENCE` | **PASSED** |
| `test_holdout_isolation_in_operator_stats` | Strict isolation of final holdout metrics (`val_sharpe` only) | **PASSED** |
| `test_co_occurrence_collinearity_detection` | Flags collinear pairs ($\ge 85\%$ co-occurrence, count $\ge 3$) | **PASSED** |
| `test_persist_and_retrieve_operator_stats` | Atomic persistence and retrieval of `OperatorEfficacyRecord` | **PASSED** |
| `test_learning_loop_selects_operators_by_confidence_bounds` | Conservative LCB ranking and $N \ge 3$ gating in `SelfLearningEngine` | **PASSED** |
| `test_prompt_library_observational_disclaimer` | Prompt evidence discloses observational non-causal associations | **PASSED** |

### Complete Repository Regression Suite
- **Command**: `uv run pytest backend/tests`
- **Total Tests Collected**: 224
- **Total Tests Passed**: **224**
- **Failures / Errors**: **0**
- **Execution Time**: 34.39s

---

## 3. Files Modified for P1-09

1. `backend/src/alpha_harness/learning/memory.py`:
   - Updated `OperatorEfficacy` with `is_statistically_significant` property.
   - Corrected marginal attribution calculation so complexity discounting ($1/\sqrt{M_i}$) is applied directly to candidate lift contributions without self-weight cancellation.
   - Refined `evidence_status` classification based on conservative confidence bounds (`lcb > 0.0` for positive association, `ucb < 0.0` for negative association).
2. `backend/src/alpha_harness/learning/loop.py`:
   - Updated `best_ops` ranking to use conservative Lower Confidence Bound (LCB).
   - Updated `worst_ops` ranking to use conservative Upper Confidence Bound (UCB).
   - Added explicit `operator_evidence_method="p1_09_v1"`.
3. `backend/src/alpha_harness/prompts/library.py`:
   - Added observational association disclaimer in `render_reflection_evidence_prompt`.
4. `backend/tests/test_p1_09_operator_attribution.py`:
   - Created full 11-test regression suite verifying P1-09 across AST parsing, shrinkage, confounder controls, persistence, holdout isolation, and learning loop integration.

---

## 4. Remaining Limitations & Non-Goals

1. **Non-Causal Observational Metric**: Operator lift remains an observational association based on in-sample validation runs. True causal attribution would require randomized controlled counterfactual expression ablation experiments.
2. **Additive Linear Attribution**: Complexity weighting uses an inverse square-root discount ($1/\sqrt{M_i}$). Higher-order non-linear operator interactions (e.g., synergistic pairs like `rank` + `ts_decay_linear`) are tracked via co-occurrence flags rather than full interaction term regressions.
3. **Multiplicity in Candidate Selection**: P1-09 controls attribution for operators given evaluated candidates, but candidate winner/loser selection itself remains based on point composite score thresholds (deferred to P1-10).
