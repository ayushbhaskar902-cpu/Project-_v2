# P1-14 Final Verification Report: Diversity Generation and Portfolio Curation

**Issue ID:** P1-14  
**Canonical Title:** Alpha diversity is too downstream  
**Canonical Description:** Low-correlation curation exists, but diversity is not adequately enforced during research allocation and candidate generation. Track expression structural similarity, hypothesis family, datafield family, dataset, region, universe, operator family and return-series correlation. Allocate quotas toward under-explored categories while retaining an exploration budget. Use real common-period PnL for correlation and document missing-data policy.  
**Acceptance Criteria:** A generation round produces auditable family diversity; correlation is computed only on aligned real series; near-duplicate candidates are identified and pruned; P1-10 statistical gates and P0-03 holdout quarantine are strictly preserved.  
**Status:** ✅ **VERIFIED COMPLETE**  
**Completed Date:** 2026-10-18  

---

## 1. Executive Summary & Audit Resolution

Previously, diversity was only addressed downstream at submission planning or during correlation collisions against already submitted alphas. During research generation, every syntactically valid expression was accepted directly into simulation queues, allowing LLMs to flood simulation capacity with redundant parameter-jitter sweeps (e.g. `ts_decay_linear(close, 10)` vs `12`). Furthermore, return correlation in `submission_planner.py` incorrectly treated missing/insufficient overlap (`NaN`) as legal correlation (`under = np.isnan(rho) | (rho < ceiling)`).

For **P1-14**, we designed, implemented, and verified:
1. **Deterministic Upstream Deduplication & Fingerprinting**:
   - `extract_candidate_fingerprint()` computes canonical rendered expression, SHA-256 hash (`expression_hash`), and abstracted AST skeleton (`ast_skeleton`).
   - Abstracted AST skeleton replaces numeric constants with `CONST` and data fields with `FIELD` while strictly preserving operator names, nesting, and arity structure.
2. **Conservative Parameter-Jitter Pruning**:
   - Near-duplicates are identified when two candidates share identical AST skeletons, identical data fields, identical operator sets, and matching hypothesis categories.
   - Expressions sharing identical AST skeletons that utilize **different data fields** (e.g., `close` vs. `volume`) or represent **different hypothesis categories** (e.g., `MOMENTUM` vs. `MEAN_REVERSION`) are strictly preserved and never discarded.
3. **Balanced Quota Allocation & Exploration Reserve**:
   - `DiversitySelector` balances simulation slot allocation across hypothesis categories, datafield families, and operator families.
   - Family quotas are treated as configurable targets rather than hard blockers; shortfalls are dynamically re-allocated to available eligible candidates without candidate fabrication or empty fallback.
   - Configurable exploration reserve (`exploration_reserve_ratio`, default 20%) preserves exploration budget for underrepresented or multi-field candidates.
4. **Strict Separation of Pre-Simulation Diversity and Post-Simulation Quality**:
   - Upstream diversity filtering operates before simulation dispatch.
   - Post-simulation winner classification strictly reuses P1-10's authoritative evidence requirements ($t \ge 1.96$, $\text{LCB}_{95} > 0$, FDR $q \le 0.15$). Diversity never promotes statistically unsupported candidates to winner status.
5. **Robust Return Correlation & Holdout Quarantine**:
   - In `submission_planner.py`, fixed `under = (~np.isnan(rho)) & (rho < ceiling)`. Missing observations or pairs with overlap $< \text{MIN\_OVERLAP}$ (120/250 days) are never inferred as low correlation.
   - Added `evaluate_pairwise_correlation` explicitly reporting `INSUFFICIENT_OVERLAP` and `ZERO_VARIANCE`.
   - Strict holdout quarantine ensures holdout dates ($\ge \text{holdout\_date}$) are excluded from correlation estimates and search matrices.

---

## 2. Test Verification Evidence

### 2.1 Targeted P1-14 Suite (`tests/test_p1_14_diversity.py`)
- **Command:** `uv run pytest tests/test_p1_14_diversity.py -v`
- **Result:** **13 passed, 0 failed in 3.66s** (Exit code `0`).
- **Tests Added & Verified:**
  1. `test_exact_normalized_duplicate_rejection`: Exact duplicate expressions with varying whitespace/formatting are pruned.
  2. `test_near_duplicate_parameter_jitter_pruning`: Parameter jitter sweeps with identical AST skeleton and fields are pruned to 1 representative.
  3. `test_distinct_fields_with_matching_skeleton_preserved`: Expressions with identical AST skeletons using distinct data fields are preserved.
  4. `test_distinct_hypothesis_categories_with_matching_skeleton_preserved`: Identical skeletons across distinct hypothesis categories are preserved.
  5. `test_deterministic_diversity_selection`: Allocation is 100% deterministic given identical inputs and seed.
  6. `test_family_quotas_do_not_cause_empty_results_on_shortfall`: Quota shortfalls in one family smoothly re-allocate to available families without candidate fabrication.
  7. `test_invalid_expressions_never_enter_simulation`: Expressions failing AST validation are rejected and never dispatched.
  8. `test_diversity_cannot_override_p1_10_statistical_evidence`: Sub-threshold candidates ($t < 1.96$, $\text{LCB}_{95} \le 0$) cannot bypass P1-10 significance gates.
  9. `test_missing_or_insufficient_return_overlap_does_not_infer_low_correlation`: Missing or insufficient overlap reports `INSUFFICIENT_OVERLAP` and is not inferred as low correlation.
  10. `test_holdout_quarantine_in_submission_planner`: Holdout returns are quarantined from correlation and search matrices.
  11. `test_session_wide_deduplication`: Duplicates of previously generated expressions in a session are pruned across generation rounds.
  12. `test_exploration_reserve_allocation`: Reserve ratio allocates slots to exploratory and multi-field candidates.
  13. `test_orchestrator_generation_round_deduplication`: End-to-end SQLite integration testing persistence of `expression_hash`, `ast_skeleton`, and `pruning_reason`.

### 2.2 Cross-Issue Regression Suite (P0-03, P1-01, P1-02, P1-10, P1-11, P1-12, P1-13, P1-14)
- **Command:** `uv run pytest tests/test_p0_03_splits.py tests/test_p1_01_reflection.py tests/test_p1_02_routing.py tests/test_p1_10_multiplicity_selection.py tests/test_p1_11_fair_model_leaderboard.py tests/test_p1_12_researcher_memory.py tests/test_p1_13_prompt_validation.py tests/test_p1_14_diversity.py -q`
- **Result:** **126 passed, 0 failed, 31 warnings in 25.42s** (Exit code `0`).

### 2.3 Full Backend Test Suite
- **Command:** `uv run pytest --tb=short -q`
- **Result:** **308 passed, 0 failed, 48 warnings in 43.34s** (Exit code `0`).

---

## 3. Files Modified & Added

- [`src/alpha_harness/labs/fastexpr.py`](backend/src/alpha_harness/labs/fastexpr.py): Added `CandidateFingerprint`, `canonicalize_expression`, `compute_expression_hash`, `abstract_ast_skeleton`, `canonicalize_skeleton`, `extract_candidate_fingerprint`, and `is_near_duplicate`.
- [`src/alpha_harness/orchestrator/diversity.py`](backend/src/alpha_harness/orchestrator/diversity.py): Created `DiversitySelector`, `DiversityConfig`, `CandidateItem`, `CandidateAuditRecord`, and `DiversityFunnelMetrics`.
- [`src/alpha_harness/orchestrator/researchers.py`](backend/src/alpha_harness/orchestrator/researchers.py): Integrated upstream deduplication and diversity selection into `run_generation_round`.
- [`src/alpha_harness/tools/submission_planner.py`](backend/src/alpha_harness/tools/submission_planner.py): Fixed `under` logic to reject `NaN` overlap as low correlation, added `evaluate_pairwise_correlation`, and quarantined holdout dates.
- [`src/alpha_harness/db/models.py`](backend/src/alpha_harness/db/models.py): Added `expression_hash`, `ast_skeleton`, and `pruning_reason` to `ResearchCandidate`.
- [`tests/test_p1_14_diversity.py`](file:///c:/Users/LENOVO/Desktop/alpha-harness/backend/tests/test_p1_14_diversity.py): Created comprehensive test suite with 13 tests.
