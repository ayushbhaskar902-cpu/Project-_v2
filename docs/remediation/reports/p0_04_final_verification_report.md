# P0-04 Final Verification Report: Genuine Robustness Perturbation Simulations

**Remediation Item:** P0-04 — Robustness endpoint fabricates degraded results instead of re-simulating  
**Status:** **VERIFIED FIXED**  
**Environment:** Python 3.14.7 (target) / Python 3.13.9 (host fallback), Windows, SQLite, DuckDB (AlphaVault)  
**Total Regression Test Count:** 49 passed, 0 failed across P0-01, P0-02, P0-03, P0-04  

---

## 1. Executive Summary

Prior to P0-04, `run_robustness_suite` in [research.py](file:///C:/Users/LENOVO/Desktop/alpha-harness/backend/src/alpha_harness/api/research.py) fabricated degraded Sharpe and fitness numbers using `random.uniform()` without submitting real simulation requests or verifying performance series.

In P0-04, all synthetic degradation calculations have been completely eliminated from the production codebase. Robustness evaluation now routes strictly through genuine WorldQuant BRAIN perturbation simulations, backed by real simulation requests, immutable provenance records, AlphaVault verified series loading, and evaluation over the exact baseline validation window.

---

## 2. Files Changed and Rationale

| File | Nature of Changes | Rationale |
| :--- | :--- | :--- |
| [backend/src/alpha_harness/db/models.py](file:///C:/Users/LENOVO/Desktop/alpha-harness/backend/src/alpha_harness/db/models.py#L1042-L1095) | Additive schema extension on `ResearchRobustness` | Added `status`, `result_kind`, `is_synthetic`, `baseline_eval_id`, `brain_simulation_id`, `perturbed_alpha_id`, `perturbation_config`, `error_details`, `val_start_date`, `val_end_date`. |
| [backend/src/alpha_harness/db/sqlite.py](file:///C:/Users/LENOVO/Desktop/alpha-harness/backend/src/alpha_harness/db/sqlite.py#L173-L188) | Non-destructive legacy migration in `migrate()` | Automatically identifies historical records lacking genuine `brain_simulation_id` and marks them as `is_synthetic = 1`, `result_kind = 'LEGACY_SYNTHETIC'`. Preserves all historical metrics for audit. |
| [backend/src/alpha_harness/evaluation/robustness.py](file:///C:/Users/LENOVO/Desktop/alpha-harness/backend/src/alpha_harness/evaluation/robustness.py) | Full refactoring of `RobustnessEngine` | Implemented `validate_perturbation`, `execute_perturbation`, `run_candidate_suite`, and safe `record_test_result`. Enforces honest `NOT_RUN` when BRAIN execution is unavailable, `UNSUPPORTED` on invalid options, `FAILED` on sim errors, and `COMPLETED` on genuine series verification. Excludes synthetic records from genuine summary statistics. |
| [backend/src/alpha_harness/api/research.py](file:///C:/Users/LENOVO/Desktop/alpha-harness/backend/src/alpha_harness/api/research.py#L860-L925) | Removed `random.uniform` and connected to engine | Removed all random degradation calculations and replaced with `state.robustness.run_candidate_suite`. Returns full execution and provenance metadata in API responses. |
| [backend/src/alpha_harness/state.py](file:///C:/Users/LENOVO/Desktop/alpha-harness/backend/src/alpha_harness/state.py#L136-L144) | Service composition injection | Injected `vault=self.alphas`, `endpoints=self.endpoints`, `tracker=self.tracker`, `engine=self.engine`, and `auth=self.auth` into `RobustnessEngine`. |
| [backend/tests/test_p0_04_robustness.py](file:///C:/Users/LENOVO/Desktop/alpha-harness/backend/tests/test_p0_04_robustness.py) | New comprehensive regression suite | 11 automated regression tests covering all 12 verification criteria specified by the prompt. |

---

## 3. How Genuine Perturbation Simulations Are Dispatched and Retrieved

1. **Parameter Isolation:** `execute_perturbation()` takes the candidate's actual expression and baseline configuration (`region`, `universe`, `delay`, `decay`, `neutralization`, `truncation`, etc.) and varies **strictly** the single parameter designated by the perturbation plan.
2. **Deterministic Validation:** If the parameter is unsupported by BRAIN (e.g., unrecognized neutralization enum or out-of-range decay), `UNSUPPORTED` is recorded and returned without dispatching.
3. **Availability Gate:** If BRAIN credentials or endpoints are unavailable or unauthenticated, the engine records and returns `NOT_RUN` with an honest explanatory message. No metrics are fabricated.
4. **Real Simulation Dispatch:** When execution is available, a genuine `SimulationRequest` is constructed and submitted via `POST /simulations`. The platform-assigned simulation ID is extracted from the `Location` header and recorded durably.
5. **Execution Tracking & Terminal Resolution:** The engine polls the simulation through the platform API until completion. If the simulation errors or times out, it records `FAILED` along with the cluster error details.
6. **AlphaVault Verification:** Once completed, the resulting `perturbed_alpha_id` performance series is ingested and verified through `AlphaVault.load_verified_series()`. If the series lacks required observations or fails verification, the test records `FAILED`.

---

## 4. Parameter Variation Specifics

- **Neutralization:** Validated against `{"MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY"}`. Modifies only `settings.neutralization`; keeps universe, decay, delay, and expression intact.
- **Decay:** Validated as integer in `[1, 100]`. Modifies only `settings.decay`; keeps neutralization, universe, and other settings intact.
- **Universe:** Validated against `{"TOP3000", "TOP1000", "TOP500", "TOP200", "ILLIQUID"}`. Modifies only `settings.universe`; keeps other settings intact.
- **Exact Configuration Provenance:** The full before/after delta and resulting configuration dictionary are serialized into `perturbation_config` on `ResearchRobustness`.

---

## 5. Comparison of Baseline and Perturbed Results

- **Validation Window Alignment:** Perturbed metrics are evaluated strictly over the **same validation dates** used by the baseline evaluation (`is_end_date` to `val_end_date` in 3-way chronological partitioning). Observations outside this window (such as IS or final holdout) are strictly excluded from the perturbation Sharpe and fitness calculation.
- **Metric Delta:** `sharpe_delta = round(perturbed_sharpe - baseline_sharpe, 4)` and `fitness_delta = round(perturbed_fitness - baseline_fitness, 4)`.
- **Pass/Fail Criteria:**
  - Perturbed Sharpe must remain $\ge 0.80$.
  - Sharpe degradation must not exceed $40\%$ of baseline Sharpe.
  - Candidate passes only if both conditions are satisfied.

---

## 6. Legacy Synthetic Records Migration

- **Non-Destructive Backfill:** During `Database.create_all()` / `sqlite.migrate()`, the following DDL and data migration executes automatically:
  ```sql
  UPDATE research_robustness
  SET is_synthetic = 1, result_kind = 'LEGACY_SYNTHETIC'
  WHERE (brain_simulation_id IS NULL OR brain_simulation_id = '')
  AND (result_kind = 'GENUINE' OR result_kind IS NULL)
  ```
- **Historical Audit Preserved:** All existing columns (`perturbed_sharpe`, `sharpe_delta`, `message`, `tested_at`) remain intact for audit inspection.
- **Strict Production Exclusion:** In `get_candidate_robustness_summary()`, all records where `is_synthetic == True` or `result_kind == 'LEGACY_SYNTHETIC'` are completely excluded from `pass_rate`, `total_tests`, `all_passed`, and `tested` calculations. They cannot masquerade as genuine test evidence.

---

## 7. Test Commands and Pass/Fail Evidence

### Test Execution Commands:
1. P0-04 suite:
   ```powershell
   uv run --directory backend --with pytest python -m pytest tests/test_p0_04_robustness.py -v
   ```
   **Result:** **11 passed in 2.66s**
2. Full P0 Regression Suite (P0-01, P0-02, P0-03, P0-04):
   ```powershell
   uv run --directory backend --with pytest python -m pytest tests/test_p0_01_provenance.py tests/test_p0_02_provenance.py tests/test_p0_03_splits.py tests/test_p0_04_robustness.py -v
   ```
   **Result:** **49 passed in 8.72s (0 failures)**
   - P0-01: 9 passed
   - P0-02: 13 passed
   - P0-03: 16 passed
   - P0-04: 11 passed
3. Python 3.13 Host Runtime Check:
   ```powershell
   $env:PYTHONPATH="src"; python -m pytest tests/test_p0_04_robustness.py -v
   ```
   **Result:** **11 passed in 3.26s**

### Mock vs. Live BRAIN Execution:
- Automated deterministic regression tests utilize explicit, typed mock wrappers (`MockBrainEndpoints` and `AlphaVault` test fixtures) simulating cluster success, cluster failure, and validation rejections.
- Live BRAIN integration tests are strictly gated by authenticated session availability; when live credentials are not present, production code deterministically and honestly emits `NOT_RUN` rather than claiming fake successes or fabricating numbers.

---

## 8. Remediation Backlog Integrity

All 36 original issues from `Project_v2_Remediation_Backlog_and_LLM_Prompt.md` remain strictly tracked and preserved in canonical order:

- **P0-01**: In-Sample / Out-of-Sample evaluation split dates inverted *(VERIFIED FIXED)*
- **P0-02**: Candidate evaluation score calculated before full backtest completes *(VERIFIED FIXED)*
- **P0-03**: Final holdout period peeked during alpha generation *(VERIFIED FIXED)*
- **P0-04**: Robustness endpoint fabricates degraded results instead of re-simulating *(VERIFIED FIXED)*
- **P0-05**: Persistent autonomous end-to-end experiment API *(NEXT UP)*
- **P1-01 through P1-16**: Outstanding and tracked
- **P2-01 through P2-13**: Outstanding and tracked
- **P3-01 through P3-02**: Outstanding and tracked
