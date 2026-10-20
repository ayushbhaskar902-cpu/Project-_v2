# P1-04 Verification Report — Authoritative Dataset Registry & Autonomous Search Policy

## Status: ✅ VERIFIED FIXED

**Issue**: P1-04 — Dataset selection is not a true part of the autonomous search policy.

**Prior Verification Confirmation (P1-03)**:
- Targeted Suite: `21 passed, 0 failed` in `test_p1_03_prompt_registry.py` (Command exited with code 0).
- Full Regression Suite at P1-03: `107 passed, 0 failed` (Command exited with code 0).

**Current Verification (P1-04)**:
- Targeted Suite: `15 passed, 0 failed` in [test_p1_04_dataset_selection.py](file:///c:/Users/LENOVO/Desktop/alpha-harness/backend/tests/test_p1_04_dataset_selection.py) (Command exited with code 0 in 1.60s).
- Full Regression Suite: `122 passed, 0 failed` across all tests in `backend/tests/` (Command exited with code 0 in 31.47s).

---

## 1. Files Changed
1. [discovery/catalog.py](backend/src/alpha_harness/discovery/catalog.py):
   - Implemented `DatasetRegistry` with deterministic catalog fingerprinting (`cat_v1_<sha256>`).
   - Defined `DatasetRegistryError`, `DatasetNotFoundError`, and `DatasetEligibilityError`.
   - Defined `DatasetAllocationPolicy` (`EXPLICIT`, `CONTROLLED_EXPLORATION`, `CATEGORY_ALIGNED`) and `DatasetSelectionContext`.
   - Enforced strict regional and delay eligibility checks failing closed.
   - Implemented deterministic controlled exploration rotation without mutable in-memory counters.
   - Documented explicit category-aligned mappings with safe fallbacks for unmapped categories.
   - Enhanced `DiscoveredDataset` with default simulation and regional scope attributes.
   - Wired `DatasetRegistry` into `CatalogDiscovery`.
2. [generator/pipeline.py](backend/src/alpha_harness/generator/pipeline.py):
   - Added `target_dataset_id`, `catalog_version`, `dataset_selection_policy`, and `dataset_selection_reason` to `GenerationBatch`.
   - Connected `target_dataset_id` through `generate_batch`: validates eligibility against `DatasetRegistry` *before* invoking LLM generation (pre-LLM fail-closed safety).
   - Injected targeted dataset metadata and candidate fields directly into rendered prompt context.
   - Populated `GeneratedCandidate.target_dataset` from the selection context while distinguishing it from actual expression `fields_used`.
3. [db/models.py](backend/src/alpha_harness/db/models.py):
   - Added `target_dataset_id` and `dataset_allocation_policy` to `ResearchResearcher`.
   - Added `target_dataset_id`, `catalog_version`, `dataset_selection_policy`, and `dataset_selection_reason` to `GenerationAttempt`.
   - Added `target_dataset_id`, `catalog_version`, `dataset_selection_policy`, and `dataset_selection_reason` to `ResearchCandidate`.
4. [db/sqlite.py](backend/src/alpha_harness/db/sqlite.py):
   - Added non-destructive, idempotent migration backfilling `dataset_selection_policy = 'LEGACY_UNSPECIFIED'` for existing records without fabricating historical data.
5. [orchestrator/researchers.py](backend/src/alpha_harness/orchestrator/researchers.py):
   - Added `dataset_allocation_policy` to `ResearcherConfig`.
   - In `initialize_session_researchers` and `add_researcher`: stored `target_dataset_id` and `dataset_allocation_policy` on researcher records.
   - In `run_generation_round`: allocated dataset selection context per researcher via `DatasetRegistry` and propagated through `_dispatch_researcher`.
   - Persisted complete dataset lineage and catalog versions to `GenerationAttempt` and `ResearchCandidate`.
6. [tests/test_p1_04_dataset_selection.py](file:///c:/Users/LENOVO/Desktop/alpha-harness/backend/tests/test_p1_04_dataset_selection.py):
   - 15 comprehensive regression tests validating all acceptance criteria.

---

## 2. Dataset Registry & Eligibility Behavior
- **Authoritative Catalog Version**: Deterministic SHA-256 fingerprint generated from canonical sorted JSON representation of all datasets and fields. Two separate instances with identical catalog definitions generate the exact same catalog version string.
- **Fail-Closed Validation**:
  - Unknown dataset IDs raise `DatasetNotFoundError`.
  - Region-incompatible datasets (e.g. requesting `fundamental_eu` for `USA`, or `fundamental_us` for `EUR`) raise `DatasetEligibilityError`.
  - Pre-LLM enforcement: Any invalid target request fails immediately before prompt generation, LLM token consumption, or simulation queuing.

---

## 3. Allocation Policies & Reproducibility Guarantees
- **`EXPLICIT`**: Validates requested `target_dataset_id` against regional constraints and assigns it directly with reason `"Explicitly requested target dataset '...'"`
- **`CONTROLLED_EXPLORATION`**: Eliminates process-randomized and volatile in-memory counters. Uses deterministic hashing over `(session_id, generation, researcher_id/index)`:
  $$\text{slot} = \text{int}(\text{SHA256}(s\_id : gen : r\_id)) \pmod{N}$$
  Guarantees:
  1. Identical inputs produce the identical dataset allocation across process restarts.
  2. Distinct concurrent researchers in a round explore diverse eligible datasets without starvation.
- **`CATEGORY_ALIGNED`**:
  - Uses documented mappings:
    - `value` $\to$ `Fundamental`
    - `quality` $\to$ `Fundamental`
    - `momentum` $\to$ `Model Data` / `Price Volume`
    - `volatility` $\to$ `Model Data` / `Price Volume`
    - `sentiment` $\to$ `Sentiment` / `Flow & Sentiment`
    - `analyst` $\to$ `Analyst`
    - `short_interest` $\to$ `Short Interest`
  - Unmapped categories (`general` or custom) document the limitation and fall back safely to `CONTROLLED_EXPLORATION` with explicit audit rationale.

---

## 4. Pipeline Integration & Distinction from Datasets Used
- Trace from Researcher $\to$ Dispatch $\to$ `generate_batch` $\to$ Prompt:
  1. Researcher persona defines `target_dataset_id` or allocation policy.
  2. `DatasetRegistry.allocate_dataset` validates eligibility and produces `DatasetSelectionContext`.
  3. Context is passed into `AlphaGenerationPipeline.generate_batch`.
  4. Prompt renderer receives targeted dataset and candidate datafield definitions in `catalog_context`.
  5. Candidates retain `c.target_dataset` as the selected target dataset, while `c.validation.fields_used` and `cand_row.datasets_used` track what the alpha expression actually referenced.

---

## 5. Actual Test Execution Output

### P1-04 Targeted Suite (15/15 passed)
```
$env:PYTHONPATH="backend/src"; uv run pytest backend/tests/test_p1_04_dataset_selection.py -v
============================= test session starts =============================
platform win32 -- Python 3.13.9, pytest-8.4.2, pluggy-1.5.0
backend\tests\test_p1_04_dataset_selection.py::test_deterministic_catalog_versioning PASSED [  6%]
backend\tests\test_p1_04_dataset_selection.py::test_registry_metadata_and_version_stability PASSED [ 13%]
backend\tests\test_p1_04_dataset_selection.py::test_successful_eligibility_validation PASSED [ 20%]
backend\tests\test_p1_04_dataset_selection.py::test_rejection_of_unknown_dataset_ids PASSED [ 26%]
backend\tests\test_p1_04_dataset_selection.py::test_rejection_of_demonstrably_incompatible_dataset_requests PASSED [ 33%]
backend\tests\test_p1_04_dataset_selection.py::test_invalid_requests_fail_before_any_llm_call PASSED [ 40%]
backend\tests\test_p1_04_dataset_selection.py::test_explicit_target_dataset_id_reaches_prompt_generation PASSED [ 46%]
backend\tests\test_p1_04_dataset_selection.py::test_controlled_exploration_reproducibility_across_restarts PASSED [ 53%]
backend\tests\test_p1_04_dataset_selection.py::test_controlled_exploration_distributes_across_researchers_no_starvation PASSED [ 60%]
backend\tests\test_p1_04_dataset_selection.py::test_concurrent_allocation_consistency PASSED [ 66%]
backend\tests\test_p1_04_dataset_selection.py::test_category_alignment_uses_documented_mappings_only PASSED [ 73%]
backend\tests\test_p1_04_dataset_selection.py::test_safe_fallback_when_category_unmapped PASSED [ 80%]
backend\tests\test_p1_04_dataset_selection.py::test_candidate_and_attempt_persist_dataset_lineage PASSED [ 86%]
backend\tests\test_p1_04_dataset_selection.py::test_candidate_lineage_distinguishes_target_from_datasets_used PASSED [ 93%]
backend\tests\test_p1_04_dataset_selection.py::test_sqlite_migration_idempotent_and_legacy_compatible PASSED [100%]
============================= 15 passed in 1.60s ==============================
```

### Full Backend Regression Suite (122/122 passed)
```
$env:PYTHONPATH="backend/src"; uv run pytest backend/tests/ -v
====================== 122 passed, 12 warnings in 31.47s ======================
```
All suites passed:
- `test_p0_01_provenance.py`: PASSED
- `test_p0_02_provenance.py`: PASSED
- `test_p0_03_splits.py`: PASSED
- `test_p0_04_robustness.py`: PASSED
- `test_p0_05_experiment.py`: PASSED
- `test_p1_01_reflection.py`: PASSED
- `test_p1_02_routing.py`: PASSED (14 tests)
- `test_p1_03_prompt_registry.py`: PASSED (21 tests)
- `test_p1_04_dataset_selection.py`: PASSED (15 tests)

---

## 6. Verdict
**✅ FULLY SATISFIED (P1-04)**
