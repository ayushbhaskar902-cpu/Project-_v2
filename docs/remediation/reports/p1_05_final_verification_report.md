# P1-05 Final Verification Report: Versioned Datafield Catalog Integration

## 1. Executive Summary
- **Status:** **FULLY SATISFIED**
- **Objective:** Integrate an authoritative, versioned datafield catalog into dataset-scoped prompt generation, candidate pre-simulation validation, `GenerationAttempt` / `ResearchCandidate` lineage, and expression-repair workflows. Fix the critical validation defect in `generator/pipeline.py` where referenced fields self-authorized.
- **Verification Result:**
  - Targeted P1-05 suite (`backend/tests/test_p1_05_datafield_catalog.py`): **20/20 PASSED** (Exit code 0).
  - Related P1 suites (`P1-02`, `P1-03`, `P1-04`, `P1-05`): **72/72 PASSED** (Exit code 0).
  - Complete backend suite (`backend/tests/`): **144/144 PASSED** (Exit code 0).

---

## 2. Files Modified & Created

| File | Changes Made |
| :--- | :--- |
| `backend/src/alpha_harness/generator/pipeline.py` | **Fixed field-validation defect**: Removed `names_to_check = (valid_names or set()) \| set(used_fields)`. Referenced fields are now strictly checked against `effective_allowed` from `DatafieldCatalog`. Fails closed with `"Unknown or ineligible data field: <field_name>"` or catalog error reasons. Integrated `df_catalog` and `allowed_fields` into `generate_batch`, `_repair_expression`, and `GenerationBatch`. |
| `backend/src/alpha_harness/discovery/catalog.py` | Implemented `DatafieldCatalog` with deterministic SHA-256 fingerprinting (`df_v1_<hash[:16]>`). Added `CORE_MARKET_FIELD_NAMES` and `STANDARD_GROUPING_FIELD_NAMES`. Added `get_datafield_catalog` to `DatasetRegistry` and `CatalogDiscovery`. Updated `allocate_dataset` to populate `datafield_catalog_version` and `allowed_field_names`. Updated `format_catalog_context` to use `DatafieldCatalog.format_prompt_context`. |
| `backend/src/alpha_harness/db/models.py` | Added `datafield_catalog_version` (`String(64)`) and `allowed_fields` (`JSON`) to `GenerationAttempt`. Added `datafield_catalog_version` (`String(64)`) and `field_families` (`JSON`) to `ResearchCandidate`. |
| `backend/src/alpha_harness/orchestrator/researchers.py` | Updated `run_generation_round` to persist `datafield_catalog_version` and `allowed_fields` on `GenerationAttempt`, and `datafield_catalog_version` and `field_families` on `ResearchCandidate`. |
| `backend/tests/test_p1_05_datafield_catalog.py` | Created 20 regression tests covering fingerprint determinism, sensitivity, scope resolution, acceptance/rejections, bug fix verification, prompt context, pre-sim gating, repair feedback & re-validation, lineage persistence, fallback behavior, fail-closed handling, and migration idempotency. |

---

## 3. Architecture & Functional Details

### A. Authoritative Field Validation Defect Fix
In `pipeline.py`, previous code had:
```python
names_to_check = (valid_names or set()) | set(used_fields)
```
This allowed arbitrary expressions referencing hallucinated fields to bypass validation. This was completely removed.
Now:
1. `data_fields(tree, grouping=True)` extracts all referenced fields from the candidate expression.
2. `effective_allowed` is derived independently from `allowed_field_names` or `datafield_catalog.get_allowed_field_names()`.
3. If any referenced field is not in `effective_allowed`, the validator fails closed with:
   - Specific catalog diagnostic reason (e.g. wrong dataset, region mismatch, delay mismatch), or
   - `"Unknown or ineligible data field: <field_name>"`.
4. Operator validation checks `names_to_check = effective_allowed`, preventing any expression identifiers from expanding allowed names.

### B. Authoritative Versioned Datafield Catalog (`DatafieldCatalog`)
- **Deterministic Fingerprinting (`df_v1_<sha256[:16]>`)**: Normalized JSON payload containing dataset ID, region, universe, delay, instrument type, sorted field definitions (category, coverage, description, type), core market fields, and grouping fields.
- **Allowed Field Resolution**:
  $$\text{Allowed} = \text{Core Market Fields} \cup \text{Grouping Fields} \cup \text{Eligible Dataset Fields for }(region, delay, universe)$$
- **Scope Compatibility**: Rejects fields belonging to other datasets, regions with mismatch (`EUR` vs `USA`), or higher delays (`delay 2` requested under `delay 1`).
- **Core Market & Grouping Policy**: Universal market fields (`close`, `open`, `high`, `low`, `volume`, `vwap`, `adv20`, `returns`, `cap`) and grouping fields (`sector`, `industry`, `subindustry`, `market`, etc.) are recognized and permitted without counting grouping fields as data fields against BRAIN's 3-data-field limit.

### C. Prompt Context Integration
- Prompt generation consumes `DatafieldCatalog.format_prompt_context()` which injects:
  1. Exact catalog version fingerprint.
  2. Core market fields with coverage.
  3. Grouping fields with usage instructions.
  4. Selected dataset fields with type, description, and coverage.
  5. Strict datafield constraints instructing models not to reference unlisted fields.

### D. Pre-Simulation Validation Gate
- In `generate_batch` and `MultiLLMOrchestrator.run_generation_round`, expressions are validated before candidate persistence and simulation dispatch.
- Invalid candidates receive `status = CandidateStatus.INVALID` and are strictly excluded from `valid_candidate_ids`, preventing any unverified or hallucinated expression from reaching the simulation API.

### E. Expression-Repair Loop
- The repair loop receives `_targeted_hint` detailing:
  - The exact failure reason (`"Unknown or ineligible data field: <field>"`).
  - A sample list of authorized alternative fields for that scope.
- Repaired expressions are re-validated against the exact same catalog version and allowed-field set (`validate_expression(..., allowed_field_names=allowed_fields, datafield_catalog=df_catalog, check_operators=True)`).

### F. Lineage Persistence & Migration Safety
- `GenerationAttempt`: Persists `datafield_catalog_version` and `allowed_fields` (JSON list).
- `ResearchCandidate`: Persists `datafield_catalog_version` and `field_families` (JSON mapping of `{field_name: category}`).
- SQLite migrations via `migrate()` are additive and idempotent. Legacy records preserve honest `NULL` defaults without fabricated catalog versions.

---

## 4. Verification & Test Execution Results

```powershell
$env:PYTHONPATH="backend/src"; uv run pytest backend/tests/test_p1_05_datafield_catalog.py -v --tb=short
# Result: 20 passed in 2.06s (Exit code 0)

$env:PYTHONPATH="backend/src"; uv run pytest backend/tests/test_p1_02_routing.py backend/tests/test_p1_03_prompt_registry.py backend/tests/test_p1_04_dataset_selection.py backend/tests/test_p1_05_datafield_catalog.py -v --tb=short
# Result: 72 passed in 9.05s (Exit code 0)

$env:PYTHONPATH="backend/src"; uv run pytest backend/tests/ -v --tb=short
# Result: 144 passed in 27.21s (Exit code 0)
```
