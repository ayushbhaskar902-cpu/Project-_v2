# P1-07 Final Verification Report

> **Status: ✅ COMPLETE**  
> **Date:** 2026-10-13  
> **Targeted Tests:** 16 passed — `python -m pytest tests/test_p1_07_language_scope.py -v --tb=short` — exit code 0  
> **P1 Regression Suite:** 127 passed, 2 warnings — `python -m pytest tests/test_p1_*.py -v --tb=short` — exit code 0  
> **Full Backend Suite:** 187 passed, 22 warnings — `python -m pytest tests/ -v --tb=short` — exit code 0  

---

## 1. Summary of Changes

| Component | File | Changes Implemented |
|---|---|---|
| **Language Policy Definition** | [pipeline.py](backend/src/alpha_harness/generator/pipeline.py) | Defined `SUPPORTED_ALPHA_LANGUAGES = frozenset({"FASTEXPR"})`, `DEFAULT_ALPHA_LANGUAGE = "FASTEXPR"`, `UnsupportedAlphaLanguageError`, and `validate_alpha_language(language: str \| None) -> str`. |
| **Package Exports** | [generator/__init__.py](backend/src/alpha_harness/generator/__init__.py) | Exported `SUPPORTED_ALPHA_LANGUAGES`, `DEFAULT_ALPHA_LANGUAGE`, `UnsupportedAlphaLanguageError`, and `validate_alpha_language`. |
| **Validation Gating** | [pipeline.py](backend/src/alpha_harness/generator/pipeline.py) | `validate_expression()` checks `validate_alpha_language(language)` before any parsing or processing. Rejects non-FASTEXPR requests immediately. |
| **Generation Pipeline** | [pipeline.py](backend/src/alpha_harness/generator/pipeline.py) | `AlphaGenerationPipeline.generate_batch()` validates `language` parameter at invocation entry; rejects unsupported languages before making any LLM calls. Sets `language=norm_language` on `GenerationBatch`. |
| **Repair Loop** | [pipeline.py](backend/src/alpha_harness/generator/pipeline.py) | `_repair_expression()` propagates validated language to `validate_expression()`. |
| **Data Models & Documentation** | [models.py](backend/src/alpha_harness/db/models.py) | Updated `ResearchCandidate.expression` and `ResearchCandidate.language` docstrings to explicitly state that WorldQuant BRAIN Fast Expression (`FASTEXPR`) is the sole supported generation language, and Python alpha generation is out of scope. |
| **Orchestrator Enforcement** | [researchers.py](backend/src/alpha_harness/orchestrator/researchers.py) | `ResearchCandidate` persistence validates `language=validate_alpha_language(batch.language)`. `enqueue_candidates_for_simulation()` validates candidate language before queueing. |
| **Simulation Schemas** | [schemas.py](backend/src/alpha_harness/brain/schemas.py) | Added `@field_validator("language", mode="before")` on `SimulationSettings` calling `validate_alpha_language`. Rejects `language="PYTHON"` during schema validation. |
| **Engine Queue & Packer** | [slots.py](backend/src/alpha_harness/engine/slots.py), [packer.py](backend/src/alpha_harness/engine/packer.py) | `BatchEngine.enqueue()` verifies all simulation requests use supported language. `packer.key_of()` derives and normalizes batch key language. |
| **API Endpoints** | [research.py](backend/src/alpha_harness/api/research.py) | Added `language: str \| None = "FASTEXPR"` with validator to `CreateSessionIn`, `TriggerGenerationIn`, and `CandidateDetailOut`. Rejects unsupported languages at API boundary. |
| **Public Documentation** | [README.md](file:///c:/Users/LENOVO/Desktop/alpha-harness/README.md#L17-L22) | Explicitly documents WorldQuant BRAIN Fast Expression (`FASTEXPR`) as the sole supported alpha generation target. |
| **Regression Tests** | [test_p1_07_language_scope.py](file:///c:/Users/LENOVO/Desktop/alpha-harness/backend/tests/test_p1_07_language_scope.py) | Created 16 comprehensive tests verifying language normalization, rejection, pipeline gating, persistence, simulation safety, and backward compatibility. |

---

## 2. Test Execution Verification

### A. Targeted P1-07 Test Suite (16/16 Passed)
```bash
python -m pytest tests/test_p1_07_language_scope.py -v --tb=short
============================= 16 passed in 1.50s ==============================
```

- `test_fastexpr_is_accepted`: Verifies FASTEXPR is recognized in `SUPPORTED_ALPHA_LANGUAGES`.
- `test_case_insensitive_fastexpr_normalization`: Verifies lowercase/mixed-case inputs (`fastexpr`, `FastExpr`) normalize cleanly.
- `test_none_defaults_to_fastexpr`: Verifies `None` cleanly resolves to `FASTEXPR`.
- `test_python_and_lowercase_python_raise_unsupported_language_error`: Verifies `PYTHON`/`python` raise `UnsupportedAlphaLanguageError` stating Python is out of scope.
- `test_arbitrary_unsupported_languages_are_rejected`: Verifies `R`, `C++`, `JULIA`, `rust`, etc. are rejected.
- `test_generate_batch_fails_before_llm_call_when_unsupported_language`: Proves pipeline blocks unsupported languages with zero LLM API calls.
- `test_validate_expression_rejects_unsupported_languages_before_parsing`: Proves `validate_expression` rejects before AST parsing.
- `test_validate_expression_accepts_valid_fastexpr_language`: Proves FASTEXPR validation succeeds.
- `test_candidate_creation_enforces_fastexpr_language`: Proves orchestrator persists candidates with `language="FASTEXPR"`.
- `test_simulation_request_preserves_fastexpr_language`: Proves outbound requests preserve `FASTEXPR`.
- `test_simulation_settings_rejects_unsupported_language`: Proves Pydantic rejects `PYTHON` on `SimulationSettings`.
- `test_unsupported_language_never_reaches_simulation_dispatch`: Proves non-FASTEXPR candidates never reach `batch_engine.enqueue`.
- `test_existing_callers_omitting_language_remain_compatible`: Proves 100% backward compatibility for legacy callers.
- `test_p1_05_datafield_validation_still_works_under_language_policy`: Proves P1-05 datafield catalog validation is intact.
- `test_p1_06_operator_validation_still_works_under_language_policy`: Proves P1-06 operator metadata and contract validation is intact.
- `test_full_lineage_preservation_through_generation_batch`: Proves prompt-version, dataset, datafield, and operator lineage remain intact.

### B. All P1 Regression Tests (127/127 Passed)
```bash
python -m pytest tests/test_p1_01_reflection.py tests/test_p1_02_routing.py tests/test_p1_03_prompt_registry.py tests/test_p1_04_dataset_selection.py tests/test_p1_05_datafield_catalog.py tests/test_p1_06_operator_catalog.py tests/test_p1_07_language_scope.py -v --tb=short
====================== 127 passed, 2 warnings in 22.02s =======================
```

### C. Full Backend Suite (187/187 Passed)
```bash
python -m pytest tests/ -v --tb=short
====================== 187 passed, 22 warnings in 38.60s ======================
```
*(All 22 warnings are pre-existing `aiosqlite` event-loop teardown warnings occurring during test worker termination).*

---

## 3. Explicit Verdict
**P1-07 is FULLY SATISFIED (FIXED).**
