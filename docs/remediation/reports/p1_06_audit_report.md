# P1-06 Audit Report: Authoritative and Versioned Operator Signatures, Arity, and Metadata Validation

## 1. Backlog Specification & Context
- **Canonical Item:** `P1-06 — Operator metadata/selection needs authoritative validation`
- **Area:** `backend/src/alpha_harness/generator/pipeline.py`, `backend/src/alpha_harness/discovery/catalog.py`, `backend/src/alpha_harness/labs/fastexpr.py`, `backend/src/alpha_harness/brain/endpoints.py`, `backend/src/alpha_harness/db/models.py`.
- **Problem Statement:**
  A local, hardcoded operator table and prompt constraints can drift from currently supported WorldQuant BRAIN operators, arities, and signatures. Currently:
  1. `CORE_BRAIN_OPERATOR_DEFS` in `pipeline.py` is a static list of 31 string definitions.
  2. `STANDARD_OPERATORS` in `catalog.py` is a static tuple of signature strings and descriptions.
  3. `OperatorInfo` in `fastexpr.py` parses parameters and required/maximum counts, but does not provide versioning, scope/level constraints, or argument semantics (such as window integer requirements, grouping slots, and keyword options).
  4. There is no versioned `OperatorCatalog` abstraction comparable to `DatafieldCatalog`.
  5. Candidates and attempts do not record an `operator_catalog_version`.
  6. The system cannot synchronize or refresh operators from `GET /operators` (`BrainEndpoints.list_operators`) or cache them in `MetadataCache`.
- **Fix Requirements:**
  - Implement `OperatorCatalog` with deterministic SHA-256 fingerprinting (`op_v1_<sha256[:16]>`).
  - Support synchronization from BRAIN API (`GET /operators`) and `MetadataCache`, with a verified deterministic built-in fallback operator registry.
  - Parse and validate operator signatures, arity (positional required vs maximum/variadic), argument types, keyword parameters, and scope (`REGULAR` vs `COMBO`).
  - Distinguish unavailable/unknown operators from expression syntax errors.
  - Never invent operator signatures or accept non-existent operators.
  - Persist `operator_catalog_version` and used operators across `GenerationAttempt` and `ResearchCandidate`.
  - Provide targeted error feedback for invalid operators in prompt generation and expression repair.

---

## 2. Current Implementation Analysis

### 2.1. `generator/pipeline.py`
- `CORE_BRAIN_OPERATOR_DEFS`: A 31-item list of definitions (`ts_rank(x, d)`, `group_neutralize(x, group)`, etc.).
- `BUILTIN_OPERATOR_TABLE`: Statically created once at module import via `operator_table(CORE_BRAIN_OPERATOR_DEFS)`.
- In `validate_expression`:
  ```python
  table_to_use = op_table or (BUILTIN_OPERATOR_TABLE if check_operators else None)
  if table_to_use:
      names_to_check = effective_allowed if effective_allowed is not None else set()
      problems = validate(tree, table_to_use, names_to_check)
  ```
  If `op_table` is not passed, it uses the static `BUILTIN_OPERATOR_TABLE`.

### 2.2. `labs/fastexpr.py`
- `operator_table(operators, scope="REGULAR")`: Parses `definition` strings using `_SIGNATURE` regex and `_signature_params`.
- `OperatorInfo`: Records `name`, `category`, `required`, `maximum`, and `params`.
- `validate(tree, table, known_names)`: Checks:
  - `node.value not in table` -> `"{node.value} is not an operator on BRAIN."`
  - `count < info.required or (info.maximum is not None and count > info.maximum)` -> `"{node.value} does not take {count} inputs."`
- **Gaps in AST validation**:
  - Does not validate that window/days parameter `d` in `ts_*` is a positive numeric constant or integer (e.g. `ts_rank(close, -5)` or `ts_rank(close, volume)` is syntactically a call with 2 arguments, but semantically invalid on BRAIN).
  - Does not validate grouping arguments for `group_*` operators (`group_rank`, `group_neutralize` require grouping fields).
  - Does not distinguish unavailable/unsupported operators for the current scope (e.g., COMBO operators in regular alphas).

### 2.3. `discovery/catalog.py`
- `format_operator_summary`: Hardcoded strings from `STANDARD_OPERATORS`.
- Accepts `extra_operators: list[dict[str, Any]] | None`, but there is no structured registry or versioning for operators.

### 2.4. `db/models.py`
- `GenerationAttempt` and `ResearchCandidate` currently lack `operator_catalog_version` and `operators_used`.
- `MetadataCache`: Exists as a table with `key` and `body: JSON`, but is not integrated into operator loading or caching.

---

## 3. Detailed Remediation Proposal for P1-06

### A. Authoritative `OperatorCatalog` Abstraction
Create `OperatorCatalog` in `discovery/catalog.py` (or dedicated module):
1. **Model & Registry**:
   - Stores authoritative `OperatorInfo` mappings including:
     - `name`: str
     - `category`: str
     - `definition`: str
     - `required_inputs`: int
     - `max_inputs`: int | None
     - `params`: tuple[str, ...]
     - `scope`: tuple[str, ...] (`REGULAR`, `COMBO`, etc.)
     - `level`: str | None
     - `description`: str
   - Deterministic SHA-256 fingerprint: `op_v1_<sha256[:16]>` computed over sorted, canonicalized definitions, arities, and scopes.
2. **Authoritative Sync & Fallback**:
   - `sync_operators(client: BrainEndpoints | None = None, db: Database | None = None)`:
     Fetches from `GET /operators` when client is authenticated; caches to `MetadataCache(key="operators:REGULAR")`.
   - Deterministic built-in fallback catalog containing the comprehensive official BRAIN operator set (cross-sectional, time-series, arithmetic, group, vector, logical).
3. **Scope & Constraint Enforcement**:
   - Strict filter by scope (rejecting COMBO-only operators in REGULAR alphas).
   - Arity validation (under-arity and over-arity).
   - Parameter constraints:
     - First argument in `ts_*`: series/data expression.
     - Second argument in `ts_*` (window `d`): positive integer/numeric literal.
     - Group argument in `group_*`: valid grouping field.
     - Keyword argument validation (e.g. `scale(x, target=1)`).

### B. Prompt Integration
- Integrate `OperatorCatalog.format_prompt_summary()` into `format_catalog_context` and prompt rendering.
- Inject exact operator signatures and version fingerprint into the prompt so the LLM researcher generates valid signatures.

### C. Pre-Simulation Validation Gate & Diagnostic Messages
- In `validate_expression` and `generate_batch`:
  - When an unknown operator appears: Return explicit diagnostic: `"Unknown or unsupported operator: <op_name>"`.
  - When arity is incorrect: Return explicit diagnostic: `"Invalid operator arity: <op_name> requires <req> inputs (got <count>)"`.
  - When grouping or window constraints fail: Return explicit diagnostic: `"Invalid operator argument: <op_name> expects <param_spec>"`.
  - Fails closed before persistence and simulation queueing.

### D. Expression-Repair Integration
- In `pipeline._targeted_hint`:
  - Detect operator rejection reasons.
  - Suggest correct canonical signature and argument specifications (e.g. `ts_rank(x, d) requires a positive day window`).

### E. Provenance & Lineage Persistence
- Add `operator_catalog_version: Mapped[str | None]` to `GenerationAttempt` and `ResearchCandidate`.
- Add `operators_used: Mapped[list[str] | None]` to `ResearchCandidate`.
- Additive, idempotent SQLite migration with honest `NULL` legacy defaults.

---

## 4. Test Strategy for P1-06 (`test_p1_06_operator_catalog.py`)
1. Deterministic operator catalog version fingerprinting.
2. Distinct fingerprints for different operator sets.
3. Acceptance of valid core, time-series, and cross-sectional operators.
4. Rejection of unknown / hallucinated operators.
5. Rejection of under-arity calls.
6. Rejection of over-arity calls.
7. Rejection of COMBO-only operators in REGULAR alpha scope.
8. Validation of time-series window arguments (`d > 0`).
9. Validation of group arguments in `group_*` operators.
10. Keyword argument validation.
11. Pre-simulation rejection of invalid operator candidates.
12. Useful operator-repair hints and re-validation.
13. Deterministic built-in fallback behavior when BRAIN is offline.
14. Refresh / sync from `MetadataCache` / `GET /operators`.
15. Lineage persistence (`operator_catalog_version`, `operators_used`).
16. Idempotent SQLite migrations.
