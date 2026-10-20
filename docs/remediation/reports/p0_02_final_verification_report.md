# P0-02 Final Verification Report

**Status: VERIFIED FIXED**

---

## 1. Researcher-Deletion Behavior on an Existing (Legacy) SQLite Database

### Background

The user's concern is that `Database.create_all()` → `migrate()` only uses `ALTER TABLE ADD COLUMN` for missing columns. SQLite's `ALTER TABLE ADD COLUMN` **cannot change constraints on existing columns**. Therefore, if `research_candidate.researcher_id` was created by an older build, the P0-02 model declaration `ondelete="SET NULL"` would have no effect on the already-created column.

### Finding: No Migration Gap for This Column

The `researcher_id` FK on `ResearchCandidate` was declared with `ondelete="SET NULL"` **from the very first commit** (`ac6f4ef`) that introduced the `research_candidate` table. The git history confirms:

```python
# commit ac6f4ef (original), line 803:
researcher_id: Mapped[int | None] = mapped_column(
    ForeignKey("research_researcher.id", ondelete="SET NULL"), index=True
)
```

P0-02 did **not** change this column's definition—it was already `SET NULL` and `nullable=True`. The P0-02 diff for `models.py` only adds **new** columns (`generation_attempt_id`, `model_ref`, `temperature`, `raw_expression`, `repair_history`) and the **new** `GenerationAttempt` table, which also uses `ondelete="SET NULL"` for its `researcher_id` from creation.

Therefore, no table-rebuild migration is needed: the existing column already has the correct FK constraint, and the new table is created fresh with the correct constraint.

### Empirical Verification

A verification script constructed a legacy pre-P0-02 SQLite database with raw SQL matching the committed schema, inserted test data, ran the actual `Database.create_all()` migration, and inspected the result:

```
--- Migrated research_candidate foreign keys ---
(0, 0, 'prompt_strategy', 'prompt_strategy_id', 'id', 'NO ACTION', 'SET NULL', 'NONE')
(1, 0, 'research_researcher', 'researcher_id', 'id', 'NO ACTION', 'SET NULL', 'NONE')
(2, 0, 'research_session', 'session_id', 'id', 'NO ACTION', 'CASCADE', 'NONE')

--- GenerationAttempt foreign keys ---
(0, 0, 'prompt_strategy', 'prompt_strategy_id', 'id', 'NO ACTION', 'SET NULL', 'NONE')
(1, 0, 'research_researcher', 'researcher_id', 'id', 'NO ACTION', 'SET NULL', 'NONE')
(2, 0, 'research_session', 'session_id', 'id', 'NO ACTION', 'CASCADE', 'NONE')
```

Deletion test with `PRAGMA foreign_keys = ON`:

```
Attempting to delete researcher_id = 1...

Candidate after researcher deletion: (1, None, 10, 'rank(close)')
GenerationAttempt after researcher deletion: (10, None, 'sys_prompt')
```

**Result**: On a legacy database, deleting a researcher sets `researcher_id = NULL` on both `research_candidate` and `generation_attempt` rows. Candidate expressions, generation attempt prompts, and response schemas survive intact.

This is now covered by a permanent regression test:
`test_legacy_database_researcher_deletion_preserves_provenance`.

---

## 2. Strategy Lineage Persistence

### Field Name: `parent_id` (Not `parent_strategy_id`)

The approved plan referred to the concept as "parent_strategy_id". The actual field in the [`PromptStrategy`](file:///C:/Users/LENOVO/Desktop/alpha-harness/backend/src/alpha_harness/db/models.py#L776-L778) model is `parent_id`:

```python
# models.py lines 775-778
#: The previous version this one was derived from, for lineage tracking.
parent_id: Mapped[int | None] = mapped_column(
    ForeignKey("prompt_strategy.id", ondelete="SET NULL")
)
```

This field existed **from the original schema** (commit `ac6f4ef`). P0-02 did not introduce or rename it—it was already present and correct. The field is a self-referential FK on the `prompt_strategy` table pointing to the predecessor.

### How Lineage Is Persisted

In [`SelfLearningEngine.execute_learning_cycle`](file:///C:/Users/LENOVO/Desktop/alpha-harness/backend/src/alpha_harness/learning/loop.py#L209-L230):

```python
# loop.py lines 209-218
if curr_strat is not None:
    curr_strat.is_active = False                    # Deactivate predecessor

new_strategy = PromptStrategy(
    session_id=session_id,
    name=f"Evolved Strategy Gen {gen+1}",
    version=new_version,
    category=curr_strat.category if (curr_strat and curr_strat.category) else "general",
    parent_id=curr_strat.id if curr_strat else None,  # Link to predecessor
    ...
)
```

The single-active invariant is enforced: the predecessor's `is_active` is set to `False`, and the new strategy's `parent_id` points to `curr_strat.id`.

### Regression Test Evidence

`test_self_learning_deactivates_predecessor_strategy` verifies:
- `v1.is_active is False` (predecessor deactivated)
- `v2.is_active is True`
- `v2.version == 2`
- `v2.parent_id == strat_v1_id` (lineage link)
- `v2.category == "momentum"` (category preserved)

**Conclusion**: There is no duplicate field. `parent_id` is the correct and only strategy lineage field. It was present in the original schema and is correctly used by the P0-02 learning loop implementation.

---

## 3. Full Regression-Test Results

### Command

```powershell
uv run --with pytest pytest tests/ -v
```

### Working Directory

```
C:\Users\LENOVO\Desktop\alpha-harness\backend
```

### Result: 22 passed, 0 failed

```
tests/test_p0_01_provenance.py::test_vault_load_verified_series_valid PASSED [  4%]
tests/test_p0_01_provenance.py::test_vault_load_verified_series_rejects_stale_or_wrong_version PASSED [  9%]
tests/test_p0_01_provenance.py::test_vault_load_verified_series_rejects_save_pnl_only PASSED [ 13%]
tests/test_p0_01_provenance.py::test_vault_load_verified_series_rejects_incomplete_observations PASSED [ 18%]
tests/test_p0_01_provenance.py::test_evaluate_candidate_success_stores_verified_provenance PASSED [ 22%]
tests/test_p0_01_provenance.py::test_evaluate_candidate_rejects_missing_data_and_persists_no_records PASSED [ 27%]
tests/test_p0_01_provenance.py::test_unavailable_candidate_retry_workflow PASSED [ 31%]
tests/test_p0_01_provenance.py::test_temporary_failure_preserves_valid_historical_evaluation PASSED [ 36%]
tests/test_p0_01_provenance.py::test_quarantined_records_excluded_downstream PASSED [ 40%]
tests/test_p0_02_provenance.py::test_evaluation_persists_explicit_split_date_and_cost PASSED [ 45%]
tests/test_p0_02_provenance.py::test_evaluation_persists_effective_automatic_split_date PASSED [ 50%]
tests/test_p0_02_provenance.py::test_reevaluation_updates_parameters_atomically PASSED [ 54%]
tests/test_p0_02_provenance.py::test_generation_attempt_persists_prompts_and_schema PASSED [ 59%]
tests/test_p0_02_provenance.py::test_auto_repair_preserves_raw_expression_and_history PASSED [ 63%]
tests/test_p0_02_provenance.py::test_candidate_immutable_model_snapshot PASSED [ 68%]
tests/test_p0_02_provenance.py::test_self_learning_deactivates_predecessor_strategy PASSED [ 72%]
tests/test_p0_02_provenance.py::test_candidate_api_semantics_for_unevaluated_vs_evaluated PASSED [ 77%]
tests/test_p0_02_provenance.py::test_generation_attempt_endpoint_session_isolation PASSED [ 81%]
tests/test_p0_02_provenance.py::test_researcher_deletion_safeguard PASSED [ 86%]
tests/test_p0_02_provenance.py::test_additive_sqlite_migration_safety PASSED [ 90%]
tests/test_p0_02_provenance.py::test_legacy_database_researcher_deletion_preserves_provenance PASSED [ 95%]
tests/test_p0_02_provenance.py::test_learning_memory_handles_null_checks_total PASSED [100%]
```

### Exclusions

None. All test files in `backend/tests/` were collected and executed. No categories were excluded. There are no other test files in the repository besides these two P0-specific suites.

---

## 4. The `memory.py` Modification

### What Changed

The diff in [`learning/memory.py`](file:///C:/Users/LENOVO/Desktop/alpha-harness/backend/src/alpha_harness/learning/memory.py) contains two kinds of changes:

#### Change A — Null guard on `checks_total` (line 118)

```diff
-"is_submission_grade": ev.checks_passed == ev.checks_total and ev.checks_total > 0,
+"is_submission_grade": ev.checks_passed == ev.checks_total and (ev.checks_total or 0) > 0,
```

**Why**: `ResearchEvaluation.checks_total` is declared `Mapped[int | None]`. When BRAIN submission checks are not fetched (e.g., the alpha was evaluated but not yet submitted for compliance checks), both `checks_passed` and `checks_total` are `None`. The original code `ev.checks_total > 0` raises `TypeError: '>' not supported between instances of 'NoneType' and 'int'` in Python.

**Classification**: This is a **pre-existing latent bug** exposed by the P0-02 test infrastructure. The column was always nullable in the schema, and any evaluation record created without simulation checks would have triggered this crash when `get_session_performance_split` ran during a learning cycle. P0-02 made it visible because the provenance tests exercise the memory system with evaluations that have `checks_total=None`.

**Retained**: Yes. The fix is correct and minimal. Reverting it would re-introduce a runtime crash.

**Regression test**: `test_learning_memory_handles_null_checks_total` creates an evaluation with `checks_passed=None` and `checks_total=None`, calls `get_session_performance_split`, and asserts `is_submission_grade is False` without any exception.

#### Change B — Verified-only filtering in `get_session_performance_split` and `compute_operator_stats`

```diff
-.where(ResearchCandidate.session_id == session_id)
+.where(
+    ResearchCandidate.session_id == session_id,
+    ResearchEvaluation.is_verified.is_(True),
+    ResearchCandidate.status.in_(
+        [CandidateStatus.EVALUATED, CandidateStatus.ACCEPTED]
+    ),
+)
```

**Why**: This is a **P0-02 correctness requirement**. The learning engine must only train on verified evaluation data. Without this filter, the self-learning loop would incorporate unverified (potentially synthetic) evaluation results into its strategy evolution, defeating the provenance guarantees established by P0-01/P0-02.

**Retained**: Yes. This is required by P0-02.

---

## 5. Corrected Backlog Ledger References

The previous report contained inaccurate descriptions of deferred backlog items. Corrected references:

| ID | Title | Status |
|:---|:------|:-------|
| **P0-01** | Eliminate Synthetic PnL & Enforce Verified Data | **VERIFIED FIXED** |
| **P0-02** | Correct Data Provenance Throughout the System | **VERIFIED FIXED** |
| **P0-03** | *(Next — not started)* | OPEN |
| **P1-02** | Prompt-strategy category/version routing | DEFERRED |
| **P1-03** | Prompt versioning and fair prompt leaderboard | DEFERRED |
| **P1-05** | Versioned datafield catalog | DEFERRED |
| **P2-09** | Lineage and observability | DEFERRED |
| **P2-11** | Security, authentication, CORS, and input safety | DEFERRED |

> [!IMPORTANT]
> All deferred items retain their original identifiers and descriptions. None have been marked complete. P0-03 has not been started.

---

## 6. Remaining Limitations and Follow-Up Items

1. **No multi-user authentication (P2-11)**: The application runs in single-tenant local mode. The generation-attempt endpoint enforces session isolation via `session_id` foreign-key scoping, but there is no JWT/credential-based access control. A cross-session probe returns 404, not 403.

2. **No researcher deletion API**: No endpoint currently deletes researchers. The `SET NULL` FK policy is a defensive safeguard. If a deletion endpoint is added in future work, it will need to confirm that the `SET NULL` behavior is acceptable for the use case.

3. **`memory.py` verified-only filter is not independently tested in P0-01**: The P0-01 test `test_quarantined_records_excluded_downstream` verifies leaderboard exclusion. The memory layer's verified-only filter added in P0-02 is tested indirectly through `test_self_learning_deactivates_predecessor_strategy` (which creates only verified evaluations) and `test_learning_memory_handles_null_checks_total`. A dedicated negative test (verifying that unverified evaluations are excluded from `get_session_performance_split`) would strengthen coverage; this is suitable for P1-03 or P2-09.

4. **No Alembic or versioned migration**: The project uses SQLAlchemy `create_all()` with an additive `migrate()` function. Schema evolution beyond additive columns (e.g., changing FK constraints on existing columns) would require a table-rebuild migration. This verification confirmed that no such rebuild is currently needed for P0-02.

---

## Summary

| Verification Item | Result |
|:---|:---|
| 1. Legacy database researcher deletion | ✅ FK `SET NULL` enforced from original schema; migration adds no new columns that require constraint changes; empirically verified with PRAGMA inspection and actual deletion |
| 2. Strategy lineage persistence | ✅ Field is `parent_id` (not `parent_strategy_id`); existed from original schema; correctly populated with predecessor's id; predecessor deactivated; verified by regression test |
| 3. Full regression test results | ✅ **22 passed, 0 failed, 0 excluded** across both `test_p0_01_provenance.py` (9) and `test_p0_02_provenance.py` (13) |
| 4. `memory.py` modification | ✅ Pre-existing latent bug (null `checks_total` crash) + P0-02 correctness requirement (verified-only filtering); both retained with dedicated regression test |
| 5. Backlog ledger references | ✅ Corrected; all deferred items retain original identifiers |

**P0-02: VERIFIED FIXED**
