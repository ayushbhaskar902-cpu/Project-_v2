# P1-13 Audit & Remediation Plan: Prompt Evolution Validation & Rollback

**Issue ID:** P1-13  
**Canonical Title:** Prompt version persistence is not the same as prompt evolution validation  
**Status:** 🔍 AUDITED — PENDING REVIEW & APPROVAL  
**Affected Area:** `alpha_harness.prompts.library`, `alpha_harness.prompts.versioning`, `alpha_harness.learning.loop`, `alpha_harness.db.models`, `alpha_harness.orchestrator.researchers`  

---

## 1. Executive Summary & Problem Analysis

In P1-02 and P1-03, we implemented immutable prompt version snapshots (`PromptVersion`) and category-scoped prompt routing. However, the system currently suffers from a critical validation gap:
1. **Immediate Unvalidated Promotion:** When the self-learning reflection cycle (`SelfLearningEngine.execute_learning_cycle`) evolves a new prompt strategy, it automatically persists it and marks it active (`is_active=True`), immediately retiring the previous version.
2. **Persistence Conflated with Quality:** Creating an immutable checksummed version does **not** prove that the new prompt generates better alphas. LLM-generated reflection feedback can hallucinate ineffective guidelines or brittle constraints that degrade subsequent generation rounds.
3. **Absence of Controlled Shadow Benchmarking:** There is no mechanism to deploy an evolved prompt into an `EXPERIMENTAL` or `SHADOW` state where its candidates are benchmarked against the incumbent baseline on comparable allocations before deciding whether to promote.
4. **No Predeclared Promotion Gates:** Promotion is not gated on statistical criteria (e.g., non-inferiority or superior validation Sharpe, AST validity rate, diversity, token cost).
5. **No Deterministic Rollback Capability:** If an active prompt degrades alpha production, there is no formal, auditable rollback API to safely demote the failing prompt and restore the prior validated baseline.

---

## 2. Canonical Specification & Scope Boundaries

### Canonical Problem
> Evolving few-shot examples and anti-patterns does not prove a newer prompt performs better.

### Canonical Fix
> Immutable version IDs and checksums; benchmark new strategy against baseline using comparable allocations; promote only when it passes predeclared criteria on validation data and does not violate cost/reliability/diversity constraints. Preserve rollback.

### Canonical Acceptance Tests
> An untested prompt remains experimental; promotion requires recorded validation; rollback restores prior prompt behavior.

### Strict Scope Boundaries
- **P1-13 (In Scope):**
  - Prompt validation status lifecycle: `EXPERIMENTAL`, `SHADOW_TESTING`, `PROMOTED`, `REJECTED`, `ROLLED_BACK`.
  - Untested prompts start in `EXPERIMENTAL` status (`is_active=False` for production routing).
  - Controlled prompt benchmark tournament: allocates comparable candidate quotas to experimental candidate prompts alongside the incumbent baseline.
  - Predeclared promotion gate: requires minimum sample size, equal or superior validation Sharpe / discovery rate, AST validity rate $\ge 90\%$, and token cost within budget.
  - Deterministic rollback API (`rollback_prompt_version`): demotes degraded prompt and atomically reactivates the previous stable promoted version.
  - Strict holdout quarantine (P0-03): all benchmark scores and promotion criteria evaluate exclusively on validation partition data.
- **P1-14 (Out of Scope):** Upstream candidate diversity, syntactic expression clustering, correlation break quotas during generation.
- **P1-15 (Out of Scope):** Candidate simulation settings mutation and parameter grids.
- **P1-16 (Out of Scope):** BRAIN submission grade vs. internal simulation checks.

---

## 3. Current Architecture & Code Audit

### 3.1 `backend/src/alpha_harness/db/models.py`
- `PromptVersion`:
  - Contains immutable snapshot components: `template_text`, `guidance`, `examples`, `anti_patterns`, `content_hash`.
  - **Defect:** Lacks validation lifecycle status (`validation_status`), benchmark trial counters, and validation trial metrics (`benchmark_sample_count`, `benchmark_val_sharpe`, `benchmark_valid_rate`).
- `PromptStrategy`:
  - Currently contains `is_active` (boolean).
  - **Defect:** Any evolved strategy becomes `is_active = True` immediately upon creation in `learning/loop.py`.

### 3.2 `backend/src/alpha_harness/learning/loop.py`
- Lines 550–650 in `execute_learning_cycle`:
  - When LLM reflection produces an evolved prompt, `_persist_evolved_strategy` creates a new `PromptStrategy`, deactivates all other versions (`is_active = False`), and activates the new one (`is_active = True`).
  - **Defect:** Completely bypasses validation. An untested prompt is immediately promoted.

### 3.3 `backend/src/alpha_harness/orchestrator/researchers.py`
- In `_resolve_strategy_version`:
  - Resolves active strategy using 5-tier fallback precedence.
  - **Defect:** Resolves whatever is marked `is_active=True`, including untested experimental mutations.

---

## 4. Proposed Solution Design for P1-13

### 4.1 Data Model Enhancements
Update `PromptVersion` and `PromptStrategy` in `backend/src/alpha_harness/db/models.py`:
```python
class PromptValidationStatus(StrEnum):
    EXPERIMENTAL = "EXPERIMENTAL"        # Newly generated mutation, untested
    SHADOW_TESTING = "SHADOW_TESTING"    # Currently undergoing controlled benchmark trials
    PROMOTED = "PROMOTED"                # Passed predeclared validation criteria; active in production
    REJECTED = "REJECTED"                # Failed promotion criteria (lower Sharpe, invalid syntax, high cost)
    ROLLED_BACK = "ROLLED_BACK"          # Previously promoted, but subsequently demoted due to degradation
```

Add tracking fields to `PromptVersion`:
- `validation_status`: `Mapped[str]` (default `"EXPERIMENTAL"`)
- `benchmark_sample_count`: `Mapped[int]` (default `0`)
- `benchmark_val_sharpe`: `Mapped[float | None]` (median validation Sharpe in benchmark trials)
- `benchmark_valid_rate`: `Mapped[float | None]` (AST validity rate in benchmark trials)
- `promoted_at`: `Mapped[datetime | None]`
- `promoted_by`: `Mapped[str | None]`
- `promotion_criteria`: `Mapped[dict[str, Any] | None]` (predeclared thresholds and recorded comparison evidence)

### 4.2 Controlled Benchmark Tournament & Predeclared Promotion Gate
Create `alpha_harness.prompts.validation` service:
- `PromptValidator`:
  - `register_experimental_version(...)`: Creates version with `validation_status = 'EXPERIMENTAL'`, `is_active = False`.
  - `evaluate_prompt_tournament(...)`:
    * Evaluates candidates produced by the experimental prompt against the incumbent baseline on identical simulation cohorts and validation windows.
    * Evaluates:
      1. Quality: $\text{SR}_{\text{exp}} \ge \text{SR}_{\text{base}} - \epsilon$ and $LCB_{95} > 0$.
      2. Reliability: AST valid rate $\ge 85\%$.
      3. Cost: Average token consumption $\le 1.2 \times \text{baseline}$.
      4. Sample size: $N \ge N_{\text{min}}$ (predeclared minimum trials, e.g. 5).
  - `promote_prompt_version(...)`:
    * Verifies that predeclared criteria are met.
    * Atomically marks experimental version as `PROMOTED` and `is_active = True`.
    * Retires previous active version to `REPLACED` (or `ACTIVE = False`).
  - `rollback_prompt_version(...)`:
    * Demotes the current version to `ROLLED_BACK`.
    * Atomically reactivates the most recent prior version with `validation_status == 'PROMOTED'`.

### 4.3 Holdout Isolation Safeguard (P0-03)
- Benchmark tournaments and promotion decisions strictly evaluate `val_sharpe`, `val_fitness`, and validation AST validity.
- Holdout partition metrics (`holdout_sharpe`, `holdout_returns`) are explicitly quarantined and inaccessible to prompt benchmarking or promotion calculations.

---

## 5. Acceptance Criteria Checklist for P1-13

1. **Untested Prompts Remain Experimental:**
   - Newly evolved prompt strategies and versions start in `EXPERIMENTAL` status with `is_active = False`.
   - Production generation rounds resolve only `PROMOTED` active versions.
2. **Controlled Benchmark Allocation:**
   - Experimental prompts undergo controlled allocation trials alongside baselines.
3. **Predeclared Promotion Gates:**
   - Promotion requires recorded validation evidence meeting predeclared thresholds on validation data.
   - Prompts failing criteria are marked `REJECTED` and never promoted to production.
4. **Deterministic Rollback:**
   - Calling `rollback_prompt_version` immediately demotes the active prompt and restores the prior stable prompt version.
5. **Strict Holdout Quarantine:**
   - Changing final holdout data does not affect prompt benchmarking, promotion, or rollback.
6. **Backward Compatibility & Regression Suite:**
   - Full test suite passes without regressions.

---

## 6. Request for User Approval

Issue P1-12 is complete and verified with all 271 backend tests passing. Per the workflow instructions, this P1-13 audit is submitted for review and approval before any code or schema changes are initiated.
