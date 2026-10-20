# P1-16 Canonical Audit Report: Distinguishing Internal Quality Thresholds from BRAIN Eligibility & Submission Status

**Issue ID:** P1-16  
**Canonical Title:** “Submission grade” is an internal heuristic, not proof of final submitability  
**Canonical Problem Statement:** Distinguish internal quality thresholds and statistical acceptance from actual WorldQuant BRAIN platform eligibility, submission checks, and permanent submission status.  
**Backlog Status:** 🔍 **AUDITED — PENDING APPROVAL** (Do not implement before formal user approval)  
**Affected Subsystems:** `alpha_harness.evaluation.metrics`, `alpha_harness.db.models`, `alpha_harness.leaderboard.service`, `alpha_harness.learning.memory`, `alpha_harness.api.research`, `alpha_harness.brain.endpoints`  

---

## 1. Current Definitions & Nomenclature in Active Codebase

Across the active codebase, five conceptually distinct stages of candidate evaluation are conflated into ambiguous boolean flags (`is_submission_grade`, `status == "ACCEPTED"`, `checks_passed == checks_total`):

| Concept | True Semantic Definition | Current Code Realization | Flaw / Conflation |
|---------|--------------------------|--------------------------|-------------------|
| **1. Internal Candidate Quality** | Whether an alpha candidate passes empirical performance thresholds on internal sample data (e.g., validation Sharpe $\ge 1.25$, validation fitness $\ge 1.0$, turnover $\in [0.01, 0.70]$, drawdown $\le 50\%$). | Labelled as `scorecard.is_submission_grade` ([`metrics.py:272, 386`](backend/src/alpha_harness/evaluation/metrics.py)). | Confuses internal sample heuristic pass with platform "submission grade". |
| **2. Statistical Acceptance / Selection** | Whether an alpha survives multiplicity-corrected statistical testing (e.g., White's Reality Check, Hansen's SPA, Bonferroni-corrected $\text{LCB} > 0$ under P1-10 / P1-13) and holdout evaluation (P0-03). | Transitions candidate to `CandidateStatus.ACCEPTED` purely if `scorecard.is_submission_grade == True` ([`metrics.py:600-602`](backend/src/alpha_harness/evaluation/metrics.py)). | Equates passing 5 arbitrary heuristics with statistical and lifecycle acceptance. |
| **3. Technical BRAIN Platform Eligibility** | Whether an alpha satisfies BRAIN platform pre-submission constraints via live platform simulation checks (`POST /alphas/{id}/check` returning `CheckResult.PASS` on sub-industry concentration, turnover limits, self-correlation $< 0.70$, etc.). | Completely absent from candidate lifecycle; `EvaluationRecord.checks` is documented as `#: BRAIN submission check results, if fetched` ([`models.py:1138`](backend/src/alpha_harness/db/models.py)) but actually stores the 5 internal Python heuristic booleans! | Overwrites and mimics platform checks with synthetic local checks. |
| **4. Submission Readiness / Portfolio Admissibility** | Whether an eligible alpha improves an existing submitted pool without violating pairwise correlation ceilings (pairwise Pearson $r < 0.50$ via `submission_planner.py` beam search or BRAIN Power Pool constraints). | Evaluated only in legacy Optuna tool (`tools/submission_planner.py`); completely disconnected from orchestrator `ResearchCandidate` workflow. | Researchers and leaderboard advertise candidates as "submission grade" without testing pairwise portfolio conflict. |
| **5. Actual BRAIN Submission Status** | Evidence of an authoritative, permanent platform submission event recorded by BRAIN API (`POST /alphas/{id}/submit` or user-verified manual record in `Submission` table). | Represented only in standalone `Submission` table ([`models.py:599`](backend/src/alpha_harness/db/models.py)), while `ResearchCandidate.status` has no `SUBMITTED` state. API research routes return `isSubmissionGrade` to frontends as a submission indicator. | UI users and automated LLMs assume an alpha marked "submission grade" is or can be immediately submitted without platform verification. |

---

## 2. Actual Code Paths Assigning, Persisting, Displaying, and Controlling Actions

### 2.1 Assignment in Metric Evaluation
In [`alpha_harness/evaluation/metrics.py`](backend/src/alpha_harness/evaluation/metrics.py):
- **Lines 261–272 (2-way split)** and **Lines 376–386 (3-way split)**:
  ```python
  checks["is_sharpe_positive"] = i_sharpe > 0.0
  checks["val_sharpe_gte_1_25"] = v_sharpe >= DEFAULT_MIN_SHARPE
  checks["val_fitness_gte_1_00"] = v_fitness >= DEFAULT_MIN_FITNESS
  checks["val_turnover_valid"] = DEFAULT_MIN_TURNOVER <= v_turnover <= DEFAULT_MAX_TURNOVER
  checks["val_drawdown_lte_50pct"] = v_drawdown <= DEFAULT_MAX_DRAWDOWN

  passed = sum(1 for v in checks.values() if v)
  scorecard.checks = checks
  scorecard.checks_passed = passed
  scorecard.checks_total = len(checks)
  scorecard.is_submission_grade = passed == len(checks)
  ```
- **Finding**: Five arbitrary internal thresholds are computed. If all 5 pass, `is_submission_grade` is set to `True`.

### 2.2 Persistence and State Mutation
In [`alpha_harness/evaluation/metrics.py:L574-614`](backend/src/alpha_harness/evaluation/metrics.py):
- **Lines 578–580**:
  ```python
  existing_eval.checks = scorecard.checks
  existing_eval.checks_passed = scorecard.checks_passed
  existing_eval.checks_total = scorecard.checks_total
  ```
- **Lines 599–603**:
  ```python
  new_status = (
      CandidateStatus.ACCEPTED
      if scorecard.is_submission_grade
      else (CandidateStatus.EVALUATED if (target_sharpe or 0) > 0 else CandidateStatus.REJECTED)
  )
  ```
- **Finding**:
  1. The internal heuristic dictionary is persisted into `EvaluationRecord.checks` (which `models.py:1138` documents as BRAIN submission check results).
  2. The candidate's lifecycle status is mutated to `CandidateStatus.ACCEPTED` purely based on `scorecard.is_submission_grade`.

### 2.3 Graduation Gate Control
In [`alpha_harness/evaluation/metrics.py:L624-645`](backend/src/alpha_harness/evaluation/metrics.py):
- **Lines 641–644**:
  ```python
  if cand.status != CandidateStatus.ACCEPTED:
      raise RuntimeError(f"Candidate {candidate_id} must be ACCEPTED for graduation (current: {cand.status})")
  ```
- **Finding**: Final holdout evaluation (graduation) is gated on `CandidateStatus.ACCEPTED`, which is currently coupled directly to the internal heuristic `is_submission_grade`.
- **Finding**: In [`metrics.py:433-443`](backend/src/alpha_harness/evaluation/metrics.py), `build_canonical_spec` uses hardcoded default values (`"truncation": 0.08`, `"pasteurization": "ON"`, `"nan_handling": "OFF"`) rather than reading from `candidate.effective_simulation_settings()`.

### 2.4 Episodic Memory Contamination
In [`alpha_harness/learning/memory.py:L476, L595`](backend/src/alpha_harness/learning/memory.py):
- ```python
  "is_submission_grade": ev.checks_passed == ev.checks_total and (ev.checks_total or 0) > 0,
  ```
- **Finding**: Researcher reflection prompts and episodic strategy updates are fed "submission grade" exemplars that have never been validated by BRAIN. LLMs learn to optimize for the 5 internal heuristic thresholds under the hallucinated assumption that they are creating platform-submittable alphas.

### 2.5 Leaderboard and Aggregation Skew
In [`alpha_harness/leaderboard/service.py`](backend/src/alpha_harness/leaderboard/service.py):
- **Line 437**:
  ```python
  is_submission_grade=bool(ev.checks_passed == ev.checks_total and (ev.checks_total or 0) > 0)
  ```
- **Line 718**:
  ```python
  submission_grade_count=fd["simulation_pass_count"]
  ```
- **Finding**: The leaderboard displays `is_submission_grade` based on internal checks. Furthermore, in researcher summary stats, `submission_grade_count` is conflated directly with raw `simulation_pass_count`!

### 2.6 API & UI Presentation
In [`alpha_harness/api/research.py`](backend/src/alpha_harness/api/research.py):
- **Lines 639, 809, 847, 932, 1186, 1211, 1237**:
  Exposes `isSubmissionGrade`, `submission_grade_only`, and `submissionGradeCount` directly to clients.
- **Finding**: A frontend user filtering by `submission_grade_only=True` receives alphas that may fail BRAIN sub-industry limits, fail self-correlation, or carry invalid settings, because no platform check was ever executed.

---

## 3. Conflation Analysis

The codebase suffers from four distinct conflations:

1. **Quality Threshold vs. Platform Eligibility**:
   - Internal metrics check only aggregate sample Sharpe, fitness, turnover, drawdown.
   - WorldQuant BRAIN evaluates ~19 platform checks (e.g., `CHECK_TURNOVER`, `CHECK_SHARPE`, `CHECK_FITNESS`, `CHECK_MARGIN`, `CHECK_SUB_INDUSTRY_CONCENTRATION`, `CHECK_WEIGHTED_CORRELATION`, `CHECK_SELF_CORRELATION`, `CHECK_LOW_SUBMISSION_FREQ`).
   - Conflation: Passing the 5 internal metrics is claimed to be "submission grade", masking all 14+ remaining platform constraints.

2. **Statistical Acceptance vs. Platform Submission**:
   - `CandidateStatus.ACCEPTED` signifies that an alpha has survived scientific discovery and validation.
   - It does *not* mean the user has submitted the alpha to BRAIN or that BRAIN has accepted it into a compensation pool.

3. **Intention vs. Recorded Execution Evidence**:
   - An alpha can be generated targeting a submittable hypothesis, but until BRAIN's `POST /alphas/{id}/check` or `POST /alphas/{id}/submit` succeeds, platform eligibility and submission are unproven.

4. **Internal Check Storage in Platform Check Column**:
   - `EvaluationRecord.checks` was designed for BRAIN `Check` objects (`list[Check]`), but is filled with `{ "sharpe_gte_1_25": True, ... }`.

---

## 4. Verification Gaps & False Labeling Risk

A candidate currently CAN be labelled as "submission grade" without evidence of satisfying requirements:

1. **Synthetic / Mock Runs**: An unverified synthetic run with fake PnL passing Sharpe $\ge 1.25$ is marked `is_submission_grade = True`.
2. **Untested Correlations**: A candidate with Sharpe 2.0 that has $0.95$ correlation with an existing production alpha is marked `is_submission_grade = True` and `CandidateStatus.ACCEPTED`, even though BRAIN strictly rejects pairwise correlations $> 0.50$ (or $0.70$).
3. **Unvalidated Settings**: If a candidate ran under fallback settings, it still receives `is_submission_grade = True` if its sample numbers pass.
4. **No BRAIN Token Required**: An environment completely disconnected from BRAIN API produces "submission grade" candidates.

---

## 5. Subsystem Interactions & Safeguard Boundaries

1. **P0-02 Provenance Linkage**:
   - `EvaluationRecord.is_verified` tracks whether PnL originated from genuine BRAIN vault series.
   - *Requirement*: An alpha MUST NOT receive verified BRAIN eligibility status if `is_verified` is False or provenance is missing.

2. **P0-03 Holdout Isolation Safeguard**:
   - `graduate_candidate()` reveals holdout data for one-way assessment.
   - *Requirement*: Eligibility checks and submission readiness MUST be determined strictly from development (In-Sample and Validation) and platform check endpoints. Holdout metrics MUST NEVER influence or retroactively grant eligibility.

3. **P1-10 Multiplicity-Corrected Selection**:
   - P1-10 selects statistical winners based on multiplicity-corrected confidence bounds ($\text{LCB} > 0$).
   - *Requirement*: Statistical acceptance is an internal quality gate; it must not be conflated with platform submission.

4. **P1-11 Fair Researcher Leaderboard**:
   - Leaderboard tracks model efficacy across matched simulation cohorts.
   - *Requirement*: Leaderboard must report internal quality pass rates separately from verified BRAIN eligibility.

5. **P1-15 Explicit Simulation Settings**:
   - P1-15 verified settings fingerprinting and preflight schema validation.
   - *Requirement*: An alpha with invalid or `LEGACY_UNVERIFIED` settings cannot claim verified platform eligibility. Furthermore, `build_canonical_spec` in `metrics.py` must use `candidate.effective_simulation_settings()` rather than hardcoded settings.

---

## 6. Minimal Remediation Architecture for P1-16

To fix P1-16 without breaking existing API contracts:

### 6.1 Disentangle Lifecycle & Eligibility States
Introduce explicit, distinct concepts on `ResearchCandidate` and `EvaluationRecord`:

1. **Internal Quality Gate (Heuristic Pass)**:
   - Rename concept internally to `is_internal_quality_pass` or `meets_internal_thresholds`.
   - Keep `scorecard.is_submission_grade` as a deprecated property mirroring `meets_internal_thresholds` for backward API compatibility, but document it explicitly as internal heuristic only.

2. **BRAIN Technical Eligibility (Platform Check)**:
   - Introduce `platform_eligibility_status: Mapped[str]` on `ResearchCandidate`:
     - `UNCHECKED`: No platform checks run.
     - `CHECK_PENDING`: Async `check_alpha` initiated.
     - `ELIGIBLE`: Real BRAIN `check_alpha` returned all `PASS` / `WARNING`.
     - `INELIGIBLE`: Real BRAIN `check_alpha` returned `FAIL`.
     - `EXEMPT_OFFLINE`: Offline/test environment where platform check is mocked/unavailable.
   - Introduce `brain_platform_checks: Mapped[dict[str, Any] | None]` on `EvaluationRecord` to hold genuine BRAIN `Check` objects separately from internal quality checks (`internal_checks`).

3. **Platform Submission Status**:
   - Clarify `CandidateStatus`: Add explicit `SUBMITTED` status if automated submission is enabled, or maintain linkage to `Submission` table with `is_submitted: bool` property verified by `Submission.alpha_id`.
   - A candidate is `SUBMITTED` ONLY when an authoritative submission record exists in the DB or from BRAIN API.

### 6.2 Fix `build_canonical_spec` in `metrics.py`
Replace the hardcoded `sim_settings` dict in [`metrics.py:433-443`](backend/src/alpha_harness/evaluation/metrics.py) with `candidate.effective_simulation_settings()`, ensuring canonical spec accurately fingerprints actual candidate settings.

### 6.3 API Contract Preservation
- In [`api/research.py`](backend/src/alpha_harness/api/research.py):
  - Retain `isSubmissionGrade` field in JSON output for backward compatibility, but ensure its docstring / OpenAPI schema explicitly describes it as `meetsInternalThresholds`.
  - Add explicit fields: `platformEligibility: str` (`UNCHECKED`, `ELIGIBLE`, `INELIGIBLE`) and `isSubmitted: bool`.
  - Update `leaderboard/service.py` to fix `submission_grade_count` so it reflects actual heuristic pass count rather than raw `simulation_pass_count`.

---

## 7. Explicit Acceptance Criteria & Required Regression Tests

When P1-16 is approved for implementation, the following acceptance criteria and tests must be satisfied:

1. **Internal Quality vs. BRAIN Eligibility Disentanglement**:
   - Test proving high internal Sharpe ($\ge 1.25$) and fitness ($\ge 1.0$) does NOT automatically assign `platform_eligibility_status = "ELIGIBLE"`.
   - Without live BRAIN check evidence, status remains `UNCHECKED`.

2. **Statistical Acceptance vs. Submission Separation**:
   - Test proving candidate transitioning to `ACCEPTED` or surviving P1-10 selection does not transition to `SUBMITTED`.
   - Submission requires recorded platform submission evidence.

3. **Provenance Integrity Boundary**:
   - Test proving an unverified candidate (`is_verified = False` or synthetic provenance) cannot be assigned `platform_eligibility_status = "ELIGIBLE"`.

4. **Simulation Settings Gate**:
   - Test proving a candidate with invalid or unverified simulation settings is rejected from platform eligibility and submission actions.
   - Test proving `build_canonical_spec` incorporates candidate's actual `effective_simulation_settings()`.

5. **Holdout Independence**:
   - Test proving holdout assessment results cannot alter or grant eligibility or submission status.

6. **Full Suite Non-Regression**:
   - All 325 existing backend tests continue to pass with zero regressions.

---

## 8. Unresolved Assumptions, Migration Concerns & Non-Blocking Debt

1. **Airgapped / Offline Test Fixtures**:
   - When running offline without active BRAIN credentials, `check_alpha` cannot poll the live API.
   - Solution: Explicit `EXEMPT_OFFLINE` / `TEST_FIXTURE` eligibility state rather than fabricating live platform passes.
2. **Historical Data Migration**:
   - Existing candidate rows have `checks` populated with internal metrics.
   - Migration policy: Mark existing rows as `platform_eligibility_status = "UNCHECKED"` with legacy notice; do not attempt to retroactively reclassify without re-running checks.
3. **Non-Blocking Technical Debt**:
   - Log `DEFERRED-SETT-01` (periodic schema refresh) and `DEFERRED-TEST-01` (thread cleanup warnings).
