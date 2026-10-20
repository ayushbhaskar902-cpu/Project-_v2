# AI Handoff & Architecture Guide — Alpha Harness (Project-_v2)

**Repository:** `https://github.com/ayushbhaskar902-cpu/Project-_v2.git`  
**Target Branch:** `main`  
**Current Date:** October 20, 2026  
**Current Program State:** P0-01 through P0-05 and P1-01 through P1-16 are **VERIFIED COMPLETE** (329 backend tests passing). **P2-01 is AUDITED — PENDING APPROVAL** as the next implementation task.

---

## 1. Project Identity & Quickstart

Alpha Harness (`Project-_v2`) is an autonomous, self-learning quantitative research harness designed for WorldQuant BRAIN alpha expression generation, multi-stage AST validation, preflight simulation settings verification, multi-LLM orchestration, statistical evaluation, and portfolio curation.

### Project Layout
- **Backend:** `backend/` — Python 3.12+, FastAPI, SQLAlchemy (asyncio + aiosqlite), Pydantic v2, DuckDB, NumPy, SciPy, Pandas. Managed by `uv`.
- **Frontend:** `frontend/` — React 18, Vite, TypeScript, Tailwind CSS, Lucide icons.
- **Canonical Backlog:** `Project_v2_Remediation_Backlog_and_LLM_Prompt.md`
- **Canonical Deferred Issues Log:** `deferred_issues_log.md`
- **Remediation Reports:** `docs/remediation/reports/` (all P0 and P1 audit and verification reports)

### Installation & Test Commands

#### Backend Setup & Test
```bash
cd backend
# Install dependencies into virtual environment managed by uv
uv sync

# Run complete backend test suite (authoritative verification)
uv run pytest --tb=short -q

# Run targeted P1-15 simulation settings test suite
uv run pytest tests/test_p1_15_simulation_settings.py -v

# Start local backend dev server
uv run uvicorn alpha_harness.main:app --reload --port 8000
```

#### Frontend Setup & Run
```bash
cd frontend
npm install
npm run dev
```

---

## 2. Core Architecture Map

| Subsystem | Primary Code Locations | Responsibilities |
|---|---|---|
| **FASTEXPR Generation & AST Validation** | [`backend/src/alpha_harness/generator/pipeline.py`](file:///c:/Users/LENOVO/Desktop/alpha-harness/backend/src/alpha_harness/generator/pipeline.py)<br>[`backend/src/alpha_harness/generator/fastexpr.py`](file:///c:/Users/LENOVO/Desktop/alpha-harness/backend/src/alpha_harness/generator/fastexpr.py) | Parses and validates FASTEXPR syntax, enforces operator grammar constraints, calculates operator complexity, counts fields/operators, and runs repair strategies without hallucinated tokens. |
| **Researcher Orchestration & Diversity Selection** | [`backend/src/alpha_harness/orchestrator/researchers.py`](file:///c:/Users/LENOVO/Desktop/alpha-harness/backend/src/alpha_harness/orchestrator/researchers.py)<br>[`backend/src/alpha_harness/orchestrator/diversity.py`](file:///c:/Users/LENOVO/Desktop/alpha-harness/backend/src/alpha_harness/orchestrator/diversity.py) | Autonomous researcher agents generate hypothesis-driven alpha candidates. Diversity gate (P1-14) enforces AST skeleton fingerprinting (`ast_skeleton`, `expression_hash`), sub-industry/field family limits, and structural deduplication before simulation queueing. |
| **Simulation Settings & Preflight Validation** | [`backend/src/alpha_harness/orchestrator/simulation_settings.py`](file:///c:/Users/LENOVO/Desktop/alpha-harness/backend/src/alpha_harness/orchestrator/simulation_settings.py)<br>[`backend/src/alpha_harness/brain/settings_schema.py`](file:///c:/Users/LENOVO/Desktop/alpha-harness/backend/src/alpha_harness/brain/settings_schema.py)<br>[`backend/src/alpha_harness/brain/schemas.py`](file:///c:/Users/LENOVO/Desktop/alpha-harness/backend/src/alpha_harness/brain/schemas.py) | P1-15 3-tier settings precedence (Overrides > Session > Profile). Preflight validation rejects dataset region mismatches and delay incompatibilities. Generates canonical SHA-256 `settings_hash` and verified settings snapshots. |
| **Evaluation, Robustness & Holdout Isolation** | [`backend/src/alpha_harness/evaluation/metrics.py`](file:///c:/Users/LENOVO/Desktop/alpha-harness/backend/src/alpha_harness/evaluation/metrics.py)<br>[`backend/src/alpha_harness/evaluation/robustness.py`](file:///c:/Users/LENOVO/Desktop/alpha-harness/backend/src/alpha_harness/evaluation/robustness.py) | Computes In-Sample (IS), Validation (Val), and sealed Holdout metrics under P0-03 strict 3-way chronological partitioning. `graduate_candidate` enforces one-way, walk-forward holdout access with zero historical overlap and program-wide contamination checks. |
| **Multiplicity-Corrected Selection** | [`backend/src/alpha_harness/evaluation/metrics.py`](file:///c:/Users/LENOVO/Desktop/alpha-harness/backend/src/alpha_harness/evaluation/metrics.py)<br>[`backend/src/alpha_harness/learning/prompt_validation.py`](file:///c:/Users/LENOVO/Desktop/alpha-harness/backend/src/alpha_harness/learning/prompt_validation.py) | P1-10 statistical selection using Lo (2002) frequency-adjusted Sharpe standard errors, White's Reality Check, and Hansen's Superior Predictive Ability (SPA). P1-13 paired-t tournament non-inferiority gate for prompt variants. |
| **Provenance & Database Models** | [`backend/src/alpha_harness/db/models.py`](file:///c:/Users/LENOVO/Desktop/alpha-harness/backend/src/alpha_harness/db/models.py)<br>[`backend/src/alpha_harness/orchestrator/runner.py`](file:///c:/Users/LENOVO/Desktop/alpha-harness/backend/src/alpha_harness/orchestrator/runner.py) | Immutable tables for `GenerationAttempt` (P0-01), `ResearchCandidate` (with SHA-256 `candidate_provenance_hash` and `settings_hash`), `EvaluationRecord` (with `is_verified` and `provenance`), `HoldoutAssessment`, `ExperimentRun`, and `PromptVariant`. |
| **Researcher Episodic Memory & Learning** | [`backend/src/alpha_harness/learning/memory.py`](file:///c:/Users/LENOVO/Desktop/alpha-harness/backend/src/alpha_harness/learning/memory.py)<br>[`backend/src/alpha_harness/learning/loop.py`](file:///c:/Users/LENOVO/Desktop/alpha-harness/backend/src/alpha_harness/learning/loop.py) | P1-12 persistent researcher memory storing structured episodic records of successful hypotheses and failure modes. Prevents repeated exploration of dead ends; enforces strict model/researcher isolation. |
| **Model Leaderboard & Benchmarking** | [`backend/src/alpha_harness/leaderboard/service.py`](file:///c:/Users/LENOVO/Desktop/alpha-harness/backend/src/alpha_harness/leaderboard/service.py) | P1-11 fair researcher leaderboard using strict matched-cohort stratification (`SimulationCohort`) and Empirical Bayes shrinkage to prevent survivorship and sample bias across models. |
| **BRAIN Client & Platform Communication** | [`backend/src/alpha_harness/brain/client.py`](file:///c:/Users/LENOVO/Desktop/alpha-harness/backend/src/alpha_harness/brain/client.py)<br>[`backend/src/alpha_harness/brain/endpoints.py`](file:///c:/Users/LENOVO/Desktop/alpha-harness/backend/src/alpha_harness/brain/endpoints.py) | Authenticated HTTP client communicating with WorldQuant BRAIN platform API (`/simulations`, `/alphas/{id}/check`, `/correlations`, etc.) with asynchronous polling and Retry-After handling. |
| **Portfolio Curation & Submissions** | [`backend/src/alpha_harness/tools/submission_planner.py`](file:///c:/Users/LENOVO/Desktop/alpha-harness/backend/src/alpha_harness/tools/submission_planner.py)<br>[`backend/src/alpha_harness/db/models.py`](file:///c:/Users/LENOVO/Desktop/alpha-harness/backend/src/alpha_harness/db/models.py) (`Submission`) | Beam search portfolio optimizer selecting alphas satisfying pairwise correlation ceilings ($r < 0.50$) against existing submitted pool. |
| **API Endpoints & Orchestration** | [`backend/src/alpha_harness/api/research.py`](file:///c:/Users/LENOVO/Desktop/alpha-harness/backend/src/alpha_harness/api/research.py)<br>[`backend/src/alpha_harness/api/experiments.py`](file:///c:/Users/LENOVO/Desktop/alpha-harness/backend/src/alpha_harness/api/experiments.py) | P0-05 autonomous end-to-end experiment orchestration routes (`POST /experiments`), candidate queries, and leaderboard endpoints. |

---

## 3. Canonical Remediation Backlog: 36 Issues

The project adheres to the 36 canonical issues defined in `Project_v2_Remediation_Backlog_and_LLM_Prompt.md`:

| ID | Title / Problem Statement | Verified Status |
|---|---|---|
| **P0-01** | Generation-attempt provenance tracking | ✅ **VERIFIED COMPLETE** |
| **P0-02** | Candidate provenance fingerprinting & data integrity | ✅ **VERIFIED COMPLETE** |
| **P0-03** | Sealed train/validation/holdout split isolation | ✅ **VERIFIED COMPLETE** |
| **P0-04** | Robustness evaluation with real parameter re-simulation | ✅ **VERIFIED COMPLETE** |
| **P0-05** | Autonomous end-to-end experiment orchestrator API | ✅ **VERIFIED COMPLETE** |
| **P1-01** | Autonomous reflection & strategy self-learning loop | ✅ **VERIFIED COMPLETE** |
| **P1-02** | Multi-LLM provider routing & model abstraction | ✅ **VERIFIED COMPLETE** |
| **P1-03** | Prompt strategy registry & version control | ✅ **VERIFIED COMPLETE** |
| **P1-04** | Target dataset selection & catalog coverage validation | ✅ **VERIFIED COMPLETE** |
| **P1-05** | Datafield catalog with region/delay awareness | ✅ **VERIFIED COMPLETE** |
| **P1-06** | Authoritative operator catalog & AST attribution | ✅ **VERIFIED COMPLETE** |
| **P1-07** | Language scope enforcement (FASTEXPR grammar) | ✅ **VERIFIED COMPLETE** |
| **P1-08** | Multi-stage candidate validation pipeline | ✅ **VERIFIED COMPLETE** |
| **P1-09** | Operator attribution & expression complexity scoring | ✅ **VERIFIED COMPLETE** |
| **P1-10** | Multiplicity-aware candidate selection (SPA/Reality Check) | ✅ **VERIFIED COMPLETE** |
| **P1-11** | Fair model leaderboard with matched-cohort stratification | ✅ **VERIFIED COMPLETE** |
| **P1-12** | Researcher-specific persistent episodic memory | ✅ **VERIFIED COMPLETE** |
| **P1-13** | Prompt evolution paired non-inferiority tournament gate | ✅ **VERIFIED COMPLETE** |
| **P1-14** | Diversity-aware candidate selection & skeleton deduplication | ✅ **VERIFIED COMPLETE** |
| **P1-15** | Explicit, validated, persisted simulation settings | ✅ **VERIFIED COMPLETE** |
| **P1-16** | “Submission grade” is an internal heuristic, not proof of final submitability | ✅ **VERIFIED COMPLETE** |
| **P2-01** | Frontend client state synchronization & WebSocket feeds | 🔍 **AUDITED — PENDING APPROVAL** (Next Implementation) |
| **P2-02** | Simulation timeout & cancellation handling | ⬜ Tracked |
| **P2-03** | SQLite write contention & connection pool lifecycle | ⬜ Tracked |
| **P2-04** | Rate-limit handling & exponential backoff queueing | ⬜ Tracked |
| **P2-05** | Cache invalidation on dataset/field catalog changes | ⬜ Tracked |
| **P2-06** | Datafield search semantic indexing & ranking | ⬜ Tracked |
| **P2-07** | Automated repair logging & repair cycle detection | ⬜ Tracked |
| **P2-08** | Multi-session concurrency & researcher pool isolation | ⬜ Tracked |
| **P2-09** | Alpha expression normalization & semantic equivalence | ⬜ Tracked |
| **P2-10** | Metric distribution outlier detection & data hygiene | ⬜ Tracked |
| **P2-11** | Dynamic token budget management across LLM providers | ⬜ Tracked |
| **P2-12** | Cold-start researcher seed generation without bias | ⬜ Tracked |
| **P2-13** | Error telemetry & diagnostic log capture | ⬜ Tracked |
| **P3-01** | OpenAPI schema fidelity & client SDK generation | ⬜ Tracked |
| **P3-02** | Deployment packaging, Dockerfile & environment templates | ⬜ Tracked |

---

## 4. Next Implementation Step: P2-01

P1-16 implementation and verification are complete. The next audit is documented in [`docs/remediation/reports/p2_01_audit_report.md`](docs/remediation/reports/p2_01_audit_report.md) and is pending user approval before implementation.

### P1-16 Verification Summary
P1-16 now distinguishes internal quality thresholds, statistical/lifecycle acceptance, verified BRAIN eligibility, and actual submission evidence. The compatibility field `scorecard.is_submission_grade` is explicitly an internal-quality alias only.
Previously, the codebase conflated 5 distinct concepts into a boolean flag (`scorecard.is_submission_grade`):
1. **Internal Heuristic Thresholds**: Passing 5 sample checks (`val_sharpe >= 1.25`, `val_fitness >= 1.0`, etc.) sets `is_submission_grade = True`.
2. **Lifecycle Status Mutation**: Setting `is_submission_grade = True` automatically transitions `candidate.status = CandidateStatus.ACCEPTED`.
3. **Database Misattribution**: `EvaluationRecord.checks` (documented as BRAIN platform checks) is populated with internal Python metric checks.
4. **Episodic Memory Contamination**: LLM prompts receive "submission grade" exemplars that never underwent BRAIN platform checks.
5. **Leaderboard Distortion**: `submission_grade_count` in researcher rankings counts raw simulation passes.

### Approved Remediation Direction for P1-16
1. **Semantic Disentanglement**:
   - Model internal heuristic pass explicitly as `meets_internal_thresholds` / `is_internal_quality_pass`. Preserve `is_submission_grade` property as a deprecated alias for backward compatibility.
   - Add `platform_eligibility_status` (`UNCHECKED`, `CHECK_PENDING`, `ELIGIBLE`, `INELIGIBLE`, `EXEMPT_OFFLINE`) to `ResearchCandidate`.
   - Store real BRAIN `Check` objects in a separate column (`brain_platform_checks`), keeping internal checks distinct.
   - A candidate is marked `SUBMITTED` only upon authoritative execution evidence in `Submission` table or BRAIN API response.
2. **Canonical Spec Correction**:
   - In [`metrics.py:433-443`](file:///c:/Users/LENOVO/Desktop/alpha-harness/backend/src/alpha_harness/evaluation/metrics.py#L433-L443), replace hardcoded simulation settings with `candidate.effective_simulation_settings()`.
3. **Holdout & Provenance Safeguards**:
   - Holdout performance cannot influence eligibility or submission.
   - Candidates without verified provenance (`is_verified = False`) cannot be marked platform-eligible.

**Established Workflow After P1-16**:
1. Implement P1-16 and run targeted + full regression suite (325+ tests passing).
2. Create `p1_16_final_verification_report.md`.
3. Audit P2-01, present findings to the user, and wait for approval before implementing P2-01.

---

## 5. Non-Negotiable Engineering Safeguards

The next LLM engineer must strictly uphold these inviolable principles:

1. **Zero Fabrication**: Never fabricate PnL observations, Sharpe ratios, fitness, turnover, provenance, platform check results, or submission confirmations.
2. **Strict Chronological Holdout Isolation (P0-03)**: Final holdout data is sealed and accessible ONLY via explicit, one-way `graduate_candidate()`. Holdout data must never be exposed to candidate generation, prompt tuning, strategy updates, or routine leaderboards.
3. **Truthful Data Provenance (P0-02)**: Provenance flags (`REAL_BRAIN_SERIES`, `SYNTHETIC_TEST_FIXTURE`, `MISSING`) must accurately reflect data origin. Synthetic series must never masquerade as real BRAIN data.
4. **Distinguish Quality from Platform Eligibility (P1-16)**: High internal backtest quality does not equal BRAIN platform eligibility or platform submission. Never claim an alpha is submitted without recorded platform evidence.
5. **Deterministic Settings Fingerprinting (P1-15)**: Simulation settings must be explicitly resolved, preflight-validated, persisted in full JSON, and deterministically hashed with SHA-256 (`settings_hash`). Ambient defaults must never silently override candidate settings.
6. **Researcher Memory Isolation (P1-12)**: Episodic memories and anti-patterns must remain partitioned by researcher/model to prevent cross-model bias and ensure fair evaluation.
7. **Multiplicity & Overfitting Corrections (P1-10)**: Account for data mining and multiple testing bias using established statistical corrections (Lo SE, White's Reality Check, Hansen's SPA) before declaring statistical superiority.
8. **Diversity Without Quality Compromise (P1-14)**: Structural diversification and AST skeleton pruning must operate as pre-simulation filters; diversity must never bypass statistical quality gates.
9. **Backward-Compatible Preservation**: Preserve all existing test fixtures, working-tree modifications, database schemas, and API routes. Do not reset or discard prior work.
10. **Report Honest Verification**: Always execute tests and verify exit codes directly. Never assume a command succeeded or hide test warnings.

---

## 6. Deferred Issues Log

The following technical debt items are actively tracked in `deferred_issues_log.md` and must be preserved:

- **`DEFERRED-STAT-01`**: Robust Sharpe Uncertainty Estimation (Lo 2002 Theorem 2 HAC Newey-West estimator, Mertens/Opdyke higher-moments kurtosis adjustment, stationary block bootstrap for production candidates).
- **`DEFERRED-BENCH-01`**: Cross-Cohort Hierarchical Meta-Ranking & Shrinkage Calibration (Bayesian mixed-effects Bradley-Terry / Elo latent ability modeling across non-overlapping cohorts).
- **`DEFERRED-MEM-01`**: Semantic Vector Embeddings for Episodic Memory Retrieval (hybrid retrieval combining exact metadata filters with dense vector cosine similarity via DuckDB VSS / sqlite-vec).
- **`DEFERRED-TEST-01`**: Test Suite Thread Cleanup & Async Lifecycle Warnings (`aiosqlite` background connection threads emitting `PytestUnhandledThreadExceptionWarning` during event-loop teardown).
- **`DEFERRED-PROMPT-01`**: Continuous Prompt Multi-Armed Bandits & Adaptive Traffic Allocation (Contextual Thompson Sampling / UCB-1 bandit algorithms for prompt routing).
- **`DEFERRED-STAT-02`**: Small-Sample Paired-t Distributional Assumptions & Robust Estimators (Wilcoxon signed-rank non-parametric fallback and studentized paired bootstrap for small matched cohorts).
- **`DEFERRED-SETT-01`**: Dynamic Platform Options Schema Periodic Refresh & Offline Synchronization (Background TTL refresh task with exponential backoff and encrypted local disk snapshots for offline fallback).

---

## 7. Verified Test History

Authoritative test runs executed on the codebase:

| Date | Command | Tests Passed | Warnings | Status |
|---|---|---|---|---|
| 2026-10-18 | `uv run pytest tests/test_p1_15_simulation_settings.py -v` | **17 passed** | 0 | Exit Code 0 ✅ |
| 2026-10-19 | `uv run pytest --tb=short -q` | **325 passed** | 19 | Exit Code 0 ✅ |

*Note on warnings:* The 19 reported warnings are `PytestUnhandledThreadExceptionWarning` from `aiosqlite` background worker threads closing after the pytest asyncio loop tears down (tracked under `DEFERRED-TEST-01`). Earlier test runs prior to test fixture standardization reported 47 warnings; the current count is verified at 19 warnings with 0 failures and 0 errors.
