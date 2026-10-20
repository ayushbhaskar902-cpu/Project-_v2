# Issue P1-11 Audit Report: Model/Researcher Leaderboard is Not a Fair Model Benchmark

**Issue**: P1-11 — Model/researcher leaderboard is not a fair model benchmark  
**Date**: 2026-10-17  
**Status**: AUDITED — AWAITING USER REVIEW & APPROVAL BEFORE IMPLEMENTATION  
**Repository Working Tree**: `c:\Users\LENOVO\Desktop\alpha-harness`  
**Primary Files Under Investigation**:
- `backend/src/alpha_harness/leaderboard/service.py`
- `backend/src/alpha_harness/db/models.py`
- `backend/src/alpha_harness/api/research.py`
- `backend/src/alpha_harness/orchestrator/researchers.py`

---

## 1. Executive Summary & Problem Diagnosis

The system maintains a multi-agent quantitative researcher system where diverse LLM model architectures (e.g., `google:gemini-2.5-flash`, `anthropic:claude-3-5-sonnet`, `deepseek:deepseek-reasoner`) generate alpha expressions under assigned researcher personas.

The leaderboard service in [`backend/src/alpha_harness/leaderboard/service.py:435-502`](backend/src/alpha_harness/leaderboard/service.py) provides `get_researcher_leaderboard()` to benchmark and rank these LLM researchers. However, the current benchmarking methodology is fundamentally invalid and unfair as an experimental scientific instrument:

```python
# Current ranking logic in leaderboard/service.py:448-498
for r in researchers:
    tot_gen = r.total_generated or 0
    tot_val = r.total_valid or 0
    v_rate = round(tot_val / tot_gen, 2) if tot_gen else 0.0
    ...
    # Count raw submission grade alphas
    c_stmt = (
        select(func.count(ResearchCandidate.id))
        .join(ResearchEvaluation, ResearchEvaluation.candidate_id == ResearchCandidate.id)
        .where(
            ResearchCandidate.researcher_id == r.id,
            ResearchEvaluation.is_verified.is_(True),
            ...
        )
    )
    ...
# Rank strictly by raw count descending, then raw avg_sharpe descending
entries.sort(key=lambda e: (e.submission_grade_count, e.avg_sharpe), reverse=True)
```

### Critical Flaws in the Current Leaderboard:

1. **Volume / Sample-Size Confounding (Count vs. Quality)**:
   - Researchers are ranked primarily on `submission_grade_count` (raw volume of passed checks). A model given a high generation quota (e.g., $N = 200$) easily outperforms a superior model given a smaller quota (e.g., $N = 20$), even if the latter has a 10x higher yield efficiency and superior risk-adjusted alpha.
2. **Task and Regime Heterogeneity (Uncontrolled Cohorts)**:
   - Models are assigned different `strategy_focus` (e.g., `momentum`, `mean_reversion`, `volatility`, `liquidity`), universes (e.g., `TOP3000` vs `TOP500`), and evaluation regimes.
   - Just as P1-03 proved that prompt versions cannot be compared across different simulation cohorts, model performance cannot be fairly benchmarked when one model operates in a high-Sharpe regime/universe while another operates in a high-decay or illiquid regime.
3. **No Uncertainty Quantification or Sample Gating**:
   - `avg_sharpe` and `best_sharpe` are reported without sample size weighting, confidence intervals, or standard errors. A model with 2 lucky candidates and an average Sharpe of 2.1 ranks above a model with 80 candidates and an average Sharpe of 1.95 (which has vastly stronger statistical evidence).
4. **Denominators and Missingness Ignored**:
   - Syntax errors, rejected AST structures, and failed simulations are inconsistently handled. The current `valid_rate` only measures `total_valid / total_generated` without distinguishing between syntax validity, simulator check passage, and economic viability.
5. **Holdout Boundary Protection (P0-03)**:
   - The researcher entity in `ResearchResearcher.avg_sharpe` and `ResearchResearcher.best_sharpe` was historically populated from legacy evaluations without explicit validation-only gating. Leaderboard rankings must strictly verify that holdout metrics never contaminate model ranking or comparison.

---

## 2. Mathematical Framework for Fair Model Benchmarking

To make the Model/Researcher Leaderboard a statistically justified, reproducible experimental benchmark, we must apply principles analogous to P1-03 (controlled cohort isolation) and P1-09/P1-10 (statistical shrinkage and multiplicity control):

### 2.1 Controlled Cohort Normalization (Relative Lift)
For candidate $i$ produced by model $m$ within simulation cohort $c$ (defined by identical region, universe, decay, delay, neutralization, and evaluation dates):
$$\Delta \text{SR}_{i, c} = \text{SR}_{i, c} - \overline{\text{SR}}_c$$
Where $\overline{\text{SR}}_c$ is the cohort baseline median across all models evaluated on that identical task. This removes regime and universe advantages.

### 2.2 Empirical Bayes Shrinkage for Model Performance
For a model with $N_m$ validated evaluations and mean relative lift $\overline{\Delta \text{SR}}_m$:
$$\widehat{\mu}_m = \frac{N_m}{N_m + k_{\text{model}}} \overline{\Delta \text{SR}}_m$$
Where $k_{\text{model}}$ is a shrinkage parameter (e.g., $k = 5$), preventing models with tiny samples ($N = 1$ or $2$) from dominating rankings with fluke point estimates.

### 2.3 Statistical Confidence Bounds
The standard error of the model's mean performance:
$$\text{SE}_m = \frac{\sigma_m}{\sqrt{N_m}}$$
Models should be ranked by their conservative **Lower Confidence Bound ($\text{LCB}_{90} = \widehat{\mu}_m - 1.645 \cdot \text{SE}_m$)** or by a composite quality-efficiency score, rather than raw counts.

### 2.4 Multi-Stage Efficiency Funnel
Each researcher/model must report full funnel metrics:
- **Generation Count ($N_{\text{gen}}$)**: Total raw expression attempts.
- **Syntactic Validity Rate ($R_{\text{ast}}$)**: Parsable AST matching catalog grammar.
- **Simulation Pass Rate ($R_{\text{sim}}$)**: Candidates passing simulation checks.
- **Statistical Discovery Rate ($R_{\text{fdr}}$)**: Candidates achieving `SIGNIFICANT_WINNER` under P1-10 FDR.
- **Efficiency-Adjusted Cost / Token Productivity**: Alpha lift per unit of token expenditure.

---

## 3. Proposed Remediation Plan for P1-11

### Step 1: Extend Researcher Ranking Schemas & DTOs
In `backend/src/alpha_harness/leaderboard/service.py`:
- Update `ResearcherRankEntry` to include:
  - `sample_count`: Total evaluated candidates.
  - `mean_validation_sharpe`: Validation-only average Sharpe.
  - `shrunk_relative_lift`: Empirical Bayes shrunk lift relative to cohort baseline.
  - `lcb_sharpe`: Lower confidence bound of model Sharpe.
  - `statistical_winner_count`: Count of FDR-verified `SIGNIFICANT_WINNER` candidates (P1-10).
  - `discovery_rate`: Ratio of statistical winners to total evaluated.
  - `status`: `"RANKED"`, `"INSUFFICIENT_SAMPLE"` ($N < 5$), or `"INCOMPARABLE"`.

### Step 2: Refactor `LeaderboardService.get_researcher_leaderboard()`
- Replace raw count sorting with a controlled multi-cohort ranking engine.
- Group candidate evaluations by `SimulationCohort` (reusing `compute_cohort_fingerprint`).
- Measure each model's excess performance against cohort medians.
- Enforce a minimum sample threshold ($N \ge 5$) before assigning an official leaderboard rank.
- Strictly exclude holdout partition data (enforcing P0-03).

### Step 3: API & Snapshot Persistence
- Update `capture_snapshot()` to record shrunk relative lift, cohort denominators, and discovery rates in `LeaderboardSnapshot.entries["top_researchers"]`.
- Update API endpoints in `backend/src/alpha_harness/api/research.py` to expose fair benchmarking filters (e.g., `cohort_id`, `category`, `min_sample`).

### Step 4: Regression Test Suite (`tests/test_p1_11_fair_model_leaderboard.py`)
Add dedicated tests verifying:
1. High-volume models with low Sharpe cannot outrank low-volume models with high-conviction, statistically significant alphas.
2. Models operating on different simulation cohorts are compared via relative lift or flagged incomparable.
3. Models with tiny samples ($N < 5$) receive `INSUFFICIENT_SAMPLE` and do not occupy top rank through lucky point estimates.
4. Holdout metrics never contaminate researcher leaderboard calculations.
5. Snapshot persistence captures complete funnel and statistical fields idempotently.

---

## 4. Verification & Safety Constraints
- **Preserve Git Tree**: Do not modify any uncommitted files or unrelated modules.
- **No Early Implementation**: Code changes for P1-11 will only begin after the user explicitly reviews and approves this audit plan.
