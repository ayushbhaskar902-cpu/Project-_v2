# Issue P1-10 Audit Report: Research Multiplicity & Uncertainty in Winner/Loser Selection

**Issue**: P1-10 — Winner/loser selection ignores research multiplicity and uncertainty  
**Date**: 2026-10-16  
**Status**: AUDITED — AWAITING USER REVIEW & APPROVAL BEFORE IMPLEMENTATION  
**Repository Working Tree**: `c:\Users\LENOVO\Desktop\alpha-harness`  

---

## 1. Executive Summary & Problem Diagnosis

In the closed-loop research engine, `ResearchMemory.get_session_performance_split()` partitions evaluated candidates into **winners** (which become few-shot prompt exemplars and prompt strategy archetypes) and **losers** (which become anti-pattern warnings).

The current implementation in [`backend/src/alpha_harness/learning/memory.py:206-213`](backend/src/alpha_harness/learning/memory.py) is statistically naive:

```python
# Sort by composite score descending
records.sort(key=lambda r: float(r["composite_score"]), reverse=True)

cutoff_top = max(1, len(records) // 4)
winners = records[:cutoff_top]
losers = [r for r in records[cutoff_top:] if r["sharpe"] < 0.8 or r["sharpe_decay"] > 0.4]
```

### Critical Flaws:
1. **Neglect of Multiple Testing (Data Mining Bias / Multiplicity)**:
   - When an LLM generates $N = 80$ or $N = 500$ alpha expressions across a campaign, extreme Sharpe ratios arise purely through sample variation and random luck.
   - Simply taking the top 25% (`records[:cutoff_top]`) of point estimates guarantees that random statistical flukes are celebrated as "winners" and codified into few-shot prompts.
2. **Ignoring Sample Uncertainty ($T$, Skewness, Kurtosis)**:
   - Point estimates of Sharpe and composite score treat all candidates equally, ignoring sample size $T$ (number of trading days).
   - A candidate with a 6-month validation series has a huge confidence interval relative to a 5-year series, yet both are ranked on raw point estimates.
3. **Rigid, Uncalibrated Thresholds for Losers**:
   - `sharpe < 0.8` or `sharpe_decay > 0.4` are arbitrary static constants. In difficult market regimes (e.g., bear markets where the universe baseline Sharpe is negative or 0.1), solid signals with Sharpe 0.75 are incorrectly branded as "losers" and flagged as anti-patterns.
   - Conversely, in bull regimes where baseline Sharpe is 1.5, ineffective alphas with Sharpe 0.9 escape classification as underperformers.

---

## 2. Mathematical Framework for Multiplicity Control

To prevent lucky flukes from contaminating prompt strategies, selection must control for research multiplicity using established financial econometrics:

### 2.1 Standard Error of the Sharpe Ratio (Lo 2002, Mertens 2002)
For a candidate with annualized Sharpe $\widehat{\text{SR}}$, observation length $T$ (in years), return skewness $\gamma_3$, and kurtosis $\gamma_4$:
$$\text{SE}(\widehat{\text{SR}}) = \sqrt{\frac{1}{T} \left(1 + \frac{1}{2} \widehat{\text{SR}}^2 - \gamma_3 \widehat{\text{SR}} + \frac{\gamma_4 - 3}{4} \widehat{\text{SR}}^2 \right)}$$
Under normal returns ($\gamma_3 = 0, \gamma_4 = 3$), this simplifies to:
$$\text{SE}(\widehat{\text{SR}}) \approx \sqrt{\frac{1}{T} \left(1 + \frac{1}{2} \widehat{\text{SR}}^2 \right)}$$

### 2.2 Benjamini-Hochberg False Discovery Rate (FDR) Procedure
Given $M$ tested candidate alphas in the research generation/session:
1. For each candidate $i$, calculate two-tailed $p$-value testing $H_0: \text{SR}_i \le \text{SR}_{\text{baseline}}$:
   $$z_i = \frac{\widehat{\text{SR}}_i - \text{SR}_{\text{baseline}}}{\text{SE}_i}, \quad p_i = 1 - \Phi(z_i)$$
2. Rank $p$-values in ascending order: $p_{(1)} \le p_{(2)} \le \dots \le p_{(M)}$.
3. For a target FDR level $q = 0.10$ (or $0.15$), find the largest $k$ such that:
   $$p_{(k)} \le \frac{k}{M} \cdot q$$
4. Candidates with $p_i \le p_{(k)}$ reject $H_0$ and qualify as statistically significant discoveries.

### 2.3 Deflated Sharpe Ratio (DSR, Bailey & López de Prado 2014)
Adjusts the Sharpe ratio threshold for the expected maximum Sharpe ratio from $M$ independent trials:
$$\text{E}[\max_{m=1\dots M} \text{SR}_m] \approx \left( (1 - \gamma) \cdot \Phi^{-1}\left(1 - \frac{1}{M}\right) + \gamma \cdot \Phi^{-1}\left(1 - \frac{1}{M \cdot e}\right) \right) \cdot \text{SE}$$
A candidate is only a genuine winner if its estimated Sharpe exceeds the expected maximum from noise: $\text{DSR}_i = \Phi\left(\frac{\widehat{\text{SR}}_i - \text{E}[\max \text{SR}]}{\text{SE}_i}\right) \ge 0.95$.

---

## 3. Proposed Remediation Plan for P1-10

### 3.1 Refactor `ResearchMemory.get_session_performance_split()`
1. **Dynamic Cohort Baseline**:
   - Compute cohort median and standard deviation of validation metrics across the session/category.
2. **Standard Error & Multiplicity Calculation**:
   - Compute standard error for each candidate based on validation observation days $T_{\text{days}}$ (retrieved from `ResearchEvaluation` or `AlphaVault`).
   - Calculate candidate $p$-values against the cohort baseline.
   - Apply Benjamini-Hochberg adjustment to obtain FDR $q$-values.
3. **Winner Qualification Gates**:
   - Candidate must have $q \le 0.15$ (or exceed DSR threshold of 0.90).
   - Candidate must have positive validation Sharpe above cohort baseline ($\Delta \text{Sharpe} > 0$).
   - Fallback for small batches ($M < 5$): require $t \ge 1.96$ against zero with small-sample shrinkage.
4. **Regime-Relative Loser Selection**:
   - Classify as losers candidates with statistically significant underperformance ($z \le -1.645$ relative to cohort mean) OR high turnover decay exceeding regime norms.
5. **Preserve Holdout Isolation**:
   - Multiplicity calculations use strictly in-sample/validation metrics (`val_sharpe`, `val_composite_score`, `is_val_decay`).
   - Final holdout data remains completely withheld.

### 3.2 Metadata & Database Lineage
Extend `ResearchEvaluation` or return dictionary with:
- `multiplicity_trials_count`: Total trials $M$ evaluated in this cohort.
- `statistical_p_value`: Raw $p$-value.
- `fdr_q_value`: Benjamini-Hochberg adjusted discovery rate.
- `selection_tier`: `SIGNIFICANT_WINNER`, `CANDIDATE_POOL`, `SIGNIFICANT_LOSER`.

### 3.3 Downstream Prompt Context
In `SelfLearningEngine`, inform the reflection prompt of:
- Number of trials evaluated $M$.
- Statistical significance tier of few-shot examples (ensuring the LLM does not emulate overfitted lucky flukes).

---

## 4. Verification Plan for P1-10

A dedicated test suite `backend/tests/test_p1_10_multiplicity_selection.py` will verify:
1. `test_benjamini_hochberg_fdr_rejects_spurious_candidates`: Verifies that in a batch of 100 random noise candidates, spurious top-decile alphas are rejected by FDR and not marked as winners.
2. `test_sample_duration_uncertainty_weighting`: Verifies that a short 30-day candidate with Sharpe 2.0 does not outrank a statistically solid 500-day candidate with Sharpe 1.6.
3. `test_regime_relative_loser_classification`: Verifies that losers are identified relative to the cohort distribution rather than hardcoded 0.8 Sharpe.
4. `test_small_sample_conservative_fallback`: Verifies graceful behavior when batch size $M < 5$.
5. `test_holdout_isolation_in_multiplicity_split`: Verifies holdout metrics are never accessed during winner/loser partitioning.
6. `test_reflection_loop_incorporates_fdr_exemplars`: Verifies few-shot exemplars in evolved strategies come from FDR-verified discoveries.

---

## 5. Next Steps

- **Awaiting User Review**: In accordance with the canonical remediation workflow, implementation of P1-10 will not begin until this audit is reviewed and approved.
- Upon approval, implementation will proceed in `backend/src/alpha_harness/learning/memory.py`, `backend/src/alpha_harness/learning/loop.py`, and `backend/tests/test_p1_10_multiplicity_selection.py`.
