# Canonical Deferred Issues & Technical Debt Log

This log persistently tracks non-blocking enhancements, advanced statistical estimators, and deferred technical debt across the remediation program.

## 8. Incremental API Decomposition Remainder

- **Issue ID**: `DEFERRED-P2-02-01`
- **Origin**: P2-02 service extraction.
- **Details**: `research.py` still contains read-only query and response-shaping handlers by design. A broader decomposition, browser-level frontend harness, and async teardown cleanup remain separate work; no deferred issue was removed or closed by P2-02.

---

## 1. Robust Sharpe Uncertainty Estimation (Deferred from P1-10)

- **Issue ID**: `DEFERRED-STAT-01`
- **Origin**: P1-10 Multiplicity-Corrected Selection & Uncertainty Audit.
- **Current Baseline**: Lo (2002) Theorem 1 frequency-adjusted standard error under IID-normal returns:
  $$\text{SE}(\text{SR}_a) = \sqrt{\frac{1 + \text{SR}_a^2 / (2q)}{Y}}$$
  where $q = 250$ and $Y$ is validation duration in trading years.
- **Limitation**: Assumes serially independent, mesokurtic returns. Does not explicitly estimate serial autocorrelation ($\rho_k$) or excess kurtosis ($\gamma_4$).
- **Deferred Work Plan**:
  1. Implement Lo (2002) Theorem 2 (GMM/Newey-West HAC estimator for auto-correlated returns).
  2. Implement Mertens (2002) / Opdyke (2007) higher-moments correction:
     $$\text{Var}(\text{SR}) = \frac{1}{T} \left(1 + \frac{1}{2}\text{SR}^2 - \gamma_3 \text{SR} + \frac{\gamma_4 - 3}{4}\text{SR}^2\right)$$
  3. Implement stationary block bootstrap for empirical distribution inference on high-conviction production graduation candidates.

---

## 2. Cross-Cohort Hierarchical Meta-Ranking & Shrinkage Calibration (Deferred from P1-11)

- **Issue ID**: `DEFERRED-BENCH-01`
- **Origin**: P1-11 Fair Researcher/Model Leaderboard.
- **Current Baseline**: Strict matched-cohort stratification. Models must be compared within an identical `SimulationCohort` (identical universe, region, neutralization, decay, delay, and evaluation dates). Disjoint cohorts are classified as `INCOMPARABLE` or `INSUFFICIENT_EVIDENCE`. Empirical Bayes shrinkage parameter is set to fixed prior weight $k = 5.0$.
- **Limitation**: When researchers operate across multiple partially overlapping cohorts (e.g. some in `USA/TOP3000` and others in `EUR/TOP1200`), direct cross-cohort global ranking is intentionally withheld to prevent confounding.
- **Deferred Work Plan**:
  1. Implement a hierarchical Bayesian mixed-effects model (e.g. Bradley-Terry / Elo latent ability model or random-effects meta-analysis) estimating latent model capability $\theta_m$ while controlling for random cohort difficulty $\gamma_c$.
  2. Calibrate shrinkage parameter $k$ dynamically using marginal maximum likelihood (MML) or cross-validation across historical cohorts.
  3. Implement dynamic provider-level token pricing rather than static aggregate estimation ($0.000002/token).
  4. Record and expose millisecond-precision generation latency distributions per model.

---

## 3. Semantic Vector Embeddings for Episodic Memory Retrieval (Deferred from P1-12)

- **Issue ID**: `DEFERRED-MEM-01`
- **Origin**: P1-12 Researcher-Specific Persistent Memory.
- **Current Baseline**: Deterministic SQL-based structured episodic memory retrieval using metadata filters (generation, category, dataset, region, universe, hypothesis family, outcome, and SHA-256 expression hash deduplication).
- **Limitation**: Does not calculate soft semantic similarity between hypotheses (e.g., detecting that "short-term momentum breakout" and "5-day moving average acceleration" test functionally identical economic rationales without exact string matches).
- **Deferred Work Plan**:
  1. Store dense embedding vectors for `hypothesis` and `rationale` in an external vector index or DuckDB VSS / sqlite-vec extension.
  2. Enable hybrid retrieval combining exact metadata filters with cosine similarity ranking.
  3. Cluster historical failures into higher-level conceptual anti-patterns.

---

## 4. Test Suite Thread Cleanup & Async Lifecycle Warnings

- **Issue ID**: `DEFERRED-TEST-01`
- **Origin**: Observed across P0, P1-10, P1-11, P1-12 test execution.
- **Details**: `PytestUnhandledThreadExceptionWarning` emitted when `aiosqlite` background connection worker threads terminate after an event loop has closed during pytest tear-down. Tests pass with exit code 0, but warnings appear in summaries.
- **Remediation**: Standardize async test fixtures across the test suite to explicitly call `await db.dispose()` in an explicit teardown yield block.

---

## 5. Continuous Prompt Multi-Armed Bandits & Adaptive Traffic Allocation (Deferred from P1-13)

- **Issue ID**: `DEFERRED-PROMPT-01`
- **Origin**: P1-13 Prompt Evolution Validation.
- **Current Baseline**: Deterministic tournament batches comparing experimental prompt variants against the incumbent on matched evaluation cohorts with predeclared non-inferiority margins $\delta$ and Bonferroni adjustments.
- **Limitation**: Allocates fixed sample splits across candidates during tournament evaluation rather than adapting dynamically as evidence accumulates.
- **Deferred Work Plan**:
  1. Implement contextual Thompson Sampling or UCB-1 bandit algorithms to dynamically route generation tasks to higher-performing prompt variants while guaranteeing exploration floors.
  2. Implement automated safety tripwires to immediately halt traffic to any prompt variant that drops below an emergency AST validity or degradation threshold.

---

## 6. Small-Sample Paired-t Distributional Assumptions & Robust Estimators (Deferred from P1-13)

- **Issue ID**: `DEFERRED-STAT-02`
- **Origin**: P1-13 Prompt Evolution Non-Inferiority Gate.
- **Current Baseline**: Matched-cohort paired difference $\Delta_i = \text{SR}_{\text{exp}, i} - \text{SR}_{\text{inc}, i}$ with Student's t critical values ($df = N - 1$) and Bonferroni multiplicity adjustment:
  $$\text{LCB}_{1-\alpha} = \bar{\Delta} - t_{1-\alpha/M, N-1} \cdot \frac{s_\Delta}{\sqrt{N}}$$
- **Distributional Assumptions**:
  1. **Normality of Differences**: Differences $\Delta_i$ across matched cohorts are assumed to be approximately normally distributed.
  2. **Inter-Cohort Independence**: Evaluated cohorts are treated as independent draws from the target simulation distribution.
  3. **Identical Scale across Cohorts**: Cohort-specific variances are pooled via the sample standard deviation $s_\Delta$.
- **Limitations**:
  - Heavy-tailed distribution of Sharpe differences across volatile regimes may inflate sample variance or lead to undercoverage in small samples ($N < 10$).
  - Overlapping time windows across evaluation cohorts may introduce serial cohort correlation, violating exact sample independence.
- **Deferred Work Plan**:
  1. Implement a paired Wilcoxon signed-rank test or sign test as a non-parametric fallback when sample differences fail Shapiro-Wilk normality tests ($p < 0.05$).
  2. Implement a studentized paired bootstrap (Efron & Tibshirani, 1993) for empirical confidence intervals on small, non-normal matched samples ($N \ge 8$).
  3. Implement Huber / trimmed M-estimators of paired location differences to prevent single-cohort regime outliers from dominating non-inferiority decisions.

---

## 7. Dynamic Platform Options Schema Periodic Refresh & Offline Synchronization (Deferred from P1-15)

- **Issue ID**: `DEFERRED-SETT-01`
- **Origin**: P1-15 Explicit, Validated, and Persisted Simulation Settings.
- **Current Baseline**: Cached schema from `OPTIONS /simulations` stored in SQLite `MetadataCache` with 1-hour TTL on login, complemented by explicit offline test fixtures for synthetic runs.
- **Limitation**: When long-running worker processes execute indefinitely across sessions, schema options may expire or become stale if network access to BRAIN API is intermittently unavailable.
- **Deferred Work Plan**:
  1. Implement a background asynchronous TTL refresh task with exponential backoff and connection liveness checks.
  2. Implement an encrypted local disk snapshot of verified platform options for graceful cold-start recovery in offline or airgapped environments.
  3. Support per-user permissioned schema scoping (identifying which universes/features are enabled per API token).

