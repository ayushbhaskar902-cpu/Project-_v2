# Project-_v2: Remediation Backlog and LLM Coding Prompt

**Repository:** https://github.com/ayushbhaskar902-cpu/Project-_v2  
**Baseline:** Public `main` branch reviewed on 10 October 2026.  
**Purpose:** Turn the forensic audit into a traceable fix plan. The issue list includes code-level findings, missing requirements, and areas that require execution-level confirmation. Do not assume the list is exhaustive of bugs that only appear at runtime.

## Important source locations

- Research API / evaluation / robustness: `backend/src/alpha_harness/api/research.py`
- Researcher orchestration: `backend/src/alpha_harness/orchestrator/researchers.py`
- Evaluation metrics: `backend/src/alpha_harness/evaluation/metrics.py`
- Self-learning: `backend/src/alpha_harness/learning/loop.py`
- Research memory: `backend/src/alpha_harness/learning/memory.py`
- Alpha generation/AST validation: `backend/src/alpha_harness/generator/pipeline.py`
- BRAIN client/integration: `backend/src/alpha_harness/brain/`
- Shared application wiring: `backend/src/alpha_harness/state.py`
- Database models: `backend/src/alpha_harness/db/models.py`
- Frontend: `frontend/`
- Product claims / workflow: `README.md`, `DESIGN.md`, `RESEARCH_ENGINE_CHANGELOG.md`

## Priority definitions

- **P0:** Invalidates core results or prevents the defining project requirement from working.
- **P1:** Major required capability missing or incorrectly connected.
- **P2:** Reliability, reproducibility, maintainability, or methodological weakness that must be addressed before production use.
- **P3:** Documentation and lower-risk maintainability polish.

---

# 1. Remediation backlog: issue-by-issue

## P0 — fix before trusting research results

### P0-01 — Synthetic PnL is passed through the ordinary evaluation engine

**Area:** `backend/src/alpha_harness/api/research.py`, evaluation pipeline.  
**Problem:** When the real daily series contains fewer than 20 points, the code can synthesize roughly 2,500 daily PnL observations from the alpha's summary Sharpe and noise, then send those observations to `EvaluationEngine`. The resulting derived metrics can look like measured IS/OOS statistics even though no observed daily return series was evaluated.  
**Fix:** Remove this synthetic fallback from every production evaluation/selection path. If real daily returns/turnover are unavailable, mark the candidate `WAITING_FOR_SERIES` or `EVALUATION_UNAVAILABLE`, return a structured reason, and do not calculate a scorecard or let the candidate participate in leaderboards, learning, or portfolio curation. Synthetic series may exist only in an explicitly named test fixture or demo mode that cannot be mistaken for real data.  
**Acceptance tests:** (1) Missing/short series yields no evaluation record; (2) no synthetic values are sent to `EvaluationEngine`; (3) candidate cannot influence learning/leaderboard/portfolio; (4) real series still evaluates normally.

### P0-02 — Data provenance is misreported

**Area:** `backend/src/alpha_harness/api/research.py`.  
**Problem:** The evaluation response currently returns `realBrainData: True` even when the fallback above creates synthetic observations.  
**Fix:** Persist explicit provenance with every raw data artifact and scorecard, e.g. `REAL_BRAIN_SERIES`, `BRAIN_SUMMARY_ONLY`, `SYNTHETIC_TEST_FIXTURE`, `MISSING`. Build response fields from persisted provenance; never hard-code a real-data flag. A scorecard based only on summary metrics must be labeled as summary-only and must not pretend to contain independently computed daily-series statistics.  
**Acceptance tests:** Provenance survives storage/reload; response status agrees with the exact data path used; learning and leaderboard queries exclude ineligible provenance by default.

### P0-03 — IS/OOS separation is not a sealed evaluation protocol

**Area:** `backend/src/alpha_harness/evaluation/metrics.py`, `learning/loop.py`, research APIs.  
**Problem:** IS and OOS metrics are computed, but evaluated candidates can feed the learning/strategy-update cycle. Reusing the same OOS window for repeated prompt/strategy choices turns it into another optimization set.  
**Fix:** Define immutable experiment-level time boundaries and data access rules. Use chronological train/validation/holdout windows or a walk-forward protocol. During normal strategy learning, expose only training and validation results. Keep final holdout data inaccessible to prompt generation, strategy updates, hyperparameter selection, and routine leaderboards. Any holdout access must be a separately recorded, one-way final evaluation. If the platform does not support independent OOS simulation, state that limitation clearly rather than calling the result sealed OOS.  
**Acceptance tests:** A test proves the learning service cannot query the protected holdout; changing holdout results cannot change prompt/strategy selection; final evaluation is recorded separately; split dates are persisted and validated against leakage.

### P0-04 — Robustness endpoint fabricates degraded results instead of re-simulating

**Area:** `backend/src/alpha_harness/api/research.py`, robustness service.  
**Problem:** Robustness tests use random degradation multipliers based on baseline metrics rather than executing BRAIN simulations with changed settings. The endpoint itself documents that these are synthetic results.  
**Fix:** For each perturbation, create a real BRAIN simulation request changing exactly the intended variable (neutralization, decay, universe, region where supported, delay, cost assumption if supported). Persist the baseline ID, perturbation config, BRAIN simulation ID, raw results, status, and provenance. If the environment cannot run a perturbation, return `UNSUPPORTED`/`NOT_RUN`; never manufacture a passing/failing metric. Preserve synthetic tests only as tests with a synthetic label.  
**Acceptance tests:** A robustness result can be traced to a completed simulation ID and distinct simulation settings; unavailable tests are explicitly not run; no random degradation is included in production scores.

### P0-05 — Autonomous end-to-end API is missing

**Area:** research API, orchestration, frontend.  
**Problem:** Generation, enqueueing, evaluation, learning, robustness, and curation are separate operations requiring manual sequencing. That does not meet the one-request, prompt-and-parameters-to-result goal.  
**Fix:** Add an experiment/job orchestrator and an API such as `POST /experiments` that validates configuration, creates a persistent job, runs stages according to state transitions, retries safely, waits/polls BRAIN through its real lifecycle, evaluates only after data is ready, and returns a job ID/status. Add `GET /experiments/{id}` and optionally cancel/resume endpoints. A long-running job should not hold an HTTP request open.  
**Acceptance tests:** One API request can drive a mock end-to-end pipeline from creation to completed evaluation; job survives process restart; duplicate requests/idempotency rules are tested; errors surface as failed stages, not false success.

## P1 — core research capabilities and learning correctness

### P1-01 — Learning cycle is heuristic rather than actual LLM reflection

**Area:** `backend/src/alpha_harness/learning/loop.py`.  
**Problem:** The class accepts an LLM service, but the observed cycle mainly partitions winners/losers, aggregates operators, and emits hard-coded insight text; the injected LLM is not part of the actual reflection call.  
**Fix:** Implement a separate reflection service that receives a bounded, structured experiment report: provenance-qualified metrics, hypotheses, expression metadata, failures, datafield/operator family, configuration, costs and uncertainty. Ask the selected LLM to propose testable insights and next experiments in a strict schema. Treat LLM conclusions as hypotheses, not facts; validate them through controlled experiments before marking them as learned rules. Keep deterministic analysis as evidence and provide a non-LLM fallback that is explicitly labeled.  
**Acceptance tests:** Unit tests prove the configured reflection provider is called; malformed model output fails safely; each insight references evidence IDs; proposed rules are not marked validated until an experiment tests them.

### P1-02 — Evolved prompt strategy may be saved under the wrong category

**Area:** `backend/src/alpha_harness/learning/loop.py` and `orchestrator/researchers.py`.  
**Problem:** The learning loop stores an evolved strategy with `category="general"`, whereas generation fetches active strategies by each researcher's category.  
**Fix:** Make strategy updates explicitly scoped to a researcher/category/strategy family, or create a deliberate global strategy plus well-defined category-specific derivation. Define how a new strategy supersedes the prior active version and ensure only one intended active version exists per scope.  
**Acceptance tests:** A learning cycle for a momentum researcher changes the strategy ID actually loaded by the next momentum run; other categories remain unchanged unless explicitly updated; tests cover activation and version ordering.

### P1-03 — No rigorous prompt-level leaderboard / controlled prompt experiment

**Area:** prompt library, database models, research APIs, frontend.  
**Problem:** Researcher-level aggregate metrics exist, but prompt versions are not benchmarked as first-class, fairly compared units.  
**Fix:** Persist prompt/strategy version, checksum, model/provider, task assignment, token/cost, latency, valid-generation rate, evaluation provenance, sample size, confidence interval, and resulting candidate IDs. Add controlled prompt experiments with matched tasks/allocations and a prompt leaderboard. Do not compare different prompts unless allocation and experiment differences are visible.  
**Acceptance tests:** Every candidate resolves to an immutable prompt version; leaderboard can group by prompt and show denominators, cost, uncertainty, and validation status; prompt versions cannot be mutated in place.

### P1-04 — Dataset selection is not a true part of the autonomous search policy

**Area:** `orchestrator/researchers.py`, `discovery/catalog.py`, generation prompts.  
**Problem:** The main generation flow is session-scoped and does not show a robust policy that actively chooses and allocates datasets for each hypothesis.  
**Fix:** Introduce a dataset catalog/registry with authoritative BRAIN metadata and eligibility constraints. Require experiment design to choose a dataset deliberately and persist the choice, reason, candidate datafields, and catalog version. Add dataset allocation policies that support controlled exploration. If BRAIN does not expose a given catalog capability, document the boundary and accept explicit user configuration instead of pretending it is automated.  
**Acceptance tests:** Generated candidates have a valid catalog-backed dataset context; invalid/unavailable dataset IDs are rejected before LLM generation or simulation; cross-dataset experiments are reproducible.

### P1-05 — Datafield selection/discovery is weakly integrated

**Area:** `discovery/catalog.py`, `generator/pipeline.py`, orchestrator.  
**Problem:** Expression parsing and field tracking exist, but the system does not sufficiently prove that live datafield metadata constrains prompt creation and candidate validity.  
**Fix:** Create a cached, versioned BRAIN datafield catalog per dataset/region/universe where supported. Provide type, unit, coverage, delay, availability and category metadata when the platform exposes it. Inject only eligible fields into the model prompt; validate every field against the exact experiment catalog version before simulation. Track family-level and field-level usage/outcomes.  
**Acceptance tests:** Unknown or wrong-dataset fields are rejected; catalog version is attached to candidates; no candidate goes to BRAIN with unverified fields.

### P1-06 — Operator metadata/selection needs authoritative validation

**Area:** `generator/pipeline.py`, `discovery/catalog.py`.  
**Problem:** A local operator table and prompt constraints can drift from currently supported BRAIN operators/arity/signatures.  
**Fix:** Sync supported operator metadata where allowed, version the registry, validate name, arity, argument types/units, and operator-specific constraints. Distinguish unavailable operators from syntax errors. Never invent operator signatures.  
**Acceptance tests:** Unknown operators and invalid arity/type combinations fail before submission; tests cover catalog updates and unsupported operators.

### P1-07 — Only Fast Expression appears to be implemented as a generation target

**Area:** generation pipeline.  
**Problem:** The stated project scope mentions expressions or Python alphas, but the traced pipeline is oriented around BRAIN Fast Expression.  
**Fix:** Decide whether Python alpha generation is truly required for the intended BRAIN workflow. If yes, implement a separate, sandboxed Python-alpha interface with an explicit supported contract, dependency/security restrictions, deterministic execution and result adapters. Do not execute arbitrary LLM Python in the application process. If it is out of scope, remove the claim from documentation and API contract.  
**Acceptance tests:** Either tested safe Python-alpha support exists or documentation/API explicitly state Fast Expression only.

### P1-08 — AST validation is mistaken for full alpha validation

**Area:** `generator/pipeline.py`.  
**Problem:** Parsing and checking operator existence/arity do not establish field semantic validity, unit compatibility, data availability, causal availability, or research validity.  
**Fix:** Separate validation stages: syntax, operator/function signature, field catalog membership, type/unit compatibility, parameter range, BRAIN constraint validation, and preflight simulation validation. Emit structured error codes and preserve all validation results with candidate lineage. Never represent syntactic validity as statistical validity.  
**Acceptance tests:** Negative tests cover wrong units, wrong input types, wrong dataset, invalid periods, unsupported fields, and malformed expressions; each fails before simulation.

### P1-09 — Operator attribution uses naive mean Sharpe

**Area:** `learning/memory.py`, `learning/loop.py`.  
**Problem:** Operators can appear good because they co-occur with a particular researcher, datafield, region, expression complexity, or regime. Raw average Sharpe is not causal attribution.  
**Fix:** Treat single-operator summaries as descriptive only. Add matched/controlled experiments or regression/stratified analyses that control for confounders, minimum sample sizes, uncertainty, and multiple testing. Track co-occurring operator patterns and field-family context. Ensure the LLM sees the evidence and limitations.  
**Acceptance tests:** Stats include sample counts and uncertainty; sparse categories are marked inconclusive; a synthetic confounded test demonstrates raw mean Sharpe is not called causal evidence.

### P1-10 — Winner/loser selection ignores research multiplicity and uncertainty

**Area:** `learning/memory.py`, `evaluation/metrics.py`.  
**Problem:** Top quartile and threshold rules do not account for multiple testing, confidence intervals, strategy count, or repeated experimentation.  
**Fix:** Add trial-count tracking and a selection protocol appropriate to data available: walk-forward validation, bootstrap/confidence intervals, and where justified Probabilistic/Deflated Sharpe or other multiple-testing controls. Set minimum history/sample-size gates. Don't imply these tests eliminate all selection bias.  
**Acceptance tests:** Every result records trial family/count and validation method; low-sample candidates are not declared statistically superior; tests cover selection thresholds.

### P1-11 — Model/researcher leaderboard is not a fair model benchmark

**Area:** researcher metrics/leaderboard API.  
**Problem:** Aggregate Sharpe/counts alone confound quality with the number and difficulty of attempts, candidate validity, provider cost and latency.  
**Fix:** Record per-attempt denominator, task allocation, prompt version, dataset/region, tokens, cost, latency, parse/validation failures, simulation failures and evaluation provenance. Compare models using matched tasks or stratified metrics, cost-quality frontiers and uncertainty.  
**Acceptance tests:** Leaderboard discloses sample sizes and cost; unmatched populations are not silently ranked as directly comparable.

### P1-12 — Researcher-specific persistent memory is shallow

**Area:** `learning/memory.py`, database models.  
**Problem:** Per-researcher persona/configuration exists, but memory of tested hypotheses, failed strategies and context-specific outcomes is not sufficiently independent and structured.  
**Fix:** Store research episodes by researcher, prompt version, dataset/region, hypothesis family, candidate, evaluation provenance and follow-up outcome. Use retrieval with time/version filters. Keep a shared knowledge layer separate from private researcher strategies.  
**Acceptance tests:** Each researcher can retrieve its own prior experiments; retrieval filters prevent holdout contamination; a second researcher cannot have its prompt silently changed by a scoped update.

### P1-13 — Prompt version persistence is not the same as prompt evolution validation

**Area:** prompt library, learning.  
**Problem:** Evolving few-shot examples and anti-patterns does not prove a newer prompt performs better.  
**Fix:** Immutable version IDs and checksums; benchmark new strategy against baseline using comparable allocations; promote only when it passes predeclared criteria on validation data and does not violate cost/reliability/diversity constraints. Preserve rollback.  
**Acceptance tests:** An untested prompt remains experimental; promotion requires recorded validation; rollback restores prior prompt behavior.

### P1-14 — Alpha diversity is too downstream

**Area:** generation, evaluation, portfolio curation.  
**Problem:** Low-correlation curation exists, but diversity is not adequately enforced during research allocation and candidate generation.  
**Fix:** Track expression structural similarity, hypothesis family, datafield family, dataset, region, universe, operator family and return-series correlation. Allocate quotas toward under-explored categories while retaining an exploration budget. Use real common-period PnL for correlation and document missing-data policy.  
**Acceptance tests:** A generation round produces auditable family diversity; correlation is computed only on aligned real series; near-duplicate candidates are identified.

### P1-15 — Simulation settings are too hard-coded

**Area:** `orchestrator/researchers.py`, simulation request construction.  
**Problem:** Session-level region/universe and defaults such as neutralization/decay are reused across candidates, constraining diversity and potentially confusing defaults with selected research policy.  
**Fix:** Represent simulation settings as validated experiment configuration and persist them with each candidate. Create explicit experiment allocations over eligible configurations. Defaults must be visible, configurable and tagged as defaults. Do not expand settings blindly if BRAIN rate limits or API rules disallow it.  
**Acceptance tests:** The settings used in a simulation match the candidate's stored configuration; tests verify at least two eligible configurations can be evaluated where supported.

### P1-16 — “Submission grade” is an internal heuristic, not proof of final submitability

**Area:** `evaluation/metrics.py`, API/UI labels.  
**Problem:** Candidate status is decided by configured internal thresholds. Passing those checks does not by itself prove BRAIN platform acceptance or a final portfolio is eligible for submission.  
**Fix:** Name and document internal status as `internal_candidate_pass` or similar. If available and authorized, run the applicable BRAIN-side checks and store their actual results separately. Keep validation, platform eligibility, and final user submission as distinct states.  
**Acceptance tests:** UI/API distinguish internal threshold pass from verified platform status; no unsupported claims of guaranteed acceptance.

**Verification:** COMPLETE ✅ — `docs/remediation/reports/p1_16_final_verification_report.md`.

## P2 — reliability, reproducibility, maintainability, operational correctness

### P2-01 — Frontend can evaluate before BRAIN simulations finish

**Area:** `frontend/`, research APIs.  
**Problem:** Workflow can enqueue simulations and immediately call evaluate; asynchronous BRAIN completion/backfill may not be ready.  
**Fix:** Frontend subscribes/polls a persistent job state machine. Backend exposes explicit statuses such as `QUEUED`, `SUBMITTED`, `SIMULATING`, `RESULT_BACKFILL`, `EVALUABLE`, `EVALUATED`, `FAILED`. Trigger evaluation only when required artifacts are available. Do not depend on repeated manual clicks.  
**Acceptance tests:** Simulated delayed BRAIN response completes without manual re-click; transient failures are recoverable; status transitions are tested.

**Verification:** COMPLETE ✅ — `docs/remediation/reports/p2_01_final_verification_report.md`.

### P2-02 — API module contains too much business/research logic

**Area:** `backend/src/alpha_harness/api/research.py`.  
**Problem:** HTTP routing, BRAIN synchronization, synthetic fallback, evaluation and robustness logic are coupled in a large module.  
**Fix:** Move business operations into application services (`EvaluationService`, `RobustnessService`, `ExperimentOrchestrator`, etc.) and keep route handlers thin. Make this incremental, not a big-bang rewrite, and only after behavioral tests exist.  
**Acceptance tests:** Service-level tests run without HTTP routing; route tests verify schemas/status codes; behavior remains unchanged except intentional fixes.

**Verification:** COMPLETE ✅ — `docs/remediation/reports/p2_02_final_verification_report.md`.

### P2-03 — Researcher failures may be converted into missing output

**Area:** `orchestrator/researchers.py`.  
**Problem:** Concurrent dispatch uses `return_exceptions=True`; exception paths are logged but need durable per-attempt accounting so a failed model does not simply disappear.  
**Fix:** Persist a `ResearcherAttempt` record for every call with status, provider, model, error class, retries, latency, token/cost details when available, and output validation. Make partial success visible. Retry only transient, idempotent failures with bounded backoff.  
**Acceptance tests:** Provider exception creates a failed attempt record and visible UI state; sibling researchers still finish; retries do not duplicate candidate rows.

**Verification:** COMPLETE ✅ — `docs/remediation/reports/p2_03_final_verification_report.md`.

### P2-04 — BRAIN result backfill failure is not a terminally modeled state

**Area:** evaluation API/backfill integration.  
**Problem:** Backfill failures are logged, but candidates can proceed into fallback behavior instead of remaining explicitly blocked on real data.  
**Fix:** Track backfill status, attempts, last error, next retry and required artifact availability. Backoff on temporary errors; stop/review after a defined retry budget; never replace missing data with fabricated returns.  
**Acceptance tests:** Backfill failure leaves candidate unevaluated; a later successful backfill resumes evaluation; terminal failure is visible and auditable.

**Verification:** IMPLEMENTED — **UNVERIFIED** pending backend and frontend test-suite evidence in `docs/remediation/reports/p2_04_final_verification_report.md`.

### P2-05 — Reproducibility is compromised by randomized synthetic seeds

**Area:** synthetic fallback/robustness code.  
**Problem:** `hash(...)`-based random seeds may vary between Python processes and are being used in pathways that should not exist in production.  
**Fix:** First remove synthetic metrics from production. For remaining test fixtures, derive stable seeds from a stable hash such as SHA-256 of canonical identifiers/configuration. Persist all random seeds and experiment config.  
**Acceptance tests:** Same fixture/config yields identical output across fresh processes; distinct configs have distinct stable seeds.

### P2-06 — Composite score is hard-coded without calibrated justification

**Area:** `evaluation/metrics.py`.  
**Problem:** A weighted formula ranks Sharpe, Fitness, decay and turnover using fixed weights. It may be a useful heuristic, but no calibration/ablation evidence is demonstrated.  
**Fix:** Version the score definition, expose components separately, define weights in validated configuration, and benchmark it against predeclared portfolio objectives. Never hide hard thresholds in code. Report score version on every row.  
**Acceptance tests:** Score can be reproduced from stored components and version; config validation prevents nonsensical weights; ablation results are documented.

### P2-07 — State transitions and concurrency need stronger guarantees

**Area:** orchestrator, BRAIN batch engine, database.  
**Problem:** Research steps are asynchronous and result arrival/backfill can race with evaluation or learning.  
**Fix:** Implement guarded state transitions, transaction boundaries, idempotency keys, locking/unique constraints where needed, and a rule that learning only runs over a frozen, fully evaluated generation.  
**Acceptance tests:** Concurrent duplicate requests do not create duplicate simulations/evaluations; learning refuses incomplete generations; restart/resume is tested.

### P2-08 — Rate-limit and retry behavior need auditable policies

**Area:** LLM service and BRAIN client.  
**Problem:** Third-party API failures/rate limits are part of the workflow, but the audit has not established complete bounded retry, backoff, quota and circuit-breaker behavior.  
**Fix:** Verify provider-specific policies, add bounded exponential backoff with jitter for transient status codes, `Retry-After` handling, global/per-provider concurrency limits, and explicit terminal errors. Do not blindly retry non-idempotent submission requests.  
**Acceptance tests:** Mock 429/5xx/timeouts; retry budget and delays work; duplicate BRAIN submissions are prevented.

### P2-09 — Observability needs stage-level research lineage

**Area:** backend services/logging/database.  
**Problem:** Logs alone are insufficient to reconstruct which input/prompt/config produced a candidate and why it failed.  
**Fix:** Propagate `experiment_id`, `generation_id`, `candidate_id`, `researcher_attempt_id`, `prompt_version_id`, `simulation_id`, provenance and correlation IDs. Store structured stage events and errors. Avoid logging secrets or sensitive response headers.  
**Acceptance tests:** An investigator can reconstruct a candidate from API request through final evaluation and identify every failed stage.

### P2-10 — End-to-end automated tests are not sufficiently demonstrated

**Area:** backend/frontend/CI.  
**Problem:** Repository documentation reports manual verification, but the audit did not establish a full automated regression suite covering real/synthetic separation, OOS leakage, async states and strategy routing.  
**Fix:** Add unit tests, DB integration tests, mocked-provider/BRAIN integration tests, contract tests, and frontend API lifecycle tests. Add a small deterministic end-to-end CI workflow that requires no real credentials; keep optional live smoke tests separate.  
**Acceptance tests:** CI fails on synthetic PnL entering production evaluation, protected OOS leakage, category-routing regression, and premature evaluation.

### P2-11 — Security/configuration audit is still required

**Area:** environment configuration, API, LLM/BRAIN clients, launcher.  
**Problem:** This is a required audit area, but the prior source review did not conclusively establish every authentication, secret-storage, CORS, input-size and network-boundary property. Treat it as unverified rather than claiming a proven vulnerability.  
**Fix:** Confirm secrets are environment-managed and never persisted/logged; validate request sizes and parameters; apply appropriate authentication to non-local deployments; restrict CORS; ensure prompt/alpha expressions cannot execute arbitrary local code; redact credentials in errors.  
**Acceptance tests:** Secret-scanning, configuration tests, auth/CORS tests and hostile-input tests pass.

### P2-12 — Storage and schema evolution need reproducibility guarantees

**Area:** `db/models.py`, DB initialization/migrations.  
**Problem:** The database holds useful candidate/evaluation state, but the audit did not establish robust versioned migrations and immutable experiment metadata across SQLite/PostgreSQL.  
**Fix:** Verify migration strategy; avoid runtime schema assumptions; add indexes/uniqueness for idempotency; version candidate schema, catalog snapshots, prompt versions, score definitions and experiment configurations.  
**Acceptance tests:** Fresh DB setup and upgrade from prior schema pass in CI; uniqueness constraints prevent duplicates; stored experiments survive service restart.

### P2-13 — Data and model comparison must disclose missingness and denominators

**Area:** leaderboard and evaluation APIs.  
**Problem:** Averages can mislead if candidates lack PnL, have different time spans, or differ in failed attempts.  
**Fix:** Display `N attempted`, `N valid`, `N simulated`, `N real-series evaluated`, period covered, provenance counts and missingness. Never treat absent values as zero or silently remove failures from the denominator without documenting the policy.  
**Acceptance tests:** Leaderboard response exposes denominators and period; metrics are consistent when candidates are missing data.

## P3 — documentation and polish

### P3-01 — README/changelog overstate completion

**Area:** `README.md`, `RESEARCH_ENGINE_CHANGELOG.md`, `DESIGN.md`.  
**Problem:** Product language implies rigorous closed-loop learning/OOS robustness while source contains synthetic robustness and evaluation fallback paths.  
**Fix:** Update claims only after P0/P1 acceptance tests pass. Clearly label implemented, partial, planned and synthetic/demo behavior.  
**Acceptance tests:** Every major advertised capability maps to code and an automated test or is explicitly marked partial/unavailable.

### P3-02 — Error messages and result status need consistent vocabulary

**Area:** API schemas/UI.  
**Problem:** Distinct states (pending, unavailable, invalid, failed, synthetic, complete) must not collapse to generic success.  
**Fix:** Standardize enumerated job/candidate/evaluation/provenance states and display actionable reasons.  
**Acceptance tests:** API schema tests cover every state; frontend never renders unavailable data as a successful score.

---

# 2. Recommended dependency order

1. P0-01 remove synthetic PnL from production evaluation.
2. P0-02 make provenance explicit and truthful.
3. P0-03 seal the OOS protocol; prevent holdout data from influencing learning.
4. P0-04 replace synthetic robustness with actual BRAIN perturbation jobs.
5. P2-04 and P2-07 model backfill/state transitions safely.
6. P0-05 build the experiment/job orchestrator and autonomous API.
7. P1-08 strengthen semantic preflight validation.
8. P1-04/P1-05/P1-06 connect versioned catalogs to experiment design and generation.
9. P1-15 make candidate simulation settings explicit and diverse.
10. P1-01/P1-02 implement evidence-driven reflection and repair category routing.
11. P1-09/P1-10 introduce statistically defensible attribution and selection controls.
12. P1-03/P1-11/P1-13 build fair prompt/researcher benchmarking.
13. P1-12 implement persistent researcher memory.
14. P1-14 improve upstream diversity and portfolio construction.
15. P1-16 separate internal quality gates from true BRAIN submission state.
16. P2-01/P2-03/P2-08/P2-09 improve async behavior, failure records, retries and observability.
17. P2-06/P2-10/P2-12 improve metric configuration, tests and migrations.
18. P2-11 complete security/configuration audit.
19. P3-01/P3-02 update documentation and state vocabulary.
20. Re-run the complete requirement matrix; do not claim completion from code changes alone.

---

# 3. Copy-paste prompt for a coding LLM

```text
ROLE
You are the lead backend engineer, quantitative researcher, ML systems engineer, and test engineer responsible for remediating this existing repository. You must preserve working functionality and make evidence-driven, minimal changes.

REPOSITORY
https://github.com/ayushbhaskar902-cpu/Project-_v2

PROJECT GOAL
Build a reliable, self-learning multi-LLM alpha creation engine for WorldQuant BRAIN. It should generate candidates, validate them, run real BRAIN simulations, evaluate them on a statistically defensible protocol, run real robustness tests where supported, learn from trustworthy evidence, compare model/prompt strategies, and expose an autonomous API/dashboard.

NON-NEGOTIABLE RULES
1. Work against the actual current checkout and inspect `git status`, current branch, and recent commits before editing. Do not assume this remediation file perfectly describes the current branch.
2. Do not rewrite the repository, change frameworks, or refactor unrelated code.
3. Resolve ONE issue at a time. Start with P0-01 only. Do not silently continue to P0-02 or another issue until the current issue's tests and review are complete and I authorize the next issue.
4. Before editing, trace the real call path across API, service, DB, frontend and tests. List exact files/functions and explain the proposed minimal change.
5. Add or update regression tests before or alongside the fix. A code change is not complete until relevant tests run and pass.
6. Never claim a test, live BRAIN request, or full application run succeeded unless you actually executed it. Distinguish PASS, FAIL, BLOCKED and NOT RUN.
7. Never use fabricated metrics as real research evidence. Never claim synthetic data is live BRAIN data.
8. Never expose the final holdout/OOS period to prompt selection, LLM reflection, model choice, strategy updates, hyperparameter tuning, or routine leaderboards. If current BRAIN data interfaces cannot support a genuinely sealed holdout, state the limitation honestly and implement the strongest valid alternative without claiming a sealed OOS result.
9. Never execute arbitrary generated Python or shell code in the application process. Respect WorldQuant BRAIN API rules, credentials, rate limits and terms.
10. Keep changes reproducible, typed, logged and backward-compatible where practical. If a schema change is necessary, include a migration or safe migration plan.
11. If a requirement cannot be implemented because platform/API access is unavailable, create an explicit blocked/unavailable state and explain the missing dependency. Do not replace it with a mock that looks real.
12. Do not invent function names, datafields, operators, APIs, BRAIN behavior, or test results. Verify them from repository code/docs or explicitly mark them unverified.

SOURCE AREAS TO INSPECT
- `backend/src/alpha_harness/api/research.py`
- `backend/src/alpha_harness/orchestrator/researchers.py`
- `backend/src/alpha_harness/evaluation/metrics.py`
- `backend/src/alpha_harness/learning/loop.py`
- `backend/src/alpha_harness/learning/memory.py`
- `backend/src/alpha_harness/generator/pipeline.py`
- `backend/src/alpha_harness/brain/`
- `backend/src/alpha_harness/db/models.py`
- `backend/src/alpha_harness/state.py`
- frontend research workflow and its API client
- tests, migrations, configuration, README, and changelog

ISSUE BACKLOG
P0-01: Remove the synthetic daily-PnL fallback from all production evaluation and selection paths. Missing/short real series must make a candidate unevaluable, not produce a score.
P0-02: Add truthful, persisted data provenance (`REAL_BRAIN_SERIES`, `BRAIN_SUMMARY_ONLY`, `SYNTHETIC_TEST_FIXTURE`, `MISSING` or an equally precise enum). Ensure API responses, DB records, leaderboards and learning agree. Remove any hard-coded real-data claim.
P0-03: Make IS/validation/final holdout separation real. Prevent holdout leakage into learning, prompt/model selection and tuning; persist split policy and exposure/audit records.
P0-04: Replace synthetic random robustness degradation with real BRAIN re-simulations for supported perturbations; mark unsupported/unrun tests honestly.
P0-05: Add a persistent experiment/job orchestrator and an autonomous prompt+parameters API that owns generation → simulation → data availability → evaluation → learning/curation with safe resumable states.
P1-01: Implement evidence-grounded LLM reflection, with structured inputs/outputs and evidence references; keep claims as hypotheses until tested.
P1-02: Fix PromptStrategy category/version routing so the next generation actually consumes the intended updated strategy.
P1-03: Add an immutable prompt version registry and fair prompt-level benchmarking/leaderboard.
P1-04: Make dataset selection/catalog eligibility part of explicit experiment design.
P1-05: Integrate a versioned datafield catalog into prompts and pre-simulation validation.
P1-06: Integrate authoritative/versioned operator signatures, arity and type metadata.
P1-07: Decide whether Python alpha generation is in scope; implement safely in isolation or state clearly it is unsupported.
P1-08: Separate syntax validation from semantic/type/unit/dataset/BRAIN preflight validation.
P1-09: Replace naive causal claims about operator performance with controlled/stratified analyses and uncertainty.
P1-10: Track trial counts/multiple testing and use justified statistical validation gates; do not infer significance from rank alone.
P1-11: Make researcher/model comparisons fair using matched or stratified tasks, sample sizes, cost, latency, validity, provenance and uncertainty.
P1-12: Add persistent researcher-specific experiment memory and prevent cross-category contamination.
P1-13: Test prompt/strategy changes against baselines and promote only with predeclared validation criteria; support rollback.
P1-14: Enforce diversity during generation as well as portfolio curation (hypothesis, field family, operators, dataset, region, universe and measured return correlation).
P1-15: Make simulation settings explicit, validated and persisted per candidate rather than relying on hidden fixed defaults.
P1-16: Distinguish internal quality thresholds from actual BRAIN eligibility/submission status.
P2-01: Fix frontend/backend asynchronous lifecycle so evaluation starts only when required BRAIN data is ready; no manual re-click loop.
P2-02: Incrementally extract business logic from large route handlers into services after tests protect behavior.
P2-03: Persist failed LLM attempts instead of losing failures in concurrent dispatch.
P2-04: Model backfill retries and terminal failure states; missing PnL never triggers fake data.
P2-05: Use stable, persisted seeds only for explicit synthetic tests; eliminate process-randomized hash-based production behavior.
P2-06: Version composite scores/weights and benchmark the objective; disclose components.
P2-07: Add idempotency, guarded state transitions, and prevent learning on incomplete generations.
P2-08: Verify bounded rate-limit/retry behavior for providers and BRAIN, including safe retry rules.
P2-09: Add stage-level correlation IDs and auditable experiment lineage.
P2-10: Add automated unit, integration, contract and small end-to-end regression tests.
P2-11: Audit secrets, authentication, CORS, input limits, error redaction and generated-expression safety; report unverified items honestly.
P2-12: Verify DB migrations, uniqueness constraints, indexes and reproducible experiment metadata across supported DB engines.
P2-13: Expose metric denominators, missingness, date ranges and provenance in leaderboards.
P3-01: Correct README/changelog claims to reflect tested functionality only.
P3-02: Standardize status and error states across API and UI.

MANDATORY WORKFLOW FOR THE CURRENT ISSUE
A. Inspect current repository state and relevant source/test files.
B. Confirm whether P0-01 still exists on the current branch. If already fixed, verify the fix and tests before marking it done; then ask me before advancing.
C. Draw the exact data flow from BRAIN series retrieval to `EvaluationEngine`, DB persistence, leaderboards, learning and API response.
D. Propose the smallest safe implementation plan and list files to change. Do not edit until you have presented this plan and I approve it.
E. After approval, implement P0-01 only. Do not bundle unrelated refactors.
F. Add tests for at least: no series, fewer than the minimum number of points, exact minimum boundary, valid real series, and confirmation that unavailable candidates cannot enter learning/leaderboards/curation.
G. Run relevant tests, type/lint checks if configured, and inspect `git diff`. Report commands and actual outputs.
H. Check the behavior through the public API/service path, not only a helper unit test.
I. Update a remediation ledger with issue ID, status, files changed, behavior before/after, tests run, results, risks, and next dependency.
J. Stop after P0-01. Do not implement the next issue without my explicit approval.

OUTPUT FORMAT FOR EACH ISSUE
1. Issue ID and title
2. Current-branch confirmation with file/function evidence
3. Root cause and impact
4. Minimal design proposal
5. Files expected to change
6. Tests to add/update
7. Risks and migration/compatibility considerations
8. Wait for my approval before editing
9. After implementation: exact diff summary, commands executed, test results, and remaining limitations
10. Explicit verdict: FIXED / PARTIALLY FIXED / BLOCKED / NOT FIXED

QUALITY BAR
Prefer correctness and honest failure states over fabricated completeness. Do not optimize alpha metrics by repeatedly consulting the final holdout. Do not call an alpha production-grade or submittable merely because a heuristic threshold passed. Do not mark an issue complete until tests verify the behavior end-to-end at the relevant layer.

BEGIN NOW WITH P0-01 ONLY. Inspect and report first; do not edit files until I approve the implementation plan.
```

---

# 4. How to use this prompt

1. Open the actual repository in your coding LLM (for example, your local checkout in VS Code/Codex).
2. Give it this prompt and the repository workspace. If the tool cannot access the repo directly, open the repo locally or provide the relevant files; do not rely on a repository URL alone.
3. Let it inspect and propose the P0-01 plan. Review the files/tests it names before approving.
4. Make it stop after P0-01, verify the diff and test results, then authorize P0-02. Continue in dependency order.
5. Keep the remediation ledger under version control so a later LLM session can continue without assuming prior work happened.
