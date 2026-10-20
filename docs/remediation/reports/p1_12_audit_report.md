# P1-12 Audit & Remediation Plan: Researcher-Specific Persistent Memory

**Issue ID:** P1-12  
**Canonical Title:** Researcher-specific persistent memory is shallow  
**Status:** 🔍 AUDITED — PENDING REVIEW & APPROVAL  
**Affected Area:** `alpha_harness.learning.memory`, `alpha_harness.db.models`, `alpha_harness.orchestrator.researchers`, `alpha_harness.learning.loop`  

---

## 1. Executive Summary & Problem Analysis

In the current architecture, multi-LLM researcher personas exist (`ResearchResearcher` configuring `model_ref`, `strategy_focus`, `system_prompt`, `temperature`, `target_dataset_id`), and candidates are tagged with `researcher_id`. However:
1. **No Structured Episodic Memory:** There is no dedicated entity or schema storing structured research episodes for an individual researcher persona. Researchers track only shallow scalar aggregates (`total_generated`, `total_valid`, `avg_sharpe`, `best_sharpe`).
2. **Context-Specific Outcomes Are Lost:** Detailed contextual memory—what specific hypotheses were tested, what failed strategies or syntax errors were encountered, which datasets and operator patterns succeeded or hit saturation for that specific researcher—is discarded between generation cycles.
3. **Absence of Memory Retrieval Mechanism:** When an LLM researcher generates new candidates (`generate_batch`), it receives only generic session prompts and category few-shots, without access to its own historical experiments, negative lessons, or prior hypothesis progressions.
4. **Shared vs. Private Memory Conflation:** The system lacks a clear architectural separation between:
   - **Shared Knowledge Layer:** Global operator efficacy, cross-researcher empirical statistical baselines, dataset coverage facts.
   - **Private Researcher Memory:** Persona-specific hypothesis journals, persona-specific failed strategies, persona-scoped prompt evolution, and experiment histories.
5. **Holdout Contamination Risk:** In the absence of strict retrieval policies and filters, ad-hoc memory retrieval could inadvertently expose holdout metrics or post-validation outcomes to the generative prompt.

---

## 2. Canonical Specification & Scope Boundaries

### Canonical Problem
> Per-researcher persona/configuration exists, but memory of tested hypotheses, failed strategies and context-specific outcomes is not sufficiently independent and structured.

### Canonical Fix
> Store research episodes by researcher, prompt version, dataset/region, hypothesis family, candidate, evaluation provenance and follow-up outcome. Use retrieval with time/version filters. Keep a shared knowledge layer separate from private researcher strategies.

### Canonical Acceptance Tests
> Each researcher can retrieve its own prior experiments; retrieval filters prevent holdout contamination; a second researcher cannot have its prompt silently changed by a scoped update.

### Strict Scope Boundaries
- **P1-12 (In Scope):**
  - Structured `ResearchEpisode` model / memory store capturing researcher ID, session ID, prompt version, dataset/region, hypothesis family, candidate expression, AST validation/simulation status, evaluation provenance, validation performance, and follow-up reflection notes.
  - Persona-specific episodic memory retrieval API with filtering by time, generation, version, dataset, and hypothesis family.
  - Strict isolation: Private researcher episodes are inaccessible to peer researchers unless explicitly graduated to the shared knowledge layer.
  - Strict holdout isolation: Memory retrieval queries exclusively filter for validation/IS metrics; holdout partition data is quarantined and never loaded into episode context.
  - Scoped prompt safety: Researcher prompt updates and memory writes are strictly partitioned by `researcher_id` and category; updates to researcher $A$ never mutate or overwrite researcher $B$'s prompt or state.
- **P1-13 (Out of Scope):** Prompt version shadow testing, pre-deployment tournament validation, rollback criteria.
- **P1-14 (Out of Scope):** Upstream diversity enforcement, syntactic clustering, cross-series return correlation quotas.
- **P1-15 (Out of Scope):** Candidate-level simulation setting mutation and neutralization/decay grid sampling.

---

## 3. Current Architecture & Code Audit

### 3.1 `backend/src/alpha_harness/db/models.py`
- `ResearchResearcher`:
  - Contains configuration: `session_id`, `model_ref`, `label`, `strategy_focus`, `system_prompt`, `temperature`, `target_dataset_id`, `dataset_allocation_policy`.
  - Contains scalar summary counters: `total_generated`, `total_valid`, `total_simulated`, `total_accepted`, `avg_sharpe`, `avg_fitness`, `best_sharpe`, `total_tokens`.
  - **Defect:** Does not have relationship or linkage to an episodic memory log. Does not preserve context-specific outcomes or hypothesis lineage.
- `ResearchCandidate`:
  - Fields: `researcher_id`, `session_id`, `prompt_strategy_id`, `prompt_version_id`, `hypothesis`, `expression`, `fields_used`, `status`, `rejection_reason`.
  - **Defect:** Candidate records represent raw generation items, not structured research episodes containing synthesis, hypothesis progression, or follow-up outcomes.
- `OperatorEfficacyRecord`:
  - Stores aggregated operator lifts. Represents part of the *shared* layer, but is currently the only persistent memory table.

### 3.2 `backend/src/alpha_harness/learning/memory.py`
- `ResearchMemory`:
  - Manages `get_session_performance_split`, `get_eligible_candidates`, `compute_operator_stats`, and `persist_operator_stats`.
  - **Defect:** All memory queries are session-wide or category-wide. There are zero researcher-specific memory retrieval methods or episodic storage methods.

### 3.3 `backend/src/alpha_harness/orchestrator/researchers.py`
- When researchers are configured and run, prompts are populated using session-level or category-level templates (`PromptStrategy` / `PromptVersion`).
- **Defect:** No episode retrieval is injected into the context window for the researcher. A researcher running 5 iterations repeats hypotheses or falls into known syntax/expression traps encountered in iterations 1–4.

---

## 4. Proposed Solution Design

### 4.1 Data Model: `ResearchEpisode`
Add a dedicated episodic memory table to SQLite (`backend/src/alpha_harness/db/models.py`):
```python
class ResearchEpisode(Base):
    """Structured research episode tracking an individual researcher's experiment history.
    
    Stores hypothesis progression, candidate expressions, evaluation outcomes,
    failure classifications, and reflection notes scoped to a specific researcher.
    """
    __tablename__ = "research_episode"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("research_session.id", ondelete="CASCADE"), index=True)
    researcher_id: Mapped[int] = mapped_column(ForeignKey("research_researcher.id", ondelete="CASCADE"), index=True)
    candidate_id: Mapped[int | None] = mapped_column(ForeignKey("research_candidate.id", ondelete="SET NULL"), nullable=True, index=True)
    prompt_version_id: Mapped[int | None] = mapped_column(ForeignKey("prompt_version.id", ondelete="SET NULL"), nullable=True, index=True)
    
    # Context & Environment Metadata
    generation: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    dataset_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    region: Mapped[str | None] = mapped_column(String(32), nullable=True)
    universe: Mapped[str | None] = mapped_column(String(32), nullable=True)
    hypothesis_family: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    hypothesis: Mapped[str | None] = mapped_column(Text, nullable=True)
    expression: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # Execution & Evaluation Provenance
    funnel_stage_reached: Mapped[str] = mapped_column(String(32), default="GENERATED")  # GENERATED, AST_VALID, SIMULATED, EVALUATED
    provenance: Mapped[str | None] = mapped_column(String(32), default="LOCAL_SIMULATION")
    outcome: Mapped[str] = mapped_column(String(32), default="PENDING")  # SUCCESS, REJECTED_SYNTAX, REJECTED_SIMULATION, STATISTICAL_LOSER, STATISTICAL_WINNER
    
    # Strictly Non-Holdout Metrics (P0-03 Holdout Isolation Safeguard)
    validation_sharpe: Mapped[float | None] = mapped_column(Float, nullable=True)
    validation_fitness: Mapped[float | None] = mapped_column(Float, nullable=True)
    validation_turnover: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    # Structured Failure & Success Analysis
    failure_category: Mapped[str | None] = mapped_column(String(64), nullable=True)  # SYNTAX_ERROR, PREFLIGHT_FAIL, LOW_SHARPE, HIGH_TURNOVER, HIGH_DECAY
    failure_details: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=dict)
    lessons_learned: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # Privacy / Sharing Flag
    is_shared_knowledge: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow, index=True)
```

### 4.2 Researcher-Specific Episodic Memory Service
In `backend/src/alpha_harness/learning/memory.py`:
- `record_research_episode(...)`:
  - Creates and commits a `ResearchEpisode` entry upon candidate evaluation or failure.
  - Automatically captures failure reasons (e.g. AST parsing errors, simulation execution failures, statistical rejection from P1-10 multiplicity split).
  - Explicitly prevents writing holdout metrics into the episode.
- `get_researcher_episodes(...)`:
  - Retrieves episodes filtered strictly by `researcher_id` (or shared knowledge if `include_shared=True`).
  - Supports filters: `dataset_id`, `region`, `hypothesis_family`, `min_generation`, `max_generation`, `outcomes`, `limit`.
  - Enforces isolation: Researcher $B$ cannot view Researcher $A$'s private episodes (`is_shared_knowledge=False`).
- `build_researcher_memory_context(...)`:
  - Formats retrieved episodes into a structured LLM context section:
    1. **Recent Tested Hypotheses & Outcomes:** Highlighting what was tried and its empirical result.
    2. **Negative Lessons (Pitfalls to Avoid):** Specific syntax errors, high-turnover traps, or unviable expressions generated by this researcher.
    3. **Successful Discoveries:** Statistically validated winning patterns discovered by this researcher.
  - Respects token budget and window limits.

### 4.3 Shared Knowledge Layer vs. Private Memory Layer
- **Shared Knowledge Layer:**
  - Accessible to all researchers.
  - Includes `OperatorEfficacyRecord` (statistically deconfounded operator lift, P1-09).
  - High-confidence cross-cutting discoveries explicitly promoted to `is_shared_knowledge = True`.
- **Private Researcher Layer:**
  - Scoped strictly to `researcher_id`.
  - Persona-specific system prompts and updates: Scoped prompt updates to Researcher $A$ (e.g. updating system prompt or strategy focus) never mutate Researcher $B$.
  - Hypothesis scratchpad and failure logs remain confidential to the persona.

### 4.4 Verification & Holdout Quarantine
- Strict assertion in `record_research_episode` and `get_researcher_episodes`:
  - Ensure zero access to `holdout_sharpe`, `holdout_returns`, or `holdout_fitness`.
  - Test verifying that changing holdout data produces 0 difference in episode records or retrieved memory context.

---

## 5. Acceptance Criteria Checklist for P1-12

1. **Structured Episodic Storage:**
   - Research episodes are persisted in the database linked to `researcher_id`, `session_id`, `prompt_version_id`, `candidate_id`, `dataset_id`, `region`, `hypothesis_family`, `provenance`, and `outcome`.
2. **Researcher Retrieval Isolation:**
   - Each researcher can query its own past experiments.
   - Researcher $A$ cannot retrieve Researcher $B$'s private research episodes (`is_shared_knowledge=False`).
   - Shared knowledge is queryable across all researchers without leaking private episodes.
3. **Holdout Quarantine:**
   - Episode records and retrieval contexts contain exclusively in-sample, validation, or qualitative metadata.
   - Modifying holdout records does not affect episode logs, retrieval queries, or memory context.
4. **Scoped Prompt Mutation Protection:**
   - Updating or evolving Researcher $A$'s configuration or system prompt does not mutate, overwrite, or affect Researcher $B$'s configuration or prompt.
5. **Context Generation:**
   - Memory context builder produces structured markdown summaries of prior successes, failures, and hypotheses suitable for inclusion in researcher prompts.
6. **Backward Compatibility & Regression Suite:**
   - Existing DB migrations handle the new table seamlessly across SQLite.
   - Full test suite passes without regressions.

---

## 6. Request for User Approval

The P1-11 implementation is verified with all 260 tests passing. Per the workflow instructions, this P1-12 audit is submitted for review and approval before any code or schema changes are initiated.
