# P1-01 Architectural and Code Audit Report

**Remediation Item:** P1-01 — Learning cycle is heuristic rather than actual LLM reflection  
**Status:** **AUDITED — AWAITING APPROVAL**  
**Audit Date:** 2026-10-11  

---

## 1. Executive Summary

A comprehensive architectural and source code audit of Alpha Harness was performed to evaluate whether the self-learning research cycle executes genuine, evidence-grounded LLM reflection or relies on heuristic rules and handwritten templates.

### Key Finding
**The current learning cycle does NOT invoke an LLM at all.**  
While `LLMService` is passed to [`SelfLearningEngine.__init__`](backend/src/alpha_harness/learning/loop.py) and assigned to `self.llm`, `self.llm` is **never called anywhere in `learning/loop.py` or `learning/memory.py`**. The entire reflection and learning cycle is an entirely deterministic, hardcoded Python procedure that:
1. Partitions candidates by a static percentile cutoff (top 25% by composite score).
2. Performs naive regex parsing to tally Fast Expression operator frequencies.
3. Formulates insights using **static, hardcoded string templates** (e.g., verbatim output: `"Common causes: unnormalized momentum or excessive lookback without group neutralization."`).
4. Generates a new `PromptStrategy` by mechanically copying the top 3 winner hypotheses and top 3 loser expressions with hardcoded flaw labels (`"High turnover or decay"`).

Crucially, **no qualitative quantitative synthesis, economic reasoning, hypothesis refinement, or root cause diagnostic is ever performed by a language model**.

However, several foundational components are already robust and compliant:
- Candidate performance queries in [`ResearchMemory`](backend/src/alpha_harness/learning/memory.py) strictly enforce data verification (`is_verified.is_(True)`) and evaluate solely against validation partition metrics (`val_sharpe`, `val_composite_score`), strictly preserving P0-03's holdout isolation.
- The `PromptStrategy` and `ResearchInsight` database schemas are already provisioned with appropriate fields.
- The downstream generator in [`orchestrator/researchers.py`](backend/src/alpha_harness/orchestrator/researchers.py) is already wired to consume `PromptStrategy.guidance`, `examples`, and `anti_patterns`.

---

## 2. Actual Learning-Cycle Execution Flow

```
                      ┌────────────────────────────────────────┐
                      │   ExperimentRunner._phase_learn() or   │
                      │  POST /sessions/{id}/learning/cycle    │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │  SelfLearningEngine                    │
                      │  .execute_learning_cycle(session_id)   │
                      └───────────────────┬────────────────────┘
                                          │
           ┌──────────────────────────────┴──────────────────────────────┐
           ▼                                                             ▼
┌──────────────────────────────────────┐      ┌──────────────────────────────────────┐
│ 1. Memory Performance Partitioning   │      │ 2. Heuristic Operator Attribution    │
│ ResearchMemory                       │      │ ResearchMemory                       │
│ .get_session_performance_split()     │      │ .compute_operator_stats()            │
│ - Top 25% by val_composite_score     │      │ - Regex findall r"([a-z_][a-z0-9_]*)\("│
│ - Losers: Sharpe < 0.8 or decay>0.4  │      │ - Mean Sharpe across occurrences     │
└──────────────────┬───────────────────┘      └──────────────────┬───────────────────┘
                   │                                             │
                   └──────────────────────┬──────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │ 3. Handcrafted String Formatting       │
                      │ (NO LLM CALL - 100% HEURISTIC)         │
                      │ - Hardcoded template strings           │
                      │ - Fixed constants (1.5 Sharpe, 0.5 TO) │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │ 4. Persistence & Prompt Strategy Update│
                      │ - Saves ResearchInsight rows           │
                      │ - Inserts evolved PromptStrategy       │
                      │ - Increments session.current_generation│
                      └────────────────────────────────────────┘
```

### Trace Step-by-Step:
1. **Trigger:** `ExperimentRunner._phase_learn()` or `POST /api/research/sessions/{session_id}/learning/cycle` calls `SelfLearningEngine.execute_learning_cycle(session_id)`.
2. **Data Retrieval:** `ResearchMemory.get_session_performance_split()` queries `ResearchCandidate` joined with `ResearchEvaluation` for the current generation (or across all generations if none in current).
3. **Winner/Loser Partitioning:** Candidates are ordered by `val_composite_score`. The top $25\%$ ($\max(1, N // 4)$) are labelled `winners`. Candidates outside the top quartile with `sharpe < 0.8` or `sharpe_decay > 0.4` are labelled `losers`.
4. **Attribution:** `ResearchMemory.compute_operator_stats()` runs regex `re.findall(r"([a-z_][a-z0-9_]*)\s*\(", expr.lower())` over all candidate expressions and computes the arithmetic mean of their validation Sharpes.
5. **Insight Generation (The Defect):**
   - Winning summary is formed via f-string:  
     `f"Generation {gen} winning signals favored '{top_winner['hypothesis']}' with Sharpe {top_winner['sharpe']:.2f}."`
   - Failure summary is formed via static string:  
     `f"Generation {gen} observed {len(losers)} underperforming signals. Common causes: unnormalized momentum or excessive lookback without group neutralization."`
   - Fixed rules are injected: `target_sharpe_min = 1.5`, `max_allowed_turnover = 0.50`.
6. **Persistence:** `ResearchInsight` rows are inserted into SQLite. `prompt_strategy_id` and `researcher_id` are left `NULL`. No LLM prompt or response is recorded.
7. **Prompt Evolution:** A new `PromptStrategy` row is created, copying the top 3 winner dictionaries and top 3 loser expressions with static `"flaw": "High turnover or decay"`.
8. **Generation Advancement:** `ResearchSession.current_generation` is incremented by 1.

---

## 3. Confirmed Defects and Code Evidence

### Defect 1: Complete Absence of LLM Invocation in Reflection
- **Severity:** Critical
- **Status:** Confirmed defect
- **Location:** [`backend/src/alpha_harness/learning/loop.py`](backend/src/alpha_harness/learning/loop.py)
- **Code Evidence:**
  [`SelfLearningEngine.__init__`](backend/src/alpha_harness/learning/loop.py) accepts `llm: LLMService | None = None` and sets `self.llm = llm`. However, a ripgrep search for `self.llm` reveals **zero calls to `self.llm`** anywhere in the module.
- **Impact:** The engine cannot reflect on why alphas succeeded or failed, cannot propose qualitative hypothesis shifts, cannot synthesize cross-candidate patterns, and cannot adapt strategies to specific market regimes or datasets.

### Defect 2: Hardcoded Handcrafted Explanations and Fixed Metric Thresholds
- **Severity:** High
- **Status:** Confirmed defect
- **Location:** [`backend/src/alpha_harness/learning/loop.py#L115-L158`](backend/src/alpha_harness/learning/loop.py)
- **Code Evidence:**
  ```python
  # Line 140-143:
  l_summary = (
      f"Generation {gen} observed {len(losers)} underperforming signals. "
      "Common causes: unnormalized momentum or excessive lookback without group neutralization."
  )
  ...
  # Line 153:
  "max_allowed_turnover": 0.50
  ...
  # Line 192:
  "flaw": "High turnover or decay"
  ```
- **Impact:** Regardless of whether the candidates failed due to extreme correlation, low volatility, bad data fields, division by zero, or regime shifts, the system emits the exact same static diagnostic string.

### Defect 3: Lack of Reflection Persistence & Provenance
- **Severity:** Medium
- **Status:** Confirmed defect
- **Location:** [`backend/src/alpha_harness/learning/loop.py#L166-L180`](backend/src/alpha_harness/learning/loop.py)
- **Code Evidence:**
  The `ResearchInsight` model has columns for `prompt_strategy_id` and `researcher_id`, but `loop.py` leaves them unset. Furthermore, there is no storage of model reference (`model_ref`), reflection prompt, reflection token usage, or LLM response payload.
- **Impact:** Lack of auditability regarding how prompt guidance was derived.

### Defect 4: Single Category Collapse During Strategy Evolution
- **Severity:** Medium
- **Status:** Confirmed defect
- **Location:** [`backend/src/alpha_harness/learning/loop.py#L217`](backend/src/alpha_harness/learning/loop.py)
- **Code Evidence:**
  ```python
  new_strategy = PromptStrategy(
      ...,
      category=curr_strat.category if (curr_strat and curr_strat.category) else "general",
  )
  ```
  In multi-researcher sessions where researcher personas span distinct categories (`MOMENTUM`, `MEAN_REVERSION`, `VALUE`, `QUALITY`), the learning cycle evolves only one strategy row for a single category, causing other specialized categories to never receive evolved prompt guidance.

---

## 4. Existing Compliant Functionality to Preserve

1. **Strict Holdout Protection (P0-03 Compliance):**
   In [`learning/memory.py#L106-L121`](backend/src/alpha_harness/learning/memory.py), candidate performance is extracted strictly using:
   - `val_sharpe`
   - `val_fitness`
   - `val_after_cost_sharpe`
   - `is_val_decay`
   - `val_composite_score`  
   Holdout columns (`holdout_sharpe`, `holdout_fitness`, etc.) are **never accessed**. This compliance must be strictly preserved.
2. **Provenance Verification:**
   Queries in [`learning/memory.py`](backend/src/alpha_harness/learning/memory.py) enforce `ResearchEvaluation.is_verified.is_(True)` and filter by completed statuses (`EVALUATED`, `ACCEPTED`, `GRADUATED`). Unverified or synthetic evaluations are excluded.
3. **Downstream Strategy Injection:**
   [`orchestrator/researchers.py#L303-L366`](backend/src/alpha_harness/orchestrator/researchers.py) already contains the logic to query active `PromptStrategy` rows, extract `guidance`, `examples`, and `anti_patterns`, format them into `learning_feedback`, and pass them to [`pipeline.generate_candidates()`](backend/src/alpha_harness/generator/pipeline.py).
4. **Leaderboard Snapshot Integration:**
   `SelfLearningEngine` returns a `LearningCycleResult`, which triggers periodic `LeaderboardService.capture_snapshot()`.

---

## 5. Answers to Audit Questions

### A. Does the system actually call an LLM during reflection?
**No.** The execution path in `SelfLearningEngine.execute_learning_cycle` never calls `self.llm`. Reflection output is generated entirely through Python arithmetic and template strings.

### B. What evidence does reflection receive?
Reflection retrieves verified validation performance (`val_sharpe`, `val_composite_score`, `is_val_decay`), candidate hypotheses, alpha expressions, and operator appearances. It does not use holdout data (preserving P0-03). However, because no LLM is called, this evidence is only used for Python sorting and string interpolation.

### C. Is reflection scientifically grounded?
**No.** It does not explain causal relationships. It applies a static string attribute explaining all failures as `"unnormalized momentum or excessive lookback without group neutralization"`, regardless of the strategy, dataset, or mathematical construction of the alpha.

### D. How are reflections stored?
Reflections are stored in `research_insight` and `prompt_strategy` tables. However, model identity, reflection prompts, and raw reasoning are not recorded, and relational foreign keys (`prompt_strategy_id`, `researcher_id`) are left null.

### E. How does reflection influence research?
The evolved `PromptStrategy` injects the top 3 winner hypotheses/expressions and top 3 loser expressions into subsequent researcher prompts. However, the qualitative guidance is limited to a comma-separated list of top/bottom 3 regex-parsed operators.

### F. How are failures handled?
If `execute_learning_cycle` encounters an error, `ExperimentRunner` logs a warning and increments `error_count`. If candidates are missing, it falls back to cross-generation candidates. But because no LLM call exists, LLM-specific failure modes (timeouts, quota exhaustion, malformed JSON, refusal) are currently unhandled in the learning engine.

### G. Does the current implementation already satisfy any requirements?
Yes. The data gathering queries, validation partition isolation, database tables, and downstream prompt injection pipeline are functional and should be retained.

---

## 6. Proposed P1-01 Implementation Plan

To implement genuine LLM reflection without over-scoping into later backlog items (P1-03, P1-09, P1-10, P1-12, P1-13), we will implement the following minimal, clean architecture:

1. **Reflection Prompt Design & Structured Schema:**
   - Define a structured JSON response schema (`REFLECTION_RESPONSE_SCHEMA`) requesting:
     - `executive_diagnosis`: High-level scientific diagnosis of the generation's results.
     - `winning_patterns`: What economic hypothesis and operator mechanics drove the winners.
     - `failure_modes`: Specific mathematical or market flaws observed in underperforming alphas.
     - `operator_guidance`: Recommended operators to explore and operators to avoid.
     - `strategy_recommendations`: Qualitative guidance for the next generation of researchers.
     - `anti_patterns`: Concrete flawed expressions with specific explanations of their defects.
2. **LLM Invocation in `SelfLearningEngine`:**
   - Assemble verified evidence: top performers (expressions, hypotheses, validation Sharpes, turnovers, fitness) and underperformers (expressions, hypotheses, validation Sharpes, decay, turnover).
   - Format a structured evidence prompt for the reflection agent.
   - Invoke `self.llm.generate()` using the session's designated reflection model (or the primary configured LLM).
3. **Resilient Fallback Policy:**
   - If the LLM call fails, times out, or budget is exhausted: log the failure and fall back gracefully to empirical statistical summaries, marking `is_heuristic_fallback = True` on the insight record.
4. **Enriched Persistence:**
   - Record LLM model reference, structured rules, and evidence in `ResearchInsight`.
   - Update `PromptStrategy.guidance` with the LLM's genuine qualitative recommendations and operator tips.
   - Populate `PromptStrategy.anti_patterns` with the model's actual diagnostic explanations rather than hardcoded labels.

---

## 7. File-by-File Change Plan

| File | Proposed Change |
|:---|:---|
| `backend/src/alpha_harness/prompts/library.py` | Add `REFLECTION_SYSTEM_PROMPT` and `render_reflection_user_prompt(...)` defining the quantitative reflection persona and evidence schema. |
| `backend/src/alpha_harness/learning/loop.py` | Update `SelfLearningEngine.execute_learning_cycle` to: (1) assemble verified candidate evidence, (2) call `self.llm.generate()`, (3) parse structured reflection output, (4) persist model-attributed `ResearchInsight` records, (5) update `PromptStrategy` with genuine model guidance, and (6) provide deterministic heuristic fallback if LLM is unavailable. |
| `backend/src/alpha_harness/db/models.py` | Ensure `ResearchInsight` supports `model_ref` and `is_llm_generated` (nullable / server default). |
| `backend/tests/test_p1_01_reflection.py` | New regression test suite validating genuine LLM reflection, prompt construction, structured schema parsing, error fallback, holdout metric exclusion, and database persistence. |

---

## 8. Regression-Test Matrix and Acceptance Criteria

1. **Evidence-Grounded Prompting:** Verification that the prompt sent to the LLM contains real validation Sharpes, hypotheses, and expressions from the database, and zero holdout data.
2. **Structured LLM Output Parsing:** Verification that LLM-generated diagnoses, operator recommendations, and anti-patterns correctly populate `ResearchInsight` and `PromptStrategy`.
3. **Failure & Timeout Resiliency:** If the LLM throws `ProviderError` or times out, the learning cycle does not crash; it logs the failure and falls back gracefully.
4. **Downstream Injection:** Verification that subsequent generation rounds receive the LLM's evolved guidance in their prompt context.
5. **No Holdout Contamination:** Explicit check confirming that `holdout_sharpe` and other final holdout partition metrics remain unaccessed during reflection.

---

## 9. Dependencies and Deferred Backlog Preservation

The following related issues are kept strictly separate:
- **P1-03:** Rigorous prompt-level leaderboard / controlled prompt experiment.
- **P1-09:** Advanced operator attribution (addressing naive mean Sharpe).
- **P1-10:** Multiplicity-aware winner/loser selection (adjusting for false discovery rate).
- **P1-12:** Deep researcher-specific persistent memory.
- **P1-13:** Systematic prompt evolution validation.

---

## 10. Authoritative Remediation Backlog

| ID | Description | Status |
|:---|:---|:---|
| **P0-01** | Synthetic PnL is passed through the ordinary evaluation engine | ✅ VERIFIED FIXED |
| **P0-02** | Data provenance is misreported | ✅ VERIFIED FIXED |
| **P0-03** | IS/OOS separation is not a sealed evaluation protocol | ✅ VERIFIED FIXED |
| **P0-04** | Robustness endpoint fabricates degraded results instead of re-simulating | ✅ VERIFIED FIXED |
| **P0-05** | Persistent autonomous end-to-end experiment API | ✅ VERIFIED FIXED |
| **P1-01** | Learning cycle is heuristic rather than actual LLM reflection | 🔍 **AUDITED — AWAITING APPROVAL** |
| **P1-02** | Truncation operator not implemented in Fast Expression | ⬜ Outstanding |
| **P1-03** | No prompt-level leaderboard / controlled prompt experiment | ⬜ Outstanding |
| **P1-04** | Missing unit handling verification | ⬜ Outstanding |
| **P1-05** | Fast Expression operator arity mismatch | ⬜ Outstanding |
| **P1-06** | Vector operations fail to compile / collapse to scalar | ⬜ Outstanding |
| **P1-07** | Pasteurization settings ignored in simulation payload | ⬜ Outstanding |
| **P1-08** | Nan-handling setting is hardcoded to OFF | ⬜ Outstanding |
| **P1-09** | Operator attribution uses naive mean Sharpe | ⬜ Outstanding |
| **P1-10** | Winner/loser selection ignores research multiplicity and uncertainty | ⬜ Outstanding |
| **P1-11** | Tournament selection lacks fitness sharing / diversity preservation | ⬜ Outstanding |
| **P1-12** | Researcher-specific persistent memory is shallow | ⬜ Outstanding |
| **P1-13** | Prompt version persistence is not the same as prompt evolution validation | ⬜ Outstanding |
| **P1-14** | Correlation breaker uses random search instead of systematic orthogonalization | ⬜ Outstanding |
| **P1-15** | Submission planner allows non-compliant alphas into final batch | ⬜ Outstanding |
| **P1-16** | Power Pool candidate selection ignores dynamic pool exhaustion | ⬜ Outstanding |
| **P2-01** through **P2-13** | 13 original P2 issues | ⬜ Outstanding |
| **P3-01** through **P3-02** | 2 original P3 issues | ⬜ Outstanding |

---

## 11. Final Status

**AUDIT COMPLETE — AWAITING APPROVAL.**
