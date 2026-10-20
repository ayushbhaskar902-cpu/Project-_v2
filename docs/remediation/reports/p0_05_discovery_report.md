# P0-05 Discovery Report: Persistent Autonomous End-to-End Experiment API

**Remediation Item:** P0-05 — Persistent autonomous end-to-end experiment API  
**Status:** **DISCOVERY COMPLETE — AWAITING IMPLEMENTATION APPROVAL**  
**Audit Date:** 2026-10-11  

---

## 1. Executive Summary

The existing system has **all the building blocks** for an autonomous research loop — generation, simulation, evaluation, learning, leaderboards, and pool curation — but **no orchestration layer connects them into a persistent, self-driving lifecycle**.

Every phase transition today requires a separate HTTP request from an external caller. If the process restarts, no in-flight research session is resumed. The `ResearchSession.status` field exists (`IDLE → RUNNING → PAUSED → LEARNING → COMPLETE → FAILED`) but is never autonomously advanced.

---

## 2. Architectural Audit: What Exists

### 2.1 Endpoints (Request-Driven)

Each phase of the research lifecycle is exposed as a **separate, manually-triggered endpoint**:

| Phase | Endpoint | Autonomous? |
|:---|:---|:---|
| Create session | `POST /api/research/sessions` | N/A (one-shot) |
| Generate candidates | `POST /api/research/sessions/{id}/generate` | ❌ Manual trigger |
| Enqueue for simulation | `POST /api/research/sessions/{id}/enqueue` | ❌ Manual trigger |
| Evaluate candidates | `POST /api/research/sessions/{id}/evaluate` | ❌ Manual trigger |
| Run robustness | `POST /api/research/sessions/{id}/robustness` | ❌ Manual trigger |
| Learning cycle | `POST /api/research/sessions/{id}/learning/cycle` | ❌ Manual trigger |
| Graduate candidate | `POST /api/research/sessions/{id}/candidates/{cid}/graduate` | ❌ Manual trigger |
| Curate pool | `POST /api/research/pool/curate` | ❌ Manual trigger |

There is **no** `POST /api/research/sessions/{id}/run` or equivalent that starts and drives a session to completion.

### 2.2 Simulation Tracking (Autonomous ✅)

The **SimulationTracker** ([tracker.py](backend/src/alpha_harness/engine/tracker.py)) is the one component that runs autonomously:
- Continuous polling loop (`_run` / `_tick`) monitors active simulations
- Reconciles orphaned/PENDING simulations on startup
- Re-adopts RUNNING simulations after restart
- Hands finished alphas to the vault via `alpha_landed`

This is **not** connected to the research session lifecycle — it tracks simulations generically, not as part of an autonomous experiment.

### 2.3 Orchestrator ([researchers.py](backend/src/alpha_harness/orchestrator/researchers.py))

`MultiLLMOrchestrator` provides:
- `initialize_session_researchers()` — seeds researcher personas
- `run_generation_round()` — dispatches one parallel generation round
- `enqueue_candidates_for_simulation()` — submits valid candidates to BatchEngine
- `sync_session_simulations()` — synchronizes candidate status with simulation outcomes

All methods are **one-shot**: they do exactly one round/sync and return. There is no loop, no budget tracking, and no automatic advancement to the next phase.

### 2.4 Self-Learning Engine ([learning/loop.py](backend/src/alpha_harness/learning/loop.py))

`SelfLearningEngine.execute_learning_cycle()`:
- Partitions candidates into winners/losers
- Derives operator and field insights
- Mutates prompt strategies
- Advances `current_generation`

Called **only** from the `POST /api/research/sessions/{id}/learning/cycle` endpoint. Never triggered automatically.

### 2.5 Session Status Transitions

`ResearchStatus` enum defines: `IDLE → RUNNING → PAUSED → LEARNING → COMPLETE → FAILED`

But the only place status is mutated is in `run_generation_round()`:
```python
status=ResearchStatus.RUNNING  # Set once, never advanced further
```

No code transitions a session to `LEARNING`, `PAUSED`, `COMPLETE`, or `FAILED`.

### 2.6 Budget Enforcement

`ResearchSession` has:
- `simulation_budget` (default 500)
- `simulations_used` (updated in `sync_session_simulations`)
- `learning_frequency` (default 80)

**No code enforces the budget** — the caller can generate and enqueue indefinitely. No code checks `simulations_used >= simulation_budget` and stops.

### 2.7 Process Restart / Persistence

**What survives a restart:**
- All session, candidate, evaluation, and learning data (SQLite)
- Simulation records and alpha vault series (SQLite + DuckDB)
- In-flight simulations are reconciled by the tracker

**What does NOT survive a restart:**
- No record of which session was "running" or at which phase
- No background task to resume a session's experiment loop
- Session `status` could be stuck at `RUNNING` permanently after a crash

---

## 3. Gap Analysis: What Is Missing

### Gap 1: Autonomous Experiment Loop
No background task or coroutine drives the `generate → enqueue → wait → evaluate → learn → repeat` cycle. The entire flow is caller-driven.

### Gap 2: Budget-Gated Termination
No enforcement of `simulation_budget`. Sessions have no natural stopping condition.

### Gap 3: Phase-Aware Status Machine
`ResearchSession.status` is written once to `RUNNING` and never advanced. There is no state machine governing `IDLE → GENERATING → SIMULATING → EVALUATING → LEARNING → (repeat or COMPLETE)`.

### Gap 4: Simulation Completion Callbacks
When simulations finish, `SimulationTracker` calls `on_alpha` (for vault capture) and `on_change` (for WebSocket broadcast), but **never** triggers evaluation or learning. The research session is not notified.

### Gap 5: Crash Recovery for Sessions
After a restart, `tracker.reconcile()` recovers orphaned simulations, but no code looks for sessions with `status = RUNNING` and resumes their experiment loop.

### Gap 6: Pause / Resume / Cancel
No mechanism to pause a running session, resume a paused one, or cancel gracefully with cleanup.

### Gap 7: Error Isolation and Session Failure
If generation fails (all LLMs down), evaluation fails (no series available), or learning fails (LLM timeout), the session has no error-handling policy. It would silently stop.

---

## 4. Proposed Architecture

### 4.1 Core Component: `ExperimentRunner`

A new class that owns the autonomous loop for one session:

```
ExperimentRunner(session_id, db, orchestrator, evaluation, learning, ...)
├── start()      → spawns background asyncio.Task
├── pause()      → sets flag, loop yields at next phase boundary
├── resume()     → respawns task from last durable checkpoint
├── cancel()     → stops loop, marks session FAILED/CANCELLED
└── _run()       → the generate-simulate-evaluate-learn loop
```

### 4.2 Phase State Machine

```
IDLE → GENERATING → SIMULATING → EVALUATING → LEARNING → [GENERATING or COMPLETE]
         ↕              ↕             ↕            ↕
       PAUSED         PAUSED        PAUSED       PAUSED
         ↕              ↕             ↕            ↕
       FAILED         FAILED        FAILED       FAILED
```

Each transition is durably persisted to `ResearchSession.status` before work begins, so a restart can recover.

### 4.3 Simulation Completion Bridge

Register a callback with `SimulationTracker.on_change` (or a new purpose-built hook) that detects when all candidates in the current round have terminated. This triggers automatic evaluation and learning.

### 4.4 Durable Checkpointing

Add columns to `ResearchSession`:
- `current_phase` (e.g., `GENERATING`, `SIMULATING`, `EVALUATING`, `LEARNING`)
- `current_round_candidate_ids` (JSON list of candidate IDs in the current round)
- `error_count` (consecutive errors before marking FAILED)
- `last_checkpoint_at` (timestamp of last phase completion)

### 4.5 Budget Enforcement

Before each generation round:
```python
if session.simulations_used >= session.simulation_budget:
    transition to COMPLETE
```

### 4.6 API Surface

| Endpoint | Action |
|:---|:---|
| `POST /sessions/{id}/run` | Start autonomous loop |
| `POST /sessions/{id}/pause` | Pause at next phase boundary |
| `POST /sessions/{id}/resume` | Resume from last checkpoint |
| `POST /sessions/{id}/cancel` | Abort and mark FAILED |
| `GET /sessions/{id}/status` | Current phase, progress, ETA |

### 4.7 Startup Recovery

In `AppState.startup()`, after tracker reconciliation:
```python
for session in sessions_with_status_RUNNING:
    ExperimentRunner(session.id, ...).resume()
```

### 4.8 Error Policy

- **Generation failure (all LLMs):** Retry up to 3 times with backoff, then PAUSED with message
- **Simulation timeout:** Wait up to `LOST_AFTER_SECONDS` (already in tracker), then skip failed candidates
- **Evaluation failure (no series):** Mark candidates UNAVAILABLE, continue with others
- **Learning failure:** Log and skip, continue to next generation
- **Consecutive errors >= threshold:** Transition to FAILED with diagnostic message

---

## 5. Files to Create or Modify

| File | Action | Purpose |
|:---|:---|:---|
| `backend/src/alpha_harness/experiment/runner.py` | **CREATE** | `ExperimentRunner` class with autonomous loop |
| `backend/src/alpha_harness/experiment/__init__.py` | **CREATE** | Module exports |
| `backend/src/alpha_harness/db/models.py` | **MODIFY** | Add `current_phase`, `current_round_candidate_ids`, `error_count`, `last_checkpoint_at` to `ResearchSession` |
| `backend/src/alpha_harness/db/sqlite.py` | **MODIFY** | Schema migration for new columns |
| `backend/src/alpha_harness/api/research.py` | **MODIFY** | Add `/run`, `/pause`, `/resume`, `/cancel`, `/status` endpoints |
| `backend/src/alpha_harness/state.py` | **MODIFY** | Register `ExperimentRunner` registry; startup recovery |
| `backend/tests/test_p0_05_experiment.py` | **CREATE** | Regression test suite |

---

## 6. Verification Criteria

1. `POST /sessions/{id}/run` starts a background loop that autonomously generates, enqueues, waits, evaluates, and learns
2. Session transitions through all phases durably (`GENERATING → SIMULATING → EVALUATING → LEARNING`)
3. The loop stops when `simulations_used >= simulation_budget` and marks session `COMPLETE`
4. A process restart resumes any session with `status = RUNNING` from its last checkpoint
5. `POST /sessions/{id}/pause` halts the loop at the next phase boundary; `POST /sessions/{id}/resume` restarts it
6. Individual phase failures (generation, evaluation, learning) are isolated and logged without crashing the session
7. Consecutive unrecoverable errors transition the session to `FAILED` with a diagnostic message
8. Existing manual-trigger endpoints continue to work independently
9. WebSocket broadcasts session phase transitions so the dashboard reflects real-time state
10. All existing P0-01 through P0-04 tests continue to pass

---

## 7. Authoritative Remediation Backlog (Corrected)

| ID | Description | Status |
|:---|:---|:---|
| **P0-01** | Synthetic PnL is passed through the ordinary evaluation engine | ✅ VERIFIED FIXED |
| **P0-02** | Data provenance is misreported | ✅ VERIFIED FIXED |
| **P0-03** | IS/OOS separation is not a sealed evaluation protocol | ✅ VERIFIED FIXED |
| **P0-04** | Robustness endpoint fabricates degraded results instead of re-simulating | ✅ VERIFIED FIXED |
| **P0-05** | Persistent autonomous end-to-end experiment API | 🔍 DISCOVERY COMPLETE |
| **P1-01** through **P1-16** | 16 original P1 issues | ⬜ Outstanding |
| **P2-01** through **P2-13** | 13 original P2 issues | ⬜ Outstanding |
| **P3-01** through **P3-02** | 2 original P3 issues | ⬜ Outstanding |

**Total issues:** 36 (5 P0 + 16 P1 + 13 P2 + 2 P3)
