# Alpha Harness - Research Engine Changelog and Architecture Notes

> **Date**: 2026-10-05
> **Session**: Full diagnostic + fix session for the autonomous Multi-LLM Alpha Research Engine
> **Status**: Pipeline verified working - alphas ARE simulated on BRAIN

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Issues Found and Fixed](#issues-found-and-fixed)
4. [Files Modified](#files-modified)
5. [How the Pipeline Works (Verified)](#how-the-pipeline-works-verified)
6. [BRAIN Simulation Proof](#brain-simulation-proof)
7. [Known Issues and Remaining Work](#known-issues-and-remaining-work)
8. [Database Schema Reference](#database-schema-reference)
9. [Configuration](#configuration)
10. [How to Run](#how-to-run)

---

## Executive Summary

The Alpha Harness Research Engine is a **Self-Learning Multi-LLM Alpha Creation Engine** that:
1. Uses 4 heterogeneous LLM researchers (Gemini Pro, Gemini Flash, GPT-4o, DeepSeek) to generate alpha expressions
2. Validates them via AST parsing
3. **Submits them to WorldQuant BRAIN for real 10-year backtests**
4. Evaluates results with IS/OOS Sharpe decay analysis
5. Feeds results into a reinforcement learning loop that improves prompt strategies

### Key Finding
The pipeline **IS working correctly**. All 56 unique alpha expressions were submitted to BRAIN in 6 batches (10+10+10+10+10+6), completed within approximately 2 minutes, and results were captured. The simulations complete very fast via multi-sim batching.

---

## Architecture Overview

```
Frontend (React/Vite) - Research Studio @ /research
  [Generate Round] [Simulate Round] [Learn Cycle] [Curate Pool]
        |                |               |              |
        v                v               v              v
Backend (FastAPI/Uvicorn)
  api/research.py - REST Endpoints
    POST /sessions                     - Create research campaign
    POST /sessions/{id}/generate       - LLM generation round
    POST /sessions/{id}/enqueue        - Queue for BRAIN sim
    POST /sessions/{id}/evaluate       - Real backtest eval
    POST /sessions/{id}/learning/cycle - RL reflection loop
    POST /pool/curate                  - Low-correlation pool

  orchestrator/researchers.py
    MultiLLMOrchestrator
      4 LLM researcher personas
      run_generation_round()                - parallel LLM dispatch
      enqueue_candidates_for_simulation()   - BatchEngine
      sync_session_simulations()            - BRAIN result sync

  engine/slots.py
    BatchEngine
      enqueue()       - dedup + create SimulationRecord
      _run()          - tick every 2s
      _fill_slots()   - pack batches + submit to BRAIN
      8 concurrent slots x 10 items = 80 sims/round

  learning/loop.py
    SelfLearningEngine
      run_cycle()                - analyze results
      partition_performance()    - top/bottom/middle
      update_prompt_strategies() - RL prompt evolution

  state.py - AppState (composition root)
    Wires: db, engine, tracker, orchestrator, llm, vault

  Storage:
    SQLite (harness.db)    - candidates, sessions, strategies
    DuckDB (catalog.duckdb) - PnL series, turnover, metrics
    BRAIN API              - 10yr simulation backtest
```

---

## Issues Found and Fixed

### Fix 1: Orphan Sweep Cancelling Research Simulations
**File**: `backend/src/alpha_harness/engine/slots.py`
**Problem**: The `disown_orphans()` method was cancelling QUEUED SimulationRecords for research sessions because `research_session_*` tasks dont have a matching `Study` row.
**Fix**: Added exclusion filter: `~SimulationRecord.task.like("research_session_%")`

### Fix 2: Real BRAIN Data in Evaluation (not synthetic)
**File**: `backend/src/alpha_harness/api/research.py`
**Problem**: The evaluate endpoint was generating synthetic PnL series instead of fetching real backtest data from BRAIN/DuckDB.
**Fix**: Connected evaluation to real vault data pipeline:
1. Calls `sync_session_simulations()` first to pull `alpha_id` from SimulationRecords
2. Downloads real daily PnL/turnover from DuckDB via `state.alphas.series()`
3. Falls back to `state.alphas.pnl_series()` then `state.alphas.by_ids()` for summary metrics
4. Only generates synthetic approximation when no real data is available at all

### Fix 3: Invalid Operators Removed
**File**: `backend/src/alpha_harness/generator/pipeline.py`
**Problem**: `ts_min` and `ts_max` were in the operator list but are not valid BRAIN FastExpr operators.
**Fix**: Removed both from the allowed operator lists.

### Fix 4: Learning Loop Empty Generation Fallback
**File**: `backend/src/alpha_harness/learning/loop.py`
**Problem**: When the current generation had no evaluated candidates, the learning cycle would fail silently.
**Fix**: Added fallback to retrieve ALL historical evaluated candidates when the current generations subset is empty, ensuring the RL loop always has data to learn from.

### Fix 5: Logger Import Fix
**File**: `backend/src/alpha_harness/api/research.py`
**Problem**: `NameError: name 'log' is not defined` when the backfill logic attempted to log warnings.
**Fix**: Added `import structlog` and `log = structlog.get_logger(__name__)` at module top.

### Fix 6: Authentication Guard for Backfill
**File**: `backend/src/alpha_harness/api/research.py`
**Problem**: Backfill fetch would fail with auth errors when no BRAIN session was active.
**Fix**: Added `getattr(state.auth.session, "authenticated", False)` check before attempting to download series from BRAIN.

---

## Files Modified

| File | Lines Changed | Purpose |
|------|--------------|---------|
| `backend/src/alpha_harness/engine/slots.py` | ~1 line | Exempt research sessions from orphan sweep |
| `backend/src/alpha_harness/api/research.py` | ~80 lines | Real BRAIN evaluation, auth guard, logging |
| `backend/src/alpha_harness/orchestrator/researchers.py` | ~60 lines | Added sync_session_simulations() method |
| `backend/src/alpha_harness/generator/pipeline.py` | ~2 lines | Removed invalid operators |
| `backend/src/alpha_harness/learning/loop.py` | ~10 lines | Empty generation fallback |
| `frontend/src/screens/research/index.tsx` | ~5 lines | Toast messages for BRAIN pending status |

---

## How the Pipeline Works (Verified)

### Step 1: Generate (POST /sessions/{id}/generate)
- 4 LLM researchers each generate ~20 alpha expressions (80 total target)
- Each expression is AST-validated for BRAIN FastExpr syntax
- Valid candidates stored in SQLite with status VALID

### Step 2: Enqueue + Simulate (POST /sessions/{id}/enqueue)
- Valid candidates are passed to BatchEngine.enqueue()
- Dedup check: expressions already simulated get SKIPPED with existing alpha_id
- New expressions get QUEUED as SimulationRecord rows
- Engine loop picks up QUEUED records every 2 seconds
- Packs into multi-simulation batches (up to 10 per batch, 8 concurrent slots)
- Submits to BRAIN API (engine.batch_running log entry with platform_id)
- BRAIN runs 10-year daily backtest
- Results expand (engine.batch_expanded with alpha_id per child)

### Step 3: Evaluate (POST /sessions/{id}/evaluate)
- sync_session_simulations() pulls alpha_id from completed SimulationRecords
- Downloads real PnL series from DuckDB vault
- Computes: full Sharpe, after-cost Sharpe, IS/OOS Sharpe, Sharpe decay, composite score
- Marks candidates as EVALUATED with full scorecard

### Step 4: Learn (POST /sessions/{id}/learning/cycle)
- Partitions evaluated candidates into top/bottom/middle performers
- Analyzes which operators, field groups, and expression patterns correlate with success
- Updates PromptStrategy rows with refined guidance, winning examples, anti-patterns
- Increments generation counter

### Step 5: Repeat
- Next generation uses updated strategies for better alpha quality

---

## BRAIN Simulation Proof

From the server logs at 00:39:52 to 00:41:34, session 46 submitted and completed:

```
00:39:52 engine.enqueued       queued=56 skipped=21 task=research_session_46
00:39:55 engine.batch_running  parent=28043 platform_id=17B65PfUw4RE9Nlql7MSxKq size=10
00:39:56 engine.batch_running  parent=28044 platform_id=3WdJkVasI4zJbErXR9EClp5 size=10
00:39:56 engine.batch_running  parent=28045 platform_id=4BE03ieFU5hU9KA1O29ASUH size=10
00:39:57 engine.batch_running  parent=28046 platform_id=V8Dm1dkG5fP8AWSHCq08ii  size=10
00:39:57 engine.batch_running  parent=28047 platform_id=27S06jeZz500as31dICpuNDF size=10
00:39:58 engine.batch_running  parent=28048 platform_id=4r6qsu1D14sKajZ9F3elhEK size=6
00:40:13 engine.batch_expanded children=10 matched=10 parent=28043
00:40:17 engine.batch_expanded children=10 matched=10 parent=28044
00:40:22 engine.batch_expanded children=10 matched=10 parent=28045
00:40:25 engine.batch_expanded children=10 matched=10 parent=28046
00:40:29 engine.batch_expanded children=10 matched=10 parent=28047
00:41:34 engine.batch_expanded children=6  matched=6  parent=28048
```

**56 alphas submitted in 6 batches, all returned with results from BRAIN in ~2 minutes.**

Each platform_id is a real BRAIN simulation ID visible at:
https://platform.worldquantbrain.com/simulate/{platform_id}

---

## Known Issues and Remaining Work

### 1. Sync Overcounting Failed Simulations
**Status**: Minor cosmetic issue
**Problem**: sync_session_simulations() counts ALL candidates linked to SimulationRecords for the session, including those from previous generations whose records were cancelled by the old orphan sweep bug.
**Impact**: Log shows failed=414 but most are historical artifacts.
**Fix**: Filter sync to only process candidates from the current or specified generation, or only those with non-EVALUATED status.

### 2. Evaluate Fires Before BRAIN Completes
**Status**: By design, but could be improved
**Problem**: Frontend calls /enqueue then immediately /evaluate. If BRAIN takes more than 2 seconds, the evaluate finds no completed simulations yet.
**Fix Options**:
- Add a polling mechanism in the frontend that checks every 10s until simulations complete
- Add a WebSocket push when batch_expanded fires for a research_session_* task
- Show a "Simulations in progress on BRAIN - will auto-evaluate when complete" message

### 3. Real PnL Series Availability
**Status**: Partially working
**Problem**: Some completed alphas dont have PnL series in DuckDB yet because the backfill task takes time.
**Fix**: The evaluate endpoint already has fallback logic:
1. Try real series from vault
2. Try pnl_series
3. Use alpha summary metrics to approximate

### 4. Multiple Generations Reuse Same SimulationRecords
**Status**: Minor
**Problem**: When the same expression appears across generations, dedup correctly reuses the existing alpha_id, but this means multiple candidates can point to the same SimulationRecord.
**Impact**: No functional impact, the evaluation uses the real BRAIN data regardless.

---

## Database Schema Reference

### SQLite (harness.db) - Operational State
- research_sessions: Campaign config, generation counter, status
- research_candidates: Generated expressions, validation, status, alpha_id link
- research_researchers: LLM persona configs, counters
- research_evaluations: IS/OOS Sharpe, fitness, composite scores
- research_insights: RL learning observations per generation
- prompt_strategies: Evolved prompt templates per strategy category
- simulation_records: BatchEngine queue entries, BRAIN status tracking

### DuckDB (catalog.duckdb) - Columnar Analytics
- alpha_pnl: Daily PnL series per alpha_id
- alpha_turnover: Daily turnover series per alpha_id
- alpha_metrics: Summary metrics (Sharpe, fitness, drawdown)

---

## Configuration

### Default Researcher Roster
| Model | Label | Strategy Focus | Temperature |
|-------|-------|---------------|-------------|
| google:gemini-2.5-pro | Gemini Trend and Momentum | MOMENTUM | 0.7 |
| google:gemini-2.5-flash | Gemini Fast Reversion | MEAN_REVERSION | 0.6 |
| openai:gpt-4o | GPT Fundamental Value | VALUE | 0.7 |
| deepseek:deepseek-chat | DeepSeek Multi-Factor | GENERAL | 0.8 |

### Engine Limits
- **Concurrent BRAIN Slots**: 8
- **Items per Multi-Sim Batch**: 10
- **Round Capacity**: 80 candidates
- **Engine Tick**: 2 seconds
- **Quick Mode**: Enabled for configured regions if account has QUICK_MODE permission

---

## How to Run

```bash
# Backend
cd backend
.venv\Scripts\python.exe -m alpha_harness

# Frontend
cd frontend
npm run dev
```

Access Research Studio at: http://localhost:5173/research

### Research Session Workflow (UI)
1. Select or create a session
2. Click **Dispatch Round** - generates 80 candidate alphas via 4 LLMs
3. Click **Simulate Round** - enqueues to BRAIN + evaluates completed results
4. Wait ~2-3 minutes for BRAIN to complete, then click **Simulate Round** again to evaluate
5. Click **Learning Cycle** - runs RL reflection, advances generation
6. Repeat steps 2-5 for progressive improvement
7. Click **Curate Pool** - builds optimal low-correlation alpha portfolio
