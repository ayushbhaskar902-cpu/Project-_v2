# Alpha Harness - Self-Learning Multi-LLM Alpha Creation Engine

Alpha Harness is an autonomous quantitative alpha discovery and research harness. It combines heterogeneous Large Language Model (LLM) agents with real-time WorldQuant BRAIN simulation, statistical validation, out-of-sample robustness testing, and closed-loop reinforcement learning to generate and refine production-grade financial alpha factors.

---

## Key Features

1. **Heterogeneous Multi-LLM Roster**
   - Dispatches parallel generation across diverse frontier models (Gemini Pro, Gemini Flash, GPT-4o, DeepSeek).
   - Specialized personas (Momentum Specialist, Mean Reversion Specialist, Volume-Price Interaction, Cross-Asset Arbitrage).

2. **Real WorldQuant BRAIN Integration**
   - Full automated simulation lifecycle through BRAIN's batch simulation API.
   - Live monitoring of submission status, queue times, and multi-simulation batches.

3. **Rigorous AST Validation & Metrics**
   - Abstract Syntax Tree (AST) validation against the BRAIN domain-specific language (DSL).
   - In-Sample (IS) and Out-Of-Sample (OOS) Sharpe ratio calculation and Sharpe decay tracking.
   - Turnover, fitness, margin, returns, and pairwise correlation analysis.

4. **Closed-Loop Self-Learning (RL)**
   - Learning cycles dynamically analyze simulation outcomes every generation.
   - Extracts positive factor patterns and failure modes to update prompt strategies for next-generation candidates.

5. **Alpha Portfolio Curation**
   - Low-correlation alpha pool optimization.
   - Automated submittable alpha ranking and vault integration.

6. **Interactive Research Studio UI**
   - Modern React dashboard located at `/research` with real-time session controls, candidate inspection, researcher leaderboards, and learning insights.

---

## Architecture

- **Backend**: Python 3.12+, FastAPI, SQLAlchemy (SQLite/PostgreSQL), Structlog, WorldQuant BRAIN client.
- **Frontend**: React 18, TypeScript, Vite, TailwindCSS / Custom Design System.
- **Documentation**: See [RESEARCH_ENGINE_CHANGELOG.md](RESEARCH_ENGINE_CHANGELOG.md) for detailed pipeline architecture and verification logs.

---

## Getting Started

### Prerequisites
- Python 3.12+
- Node.js 18+
- WorldQuant BRAIN credentials configured in `.env`

### Backend Setup
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
python -m alpha_harness
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173/research` to access the Research Studio.
