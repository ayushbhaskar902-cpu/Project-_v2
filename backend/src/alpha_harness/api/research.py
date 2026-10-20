"""Research Studio REST API endpoints.

Exposes REST endpoints for the Self-Learning Multi-LLM Alpha Creation Engine:
- Research Sessions (campaign lifecycle)
- Heterogeneous LLM Researchers
- Candidate Alpha Factor Generation & AST Validation
- Quantitative Evaluation & IS/OOS Metric Separation
- Parameter & Market Regime Robustness Tests
- Multi-Axis Leaderboard (Alphas, Models, Prompts)
- Closed-Loop Reflection & Self-Learning Insights
- Low-Correlation Alpha Pool Curation & Combo Optimization
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import case, desc, func, or_, select

from ..application.research import ResearchSessionService, ResearchWorkflowService
from ..db.models import (
    CandidateStatus,
    EvaluationJob,
    GenerationAttempt,
    LeaderboardSnapshot,
    PromptStrategy,
    ResearchCandidate,
    ResearchEvaluation,
    ResearchInsight,
    ResearchSession,
    ResearchStatus,
    Submission,
    utcnow,
)
from ..evaluation import (
    HoldoutChronologyError,
    HoldoutImmutabilityError,
    HoldoutIndeterminacyError,
    HoldoutOverlapError,
    InsufficientDataError,
)
from ..evaluation.jobs import EvaluationJobService
from ..schemas import Out
from .deps import State, refuse

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/research", tags=["research"])


# --- Schemas -------------------------------------------------------------


class CreateSessionIn(BaseModel):
    name: str = "Alpha Research Campaign"
    prompt: str = "Discover profitable cross-sectional equity momentum and mean reversion alphas"
    instrument_type: str = "EQUITY"
    region: str = "USA"
    universe: str = "TOP3000"
    delay: int = 1
    simulation_budget: int = 500
    learning_frequency: int = 80
    model_refs: list[str] = Field(default_factory=lambda: [
        "google:gemini-2.5-pro",
        "google:gemini-2.5-flash",
        "openai:gpt-4o",
        "deepseek:deepseek-chat",
    ])
    constraints: dict[str, Any] = Field(default_factory=dict)
    language: str | None = Field(default="FASTEXPR", description="Target alpha language. Strictly restricted to FASTEXPR.")

    @field_validator("language", mode="before")
    @classmethod
    def _validate_language(cls, v: Any) -> str:
        from ..generator.pipeline import validate_alpha_language
        return validate_alpha_language(v)


class SessionSummaryOut(Out):
    id: int
    name: str | None
    prompt: str | None
    region: str | None
    universe: str | None
    delay: int | None
    status: str | None
    simulation_budget: int | None
    simulations_used: int | None
    learning_frequency: int | None
    current_generation: int | None
    total_candidates: int
    valid_candidates: int
    submission_grade_candidates: int
    best_sharpe: float | None
    created_at: datetime | None


class ResearcherOut(Out):
    id: int
    model_ref: str | None
    label: str | None
    strategy_focus: str | None
    temperature: float | None
    total_generated: int | None
    total_valid: int | None
    avg_sharpe: float | None
    best_sharpe: float | None
    enabled: bool | None


class CandidateOut(Out):
    id: int
    session_id: int | None = None
    researcher_id: int | None = None
    generation_attempt_id: int | None = None
    generation: int | None = 0
    hypothesis: str | None = None
    expression: str | None = None
    language: str = Field(default="FASTEXPR", description="Target alpha language (FASTEXPR).")
    operator_count: int | None = 0
    field_count: int | None = 0
    is_valid: bool | None = True
    validation_error: str | None = None
    retry_count: int | None = 0
    status: str | None = "DRAFT"
    full_sharpe: float | None = None
    after_cost_sharpe: float | None = None
    oos_sharpe: float | None = None
    sharpe_decay: float | None = None
    composite_score: float | None = None
    is_submission_grade: bool | None = Field(
        default=False,
        description="Deprecated alias for internal quality thresholds; not BRAIN eligibility.",
    )
    platform_eligibility: str = Field(default="UNCHECKED", description="Authoritative BRAIN check state.")
    is_submitted: bool = Field(default=False, description="True only when a Submission record exists.")
    alpha_id: str | None = None
    simulation_record_id: int | None = None
    prompt_strategy_id: int | None = None
    model_ref: str | None = None
    temperature: float | None = None
    datasets_used: list[str] = Field(default_factory=list)
    fields_used: list[str] = Field(default_factory=list)
    raw_expression: str | None = None
    is_verified: bool | None = Field(
        default=None,
        description="True if evaluation verified against real BRAIN series; False if unverified; None if unevaluated."
    )
    evaluation_provenance: str | None = Field(
        default=None,
        description="Source provenance of the evaluation (e.g. 'BRAIN_VAULT'). None if unevaluated."
    )
    evaluation_scheme: str | None = None
    is_graduated: bool = False
    val_composite_score: float | None = None
    val_sharpe: float | None = None
    val_after_cost_sharpe: float | None = None
    val_fitness: float | None = None
    val_turnover: float | None = None
    is_val_decay: float | None = None
    holdout_sharpe: float | None = None
    holdout_after_cost_sharpe: float | None = None
    holdout_fitness: float | None = None
    simulation_settings: dict[str, Any] | None = None
    settings_hash: str | None = None
    settings_verification_status: str | None = None
    candidate_provenance_hash: str | None = None
    created_at: datetime | None = None


class GenerationAttemptOut(Out):
    id: int
    session_id: int
    researcher_id: int | None = None
    prompt_strategy_id: int | None = None
    resolution_tier: str | None = None
    generation: int | None = 0
    model_ref: str | None = None
    temperature: float | None = None
    system_prompt: str | None = None
    user_prompt: str | None = None
    response_schema: dict[str, Any] | None = None
    status: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    retry_count: int | None = None
    error_class: str | None = None
    error_message: str | None = None
    latency_seconds: float | None = None
    candidate_count: int | None = None
    output_count: int | None = None
    token_usage: int | None = None
    cost: float | None = None
    validation_outcome: str | None = None
    provider_request_id: str | None = None
    created_at: datetime | None = None


class UpdateResearcherIn(BaseModel):
    enabled: bool | None = None
    temperature: float | None = None
    strategy_focus: str | None = None
    system_prompt: str | None = None


class EnqueueCandidatesIn(BaseModel):
    candidate_ids: list[int] | None = None
    task: str | None = None


class TriggerGenerationIn(BaseModel):
    total_candidates: int = 80
    language: str | None = Field(default="FASTEXPR", description="Target alpha language. Strictly restricted to FASTEXPR.")

    @field_validator("language", mode="before")
    @classmethod
    def _validate_language(cls, v: Any) -> str:
        from ..generator.pipeline import validate_alpha_language
        return validate_alpha_language(v)


class CuratePoolIn(BaseModel):
    session_id: int | None = None
    target_pool_size: int = 50
    max_pairwise_correlation: float = 0.60


# --- Endpoints -----------------------------------------------------------


@router.get("/sessions", response_model=list[SessionSummaryOut])
async def list_sessions(state: State) -> list[SessionSummaryOut]:
    """Lists all research campaigns with live generation and alpha statistics."""
    async with state.db.session() as session:
        stmt = select(ResearchSession).order_by(desc(ResearchSession.created_at))
        res = await session.execute(stmt)
        sessions = list(res.scalars().all())

        results: list[SessionSummaryOut] = []
        for s in sessions:
            # Candidate counts
            c_stmt = (
                select(
                    func.count(ResearchCandidate.id),
                    func.sum(case((ResearchCandidate.is_valid.is_(True), 1), else_=0)),
                )
                .where(ResearchCandidate.session_id == s.id)
            )
            c_res = await session.execute(c_stmt)
            tot_c, val_c = c_res.one()

            # Best Sharpe and submission grade count
            e_stmt = (
                select(
                    func.max(func.coalesce(ResearchEvaluation.val_sharpe, ResearchEvaluation.full_sharpe)),
                    func.sum(
                        case(
                            (
                                or_(ResearchEvaluation.internal_checks.is_not(None), ResearchEvaluation.checks.is_(None))
                                & (ResearchEvaluation.checks_passed == ResearchEvaluation.checks_total),
                                1,
                            ),
                            else_=0,
                        )
                    ),
                )
                .join(
                    ResearchCandidate,
                    ResearchCandidate.id == ResearchEvaluation.candidate_id,
                )
                .where(
                    ResearchCandidate.session_id == s.id,
                    ResearchEvaluation.is_verified.is_(True),
                    ResearchCandidate.status.in_([
                        CandidateStatus.EVALUATED,
                        CandidateStatus.ACCEPTED,
                        CandidateStatus.GRADUATED,
                    ]),
                )
            )
            e_res = await session.execute(e_stmt)
            max_s, sub_c = e_res.one()

            results.append(
                SessionSummaryOut(
                    id=s.id,
                    name=s.name,
                    prompt=s.prompt,
                    region=s.region,
                    universe=s.universe,
                    delay=s.delay,
                    status=s.status,
                    simulation_budget=s.simulation_budget,
                    simulations_used=s.simulations_used,
                    learning_frequency=s.learning_frequency,
                    current_generation=s.current_generation,
                    total_candidates=tot_c or 0,
                    valid_candidates=val_c or 0,
                    submission_grade_candidates=sub_c or 0,
                    best_sharpe=round(float(max_s), 2) if max_s is not None else None,
                    created_at=s.created_at,
                )
            )

        return results


@router.post("/sessions", response_model=SessionSummaryOut)
async def create_session(payload: CreateSessionIn, state: State) -> SessionSummaryOut:
    """Creates and initializes a new autonomous research session with configured LLMs."""
    session_id, _ = await ResearchSessionService(state).create(payload)

    return SessionSummaryOut(
        id=session_id,
        name=payload.name,
        prompt=payload.prompt,
        region=payload.region,
        universe=payload.universe,
        delay=payload.delay,
        status=ResearchStatus.IDLE,
        simulation_budget=payload.simulation_budget,
        simulations_used=0,
        learning_frequency=payload.learning_frequency,
        current_generation=0,
        total_candidates=0,
        valid_candidates=0,
        submission_grade_candidates=0,
        best_sharpe=None,
        created_at=utcnow(),
    )


@router.get("/sessions/{session_id}/researchers", response_model=list[ResearcherOut])
async def list_session_researchers(session_id: int, state: State) -> list[ResearcherOut]:
    """Lists researcher agents configured for a session."""
    researchers = await state.orchestrator.get_session_researchers(session_id)
    return [
        ResearcherOut(
            id=r.id,
            model_ref=r.model_ref,
            label=r.label,
            strategy_focus=r.strategy_focus,
            temperature=r.temperature,
            total_generated=r.total_generated,
            total_valid=r.total_valid,
            avg_sharpe=round(float(r.avg_sharpe), 2) if r.avg_sharpe is not None else None,
            best_sharpe=round(float(r.best_sharpe), 2) if r.best_sharpe is not None else None,
            enabled=r.enabled,
        )
        for r in researchers
    ]


@router.patch("/sessions/{session_id}/researchers/{researcher_id}")
async def update_session_researcher(
    session_id: int, researcher_id: int, payload: UpdateResearcherIn, state: State
) -> dict[str, Any]:
    """Updates researcher persona configuration and hyperparameters."""
    ok = await state.orchestrator.update_researcher(
        researcher_id=researcher_id,
        enabled=payload.enabled,
        temperature=payload.temperature,
        strategy_focus=payload.strategy_focus,
        system_prompt=payload.system_prompt,
    )
    if not ok:
        raise HTTPException(status_code=404, detail=f"Researcher {researcher_id} not found")
    return {"status": "success", "researcherId": researcher_id}


@router.post("/sessions/{session_id}/enqueue")
async def enqueue_session_candidates(
    session_id: int, payload: EnqueueCandidatesIn, state: State
) -> dict[str, Any]:
    """Enqueues valid candidates from this session for simulation in BatchEngine."""
    queued_ids = await ResearchSessionService(state).enqueue(session_id, payload)
    return {
        "status": "success",
        "sessionId": session_id,
        "queuedCount": len(queued_ids),
        "queuedCandidateIds": queued_ids,
    }


@router.post("/sessions/{session_id}/generate")
async def trigger_generation_round(
    session_id: int, payload: TriggerGenerationIn, state: State
) -> dict[str, Any]:
    """Dispatches a parallel generation round across all enabled researchers."""
    valid_ids = await ResearchSessionService(state).generate(session_id, payload.total_candidates)
    summary = getattr(valid_ids, "summary", {})
    return {
        "status": "success",
        "sessionId": session_id,
        "validCandidatesGenerated": len(valid_ids),
        "validCandidateIds": valid_ids,
        **summary,
    }


@router.post("/sessions/{session_id}/run")
async def run_experiment_session(session_id: int, state: State) -> dict[str, Any]:
    """Starts the autonomous experiment loop for a research session."""
    try:
        await ResearchSessionService(state).start(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "status": "started",
        "sessionId": session_id,
        "message": "Autonomous experiment loop started.",
    }


@router.post("/sessions/{session_id}/pause")
async def pause_experiment_session(session_id: int, state: State) -> dict[str, Any]:
    """Pauses the autonomous experiment loop at the next phase boundary."""
    try:
        await ResearchSessionService(state).pause(session_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "status": "pause_requested",
        "sessionId": session_id,
        "message": "Pause requested; experiment will pause at the next phase boundary.",
    }


@router.post("/sessions/{session_id}/resume")
async def resume_experiment_session(session_id: int, state: State) -> dict[str, Any]:
    """Resumes an autonomous experiment loop that was paused or interrupted."""
    try:
        await ResearchSessionService(state).resume(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "status": "resumed",
        "sessionId": session_id,
        "message": "Autonomous experiment loop resumed.",
    }


@router.post("/sessions/{session_id}/cancel")
async def cancel_experiment_session(session_id: int, state: State) -> dict[str, Any]:
    """Cancels an autonomous experiment loop immediately."""
    try:
        await ResearchSessionService(state).cancel(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return {
        "status": "cancelled",
        "sessionId": session_id,
        "message": "Autonomous experiment loop cancelled.",
    }


@router.get("/sessions/{session_id}/status")
async def get_experiment_session_status(session_id: int, state: State) -> dict[str, Any]:
    """Returns the current autonomous experiment lifecycle status for a session."""
    try:
        return await ResearchSessionService(state).status(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/sessions/{session_id}/candidates", response_model=list[CandidateOut])
async def list_candidates(
    session_id: int,
    state: State,
    generation: int | None = None,
    valid_only: bool = False,
    limit: int = 100,
    offset: int = 0,
) -> list[CandidateOut]:
    """Queries candidates for a research session with validation and evaluation metrics."""
    async with state.db.session() as session:
        stmt = (
            select(ResearchCandidate, ResearchEvaluation)
            .outerjoin(
                ResearchEvaluation,
                ResearchEvaluation.candidate_id == ResearchCandidate.id,
            )
            .where(ResearchCandidate.session_id == session_id)
        )
        if generation is not None:
            stmt = stmt.where(ResearchCandidate.generation == generation)
        if valid_only:
            stmt = stmt.where(ResearchCandidate.is_valid.is_(True))

        stmt = stmt.order_by(desc(ResearchCandidate.id)).limit(limit).offset(offset)
        res = await session.execute(stmt)
        rows = res.all()

        submitted_ids = set((await session.execute(
            select(Submission.alpha_id).where(Submission.alpha_id.is_not(None))
        )).scalars())
        results: list[CandidateOut] = []
        for cand, ev in rows:
            is_3way = ev is not None and ev.evaluation_scheme in ("THREE_WAY_CHRONOLOGICAL", "SPLIT_3WAY_FINAL")
            is_grad = (cand.status == CandidateStatus.GRADUATED)

            c_score = (
                ev.val_composite_score
                if (is_3way and ev.val_composite_score is not None)
                else (ev.composite_score if ev else None)
            )
            ac_sharpe = (
                ev.val_after_cost_sharpe
                if (is_3way and ev.val_after_cost_sharpe is not None)
                else (ev.after_cost_sharpe if ev else None)
            )
            oos_s = (
                ev.val_sharpe
                if (is_3way and ev.val_sharpe is not None)
                else (ev.oos_sharpe if ev else None)
            )
            decay_s = (
                ev.is_val_decay
                if (is_3way and ev.is_val_decay is not None)
                else (ev.sharpe_decay if ev else None)
            )

            # Holdout metrics are revealed ONLY if candidate is graduated
            h_sharpe = ev.holdout_sharpe if (is_grad and ev) else None
            h_ac_sharpe = ev.holdout_after_cost_sharpe if (is_grad and ev) else None
            h_fitness = ev.holdout_fitness if (is_grad and ev) else None

            results.append(
                CandidateOut(
                    id=cand.id,
                    session_id=cand.session_id,
                    researcher_id=cand.researcher_id,
                    generation_attempt_id=cand.generation_attempt_id,
                    generation=cand.generation,
                    hypothesis=cand.hypothesis,
                    expression=cand.expression,
                    operator_count=cand.operator_count,
                    field_count=cand.field_count,
                    is_valid=cand.is_valid,
                    validation_error=cand.validation_error,
                    retry_count=cand.retry_count,
                    status=cand.status,
                    full_sharpe=round(float(ev.full_sharpe), 2) if ev and ev.is_verified and ev.full_sharpe is not None else None,
                    after_cost_sharpe=round(float(ac_sharpe), 2) if ac_sharpe is not None else None,
                    oos_sharpe=round(float(oos_s), 2) if oos_s is not None else None,
                    sharpe_decay=round(float(decay_s), 2) if decay_s is not None else None,
                    composite_score=round(float(c_score), 3) if c_score is not None else None,
                    is_submission_grade=(
                        (ev.internal_checks is not None or ev.checks is None)
                        and (ev.checks_total or 0) > 0 and ev.checks_passed == ev.checks_total
                    ) if (ev and ev.is_verified) else False,
                    platform_eligibility=cand.platform_eligibility_status or "UNCHECKED",
                    is_submitted=bool(cand.alpha_id and cand.alpha_id in submitted_ids),
                    alpha_id=cand.alpha_id,
                    simulation_record_id=cand.simulation_record_id,
                    prompt_strategy_id=cand.prompt_strategy_id,
                    model_ref=cand.model_ref,
                    temperature=cand.temperature,
                    datasets_used=cand.datasets_used or [],
                    fields_used=cand.fields_used or [],
                    raw_expression=cand.raw_expression,
                    is_verified=(bool(ev.is_verified) if ev is not None else None),
                    evaluation_provenance=(ev.provenance if ev is not None else None),
                    evaluation_scheme=(ev.evaluation_scheme if ev is not None else None),
                    is_graduated=is_grad,
                    val_composite_score=round(float(ev.val_composite_score), 3) if ev and ev.val_composite_score is not None else None,
                    val_sharpe=round(float(ev.val_sharpe), 2) if ev and ev.val_sharpe is not None else None,
                    val_after_cost_sharpe=round(float(ev.val_after_cost_sharpe), 2) if ev and ev.val_after_cost_sharpe is not None else None,
                    val_fitness=round(float(ev.val_fitness), 2) if ev and ev.val_fitness is not None else None,
                    val_turnover=round(float(ev.val_turnover), 3) if ev and ev.val_turnover is not None else None,
                    is_val_decay=round(float(ev.is_val_decay), 2) if ev and ev.is_val_decay is not None else None,
                    holdout_sharpe=round(float(h_sharpe), 2) if h_sharpe is not None else None,
                    holdout_after_cost_sharpe=round(float(h_ac_sharpe), 2) if h_ac_sharpe is not None else None,
                    holdout_fitness=round(float(h_fitness), 2) if h_fitness is not None else None,
                    simulation_settings=cand.simulation_settings,
                    settings_hash=cand.settings_hash,
                    settings_verification_status=cand.settings_verification_status,
                    candidate_provenance_hash=cand.candidate_provenance_hash,
                    created_at=cand.created_at,
                )
            )

        return results


@router.get("/sessions/{session_id}/generation-attempts/{attempt_id}", response_model=GenerationAttemptOut)
async def get_generation_attempt(session_id: int, attempt_id: int, state: State) -> GenerationAttemptOut:
    """Retrieves full generation attempt context including prompt, schema, and catalog context."""
    async with state.db.session() as session:
        stmt = select(GenerationAttempt).where(
            GenerationAttempt.id == attempt_id,
            GenerationAttempt.session_id == session_id,
        )
        res = await session.execute(stmt)
        attempt = res.scalar_one_or_none()
        if not attempt:
            raise refuse(
                404,
                "generation_attempt_not_found",
                f"Generation attempt {attempt_id} was not found in session {session_id}.",
            )
        return GenerationAttemptOut(
            id=attempt.id,
            session_id=attempt.session_id,
            researcher_id=attempt.researcher_id,
            prompt_strategy_id=attempt.prompt_strategy_id,
            resolution_tier=attempt.resolution_tier,
            generation=attempt.generation or 0,
            model_ref=attempt.model_ref,
            temperature=attempt.temperature,
            system_prompt=attempt.system_prompt,
            user_prompt=attempt.user_prompt,
            response_schema=attempt.response_schema,
            status=attempt.status,
            started_at=attempt.started_at,
            completed_at=attempt.completed_at,
            retry_count=attempt.retry_count,
            error_class=attempt.error_class,
            error_message=attempt.error_message,
            latency_seconds=attempt.latency_seconds,
            candidate_count=attempt.candidate_count,
            output_count=attempt.output_count,
            token_usage=attempt.token_usage,
            cost=attempt.cost,
            validation_outcome=attempt.validation_outcome,
            provider_request_id=attempt.provider_request_id,
            created_at=attempt.created_at,
        )


# --- Evaluation & Robustness (Phase 5 & 6) --------------------------------


class EvaluateCandidatesIn(BaseModel):
    candidate_ids: list[int] | None = None
    cost_bps: float = 5.0
    simulation_attempt: int = 1


class RobustnessRunIn(BaseModel):
    candidate_ids: list[int] | None = None


@router.post("/sessions/{session_id}/evaluate")
async def evaluate_session_candidates(
    session_id: int, payload: EvaluateCandidatesIn, state: State
) -> dict[str, Any]:
    """Batch-evaluates candidates using REAL WorldQuant BRAIN backtest simulation results.

    1. Synchronizes candidate states with linked SimulationRecord outcomes from BRAIN.
    2. Identifies candidates with completed backtests (valid alpha_id).
    3. Downloads and loads real 10-year daily PnL and turnover from BRAIN/DuckDB.
    4. Computes In-Sample, Out-of-Sample, Sharpe decay, and official BRAIN submission checks.
    5. Feeds real quantitative scorecards into the database for the self-learning reflection loop.
    """
    # P2-01: durable, idempotent job coordination is authoritative; all requests use
    # persisted readiness state.
    await state.orchestrator.sync_session_simulations(session_id)
    service = EvaluationJobService(
        state.db,
        state.evaluation,
        backfill=state.backfill,
        auth=state.auth,
    )
    jobs = await service.start_session_jobs(
        session_id,
        candidate_ids=payload.candidate_ids,
        simulation_attempt=payload.simulation_attempt,
        cost_bps=payload.cost_bps,
    )
    evaluated = sum(1 for job in jobs if job["status"] == "EVALUATED")
    failed = sum(1 for job in jobs if job["status"] == "FAILED")
    pending = sum(1 for job in jobs if job["status"] not in {"EVALUATED", "FAILED"})
    return {
        # Keep the legacy status value for clients that only understand the original
        # endpoint while exposing durable per-job phases in ``jobs``.
        "status": "failed" if failed and not pending and not evaluated else ("pending_simulations" if pending else "success"),
        "sessionId": session_id,
        "evaluatedCount": evaluated,
        "submissionGradeCount": 0,
        "pendingSimulations": pending,
        "jobs": jobs,
        "scorecards": [],
        "message": "Evaluation jobs are tracked by the backend and will advance when actual platform/vault evidence is available.",
    }


@router.get("/sessions/{session_id}/evaluation-jobs")
async def list_evaluation_jobs(
    session_id: int,
    state: State,
    candidate_ids: list[int] | None = Query(default=None),
    simulation_attempt: int = 1,
) -> list[dict[str, Any]]:
    """Read-only durable evaluation progress for refresh-safe frontend polling."""
    stmt = select(EvaluationJob).where(
        EvaluationJob.session_id == session_id,
        EvaluationJob.simulation_attempt == simulation_attempt,
    )
    if candidate_ids:
        stmt = stmt.where(EvaluationJob.candidate_id.in_(candidate_ids))
    async with state.db.session() as session:
        jobs = list((await session.execute(stmt.order_by(EvaluationJob.id))).scalars().all())
    return [EvaluationJobService._view(job) for job in jobs]


@router.get("/sessions/{session_id}/evaluations")
async def list_evaluations(
    session_id: int,
    state: State,
    submission_grade_only: bool = False,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Returns detailed evaluation scorecards for candidates in a session."""
    async with state.db.session() as session:
        stmt = (
            select(ResearchCandidate, ResearchEvaluation)
            .join(
                ResearchEvaluation,
                ResearchEvaluation.candidate_id == ResearchCandidate.id,
            )
            .where(
                ResearchCandidate.session_id == session_id,
                ResearchEvaluation.is_verified.is_(True),
                ResearchCandidate.status.in_([
                    CandidateStatus.EVALUATED,
                    CandidateStatus.ACCEPTED,
                    CandidateStatus.GRADUATED,
                ]),
            )
        )
        if submission_grade_only:
            stmt = stmt.where(
                or_(ResearchEvaluation.internal_checks.is_not(None), ResearchEvaluation.checks.is_(None)),
                ResearchEvaluation.checks_passed == ResearchEvaluation.checks_total,
                ResearchEvaluation.checks_total > 0,
            )
        stmt = stmt.order_by(
            desc(
                func.coalesce(
                    ResearchEvaluation.val_composite_score,
                    ResearchEvaluation.composite_score,
                )
            )
        ).limit(limit)
        res = await session.execute(stmt)
        rows = res.all()

    results: list[dict[str, Any]] = []
    for cand, ev in rows:
        is_grad = (cand.status == CandidateStatus.GRADUATED)
        h_sharpe = ev.holdout_sharpe if is_grad else None
        h_fitness = ev.holdout_fitness if is_grad else None
        h_turnover = ev.holdout_turnover if is_grad else None
        h_returns = ev.holdout_returns if is_grad else None
        h_drawdown = ev.holdout_drawdown if is_grad else None
        h_after_cost = ev.holdout_after_cost_sharpe if is_grad else None

        results.append(
            {
                "candidateId": cand.id,
                "expression": cand.expression,
                "hypothesis": cand.hypothesis,
                "evaluationScheme": ev.evaluation_scheme or "TWO_WAY_LEGACY",
                "isGraduated": is_grad,
                "isSharpe": round(float(ev.is_sharpe), 4) if ev.is_sharpe is not None else None,
                "isFitness": round(float(ev.is_fitness), 4) if ev.is_fitness is not None else None,
                "isTurnover": round(float(ev.is_turnover), 4) if ev.is_turnover is not None else None,
                "isDrawdown": round(float(ev.is_drawdown), 4) if ev.is_drawdown is not None else None,
                "isMargin": round(float(ev.is_margin), 6) if ev.is_margin is not None else None,
                "valSharpe": round(float(ev.val_sharpe), 4) if ev.val_sharpe is not None else None,
                "valFitness": round(float(ev.val_fitness), 4) if ev.val_fitness is not None else None,
                "valTurnover": round(float(ev.val_turnover), 4) if ev.val_turnover is not None else None,
                "valReturns": round(float(ev.val_returns), 4) if ev.val_returns is not None else None,
                "valDrawdown": round(float(ev.val_drawdown), 4) if ev.val_drawdown is not None else None,
                "valAfterCostSharpe": round(float(ev.val_after_cost_sharpe), 4) if ev.val_after_cost_sharpe is not None else None,
                "valCompositeScore": round(float(ev.val_composite_score), 4) if ev.val_composite_score is not None else None,
                "isValDecay": round(float(ev.is_val_decay), 4) if ev.is_val_decay is not None else None,
                "oosSharpe": round(float(ev.oos_sharpe), 4) if ev.oos_sharpe is not None else None,
                "oosFitness": round(float(ev.oos_fitness), 4) if ev.oos_fitness is not None else None,
                "oosTurnover": round(float(ev.oos_turnover), 4) if ev.oos_turnover is not None else None,
                "fullSharpe": round(float(ev.full_sharpe), 4) if ev.full_sharpe is not None else None,
                "fullFitness": round(float(ev.full_fitness), 4) if ev.full_fitness is not None else None,
                "afterCostSharpe": round(float(ev.after_cost_sharpe), 4) if ev.after_cost_sharpe is not None else None,
                "isOosGap": round(float(ev.is_oos_gap), 4) if ev.is_oos_gap is not None else None,
                "sharpeDecay": round(float(ev.sharpe_decay), 4) if ev.sharpe_decay is not None else None,
                "compositeScore": round(float(ev.composite_score), 4) if ev.composite_score is not None else None,
                "holdoutSharpe": round(float(h_sharpe), 4) if h_sharpe is not None else None,
                "holdoutFitness": round(float(h_fitness), 4) if h_fitness is not None else None,
                "holdoutTurnover": round(float(h_turnover), 4) if h_turnover is not None else None,
                "holdoutReturns": round(float(h_returns), 4) if h_returns is not None else None,
                "holdoutDrawdown": round(float(h_drawdown), 4) if h_drawdown is not None else None,
                "holdoutAfterCostSharpe": round(float(h_after_cost), 4) if h_after_cost is not None else None,
                "checks": ev.checks,
                "internalChecks": ev.internal_checks,
                "checksPassed": ev.checks_passed,
                "checksTotal": ev.checks_total,
                "isSubmissionGrade": (
                    (ev.internal_checks is not None or ev.checks is None)
                    and ev.checks_passed == ev.checks_total and ev.checks_total > 0
                ),
                "meetsInternalThresholds": (
                    ev.internal_checks is not None or ev.checks is None
                ) and ev.checks_passed == ev.checks_total and ev.checks_total > 0,
                "platformEligibility": cand.platform_eligibility_status or "UNCHECKED",
                "isSubmitted": False,
                "evaluatedAt": ev.evaluated_at.isoformat() if ev.evaluated_at else None,
                "holdoutEvaluatedAt": ev.holdout_evaluated_at.isoformat() if ev.holdout_evaluated_at else None,
            }
        )
    return results


@router.post("/sessions/{session_id}/candidates/{candidate_id}/graduate")
async def graduate_candidate_endpoint(
    session_id: int, candidate_id: int, state: State
) -> dict[str, Any]:
    """Evaluates final holdout and graduates an ACCEPTED candidate immutably."""
    try:
        return await ResearchWorkflowService(state).graduate(session_id, candidate_id)
    except ValueError as exc:
        raise refuse(404, "candidate_not_found", str(exc))
    except HoldoutIndeterminacyError as exc:
        raise refuse(409, "holdout_indeterminacy", str(exc))
    except HoldoutChronologyError as exc:
        raise refuse(400, "holdout_chronology_violation", str(exc))
    except HoldoutOverlapError as exc:
        raise refuse(409, "holdout_overlap_conflict", str(exc))
    except HoldoutImmutabilityError as exc:
        raise refuse(409, "holdout_immutable", str(exc))
    except InsufficientDataError as exc:
        raise refuse(400, "insufficient_data", str(exc))


@router.post("/sessions/{session_id}/robustness")
async def run_robustness_suite(
    session_id: int, payload: RobustnessRunIn, state: State
) -> dict[str, Any]:
    """Generates and executes perturbation test suites for evaluated candidates.

    For each candidate, generates neutralization, decay, and universe perturbation
    tests and records perturbed metric results. In production, each perturbation
    would re-simulate on BRAIN; here synthetic results are generated from the
    baseline with realistic degradation.
    """
    return await ResearchWorkflowService(state).robustness(session_id, payload)


@router.get("/sessions/{session_id}/robustness/{candidate_id}")
async def get_candidate_robustness(
    session_id: int, candidate_id: int, state: State
) -> dict[str, Any]:
    """Returns robustness test summary for a specific candidate."""
    summary = await state.robustness.get_candidate_robustness_summary(candidate_id)
    return {"candidateId": candidate_id, **summary}


@router.post("/sessions/{session_id}/learning/cycle")
async def trigger_learning_cycle(
    session_id: int,
    state: State,
    category: str | None = Query(None),
) -> dict[str, Any]:
    """Manually triggers a closed-loop self-learning reflection cycle."""
    return await ResearchWorkflowService(state).learning_cycle(session_id, category)


@router.get("/sessions/{session_id}/learning/insights")
async def list_insights(session_id: int, state: State) -> list[dict[str, Any]]:
    """Returns learned insights and quantitative heuristics discovered in this session."""
    async with state.db.session() as session:
        stmt = (
            select(ResearchInsight)
            .where(ResearchInsight.session_id == session_id)
            .order_by(desc(ResearchInsight.id))
        )
        res = await session.execute(stmt)
        insights = list(res.scalars().all())

        return [
            {
                "id": ins.id,
                "generation": ins.generation,
                "insightType": ins.insight_type,
                "summary": ins.summary,
                "rules": ins.rules,
                "evidence": ins.evidence,
                "batchAvgSharpe": ins.batch_avg_sharpe,
                "batchBestSharpe": ins.batch_best_sharpe,
                "applied": ins.applied,
                "createdAt": ins.created_at.isoformat() if ins.created_at else None,
            }
            for ins in insights
        ]


@router.get("/sessions/{session_id}/strategies")
async def list_prompt_strategies(session_id: int, state: State) -> list[dict[str, Any]]:
    """Returns all evolved and active prompt strategies for this session."""
    async with state.db.session() as session:
        stmt = (
            select(PromptStrategy)
            .where(PromptStrategy.session_id == session_id)
            .order_by(desc(PromptStrategy.version))
        )
        res = await session.execute(stmt)
        strats = list(res.scalars().all())

        return [
            {
                "id": s.id,
                "name": s.name,
                "version": s.version,
                "category": s.category,
                "guidance": s.guidance,
                "examples": s.examples,
                "antiPatterns": s.anti_patterns,
                "avgSharpe": s.avg_sharpe,
                "isActive": s.is_active,
                "createdAt": s.created_at.isoformat() if s.created_at else None,
            }
            for s in strats
        ]


# --- Leaderboards --------------------------------------------------------


@router.get("/leaderboards/alphas")
async def get_alpha_leaderboard(
    state: State,
    session_id: int | None = None,
    min_sharpe: float | None = None,
    submission_grade_only: bool = False,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Returns the multi-metric Alpha Candidate Leaderboard."""
    ranks = await state.leaderboard.get_alpha_leaderboard(
        session_id=session_id,
        min_sharpe=min_sharpe,
        submission_grade_only=submission_grade_only,
        limit=limit,
    )
    return [
        {
            "rank": r.rank,
            "candidateId": r.candidate_id,
            "expression": r.expression,
            "hypothesis": r.hypothesis,
            "modelLabel": r.model_label,
            "generation": r.generation,
            "compositeScore": r.composite_score,
            "fullSharpe": r.full_sharpe,
            "afterCostSharpe": r.after_cost_sharpe,
            "oosSharpe": r.oos_sharpe,
            "sharpeDecay": r.sharpe_decay,
            "turnover": r.turnover,
            "fitness": r.fitness,
            "isSubmissionGrade": r.is_submission_grade,
        }
        for r in ranks
    ]


@router.get("/leaderboards/researchers")
async def get_researcher_leaderboard(
    state: State, session_id: int | None = None, cohort_id: str | None = None
) -> list[dict[str, Any]]:
    """Returns the LLM Researcher Persona Leaderboard."""
    ranks = await state.leaderboard.get_researcher_leaderboard(
        session_id=session_id, cohort_id=cohort_id
    )
    return [
        {
            "rank": r.rank,
            "researcherId": r.researcher_id,
            "label": r.label,
            "modelRef": r.model_ref,
            "strategyFocus": r.strategy_focus,
            "totalGenerated": r.total_generated,
            "totalValid": r.total_valid,
            "validRate": r.valid_rate,
            "avgSharpe": r.avg_sharpe,
            "bestSharpe": r.best_sharpe,
            "submissionGradeCount": r.submission_grade_count,
            "topDecileRate": r.top_decile_rate,
            # P1-11 Fair Benchmarking & Funnel fields
            "status": r.status,
            "cohortId": r.cohort_id,
            "sampleCount": r.sample_count,
            "meanValidationSharpe": r.mean_validation_sharpe,
            "cohortBaselineSharpe": r.cohort_baseline_sharpe,
            "relativeLift": r.relative_lift,
            "shrunkRelativeLift": r.shrunk_relative_lift,
            "sharpeSe": r.sharpe_se,
            "lcbSharpe": r.lcb_sharpe,
            "astValidCount": r.ast_valid_count,
            "astValidRate": r.ast_valid_rate,
            "preflightPassCount": r.preflight_pass_count,
            "preflightPassRate": r.preflight_pass_rate,
            "simulatedCount": r.simulated_count,
            "simulationPassCount": r.simulation_pass_count,
            "simulationPassRate": r.simulation_pass_rate,
            "statisticalWinnerCount": r.statistical_winner_count,
            "statisticalDiscoveryRate": r.statistical_discovery_rate,
            "verifiedProvenanceCount": r.verified_provenance_count,
            "verifiedProvenanceRate": r.verified_provenance_rate,
            "totalTokens": r.total_tokens,
            "efficiencyCostUsd": r.efficiency_cost_usd,
            "efficiencyLatencySec": r.efficiency_latency_sec,
            "incomparabilityReasons": r.incomparability_reasons,
        }
        for r in ranks
    ]


@router.get("/leaderboards/prompts")
async def get_prompt_leaderboard(
    state: State,
    session_id: int | None = None,
    cohort_id: str | None = None,
    category: str | None = None,
    min_evaluated: int = 3,
) -> dict[str, Any]:
    """Returns the Prompt Version Leaderboard partitioned by comparable simulation cohorts."""
    res = await state.leaderboard.get_prompt_strategy_leaderboard(
        session_id=session_id,
        cohort_id=cohort_id,
        category=category,
        min_evaluated=min_evaluated,
    )
    return {
        "activeCohorts": [
            {
                "cohortId": c.cohort_id,
                "category": c.category,
                "region": c.region,
                "universe": c.universe,
                "neutralization": c.neutralization,
                "decay": c.decay,
                "delay": c.delay,
                "truncation": c.truncation,
                "nanHandling": c.nan_handling,
                "pasteurization": c.pasteurization,
                "unitHandling": c.unit_handling,
                "costBps": c.cost_bps,
                "evaluationScheme": c.evaluation_scheme,
                "valStart": c.val_start,
                "valEnd": c.val_end,
                "datasets": c.datasets,
                "candidateCount": c.candidate_count,
            }
            for c in res.active_cohorts
        ],
        "rankings": [
            {
                "promptVersionId": r.prompt_version_id,
                "category": r.category,
                "version": r.version,
                "authoringSource": r.authoring_source,
                "contentHash": r.content_hash,
                "parentVersionId": r.parent_version_id,
                "cohortId": r.cohort_id,
                "rank": r.rank,
                "status": r.status,
                "incomparabilityReasons": r.incomparability_reasons,
                "funnel": {
                    "generated": r.funnel.generated,
                    "valid": r.funnel.valid,
                    "evaluated": r.funnel.evaluated,
                    "accepted": r.funnel.accepted,
                    "validPct": r.funnel.valid_pct,
                    "evaluatedPct": r.funnel.evaluated_pct,
                    "acceptedPct": r.funnel.accepted_pct,
                },
                "validationMetrics": {
                    "medianSharpe": r.validation_metrics.median_sharpe,
                    "meanSharpe": r.validation_metrics.mean_sharpe,
                    "bestSharpe": r.validation_metrics.best_sharpe,
                    "medianFitness": r.validation_metrics.median_fitness,
                    "medianTurnover": r.validation_metrics.median_turnover,
                },
            }
            for r in res.rankings
        ],
        "incomparableRecords": [
            {
                "promptVersionId": r.prompt_version_id,
                "category": r.category,
                "version": r.version,
                "authoringSource": r.authoring_source,
                "contentHash": r.content_hash,
                "parentVersionId": r.parent_version_id,
                "cohortId": r.cohort_id,
                "rank": r.rank,
                "status": r.status,
                "incomparabilityReasons": r.incomparability_reasons,
                "funnel": {
                    "generated": r.funnel.generated,
                    "valid": r.funnel.valid,
                    "evaluated": r.funnel.evaluated,
                    "accepted": r.funnel.accepted,
                    "validPct": r.funnel.valid_pct,
                    "evaluatedPct": r.funnel.evaluated_pct,
                    "acceptedPct": r.funnel.accepted_pct,
                },
                "validationMetrics": {
                    "medianSharpe": r.validation_metrics.median_sharpe,
                    "meanSharpe": r.validation_metrics.mean_sharpe,
                    "bestSharpe": r.validation_metrics.best_sharpe,
                    "medianFitness": r.validation_metrics.median_fitness,
                    "medianTurnover": r.validation_metrics.median_turnover,
                },
            }
            for r in res.incomparable_records
        ],
    }


@router.get("/leaderboards/snapshots")
async def get_leaderboard_snapshots(
    session_id: int, state: State
) -> list[dict[str, Any]]:
    """Returns historical ranking snapshots for generation trend plotting."""
    async with state.db.session() as session:
        stmt = (
            select(LeaderboardSnapshot)
            .where(LeaderboardSnapshot.session_id == session_id)
            .order_by(LeaderboardSnapshot.generation.asc())
        )
        res = await session.execute(stmt)
        snaps = list(res.scalars().all())

        return [
            {
                "generation": s.generation,
                "dimension": s.dimension,
                "entries": s.entries,
                "snapshotAt": s.snapshot_at.isoformat() if s.snapshot_at else None,
            }
            for s in snaps
        ]


# --- Alpha Pool Curation -------------------------------------------------


@router.post("/pool/curate")
async def curate_alpha_pool(payload: CuratePoolIn, state: State) -> dict[str, Any]:
    """Curates an optimal low-correlation Alpha Pool targeting 50+ submission-grade alphas."""
    return await ResearchWorkflowService(state).curate_pool(payload)
