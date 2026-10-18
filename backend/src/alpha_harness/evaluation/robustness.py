"""Parameter and market regime robustness testing engine.

Subject alpha candidates that pass preliminary evaluation to structured sensitivity
and perturbation tests:
- Neutralization perturbation (MARKET, SECTOR, INDUSTRY, SUBINDUSTRY)
- Decay perturbation (e.g. decay +/- 2)
- Universe perturbation (e.g. TOP3000 vs TOP1000)
- Sub-period regime stability (bull, bear, volatile regimes)

Records individual test outcomes, provenance, and delta sensitivities in SQLite (ResearchRobustness).
Fabricated metrics (such as random.uniform degradation) are strictly prohibited.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
from sqlalchemy import select

from ..brain.errors import BrainAuthError, BrainError, BrainValidationError
from ..brain.schemas import SimulationRequest, SimulationSettings, SimulationType
from ..db.models import (
    ResearchCandidate,
    ResearchEvaluation,
    ResearchRobustness,
    SimStatus,
    utcnow,
)
from ..engine.lifecycle import describe_fields, extract_simulation_id, read_outcome
from ..vault.metrics import stats

if TYPE_CHECKING:
    from ..account import AuthService
    from ..brain.endpoints import BrainEndpoints
    from ..db.sqlite import Database
    from ..engine.slots import BatchEngine
    from ..engine.tracker import SimulationTracker
    from ..vault.store import AlphaVault

log = structlog.get_logger(__name__)

VALID_NEUTRALIZATIONS = frozenset({"MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY"})
VALID_UNIVERSES = frozenset({"TOP3000", "TOP1000", "TOP500", "TOP200", "ILLIQUID"})


@dataclass(slots=True)
class RobustnessTestPlan:
    """One planned perturbation test for an alpha candidate."""

    candidate_id: int
    test_type: str  # "neutralization", "decay", "universe", "time_period"
    test_value: str
    perturbed_settings: dict[str, Any]


@dataclass(slots=True)
class RobustnessTestResult:
    """Outcome of a single perturbation test."""

    test_type: str
    test_value: str
    baseline_sharpe: float
    baseline_fitness: float
    perturbed_sharpe: float | None = None
    perturbed_fitness: float | None = None
    perturbed_turnover: float | None = None
    sharpe_delta: float | None = None
    fitness_delta: float | None = None
    passed: bool = False
    message: str = ""
    status: str = "COMPLETED"
    result_kind: str = "GENUINE"
    is_synthetic: bool = False
    candidate_id: int | None = None
    baseline_eval_id: int | None = None
    brain_simulation_id: str | None = None
    perturbed_alpha_id: str | None = None
    perturbation_config: dict[str, Any] | None = None
    error_details: str | None = None
    val_start_date: date | None = None
    val_end_date: date | None = None


class RobustnessEngine:
    """Executes and records genuine parameter and regime sensitivity tests."""

    def __init__(
        self,
        db: Database,
        vault: AlphaVault | None = None,
        endpoints: BrainEndpoints | None = None,
        tracker: SimulationTracker | None = None,
        engine: BatchEngine | None = None,
        auth: AuthService | None = None,
    ) -> None:
        self.db = db
        self.vault = vault
        self.endpoints = endpoints
        self.tracker = tracker
        self.engine = engine
        self.auth = auth

    def validate_perturbation(self, test_type: str, test_value: str) -> tuple[bool, str]:
        """Validates that the requested perturbation parameter is supported by BRAIN."""
        if test_type == "neutralization":
            if test_value.upper() not in VALID_NEUTRALIZATIONS:
                return False, f"Invalid neutralization setting: '{test_value}' (valid: {sorted(VALID_NEUTRALIZATIONS)})"
            return True, ""
        if test_type == "decay":
            try:
                d = int(test_value)
                if d < 1 or d > 100:
                    return False, f"Decay value out of supported range [1, 100]: {d}"
                return True, ""
            except (ValueError, TypeError):
                return False, f"Invalid decay value: '{test_value}' must be an integer"
        if test_type == "universe":
            if test_value.upper() not in VALID_UNIVERSES:
                return False, f"Invalid universe setting: '{test_value}' (valid: {sorted(VALID_UNIVERSES)})"
            return True, ""
        return False, f"Unsupported perturbation test_type: '{test_type}'"

    def generate_test_suite(
        self,
        candidate_id: int,
        base_neutralization: str = "INDUSTRY",
        base_decay: int = 4,
        base_universe: str = "TOP3000",
    ) -> list[RobustnessTestPlan]:
        """Generates standard perturbation suite for a promising candidate."""
        suite: list[RobustnessTestPlan] = []

        # 1. Neutralization perturbations
        neutralizations = ["SECTOR", "MARKET"] if base_neutralization == "INDUSTRY" else ["INDUSTRY", "MARKET"]
        for neut in neutralizations:
            suite.append(
                RobustnessTestPlan(
                    candidate_id=candidate_id,
                    test_type="neutralization",
                    test_value=neut,
                    perturbed_settings={"neutralization": neut},
                )
            )

        # 2. Decay perturbations (shorter and longer memory)
        decays = [max(1, base_decay - 2), base_decay + 2, base_decay + 4]
        for d in decays:
            suite.append(
                RobustnessTestPlan(
                    candidate_id=candidate_id,
                    test_type="decay",
                    test_value=str(d),
                    perturbed_settings={"decay": d},
                )
            )

        # 3. Universe perturbation (test liquidity robustness)
        if base_universe == "TOP3000":
            suite.append(
                RobustnessTestPlan(
                    candidate_id=candidate_id,
                    test_type="universe",
                    test_value="TOP1000",
                    perturbed_settings={"universe": "TOP1000"},
                )
            )

        return suite

    async def record_test_result(
        self,
        candidate_id: int,
        test_type: str,
        test_value: str,
        perturbed_sharpe: float | None = None,
        perturbed_fitness: float | None = None,
        perturbed_turnover: float | None = None,
        simulation_record_id: int | None = None,
        *,
        baseline_sharpe: float | None = None,
        baseline_fitness: float | None = None,
        sharpe_delta: float | None = None,
        fitness_delta: float | None = None,
        passed: bool | None = None,
        message: str | None = None,
        status: str = "COMPLETED",
        result_kind: str = "GENUINE",
        is_synthetic: bool = False,
        baseline_eval_id: int | None = None,
        brain_simulation_id: str | None = None,
        perturbed_alpha_id: str | None = None,
        perturbation_config: dict[str, Any] | str | None = None,
        error_details: str | None = None,
        val_start_date: date | None = None,
        val_end_date: date | None = None,
    ) -> RobustnessTestResult:
        """Evaluates sensitivity relative to baseline and persists record to SQLite."""
        async with self.db.session() as session:
            # Fetch baseline evaluation if not provided
            if baseline_sharpe is None or baseline_fitness is None:
                if baseline_eval_id is not None:
                    eval_row = await session.get(ResearchEvaluation, baseline_eval_id)
                else:
                    eval_stmt = select(ResearchEvaluation).where(
                        ResearchEvaluation.candidate_id == candidate_id
                    )
                    res = await session.execute(eval_stmt)
                    eval_row = res.scalar_one_or_none()

                if eval_row:
                    is_3way = eval_row.evaluation_scheme in ("THREE_WAY_CHRONOLOGICAL", "SPLIT_3WAY_FINAL")
                    b_sharpe = float(eval_row.val_sharpe if (is_3way and eval_row.val_sharpe is not None) else (eval_row.full_sharpe or 1.0))
                    b_fitness = float(eval_row.val_fitness if (is_3way and eval_row.val_fitness is not None) else (eval_row.full_fitness or 1.0))
                    if baseline_eval_id is None:
                        baseline_eval_id = eval_row.id
                    if val_start_date is None:
                        val_start_date = eval_row.is_end_date if is_3way else eval_row.split_date
                    if val_end_date is None and is_3way:
                        val_end_date = eval_row.val_end_date
                else:
                    b_sharpe = 1.0
                    b_fitness = 1.0
            else:
                b_sharpe = baseline_sharpe
                b_fitness = baseline_fitness

            # For completed simulation runs with genuine metrics:
            if status == "COMPLETED" and perturbed_sharpe is not None:
                if sharpe_delta is None:
                    sharpe_delta = round(perturbed_sharpe - b_sharpe, 4)
                if fitness_delta is None:
                    fitness_delta = round((perturbed_fitness or 0.0) - b_fitness, 4)

                max_allowed_drop = abs(b_sharpe) * 0.40
                if passed is None:
                    passed = perturbed_sharpe >= 0.80 and (b_sharpe - perturbed_sharpe) <= max_allowed_drop

                if message is None:
                    message = (
                        f"Passed: Sharpe {perturbed_sharpe:.2f} (delta {sharpe_delta:+.2f})"
                        if passed
                        else f"Failed: Severe performance degradation under {test_type}={test_value} "
                        f"(Sharpe: {perturbed_sharpe:.2f}, delta {sharpe_delta:+.2f})"
                    )
            else:
                passed = False if passed is None else passed
                message = message or f"Perturbation evaluation status: {status}"

            config_str = (
                json.dumps(perturbation_config)
                if isinstance(perturbation_config, dict)
                else perturbation_config
            )

            record = ResearchRobustness(
                candidate_id=candidate_id,
                test_type=test_type,
                test_value=test_value,
                simulation_record_id=simulation_record_id,
                perturbed_sharpe=perturbed_sharpe,
                perturbed_fitness=perturbed_fitness,
                perturbed_turnover=perturbed_turnover,
                sharpe_delta=sharpe_delta,
                fitness_delta=fitness_delta,
                passed=passed,
                message=message,
                status=status,
                result_kind=result_kind,
                is_synthetic=is_synthetic,
                baseline_eval_id=baseline_eval_id,
                brain_simulation_id=brain_simulation_id,
                perturbed_alpha_id=perturbed_alpha_id,
                perturbation_config=config_str,
                error_details=error_details,
                val_start_date=val_start_date,
                val_end_date=val_end_date,
                tested_at=utcnow(),
            )
            session.add(record)

        log.info(
            "robustness.test_recorded",
            candidate_id=candidate_id,
            test_type=test_type,
            test_value=test_value,
            status=status,
            result_kind=result_kind,
            passed=passed,
        )

        cfg_dict = (
            perturbation_config
            if isinstance(perturbation_config, dict)
            else (json.loads(config_str) if config_str else None)
        )

        return RobustnessTestResult(
            test_type=test_type,
            test_value=test_value,
            baseline_sharpe=b_sharpe,
            baseline_fitness=b_fitness,
            perturbed_sharpe=perturbed_sharpe,
            perturbed_fitness=perturbed_fitness,
            perturbed_turnover=perturbed_turnover,
            sharpe_delta=sharpe_delta,
            fitness_delta=fitness_delta,
            passed=passed,
            message=message,
            status=status,
            result_kind=result_kind,
            is_synthetic=is_synthetic,
            candidate_id=candidate_id,
            baseline_eval_id=baseline_eval_id,
            brain_simulation_id=brain_simulation_id,
            perturbed_alpha_id=perturbed_alpha_id,
            perturbation_config=cfg_dict,
            error_details=error_details,
            val_start_date=val_start_date,
            val_end_date=val_end_date,
        )

    async def execute_perturbation(
        self,
        candidate_id: int,
        plan: RobustnessTestPlan,
        baseline_eval_id: int | None = None,
    ) -> RobustnessTestResult:
        """Executes a real BRAIN perturbation simulation for the candidate.

        Modifies ONLY the specified perturbation parameter, submits the simulation to BRAIN,
        retrieves genuine performance data into AlphaVault, and evaluates against the baseline
        validation window. Returns NOT_RUN if BRAIN execution is unavailable, UNSUPPORTED if
        settings are invalid, or FAILED if simulation fails. Never manufactures metrics.
        """
        # 1. Validate requested perturbation parameter
        is_valid, reason = self.validate_perturbation(plan.test_type, plan.test_value)
        if not is_valid:
            return await self.record_test_result(
                candidate_id=candidate_id,
                test_type=plan.test_type,
                test_value=plan.test_value,
                baseline_eval_id=baseline_eval_id,
                status="UNSUPPORTED",
                result_kind="GENUINE",
                is_synthetic=False,
                passed=False,
                message=f"Unsupported perturbation: {reason}",
                error_details=reason,
                perturbation_config=plan.perturbed_settings,
            )

        # 2. Fetch candidate and baseline evaluation from database
        async with self.db.session() as session:
            cand = await session.get(ResearchCandidate, candidate_id)
            if cand is None:
                return await self.record_test_result(
                    candidate_id=candidate_id,
                    test_type=plan.test_type,
                    test_value=plan.test_value,
                    status="FAILED",
                    message=f"Candidate {candidate_id} not found",
                )

            if baseline_eval_id:
                ev = await session.get(ResearchEvaluation, baseline_eval_id)
            else:
                ev_res = await session.execute(
                    select(ResearchEvaluation).where(ResearchEvaluation.candidate_id == candidate_id)
                )
                ev = ev_res.scalar_one_or_none()

            if ev is None:
                return await self.record_test_result(
                    candidate_id=candidate_id,
                    test_type=plan.test_type,
                    test_value=plan.test_value,
                    status="FAILED",
                    message="Candidate has no baseline evaluation to compare against",
                )

            is_3way = ev.evaluation_scheme in ("THREE_WAY_CHRONOLOGICAL", "SPLIT_3WAY_FINAL")
            val_start_date = ev.is_end_date if is_3way else ev.split_date
            val_end_date = ev.val_end_date if is_3way else None
            baseline_sharpe = float(ev.val_sharpe if (is_3way and ev.val_sharpe is not None) else (ev.full_sharpe or 1.0))
            baseline_fitness = float(ev.val_fitness if (is_3way and ev.val_fitness is not None) else (ev.full_fitness or 1.0))
            ev_id = ev.id

        # 3. Construct the exact simulation configuration (varying ONLY the target parameter)
        region = cand.sim_region or "USA"
        universe = cand.sim_universe or "TOP3000"
        delay = cand.sim_delay if cand.sim_delay is not None else 1
        decay = cand.sim_decay if cand.sim_decay is not None else 4
        neutralization = cand.sim_neutralization or "INDUSTRY"
        truncation = getattr(cand, "sim_truncation", None) or 0.08
        pasteurization = getattr(cand, "sim_pasteurization", None) or "ON"
        unit_handling = getattr(cand, "sim_unit_handling", None) or "VERIFY"
        nan_handling = getattr(cand, "sim_nan_handling", None) or "OFF"
        language = getattr(cand, "language", None) or getattr(cand, "sim_language", None) or "FASTEXPR"

        if plan.test_type == "neutralization":
            neutralization = plan.test_value.upper()
        elif plan.test_type == "decay":
            decay = int(plan.test_value)
        elif plan.test_type == "universe":
            universe = plan.test_value.upper()

        perturbation_diff = {
            "parameter_changed": plan.test_type,
            "original_value": getattr(cand, f"sim_{plan.test_type}", None),
            "perturbed_value": plan.test_value,
            "resulting_settings": {
                "region": region,
                "universe": universe,
                "delay": delay,
                "decay": decay,
                "neutralization": neutralization,
                "truncation": truncation,
                "pasteurization": pasteurization,
                "unit_handling": unit_handling,
                "nan_handling": nan_handling,
                "language": language,
            },
        }

        # 4. Check BRAIN availability
        if self.endpoints is None or getattr(self.endpoints, "client", None) is None:
            return await self.record_test_result(
                candidate_id=candidate_id,
                test_type=plan.test_type,
                test_value=plan.test_value,
                baseline_sharpe=baseline_sharpe,
                baseline_fitness=baseline_fitness,
                baseline_eval_id=ev_id,
                status="NOT_RUN",
                result_kind="GENUINE",
                is_synthetic=False,
                passed=False,
                message="BRAIN simulation execution unavailable in current environment; synthetic evaluation prohibited",
                perturbation_config=perturbation_diff,
                val_start_date=val_start_date,
                val_end_date=val_end_date,
            )

        if self.auth is not None and not self.auth.session.authenticated:
            return await self.record_test_result(
                candidate_id=candidate_id,
                test_type=plan.test_type,
                test_value=plan.test_value,
                baseline_sharpe=baseline_sharpe,
                baseline_fitness=baseline_fitness,
                baseline_eval_id=ev_id,
                status="NOT_RUN",
                result_kind="GENUINE",
                is_synthetic=False,
                passed=False,
                message="BRAIN session unauthenticated; synthetic evaluation prohibited",
                perturbation_config=perturbation_diff,
                val_start_date=val_start_date,
                val_end_date=val_end_date,
            )

        # 5. Submit real simulation request
        sim_request = SimulationRequest(
            type=SimulationType.REGULAR,
            settings=SimulationSettings(
                instrument_type="EQUITY",
                region=region,
                universe=universe,
                delay=delay,
                decay=decay,
                neutralization=neutralization,
                truncation=truncation,
                pasteurization=pasteurization,
                unit_handling=unit_handling,
                nan_handling=nan_handling,
                language=language,
                visualization=False,
            ),
            regular=cand.expression,
        )

        try:
            resp = await self.endpoints.create_simulation(sim_request)
        except BrainValidationError as exc:
            desc = describe_fields(getattr(exc, "fields", None), exc.message)
            return await self.record_test_result(
                candidate_id=candidate_id,
                test_type=plan.test_type,
                test_value=plan.test_value,
                baseline_sharpe=baseline_sharpe,
                baseline_fitness=baseline_fitness,
                baseline_eval_id=ev_id,
                status="UNSUPPORTED",
                result_kind="GENUINE",
                is_synthetic=False,
                passed=False,
                message=f"BRAIN rejected perturbation settings: {desc}",
                error_details=str(exc),
                perturbation_config=perturbation_diff,
                val_start_date=val_start_date,
                val_end_date=val_end_date,
            )
        except BrainAuthError as exc:
            return await self.record_test_result(
                candidate_id=candidate_id,
                test_type=plan.test_type,
                test_value=plan.test_value,
                baseline_sharpe=baseline_sharpe,
                baseline_fitness=baseline_fitness,
                baseline_eval_id=ev_id,
                status="NOT_RUN",
                result_kind="GENUINE",
                is_synthetic=False,
                passed=False,
                message=f"BRAIN authentication lost: {exc}",
                error_details=str(exc),
                perturbation_config=perturbation_diff,
                val_start_date=val_start_date,
                val_end_date=val_end_date,
            )
        except Exception as exc:
            return await self.record_test_result(
                candidate_id=candidate_id,
                test_type=plan.test_type,
                test_value=plan.test_value,
                baseline_sharpe=baseline_sharpe,
                baseline_fitness=baseline_fitness,
                baseline_eval_id=ev_id,
                status="FAILED",
                result_kind="GENUINE",
                is_synthetic=False,
                passed=False,
                message=f"Simulation submission failed: {exc}",
                error_details=str(exc),
                perturbation_config=perturbation_diff,
                val_start_date=val_start_date,
                val_end_date=val_end_date,
            )

        sim_id = extract_simulation_id(resp.location)
        if not sim_id:
            return await self.record_test_result(
                candidate_id=candidate_id,
                test_type=plan.test_type,
                test_value=plan.test_value,
                baseline_sharpe=baseline_sharpe,
                baseline_fitness=baseline_fitness,
                baseline_eval_id=ev_id,
                status="FAILED",
                result_kind="GENUINE",
                is_synthetic=False,
                passed=False,
                message="No simulation ID in Location header from BRAIN",
                perturbation_config=perturbation_diff,
                val_start_date=val_start_date,
                val_end_date=val_end_date,
            )

        # 6. Poll simulation until terminal outcome
        poll_deadline = time.monotonic() + 300.0
        outcome = None
        while time.monotonic() < poll_deadline:
            try:
                poll_resp = await self.endpoints.read_simulation(sim_id)
                outcome = read_outcome(poll_resp.status, poll_resp.body, poll_resp.retry_after)
            except Exception as exc:
                outcome = None
                log.warning("robustness.poll_failed", error=str(exc))

            if outcome is not None and outcome.kind in ("finished", "gone"):
                break
            delay = getattr(outcome, "delay", None) or 1.0
            await asyncio.sleep(min(delay, 2.0))

        if outcome is None or outcome.kind not in ("finished", "gone"):
            return await self.record_test_result(
                candidate_id=candidate_id,
                test_type=plan.test_type,
                test_value=plan.test_value,
                baseline_sharpe=baseline_sharpe,
                baseline_fitness=baseline_fitness,
                baseline_eval_id=ev_id,
                status="FAILED",
                result_kind="GENUINE",
                is_synthetic=False,
                brain_simulation_id=sim_id,
                passed=False,
                message="Perturbation simulation timed out waiting for BRAIN completion",
                perturbation_config=perturbation_diff,
                val_start_date=val_start_date,
                val_end_date=val_end_date,
            )

        if outcome.status != SimStatus.COMPLETE or not outcome.alpha_id:
            return await self.record_test_result(
                candidate_id=candidate_id,
                test_type=plan.test_type,
                test_value=plan.test_value,
                baseline_sharpe=baseline_sharpe,
                baseline_fitness=baseline_fitness,
                baseline_eval_id=ev_id,
                status="FAILED",
                result_kind="GENUINE",
                is_synthetic=False,
                brain_simulation_id=sim_id,
                passed=False,
                message=f"Simulation failed on BRAIN: {outcome.message or outcome.status}",
                error_details=outcome.message,
                perturbation_config=perturbation_diff,
                val_start_date=val_start_date,
                val_end_date=val_end_date,
            )

        perturbed_alpha_id = outcome.alpha_id

        # 7. Ingest and validate performance series through AlphaVault
        if self.vault is None:
            return await self.record_test_result(
                candidate_id=candidate_id,
                test_type=plan.test_type,
                test_value=plan.test_value,
                baseline_sharpe=baseline_sharpe,
                baseline_fitness=baseline_fitness,
                baseline_eval_id=ev_id,
                status="FAILED",
                result_kind="GENUINE",
                is_synthetic=False,
                brain_simulation_id=sim_id,
                perturbed_alpha_id=perturbed_alpha_id,
                passed=False,
                message="AlphaVault unavailable to ingest/validate series",
                perturbation_config=perturbation_diff,
                val_start_date=val_start_date,
                val_end_date=val_end_date,
            )

        lacking = await self.vault.lacking_series([perturbed_alpha_id])
        if lacking:
            try:
                pnl_rs = await self.endpoints.get_recordset(perturbed_alpha_id, "pnl")
                to_rs = await self.endpoints.get_recordset(perturbed_alpha_id, "turnover")
                try:
                    yearly_rs = await self.endpoints.get_recordset(perturbed_alpha_id, "yearly-stats")
                    y_rows = yearly_rs.rows()
                except Exception:
                    y_rows = []
                await self.vault.save_pnl(perturbed_alpha_id, pnl_rs.rows(), to_rs.rows(), y_rows)
            except Exception as exc:
                return await self.record_test_result(
                    candidate_id=candidate_id,
                    test_type=plan.test_type,
                    test_value=plan.test_value,
                    baseline_sharpe=baseline_sharpe,
                    baseline_fitness=baseline_fitness,
                    baseline_eval_id=ev_id,
                    status="FAILED",
                    result_kind="GENUINE",
                    is_synthetic=False,
                    brain_simulation_id=sim_id,
                    perturbed_alpha_id=perturbed_alpha_id,
                    passed=False,
                    message=f"Failed to ingest performance series from BRAIN: {exc}",
                    error_details=str(exc),
                    perturbation_config=perturbation_diff,
                    val_start_date=val_start_date,
                    val_end_date=val_end_date,
                )

        verified = await self.vault.load_verified_series(perturbed_alpha_id)
        if verified is None:
            return await self.record_test_result(
                candidate_id=candidate_id,
                test_type=plan.test_type,
                test_value=plan.test_value,
                baseline_sharpe=baseline_sharpe,
                baseline_fitness=baseline_fitness,
                baseline_eval_id=ev_id,
                status="FAILED",
                result_kind="GENUINE",
                is_synthetic=False,
                brain_simulation_id=sim_id,
                perturbed_alpha_id=perturbed_alpha_id,
                passed=False,
                message="Perturbed performance series failed AlphaVault verification",
                perturbation_config=perturbation_diff,
                val_start_date=val_start_date,
                val_end_date=val_end_date,
            )

        dates, pnls, tos = verified

        # 8. Evaluate over the exact baseline validation dates
        d_arr = np.array(dates, dtype="datetime64[D]")
        if val_start_date and val_end_date:
            v_start = np.datetime64(val_start_date, "D")
            v_end = np.datetime64(val_end_date, "D")
            mask = (d_arr >= v_start) & (d_arr < v_end)
        elif val_start_date:
            v_start = np.datetime64(val_start_date, "D")
            mask = d_arr >= v_start
        else:
            mask = np.ones(len(dates), dtype=bool)

        if np.sum(mask) < 2:
            return await self.record_test_result(
                candidate_id=candidate_id,
                test_type=plan.test_type,
                test_value=plan.test_value,
                baseline_sharpe=baseline_sharpe,
                baseline_fitness=baseline_fitness,
                baseline_eval_id=ev_id,
                status="FAILED",
                result_kind="GENUINE",
                is_synthetic=False,
                brain_simulation_id=sim_id,
                perturbed_alpha_id=perturbed_alpha_id,
                passed=False,
                message="Fewer than 2 observations in perturbed validation window",
                perturbation_config=perturbation_diff,
                val_start_date=val_start_date,
                val_end_date=val_end_date,
            )

        pnl_arr = np.array(pnls, dtype=np.float64)
        to_arr = np.array(tos, dtype=np.float64)
        st = stats(pnl_arr[mask], to_arr[mask])
        if st is None or st.sharpe is None:
            return await self.record_test_result(
                candidate_id=candidate_id,
                test_type=plan.test_type,
                test_value=plan.test_value,
                baseline_sharpe=baseline_sharpe,
                baseline_fitness=baseline_fitness,
                baseline_eval_id=ev_id,
                status="FAILED",
                result_kind="GENUINE",
                is_synthetic=False,
                brain_simulation_id=sim_id,
                perturbed_alpha_id=perturbed_alpha_id,
                passed=False,
                message="Could not compute metrics over validation window",
                perturbation_config=perturbation_diff,
                val_start_date=val_start_date,
                val_end_date=val_end_date,
            )

        perturbed_sharpe = round(st.sharpe, 4)
        perturbed_fitness = round(st.fitness or 0.0, 4)
        perturbed_turnover = round(st.turnover, 4)

        sharpe_delta = round(perturbed_sharpe - baseline_sharpe, 4)
        fitness_delta = round(perturbed_fitness - baseline_fitness, 4)

        max_allowed_drop = abs(baseline_sharpe) * 0.40
        passed = perturbed_sharpe >= 0.80 and (baseline_sharpe - perturbed_sharpe) <= max_allowed_drop
        message = (
            f"Passed: Sharpe {perturbed_sharpe:.2f} (delta {sharpe_delta:+.2f})"
            if passed
            else f"Failed: Severe performance degradation under {plan.test_type}={plan.test_value} "
            f"(Sharpe: {perturbed_sharpe:.2f}, delta {sharpe_delta:+.2f})"
        )

        return await self.record_test_result(
            candidate_id=candidate_id,
            test_type=plan.test_type,
            test_value=plan.test_value,
            baseline_sharpe=baseline_sharpe,
            baseline_fitness=baseline_fitness,
            perturbed_sharpe=perturbed_sharpe,
            perturbed_fitness=perturbed_fitness,
            perturbed_turnover=perturbed_turnover,
            sharpe_delta=sharpe_delta,
            fitness_delta=fitness_delta,
            passed=passed,
            message=message,
            status="COMPLETED",
            result_kind="GENUINE",
            is_synthetic=False,
            baseline_eval_id=ev_id,
            brain_simulation_id=sim_id,
            perturbed_alpha_id=perturbed_alpha_id,
            perturbation_config=perturbation_diff,
            val_start_date=val_start_date,
            val_end_date=val_end_date,
        )

    async def run_candidate_suite(
        self,
        candidate_id: int,
        suite: list[RobustnessTestPlan] | None = None,
        baseline_eval_id: int | None = None,
    ) -> list[RobustnessTestResult]:
        """Runs perturbation tests for a candidate and returns individual outcomes."""
        if suite is None:
            async with self.db.session() as session:
                cand = await session.get(ResearchCandidate, candidate_id)
                if cand is None:
                    return []
                suite = self.generate_test_suite(
                    candidate_id=candidate_id,
                    base_neutralization=cand.sim_neutralization or "INDUSTRY",
                    base_decay=cand.sim_decay or 4,
                    base_universe=cand.sim_universe or "TOP3000",
                )

        results: list[RobustnessTestResult] = []
        for plan in suite:
            res = await self.execute_perturbation(
                candidate_id=candidate_id,
                plan=plan,
                baseline_eval_id=baseline_eval_id,
            )
            results.append(res)
        return results

    async def get_candidate_robustness_summary(
        self, candidate_id: int
    ) -> dict[str, Any]:
        """Returns aggregated robustness stats for an alpha candidate.

        Mandatory safeguard: legacy synthetic records and incomplete/failed/unsupported runs
        are strictly excluded from genuine pass_rate, total_tests, and all_passed determinations.
        """
        async with self.db.session() as session:
            stmt = select(ResearchRobustness).where(
                ResearchRobustness.candidate_id == candidate_id
            )
            res = await session.execute(stmt)
            tests = list(res.scalars().all())

            if not tests:
                return {
                    "tested": False,
                    "total_tests": 0,
                    "passed_tests": 0,
                    "pass_rate": 0.0,
                    "all_passed": False,
                    "tests": [],
                    "legacy_synthetic_count": 0,
                }

            # Filter genuine, completed tests for production statistics
            genuine_tests = [
                t for t in tests
                if not t.is_synthetic and t.result_kind == "GENUINE" and t.status == "COMPLETED"
            ]
            legacy_synthetic = [
                t for t in tests
                if t.is_synthetic or t.result_kind == "LEGACY_SYNTHETIC"
            ]

            passed_count = sum(1 for t in genuine_tests if t.passed)
            pass_rate = round(passed_count / len(genuine_tests), 2) if genuine_tests else 0.0
            all_passed = len(genuine_tests) > 0 and (passed_count == len(genuine_tests))

            return {
                "tested": len(genuine_tests) > 0,
                "total_tests": len(genuine_tests),
                "passed_tests": passed_count,
                "pass_rate": pass_rate,
                "all_passed": all_passed,
                "legacy_synthetic_count": len(legacy_synthetic),
                "tests": [
                    {
                        "test_type": t.test_type,
                        "test_value": t.test_value,
                        "perturbed_sharpe": t.perturbed_sharpe,
                        "sharpe_delta": t.sharpe_delta,
                        "passed": t.passed,
                        "status": t.status,
                        "result_kind": t.result_kind,
                        "is_synthetic": bool(t.is_synthetic),
                        "brain_simulation_id": t.brain_simulation_id,
                        "perturbed_alpha_id": t.perturbed_alpha_id,
                        "error_details": t.error_details,
                        "val_start_date": t.val_start_date.isoformat() if t.val_start_date else None,
                        "val_end_date": t.val_end_date.isoformat() if t.val_end_date else None,
                        "message": t.message,
                    }
                    for t in tests
                ],
            }
