"""Parameter and market regime robustness testing engine.

Subject alpha candidates that pass preliminary evaluation to structured sensitivity
and perturbation tests:
- Neutralization perturbation (MARKET, SECTOR, INDUSTRY, SUBINDUSTRY)
- Decay perturbation (e.g. decay +/- 2)
- Universe perturbation (e.g. TOP3000 vs TOP1000)
- Sub-period regime stability (bull, bear, volatile regimes)

Records individual test outcomes and delta sensitivities in SQLite (ResearchRobustness).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import select

from ..db.models import ResearchCandidate, ResearchEvaluation, ResearchRobustness, utcnow

if TYPE_CHECKING:
    from ..db.sqlite import Database

log = structlog.get_logger(__name__)


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
    perturbed_sharpe: float
    perturbed_fitness: float
    perturbed_turnover: float
    sharpe_delta: float
    fitness_delta: float
    passed: bool
    message: str


class RobustnessEngine:
    """Executes and records parameter and regime sensitivity tests."""

    def __init__(self, db: Database) -> None:
        self.db = db

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
        perturbed_sharpe: float,
        perturbed_fitness: float,
        perturbed_turnover: float,
        simulation_record_id: int | None = None,
    ) -> RobustnessTestResult:
        """Evaluates sensitivity relative to baseline and persists record to SQLite."""
        async with self.db.session() as session:
            # Fetch baseline evaluation
            eval_stmt = select(ResearchEvaluation).where(
                ResearchEvaluation.candidate_id == candidate_id
            )
            res = await session.execute(eval_stmt)
            eval_row = res.scalar_one_or_none()

            b_sharpe = float(eval_row.full_sharpe or 0.0) if eval_row else 1.0
            b_fitness = float(eval_row.full_fitness or 0.0) if eval_row else 1.0

            sharpe_delta = round(perturbed_sharpe - b_sharpe, 4)
            fitness_delta = round(perturbed_fitness - b_fitness, 4)

            # Robustness criteria:
            # 1. Perturbed Sharpe must remain positive (> 0.8)
            # 2. Sharpe drop should not exceed 40% of baseline
            max_allowed_drop = abs(b_sharpe) * 0.40
            passed = perturbed_sharpe >= 0.80 and (b_sharpe - perturbed_sharpe) <= max_allowed_drop

            message = (
                f"Passed: Sharpe {perturbed_sharpe:.2f} (delta {sharpe_delta:+.2f})"
                if passed
                else f"Failed: Severe performance degradation under {test_type}={test_value} "
                f"(Sharpe: {perturbed_sharpe:.2f}, delta {sharpe_delta:+.2f})"
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
                tested_at=utcnow(),
            )
            session.add(record)

        log.info(
            "robustness.test_recorded",
            candidate_id=candidate_id,
            test_type=test_type,
            test_value=test_value,
            passed=passed,
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
        )

    async def get_candidate_robustness_summary(
        self, candidate_id: int
    ) -> dict[str, Any]:
        """Returns aggregated robustness stats for an alpha candidate."""
        async with self.db.session() as session:
            stmt = select(ResearchRobustness).where(
                ResearchRobustness.candidate_id == candidate_id
            )
            res = await session.execute(stmt)
            tests = list(res.scalars().all())

            if not tests:
                return {"tested": False, "pass_rate": 0.0, "tests": []}

            passed_count = sum(1 for t in tests if t.passed)
            pass_rate = round(passed_count / len(tests), 2)

            return {
                "tested": True,
                "total_tests": len(tests),
                "passed_tests": passed_count,
                "pass_rate": pass_rate,
                "all_passed": passed_count == len(tests),
                "tests": [
                    {
                        "test_type": t.test_type,
                        "test_value": t.test_value,
                        "perturbed_sharpe": t.perturbed_sharpe,
                        "sharpe_delta": t.sharpe_delta,
                        "passed": t.passed,
                        "message": t.message,
                    }
                    for t in tests
                ],
            }
