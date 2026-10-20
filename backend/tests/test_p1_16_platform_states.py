import asyncio

from sqlalchemy import select

from alpha_harness.db.models import (
    CandidateStatus,
    PlatformEligibilityStatus,
    ResearchCandidate,
    ResearchEvaluation,
    Submission,
)
from alpha_harness.db.sqlite import Database
from alpha_harness.evaluation.metrics import EvaluationEngine, EvaluationScorecard, build_canonical_spec


def _platform_candidate() -> ResearchCandidate:
    return ResearchCandidate(
        alpha_id="alpha-p1-16",
        simulation_settings={"region": "USA", "universe": "TOP3000", "delay": 1},
        settings_verification_status="VERIFIED",
    )


def test_internal_quality_alias_is_not_platform_eligibility():
    scorecard = EvaluationScorecard(candidate_id=1, meets_internal_thresholds=True)
    assert scorecard.is_submission_grade is True
    candidate = ResearchCandidate(status=CandidateStatus.ACCEPTED)
    assert (candidate.platform_eligibility_status or PlatformEligibilityStatus.UNCHECKED) == PlatformEligibilityStatus.UNCHECKED


def test_platform_check_requires_verified_complete_settings():
    candidate = ResearchCandidate(
        simulation_settings={"region": "USA"},
        settings_verification_status="VERIFIED",
    )
    evaluation = ResearchEvaluation(is_verified=False, provenance="SYNTHETIC_TEST_FIXTURE")
    assert not evaluation.is_verified
    assert (candidate.platform_eligibility_status or PlatformEligibilityStatus.UNCHECKED) == PlatformEligibilityStatus.UNCHECKED


def test_canonical_spec_uses_persisted_settings():
    candidate = ResearchCandidate(
        id=7,
        alpha_id="alpha-7",
        expression="rank(ts_mean(close, 5))",
        simulation_settings={
            "region": "EUR", "universe": "TOP1200", "delay": 2, "decay": 4,
            "neutralization": "SECTOR", "truncation": 0.17,
            "pasteurization": "OFF", "unit_handling": "VERIFY", "nan_handling": "ON",
        },
    )
    scorecard = EvaluationScorecard(candidate_id=7, cost_bps=5.0)
    scorecard.is_end_date = __import__("datetime").date(2020, 1, 1)
    scorecard.val_end_date = __import__("datetime").date(2021, 1, 1)
    spec = build_canonical_spec(candidate, scorecard, [__import__("datetime").date(2018, 1, 1), __import__("datetime").date(2022, 1, 1)])
    assert spec["sim_settings"]["region"] == "EUR"
    assert spec["sim_settings"]["truncation"] == 0.17
    assert spec["sim_settings"]["pasteurization"] == "OFF"


def test_platform_check_evidence_is_persisted_separately_from_internal_checks(tmp_path):
    async def run():
        db = Database(tmp_path / "p1_16.db")
        await db.create_all()
        try:
            async with db.session() as session:
                candidate = _platform_candidate()
                session.add(candidate)
                await session.flush()
                session.add(ResearchEvaluation(
                    candidate_id=candidate.id,
                    is_verified=True,
                    provenance="BRAIN_VAULT",
                    internal_checks={"val_sharpe_gte_1_25": True},
                    checks_passed=1,
                    checks_total=1,
                ))
                candidate_id = candidate.id
            engine = EvaluationEngine(db)
            status = await engine.record_platform_eligibility(
                candidate_id,
                {"CHECK_SHARPE": {"result": "PASS"}},
            )
            async with db.session() as session:
                candidate = await session.get(ResearchCandidate, candidate_id)
                evaluation = await session.get(ResearchEvaluation, candidate_id)
                assert status == PlatformEligibilityStatus.ELIGIBLE
                assert candidate.platform_eligibility_status == PlatformEligibilityStatus.ELIGIBLE
                assert evaluation.checks == {"CHECK_SHARPE": {"result": "PASS"}}
                assert evaluation.internal_checks == {"val_sharpe_gte_1_25": True}
                assert candidate.status != "SUBMITTED"
                assert (await session.execute(select(Submission))).scalars().all() == []
        finally:
            await db.dispose()

    asyncio.run(run())
