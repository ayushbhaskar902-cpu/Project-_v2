"""Comprehensive regression tests for P1-11: Fair Researcher/Model Leaderboard.

Verifies:
1. Submission volume alone cannot improve a model's quality rank.
2. Models are compared only on eligible matched or stratified cohorts (no silent pooling).
3. Different universes, datasets, or simulation settings cannot be silently pooled as equivalent.
4. Small samples receive appropriate uncertainty treatment and INSUFFICIENT_EVIDENCE status.
5. Insufficient matched evidence does not produce a fabricated rank.
6. Missing denominators do not produce misleading funnel rates.
7. Verified and unverified provenance are distinguished.
8. Latency and cost are disclosed without silently distorting the quality score.
9. Changing final holdout data does not change leaderboard results (P0-03 holdout isolation).
10. Results and ranking calculations are deterministic.
11. Existing API consumers and relevant regression tests continue to work.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from alpha_harness.db.models import (
    Base,
    CandidateStatus,
    ResearchCandidate,
    ResearchEvaluation,
    ResearchResearcher,
    ResearchSession,
)
from alpha_harness.db.sqlite import Database
from alpha_harness.leaderboard.service import (
    LeaderboardService,
    ResearcherRankEntry,
)


@pytest.fixture
def test_db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "test_p1_11.db")
    asyncio.run(db.create_all())
    return db


def _canonical_spec(
    region: str = "USA",
    universe: str = "TOP3000",
    neutralization: str = "INDUSTRY",
    decay: int = 4,
    delay: int = 1,
    cost_bps: float = 5.0,
    val_start: str = "2018-01-01",
    val_end: str = "2020-01-01",
    datasets: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "cost_bps": cost_bps,
        "evaluation_scheme": "SPLIT_3WAY_DEV",
        "dates": {"val_start": val_start, "val_end": val_end},
        "datasets_used": datasets or ["fundamental6"],
        "sim_settings": {
            "region": region,
            "universe": universe,
            "neutralization": neutralization,
            "decay": decay,
            "delay": delay,
            "truncation": 0.08,
            "nan_handling": "OFF",
            "pasteurization": "ON",
            "unit_handling": "VERIFY",
        },
    }


def test_submission_volume_alone_cannot_improve_quality_rank(test_db: Database):
    """P1-11 Acceptance Criterion 2:
    A low-quality model with high submission volume (e.g. 30 candidates, Sharpe 0.8)
    MUST NOT outrank a high-quality model with smaller sample (e.g. 6 candidates, Sharpe 2.0)."""
    async def _run():
        async with test_db.session() as s:
            sess = ResearchSession(name="Volume Bias Test", region="USA", universe="TOP3000")
            s.add(sess)
            await s.flush()

            # Researcher A: Quality model (low volume, high Sharpe)
            r_a = ResearchResearcher(
                session_id=sess.id,
                label="Quality Model",
                model_ref="google:gemini-2.5-pro",
                strategy_focus="momentum",
                total_generated=6,
            )
            # Researcher B: Volume model (high volume, low Sharpe)
            r_b = ResearchResearcher(
                session_id=sess.id,
                label="Volume Model",
                model_ref="mock:volume-spammer",
                strategy_focus="momentum",
                total_generated=50,
            )
            s.add_all([r_a, r_b])
            await s.flush()

            spec = _canonical_spec()

            # Model A: 6 candidates with validation Sharpe 2.0
            for i in range(6):
                cand = ResearchCandidate(
                    session_id=sess.id,
                    researcher_id=r_a.id,
                    expression=f"rank(close_{i})",
                    status=CandidateStatus.EVALUATED,
                    is_valid=True,
                    generation=1,
                    datasets_used=["fundamental6"],
                )
                s.add(cand)
                await s.flush()
                ev = ResearchEvaluation(
                    candidate_id=cand.id,
                    is_verified=True,
                    provenance="BRAIN_VAULT",
                    val_sharpe=2.0 + (0.05 * (i % 3)),
                    val_fitness=1.5,
                    val_composite_score=2.0,
                    checks_passed=8,
                    checks_total=8,
                    cost_bps=5.0,
                    evaluation_scheme="SPLIT_3WAY_DEV",
                    is_end_date=date(2018, 1, 1),
                    val_end_date=date(2020, 1, 1),
                    canonical_spec=spec,
                )
                s.add(ev)

            # Model B: 30 candidates with mediocre validation Sharpe 0.8
            # Even if all 30 pass checks, submission volume must not win.
            for i in range(30):
                cand = ResearchCandidate(
                    session_id=sess.id,
                    researcher_id=r_b.id,
                    expression=f"ts_mean(volume_{i}, 5)",
                    status=CandidateStatus.EVALUATED,
                    is_valid=True,
                    generation=1,
                    datasets_used=["fundamental6"],
                )
                s.add(cand)
                await s.flush()
                ev = ResearchEvaluation(
                    candidate_id=cand.id,
                    is_verified=True,
                    provenance="BRAIN_VAULT",
                    val_sharpe=0.80 + (0.02 * (i % 5)),
                    val_fitness=1.0,
                    val_composite_score=0.8,
                    checks_passed=8,
                    checks_total=8,
                    cost_bps=5.0,
                    evaluation_scheme="SPLIT_3WAY_DEV",
                    is_end_date=date(2018, 1, 1),
                    val_end_date=date(2020, 1, 1),
                    canonical_spec=spec,
                )
                s.add(ev)

        svc = LeaderboardService(test_db)
        ranks = await svc.get_researcher_leaderboard(session_id=sess.id)

        assert len(ranks) == 2
        entry_a = next(r for r in ranks if r.researcher_id == r_a.id)
        entry_b = next(r for r in ranks if r.researcher_id == r_b.id)

        # Both have N >= 5, so both are RANKED
        assert entry_a.status == "RANKED"
        assert entry_b.status == "RANKED"

        # Model A must be RANK #1 despite Model B having 5x more submissible alphas
        assert entry_a.rank == 1, f"Expected Quality Model to rank #1, got rank {entry_a.rank}"
        assert entry_b.rank == 2, f"Expected Volume Model to rank #2, got rank {entry_b.rank}"

        assert entry_b.submission_grade_count > entry_a.submission_grade_count
        assert entry_a.lcb_sharpe > entry_b.lcb_sharpe
        assert entry_a.shrunk_relative_lift > entry_b.shrunk_relative_lift

    asyncio.run(_run())


def test_matched_cohort_stratification_no_silent_pooling(test_db: Database):
    """P1-11 Acceptance Criteria 1 & 3:
    Models evaluated on different universes (e.g. TOP3000 vs TOP500) must NOT
    be silently pooled as if they were directly comparable."""
    async def _run():
        async with test_db.session() as s:
            sess = ResearchSession(name="Stratification Test", region="USA", universe="TOP3000")
            s.add(sess)
            await s.flush()

            r1 = ResearchResearcher(session_id=sess.id, label="Model 3000", strategy_focus="momentum")
            r2 = ResearchResearcher(session_id=sess.id, label="Model 500", strategy_focus="momentum")
            s.add_all([r1, r2])
            await s.flush()

            spec_3000 = _canonical_spec(universe="TOP3000")
            spec_500 = _canonical_spec(universe="TOP500")

            # R1 in TOP3000
            for i in range(5):
                c = ResearchCandidate(session_id=sess.id, researcher_id=r1.id, is_valid=True, status=CandidateStatus.EVALUATED)
                s.add(c)
                await s.flush()
                s.add(ResearchEvaluation(
                    candidate_id=c.id, is_verified=True, provenance="BRAIN_VAULT", val_sharpe=1.8,
                    is_end_date=date(2018, 1, 1), val_end_date=date(2020, 1, 1), canonical_spec=spec_3000
                ))

            # R2 in TOP500
            for i in range(5):
                c = ResearchCandidate(session_id=sess.id, researcher_id=r2.id, is_valid=True, status=CandidateStatus.EVALUATED)
                s.add(c)
                await s.flush()
                s.add(ResearchEvaluation(
                    candidate_id=c.id, is_verified=True, provenance="BRAIN_VAULT", val_sharpe=2.5,
                    is_end_date=date(2018, 1, 1), val_end_date=date(2020, 1, 1), canonical_spec=spec_500
                ))

        svc = LeaderboardService(test_db)
        ranks = await svc.get_researcher_leaderboard(session_id=sess.id)

        # Because R1 and R2 operate on completely different universes with zero overlap,
        # they must NOT be ranked against each other on a fabricated joint leaderboard.
        for r in ranks:
            assert r.status in ("INCOMPARABLE", "INSUFFICIENT_EVIDENCE")
            assert r.rank is None, f"Expected unranked (rank=None) for incomparable cohorts, got {r.rank}"

    asyncio.run(_run())


def test_small_sample_uncertainty_and_insufficient_evidence(test_db: Database):
    """P1-11 Acceptance Criterion 4:
    A model with N < 5 candidates (even with a lucky high Sharpe) must receive
    INSUFFICIENT_EVIDENCE status and cannot occupy top rank over statistically backed models."""
    async def _run():
        async with test_db.session() as s:
            sess = ResearchSession(name="Small Sample Test", region="USA", universe="TOP3000")
            s.add(sess)
            await s.flush()

            r_lucky = ResearchResearcher(session_id=sess.id, label="Lucky Small Model", strategy_focus="value")
            r_solid = ResearchResearcher(session_id=sess.id, label="Solid Large Model", strategy_focus="value")
            s.add_all([r_lucky, r_solid])
            await s.flush()

            spec = _canonical_spec()

            # Lucky model: N = 2 candidates with extreme Sharpe = 3.5
            for i in range(2):
                c = ResearchCandidate(session_id=sess.id, researcher_id=r_lucky.id, is_valid=True, status=CandidateStatus.EVALUATED)
                s.add(c)
                await s.flush()
                s.add(ResearchEvaluation(
                    candidate_id=c.id, is_verified=True, provenance="BRAIN_VAULT", val_sharpe=3.5,
                    is_end_date=date(2018, 1, 1), val_end_date=date(2020, 1, 1), canonical_spec=spec
                ))

            # Solid model: N = 10 candidates with robust Sharpe = 1.9
            for i in range(10):
                c = ResearchCandidate(session_id=sess.id, researcher_id=r_solid.id, is_valid=True, status=CandidateStatus.EVALUATED)
                s.add(c)
                await s.flush()
                s.add(ResearchEvaluation(
                    candidate_id=c.id, is_verified=True, provenance="BRAIN_VAULT", val_sharpe=1.9,
                    is_end_date=date(2018, 1, 1), val_end_date=date(2020, 1, 1), canonical_spec=spec
                ))

        svc = LeaderboardService(test_db)
        ranks = await svc.get_researcher_leaderboard(session_id=sess.id)

        entry_lucky = next(r for r in ranks if r.researcher_id == r_lucky.id)
        entry_solid = next(r for r in ranks if r.researcher_id == r_solid.id)

        # Lucky model has N=2 < 5 -> INSUFFICIENT_EVIDENCE
        assert entry_lucky.status == "INSUFFICIENT_EVIDENCE"
        assert entry_lucky.rank is None

        # Solid model has N=10 >= 5 -> RANKED #1
        assert entry_solid.status == "RANKED"
        assert entry_solid.rank == 1

    asyncio.run(_run())


def test_transparent_evaluation_funnel_missing_denominators(test_db: Database):
    """P1-11 Acceptance Criterion 3:
    Expose transparent funnel counts and explicit rates; missing denominators must be None,
    not fake 0.0 or 1.0."""
    async def _run():
        async with test_db.session() as s:
            sess = ResearchSession(name="Funnel Test", region="USA", universe="TOP3000")
            s.add(sess)
            await s.flush()

            # Researcher with 0 generated candidates
            r_empty = ResearchResearcher(session_id=sess.id, label="Empty Model", total_generated=0)
            # Researcher with 4 generated, 2 valid, 1 simulated
            r_partial = ResearchResearcher(session_id=sess.id, label="Partial Model", total_generated=4)
            s.add_all([r_empty, r_partial])
            await s.flush()

            # R_partial:
            # Cand 1: AST invalid
            c1 = ResearchCandidate(session_id=sess.id, researcher_id=r_partial.id, is_valid=False, status=CandidateStatus.DRAFT)
            # Cand 2: AST invalid
            c2 = ResearchCandidate(session_id=sess.id, researcher_id=r_partial.id, is_valid=False, status=CandidateStatus.DRAFT)
            # Cand 3: AST valid, but not simulated
            c3 = ResearchCandidate(session_id=sess.id, researcher_id=r_partial.id, is_valid=True, status=CandidateStatus.DRAFT)
            # Cand 4: AST valid, simulated
            c4 = ResearchCandidate(session_id=sess.id, researcher_id=r_partial.id, is_valid=True, status=CandidateStatus.EVALUATED)
            s.add_all([c1, c2, c3, c4])
            await s.flush()

            ev4 = ResearchEvaluation(
                candidate_id=c4.id, is_verified=True, provenance="BRAIN_VAULT", val_sharpe=1.2,
                checks_passed=8, checks_total=8, is_end_date=date(2018, 1, 1), val_end_date=date(2020, 1, 1),
                canonical_spec=_canonical_spec()
            )
            s.add(ev4)

        svc = LeaderboardService(test_db)
        ranks = await svc.get_researcher_leaderboard(session_id=sess.id)

        entry_empty = next(r for r in ranks if r.researcher_id == r_empty.id)
        entry_partial = next(r for r in ranks if r.researcher_id == r_partial.id)

        # Empty model: missing denominators are explicitly None
        assert entry_empty.ast_valid_rate is None
        assert entry_empty.preflight_pass_rate is None
        assert entry_empty.simulation_pass_rate is None
        assert entry_empty.statistical_discovery_rate is None

        # Partial model:
        assert entry_partial.total_generated == 4
        assert entry_partial.ast_valid_count == 2
        assert entry_partial.ast_valid_rate == 0.50
        assert entry_partial.simulated_count == 1
        assert entry_partial.simulation_pass_count == 1
        assert entry_partial.simulation_pass_rate == 1.0

    asyncio.run(_run())


def test_verified_and_unverified_provenance_distinguished(test_db: Database):
    """P1-11 Acceptance Criterion 6:
    Unverified or proxy simulation results must NOT be counted as genuine verified evidence."""
    async def _run():
        async with test_db.session() as s:
            sess = ResearchSession(name="Provenance Test", region="USA", universe="TOP3000")
            s.add(sess)
            await s.flush()

            r = ResearchResearcher(session_id=sess.id, label="Mixed Model", strategy_focus="momentum")
            s.add(r)
            await s.flush()

            spec = _canonical_spec()

            # 3 verified evaluations
            for i in range(3):
                c = ResearchCandidate(session_id=sess.id, researcher_id=r.id, is_valid=True, status=CandidateStatus.EVALUATED)
                s.add(c)
                await s.flush()
                s.add(ResearchEvaluation(
                    candidate_id=c.id, is_verified=True, provenance="BRAIN_VAULT", val_sharpe=1.5,
                    is_end_date=date(2018, 1, 1), val_end_date=date(2020, 1, 1), canonical_spec=spec
                ))

            # 3 unverified / synthetic evaluations
            for i in range(3):
                c = ResearchCandidate(session_id=sess.id, researcher_id=r.id, is_valid=True, status=CandidateStatus.EVALUATED)
                s.add(c)
                await s.flush()
                s.add(ResearchEvaluation(
                    candidate_id=c.id, is_verified=False, provenance="LOCAL_PROXY", val_sharpe=4.0,
                    is_end_date=date(2018, 1, 1), val_end_date=date(2020, 1, 1), canonical_spec=spec
                ))

        svc = LeaderboardService(test_db)
        ranks = await svc.get_researcher_leaderboard(session_id=sess.id)
        entry = ranks[0]

        # Only the 3 verified evaluations can count toward sample_count
        assert entry.verified_provenance_count == 3
        assert entry.simulated_count == 6
        assert entry.verified_provenance_rate == 0.50
        assert entry.sample_count == 3

    asyncio.run(_run())


def test_efficiency_metrics_disclosed_without_distorting_quality(test_db: Database):
    """P1-11 Acceptance Criterion 5:
    Token usage and costs must be disclosed transparently without penalizing or distorting quality ranking."""
    async def _run():
        async with test_db.session() as s:
            sess = ResearchSession(name="Efficiency Test", region="USA", universe="TOP3000")
            s.add(sess)
            await s.flush()

            # R_expensive: Higher token usage, but superior alpha quality
            r_expensive = ResearchResearcher(
                session_id=sess.id, label="Expensive Model", model_ref="openai:o1",
                strategy_focus="momentum", total_tokens=500_000
            )
            # R_cheap: Low token usage, but inferior alpha quality
            r_cheap = ResearchResearcher(
                session_id=sess.id, label="Cheap Model", model_ref="google:gemini-flash",
                strategy_focus="momentum", total_tokens=10_000
            )
            s.add_all([r_expensive, r_cheap])
            await s.flush()

            spec = _canonical_spec()

            for i in range(5):
                c = ResearchCandidate(session_id=sess.id, researcher_id=r_expensive.id, is_valid=True, status=CandidateStatus.EVALUATED)
                s.add(c)
                await s.flush()
                s.add(ResearchEvaluation(
                    candidate_id=c.id, is_verified=True, provenance="BRAIN_VAULT", val_sharpe=2.2,
                    is_end_date=date(2018, 1, 1), val_end_date=date(2020, 1, 1), canonical_spec=spec
                ))

            for i in range(5):
                c = ResearchCandidate(session_id=sess.id, researcher_id=r_cheap.id, is_valid=True, status=CandidateStatus.EVALUATED)
                s.add(c)
                await s.flush()
                s.add(ResearchEvaluation(
                    candidate_id=c.id, is_verified=True, provenance="BRAIN_VAULT", val_sharpe=1.1,
                    is_end_date=date(2018, 1, 1), val_end_date=date(2020, 1, 1), canonical_spec=spec
                ))

        svc = LeaderboardService(test_db)
        ranks = await svc.get_researcher_leaderboard(session_id=sess.id)

        entry_exp = next(r for r in ranks if r.researcher_id == r_expensive.id)
        entry_chp = next(r for r in ranks if r.researcher_id == r_cheap.id)

        # Efficiency metrics are disclosed
        assert entry_exp.total_tokens == 500_000
        assert entry_exp.efficiency_cost_usd > entry_chp.efficiency_cost_usd

        # Quality rank is NOT distorted by cost: Expensive model with higher Sharpe ranks #1
        assert entry_exp.rank == 1
        assert entry_chp.rank == 2

    asyncio.run(_run())


def test_holdout_isolation_p0_03_zero_leakage(test_db: Database):
    """P1-11 Acceptance Criterion 6:
    Mutating final holdout data must produce ZERO change in researcher leaderboard rankings,
    relative lift, shrunk lift, or LCB."""
    async def _run():
        async with test_db.session() as s:
            sess = ResearchSession(name="Holdout Leakage Test", region="USA", universe="TOP3000")
            s.add(sess)
            await s.flush()

            r1 = ResearchResearcher(session_id=sess.id, label="Model 1", strategy_focus="momentum")
            r2 = ResearchResearcher(session_id=sess.id, label="Model 2", strategy_focus="momentum")
            s.add_all([r1, r2])
            await s.flush()

            spec = _canonical_spec()

            for i in range(6):
                c1 = ResearchCandidate(session_id=sess.id, researcher_id=r1.id, is_valid=True, status=CandidateStatus.EVALUATED)
                c2 = ResearchCandidate(session_id=sess.id, researcher_id=r2.id, is_valid=True, status=CandidateStatus.EVALUATED)
                s.add_all([c1, c2])
                await s.flush()

                ev1 = ResearchEvaluation(
                    candidate_id=c1.id, is_verified=True, provenance="BRAIN_VAULT", val_sharpe=2.0,
                    holdout_sharpe=0.5, is_end_date=date(2018, 1, 1), val_end_date=date(2020, 1, 1), canonical_spec=spec
                )
                ev2 = ResearchEvaluation(
                    candidate_id=c2.id, is_verified=True, provenance="BRAIN_VAULT", val_sharpe=1.2,
                    holdout_sharpe=3.0, is_end_date=date(2018, 1, 1), val_end_date=date(2020, 1, 1), canonical_spec=spec
                )
                s.add_all([ev1, ev2])

        svc = LeaderboardService(test_db)
        ranks_before = await svc.get_researcher_leaderboard(session_id=sess.id)

        # Mutate holdout metrics dramatically: reverse them
        async with test_db.session() as s:
            res = await s.execute(select(ResearchEvaluation))
            all_evs = list(res.scalars().all())
            for ev in all_evs:
                ev.holdout_sharpe = 999.0 if (ev.holdout_sharpe or 0.0) < 1.0 else -999.0
            await s.flush()

        ranks_after = await svc.get_researcher_leaderboard(session_id=sess.id)

        assert len(ranks_before) == len(ranks_after)
        for b, a in zip(ranks_before, ranks_after):
            assert b.rank == a.rank
            assert b.lcb_sharpe == a.lcb_sharpe
            assert b.shrunk_relative_lift == a.shrunk_relative_lift
            assert b.mean_validation_sharpe == a.mean_validation_sharpe

    asyncio.run(_run())


def test_deterministic_ranking_calculations(test_db: Database):
    """P1-11 Acceptance Criterion 8:
    Leaderboard results must be completely deterministic across repeated invocations."""
    async def _run():
        async with test_db.session() as s:
            sess = ResearchSession(name="Determinism Test", region="USA", universe="TOP3000")
            s.add(sess)
            await s.flush()

            for i in range(3):
                r = ResearchResearcher(session_id=sess.id, label=f"Model {i}", strategy_focus="momentum")
                s.add(r)
                await s.flush()
                for j in range(5):
                    c = ResearchCandidate(session_id=sess.id, researcher_id=r.id, is_valid=True, status=CandidateStatus.EVALUATED)
                    s.add(c)
                    await s.flush()
                    s.add(ResearchEvaluation(
                        candidate_id=c.id, is_verified=True, provenance="BRAIN_VAULT", val_sharpe=1.0 + 0.3 * i,
                        is_end_date=date(2018, 1, 1), val_end_date=date(2020, 1, 1), canonical_spec=_canonical_spec()
                    ))

        svc = LeaderboardService(test_db)
        r1 = await svc.get_researcher_leaderboard(session_id=sess.id)
        r2 = await svc.get_researcher_leaderboard(session_id=sess.id)

        assert [e.researcher_id for e in r1] == [e.researcher_id for e in r2]
        assert [e.rank for e in r1] == [e.rank for e in r2]
        assert [e.lcb_sharpe for e in r1] == [e.lcb_sharpe for e in r2]

    asyncio.run(_run())


def test_multiple_testing_p1_10_discovery_reuse(test_db: Database):
    """P1-11 Acceptance Criterion 2:
    Reuses P1-10 multiplicity-corrected split / FDR discovery logic to compute
    statistical_winner_count and statistical_discovery_rate consistently."""
    async def _run():
        async with test_db.session() as s:
            sess = ResearchSession(name="Discovery Reuse Test", region="USA", universe="TOP3000")
            s.add(sess)
            await s.flush()

            r = ResearchResearcher(session_id=sess.id, label="Discovery Model", strategy_focus="momentum")
            s.add(r)
            await s.flush()

            spec = _canonical_spec()

            # 8 candidates: 2 with long duration & high Sharpe (qualifying for P1-10 SIGNIFICANT_WINNER),
            # 6 with mediocre Sharpe (failing significance)
            for i in range(8):
                c = ResearchCandidate(session_id=sess.id, researcher_id=r.id, is_valid=True, status=CandidateStatus.EVALUATED)
                s.add(c)
                await s.flush()
                # candidates 0 and 1: Sharpe 2.5 over 4 years -> SE ≈ 0.503, t ≈ 4.97, LCB95 ≈ 1.51 > 0
                # candidates 2-7: Sharpe 0.3 over 4 years
                sr = 2.5 if i < 2 else 0.3
                s.add(ResearchEvaluation(
                    candidate_id=c.id, is_verified=True, provenance="BRAIN_VAULT", val_sharpe=sr,
                    is_end_date=date(2016, 1, 1), val_end_date=date(2020, 1, 1), canonical_spec=spec
                ))

        svc = LeaderboardService(test_db)
        ranks = await svc.get_researcher_leaderboard(session_id=sess.id)
        entry = ranks[0]

        assert entry.sample_count == 8
        assert entry.statistical_winner_count == 2
        assert entry.statistical_discovery_rate == 0.25

    asyncio.run(_run())


def test_cohort_id_filtering(test_db: Database):
    """P1-11 Acceptance Criterion 1:
    When a specific cohort_id is provided, researchers are evaluated strictly
    within that cohort, and models with no evaluations in that cohort are marked INCOMPARABLE."""
    from alpha_harness.leaderboard.service import compute_cohort_fingerprint

    async def _run():
        async with test_db.session() as s:
            sess = ResearchSession(name="Cohort Filter Test", region="USA", universe="TOP3000")
            s.add(sess)
            await s.flush()

            r_usa = ResearchResearcher(session_id=sess.id, label="USA Model", strategy_focus="momentum")
            r_eur = ResearchResearcher(session_id=sess.id, label="EUR Model", strategy_focus="momentum")
            s.add_all([r_usa, r_eur])
            await s.flush()

            spec_usa = _canonical_spec(region="USA")
            spec_eur = _canonical_spec(region="EUR")

            ev_usa_first = None
            c_usa_first = None
            for i in range(5):
                c_u = ResearchCandidate(session_id=sess.id, researcher_id=r_usa.id, is_valid=True, status=CandidateStatus.EVALUATED)
                c_e = ResearchCandidate(session_id=sess.id, researcher_id=r_eur.id, is_valid=True, status=CandidateStatus.EVALUATED)
                s.add_all([c_u, c_e])
                await s.flush()
                ev_u = ResearchEvaluation(
                    candidate_id=c_u.id, is_verified=True, provenance="BRAIN_VAULT", val_sharpe=1.8,
                    is_end_date=date(2018, 1, 1), val_end_date=date(2020, 1, 1), canonical_spec=spec_usa
                )
                ev_e = ResearchEvaluation(
                    candidate_id=c_e.id, is_verified=True, provenance="BRAIN_VAULT", val_sharpe=2.0,
                    is_end_date=date(2018, 1, 1), val_end_date=date(2020, 1, 1), canonical_spec=spec_eur
                )
                s.add_all([ev_u, ev_e])
                if i == 0:
                    ev_usa_first = ev_u
                    c_usa_first = c_u

            usa_cohort, _, _ = compute_cohort_fingerprint("momentum", ev_usa_first, c_usa_first)

        svc = LeaderboardService(test_db)
        # Explicitly filter by the USA cohort
        filtered_ranks = await svc.get_researcher_leaderboard(session_id=sess.id, cohort_id=usa_cohort)

        entry_usa = next(r for r in filtered_ranks if r.researcher_id == r_usa.id)
        entry_eur = next(r for r in filtered_ranks if r.researcher_id == r_eur.id)

        assert entry_usa.status == "RANKED"
        assert entry_usa.cohort_id == usa_cohort
        assert entry_usa.rank == 1

        assert entry_eur.status == "INCOMPARABLE"
        assert entry_eur.rank is None
        assert entry_eur.sample_count == 0

    asyncio.run(_run())


def test_api_endpoint_response_schema(test_db: Database):
    """P1-11 Acceptance Criterion 7:
    The API endpoint returns full camelCase schema preserving all legacy and new fields."""
    from types import SimpleNamespace
    from alpha_harness.api.research import get_researcher_leaderboard as api_endpoint

    async def _run():
        async with test_db.session() as s:
            sess = ResearchSession(name="API Schema Test", region="USA", universe="TOP3000")
            s.add(sess)
            await s.flush()

            r = ResearchResearcher(
                session_id=sess.id, label="API Model", model_ref="mock:test-model",
                strategy_focus="momentum", total_generated=10, total_tokens=150_000
            )
            s.add(r)
            await s.flush()

            spec = _canonical_spec()
            for i in range(5):
                c = ResearchCandidate(session_id=sess.id, researcher_id=r.id, is_valid=True, status=CandidateStatus.EVALUATED)
                s.add(c)
                await s.flush()
                s.add(ResearchEvaluation(
                    candidate_id=c.id, is_verified=True, provenance="BRAIN_VAULT", val_sharpe=2.0,
                    checks_passed=8, checks_total=8, is_end_date=date(2018, 1, 1), val_end_date=date(2020, 1, 1),
                    canonical_spec=spec
                ))

        state = SimpleNamespace(leaderboard=LeaderboardService(test_db))
        results = await api_endpoint(state=state, session_id=sess.id)

        assert len(results) == 1
        item = results[0]

        # Verify all legacy fields
        assert item["researcherId"] == r.id
        assert item["label"] == "API Model"
        assert item["modelRef"] == "mock:test-model"
        assert item["strategyFocus"] == "momentum"
        assert item["totalGenerated"] == 10
        assert item["rank"] == 1
        assert "submissionGradeCount" in item
        assert "topDecileRate" in item

        # Verify new P1-11 fields
        assert item["status"] == "RANKED"
        assert item["sampleCount"] == 5
        assert item["meanValidationSharpe"] == 2.0
        assert item["astValidCount"] == 5
        assert item["astValidRate"] == 0.50
        assert item["simulatedCount"] == 5
        assert item["simulationPassCount"] == 5
        assert item["simulationPassRate"] == 1.0
        assert item["verifiedProvenanceCount"] == 5
        assert item["verifiedProvenanceRate"] == 1.0
        assert item["totalTokens"] == 150_000
        assert item["efficiencyCostUsd"] is not None
        assert "shrunkRelativeLift" in item
        assert "lcbSharpe" in item
        assert "statisticalDiscoveryRate" in item

    asyncio.run(_run())
