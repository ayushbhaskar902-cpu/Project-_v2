"""Comprehensive regression and integration test suite for P0-03:
Proper In-Sample / Validation / Final Holdout Separation and Full-History Contamination Safeguard.
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from pathlib import Path
import pytest
from sqlalchemy import select

from alpha_harness.db.duck import Catalog
from alpha_harness.db.models import (
    CandidateStatus,
    HoldoutAssessment,
    ResearchCandidate,
    ResearchEvaluation,
    ResearchResearcher,
    ResearchSession,
    utcnow,
)
from alpha_harness.db.sqlite import Database
from alpha_harness.evaluation.metrics import (
    EvaluationEngine,
    EvaluationScorecard,
    HoldoutChronologyError,
    HoldoutImmutabilityError,
    HoldoutIndeterminacyError,
    HoldoutOverlapError,
    InsufficientDataError,
    compute_validation_composite_score,
    evaluate_pnl_series,
)
from alpha_harness.leaderboard.service import LeaderboardService
from alpha_harness.learning.loop import SelfLearningEngine
from alpha_harness.learning.memory import ResearchMemory
from alpha_harness.vault.store import AlphaVault


def make_calendar_days(num_days: int, start_date: date = date(2023, 1, 2)) -> list[date]:
    """Generates consecutive weekday trading dates."""
    dates: list[date] = []
    cur = start_date
    while len(dates) < num_days:
        if cur.weekday() < 5:
            dates.append(cur)
        cur += timedelta(days=1)
    return dates


def make_pnl_turnover_series(
    dates: list[date],
    base_daily_pnl: float = 100.0,
    turnover: float = 0.15,
) -> tuple[list[float], list[float]]:
    """Generates cumulative PnL series and flat turnover series."""
    pnl: list[float] = []
    cur = 0.0
    for i in range(len(dates)):
        cur += base_daily_pnl + (i % 3) * 10.0
        pnl.append(cur)
    to = [turnover] * len(dates)
    return pnl, to


async def populate_duckdb_vault(
    catalog: Catalog,
    alpha_id: str,
    dates: list[date],
    pnl: list[float],
    turnover: list[float],
) -> None:
    """Inserts verified series directly into DuckDB alpha_pnl table."""
    vault = AlphaVault(catalog)
    pnl_rows = [{"date": d, "pnl": p} for d, p in zip(dates, pnl)]
    to_rows = [{"date": d, "turnover": t} for d, t in zip(dates, turnover)]
    await vault.save_pnl(alpha_id, pnl_rows, to_rows, yearly_rows=[])


# ===========================================================================
# 1. Deterministic Three-Way Boundaries and Validation Calculations
# ===========================================================================


def test_three_way_deterministic_boundaries():
    """Verifies that 100 trading days split deterministically into 60% IS, 20% Val, 20% Holdout."""
    dates = make_calendar_days(100)
    pnl, to = make_pnl_turnover_series(dates, base_daily_pnl=100.0)

    scorecard = evaluate_pnl_series(
        candidate_id=1,
        dates=dates,
        pnl_series=pnl,
        turnover_series=to,
        evaluate_holdout=False,
    )

    assert scorecard.evaluation_scheme == "SPLIT_3WAY_DEV"
    assert scorecard.is_end_date == dates[60]  # first 60 observations
    assert scorecard.val_end_date == dates[80]  # next 20 observations
    assert scorecard.val_sharpe is not None
    assert scorecard.val_composite_score is not None
    # Holdout metrics remain strictly None in development
    assert scorecard.holdout_sharpe is None
    assert scorecard.holdout_fitness is None
    assert scorecard.holdout_evaluated_at is None


def test_minimum_observation_and_boundary_guards():
    """Verifies that series with fewer than 60 observations or invalid bounds are rejected."""
    # Under 60 observations
    dates = make_calendar_days(50)
    pnl, to = make_pnl_turnover_series(dates)
    with pytest.raises(InsufficientDataError, match="minimum 60 required"):
        evaluate_pnl_series(
            candidate_id=1,
            dates=dates,
            pnl_series=pnl,
            turnover_series=to,
        )

    # Inverted boundaries
    dates_100 = make_calendar_days(100)
    pnl_100, to_100 = make_pnl_turnover_series(dates_100)
    with pytest.raises(InsufficientDataError, match="must be strictly before"):
        evaluate_pnl_series(
            candidate_id=1,
            dates=dates_100,
            pnl_series=pnl_100,
            turnover_series=to_100,
            is_end_date=dates_100[70],
            val_end_date=dates_100[50],  # Inverted: val_end before is_end
        )


# ===========================================================================
# 2. Invariance Tests (Holdout Mutation & IS Mutation)
# ===========================================================================


def test_holdout_mutation_invariance_in_development():
    """Mutating final holdout segment data has ZERO effect on validation metrics or acceptance."""
    dates = make_calendar_days(100)
    pnl_orig, to = make_pnl_turnover_series(dates, base_daily_pnl=100.0)

    # Baseline evaluation
    sc_base = evaluate_pnl_series(
        candidate_id=1,
        dates=dates,
        pnl_series=pnl_orig,
        turnover_series=to,
        evaluate_holdout=False,
    )

    # Mutate only the holdout partition (last 20 days: indices 80..99)
    pnl_mutated = list(pnl_orig)
    for i in range(80, 100):
        # Massive crash or spike in holdout data
        pnl_mutated[i] = pnl_mutated[79] - 50000.0

    sc_mutated = evaluate_pnl_series(
        candidate_id=1,
        dates=dates,
        pnl_series=pnl_mutated,
        turnover_series=to,
        evaluate_holdout=False,
    )

    # Validation metrics MUST be 100% identical
    assert sc_base.val_sharpe == sc_mutated.val_sharpe
    assert sc_base.val_after_cost_sharpe == sc_mutated.val_after_cost_sharpe
    assert sc_base.val_fitness == sc_mutated.val_fitness
    assert sc_base.val_turnover == sc_mutated.val_turnover
    assert sc_base.val_composite_score == sc_mutated.val_composite_score
    assert sc_base.is_submission_grade == sc_mutated.is_submission_grade


def test_is_mutation_invariance_of_validation_score():
    """Mutating IS performance (while keeping IS Sharpe > 0) has zero effect on val_composite_score."""
    dates = make_calendar_days(100)
    pnl_orig, to = make_pnl_turnover_series(dates, base_daily_pnl=100.0)

    sc_base = evaluate_pnl_series(
        candidate_id=1,
        dates=dates,
        pnl_series=pnl_orig,
        turnover_series=to,
        evaluate_holdout=False,
    )

    # Mutate only the IS partition (indices 0..59) by adding variable returns
    pnl_mutated = list(pnl_orig)
    for i in range(60):
        pnl_mutated[i] = pnl_orig[i] + (i % 2) * 200.0

    sc_mutated = evaluate_pnl_series(
        candidate_id=1,
        dates=dates,
        pnl_series=pnl_mutated,
        turnover_series=to,
        evaluate_holdout=False,
    )

    # In-sample Sharpe changes
    assert sc_base.is_sharpe != sc_mutated.is_sharpe
    # Diagnostic decay changes
    assert sc_base.is_val_decay != sc_mutated.is_val_decay
    # BUT validation metrics and composite score remain strictly invariant!
    assert sc_base.val_sharpe == sc_mutated.val_sharpe
    assert sc_base.val_composite_score == sc_mutated.val_composite_score


# ===========================================================================
# 3. Downstream Consumers: Isolation of Learning, Leaderboard, & Pool Curation
# ===========================================================================


def test_learning_memory_uses_validation_metrics(tmp_path: Path):
    """Verifies that ResearchMemory separates winners/losers strictly on validation metrics."""
    async def _run():
        db = Database(tmp_path / "sqlite.db")
        await db.create_all()

        async with db.session() as s:
            sess = ResearchSession(name="Learning Test", prompt="test", region="USA", universe="TOP3000")
            s.add(sess)
            await s.flush()

            # Cand 1: high IS, low Val
            c1 = ResearchCandidate(session_id=sess.id, status=CandidateStatus.ACCEPTED, expression="rank(close)", hypothesis="h1")
            s.add(c1)
            await s.flush()
            ev1 = ResearchEvaluation(
                candidate_id=c1.id,
                evaluation_scheme="THREE_WAY_CHRONOLOGICAL",
                is_verified=True,
                full_sharpe=3.0,
                is_sharpe=3.0,
                val_sharpe=0.5,
                val_after_cost_sharpe=0.4,
                val_fitness=0.5,
                val_turnover=0.10,
                val_composite_score=0.36,
                is_val_decay=0.83,
            )
            s.add(ev1)

            # Cand 2: modest IS, high Val
            c2 = ResearchCandidate(session_id=sess.id, status=CandidateStatus.ACCEPTED, expression="ts_mean(close, 10)", hypothesis="h2")
            s.add(c2)
            await s.flush()
            ev2 = ResearchEvaluation(
                candidate_id=c2.id,
                evaluation_scheme="THREE_WAY_CHRONOLOGICAL",
                is_verified=True,
                full_sharpe=1.5,
                is_sharpe=1.5,
                val_sharpe=2.2,
                val_after_cost_sharpe=2.0,
                val_fitness=1.8,
                val_turnover=0.12,
                val_composite_score=1.612,
                is_val_decay=-0.46,
            )
            s.add(ev2)

        memory = ResearchMemory(db)
        winners, losers = await memory.get_session_performance_split(sess.id)

        # Winner must be Candidate 2 due to higher val_composite_score!
        assert len(winners) == 1
        assert winners[0]["candidate_id"] == c2.id
        assert winners[0]["sharpe"] == 2.2  # consumes validation Sharpe

    asyncio.run(_run())


def test_leaderboard_ranks_by_validation_composite_and_masks_holdout(tmp_path: Path):
    """Leaderboard ranks candidates by validation composite score and conceals holdout metrics."""
    async def _run():
        db = Database(tmp_path / "sqlite.db")
        await db.create_all()

        async with db.session() as s:
            sess = ResearchSession(name="LB Test", prompt="test")
            s.add(sess)
            await s.flush()

            c1 = ResearchCandidate(session_id=sess.id, status=CandidateStatus.ACCEPTED, expression="expr1")
            s.add(c1)
            await s.flush()
            s.add(ResearchEvaluation(
                candidate_id=c1.id,
                evaluation_scheme="THREE_WAY_CHRONOLOGICAL",
                is_verified=True,
                val_composite_score=1.2,
                val_sharpe=1.5,
                holdout_sharpe=4.0,  # Holdout evaluated prematurely or leaked
            ))

            c2 = ResearchCandidate(session_id=sess.id, status=CandidateStatus.ACCEPTED, expression="expr2")
            s.add(c2)
            await s.flush()
            s.add(ResearchEvaluation(
                candidate_id=c2.id,
                evaluation_scheme="THREE_WAY_CHRONOLOGICAL",
                is_verified=True,
                val_composite_score=1.8,
                val_sharpe=2.0,
                holdout_sharpe=1.0,
            ))

        lb = LeaderboardService(db)
        entries = await lb.get_alpha_leaderboard(session_id=sess.id)

        # c2 must rank #1 because val_composite_score (1.8) > (1.2)
        assert len(entries) == 2
        assert entries[0].candidate_id == c2.id
        assert entries[1].candidate_id == c1.id

        # Because candidates are ACCEPTED (not yet GRADUATED), holdout_sharpe MUST be masked as None
        assert entries[0].holdout_sharpe is None
        assert entries[1].holdout_sharpe is None

    asyncio.run(_run())


# ===========================================================================
# 4. Mandatory Safeguard: Program-Wide Unknown Legacy Contamination Block
# ===========================================================================


def test_unknown_legacy_exposure_blocks_graduation_globally(tmp_path: Path):
    """Proves that an unresolved LEGACY_CONTAMINATED_UNKNOWN blocks graduation program-wide."""
    async def _run():
        db = Database(tmp_path / "sqlite.db")
        await db.create_all()
        cat = Catalog(tmp_path / "catalog.duckdb")
        await cat.open()
        vault = AlphaVault(cat)
        engine = EvaluationEngine(db, vault)

        dates = make_calendar_days(100)
        pnl, to = make_pnl_turnover_series(dates)
        await populate_duckdb_vault(cat, "alpha_new_cand", dates, pnl, to)

        async with db.session() as s:
            # Session 1: Old session with unknown exposure
            sess1 = ResearchSession(name="Legacy Session")
            s.add(sess1)
            await s.flush()
            cand_old = ResearchCandidate(session_id=sess1.id, alpha_id="alpha_legacy_unknown", status=CandidateStatus.EVALUATED)
            s.add(cand_old)
            await s.flush()

            # Record UNKNOWN legacy exposure
            s.add(HoldoutAssessment(
                assessment_type="LEGACY_CONTAMINATED_UNKNOWN",
                candidate_id=cand_old.id,
                session_id=sess1.id,
                alpha_id=cand_old.alpha_id,
                market_scope="USA:TOP3000",
                holdout_start_date=None,
                holdout_end_date=None,
            ))

            # Session 2: Fresh session with an otherwise fully eligible candidate
            sess2 = ResearchSession(name="New Session", region="USA", universe="TOP3000")
            s.add(sess2)
            await s.flush()
            cand_new = ResearchCandidate(
                session_id=sess2.id,
                alpha_id="alpha_new_cand",
                status=CandidateStatus.ACCEPTED,
                sim_region="USA",
                sim_universe="TOP3000",
            )
            s.add(cand_new)
            await s.flush()
            s.add(ResearchEvaluation(
                candidate_id=cand_new.id,
                evaluation_scheme="THREE_WAY_CHRONOLOGICAL",
                is_verified=True,
                val_composite_score=1.5,
                val_sharpe=1.8,
                checks_passed=8,
                checks_total=8,
            ))

        # Graduation MUST fail with HoldoutIndeterminacyError
        with pytest.raises(HoldoutIndeterminacyError, match="unresolved LEGACY_CONTAMINATED_UNKNOWN"):
            await engine.graduate_candidate(cand_new.id)

    asyncio.run(_run())


def test_graduation_succeeds_after_unknown_exposure_resolved(tmp_path: Path):
    """Proves graduation becomes possible only after unknown exposure is resolved legitimately."""
    async def _run():
        db = Database(tmp_path / "sqlite.db")
        await db.create_all()
        cat = Catalog(tmp_path / "catalog.duckdb")
        await cat.open()
        vault = AlphaVault(cat)
        engine = EvaluationEngine(db, vault)

        # Dates: legacy spans Jan-Feb 2023; new candidate spans March-June 2023 (strict forward chronology)
        dates_legacy = make_calendar_days(60, start_date=date(2023, 1, 2))
        dates_new = make_calendar_days(100, start_date=date(2023, 4, 3))

        pnl_new, to_new = make_pnl_turnover_series(dates_new)
        await populate_duckdb_vault(cat, "alpha_new_cand_2", dates_new, pnl_new, to_new)

        async with db.session() as s:
            sess1 = ResearchSession(name="Session 1")
            s.add(sess1)
            await s.flush()
            cand_old = ResearchCandidate(session_id=sess1.id, alpha_id="alpha_resolved", status=CandidateStatus.EVALUATED)
            s.add(cand_old)
            await s.flush()

            # Initially UNKNOWN
            unknown_rec = HoldoutAssessment(
                assessment_type="LEGACY_CONTAMINATED_UNKNOWN",
                candidate_id=cand_old.id,
                session_id=sess1.id,
                alpha_id=cand_old.alpha_id,
                market_scope="USA:TOP3000",
                holdout_start_date=None,
                holdout_end_date=None,
            )
            s.add(unknown_rec)

            sess2 = ResearchSession(name="Session 2", region="USA", universe="TOP3000")
            s.add(sess2)
            await s.flush()
            cand_new = ResearchCandidate(
                session_id=sess2.id,
                alpha_id="alpha_new_cand_2",
                status=CandidateStatus.ACCEPTED,
                sim_region="USA",
                sim_universe="TOP3000",
            )
            s.add(cand_new)
            await s.flush()
            s.add(ResearchEvaluation(
                candidate_id=cand_new.id,
                evaluation_scheme="THREE_WAY_CHRONOLOGICAL",
                is_verified=True,
                val_composite_score=1.5,
                val_sharpe=1.8,
                checks_passed=8,
                checks_total=8,
            ))

        # Initially blocked
        with pytest.raises(HoldoutIndeterminacyError):
            await engine.graduate_candidate(cand_new.id)

        # Resolve the unknown record using reconstructed verified dates
        async with db.session() as s:
            stmt = select(HoldoutAssessment).where(HoldoutAssessment.candidate_id == cand_old.id)
            rec = (await s.execute(stmt)).scalar_one()
            rec.assessment_type = "LEGACY_CONTAMINATED"
            rec.holdout_start_date = dates_legacy[0]
            rec.holdout_end_date = dates_legacy[-1]
            rec.observation_count = len(dates_legacy)

        # Graduation must now succeed!
        scorecard = await engine.graduate_candidate(cand_new.id)
        assert scorecard.holdout_sharpe is not None
        assert scorecard.config_hash is not None

        # Verify candidate transitioned to GRADUATED
        async with db.session() as s:
            cand_row = await s.get(ResearchCandidate, cand_new.id)
            assert cand_row.status == CandidateStatus.GRADUATED

    asyncio.run(_run())


# ===========================================================================
# 5. Walk-Forward Enforcement & Overlapping Window Rejection
# ===========================================================================


def test_strict_walk_forward_and_overlapping_window_rejection(tmp_path: Path):
    """Verifies that holdout evaluation rejects non-advancing dates and overlapping windows."""
    async def _run():
        db = Database(tmp_path / "sqlite.db")
        await db.create_all()
        cat = Catalog(tmp_path / "catalog.duckdb")
        await cat.open()
        vault = AlphaVault(cat)
        engine = EvaluationEngine(db, vault)

        # Candidate 1: evaluated on Jan-May 2023 (100 days)
        # 80% boundary is index 80 (April 2023) to index 99 (May 2023)
        dates1 = make_calendar_days(100, start_date=date(2023, 1, 2))
        pnl1, to1 = make_pnl_turnover_series(dates1)
        await populate_duckdb_vault(cat, "alpha_c1", dates1, pnl1, to1)

        async with db.session() as s:
            sess = ResearchSession(name="WF Test", region="USA", universe="TOP3000")
            s.add(sess)
            await s.flush()

            c1 = ResearchCandidate(session_id=sess.id, alpha_id="alpha_c1", status=CandidateStatus.ACCEPTED)
            s.add(c1)
            await s.flush()
            s.add(ResearchEvaluation(candidate_id=c1.id, is_verified=True, val_composite_score=1.5, checks_passed=8, checks_total=8))

        # Graduate Candidate 1
        await engine.graduate_candidate(c1.id)

        # Candidate 2: attempting to evaluate overlapping period
        dates2 = make_calendar_days(100, start_date=date(2023, 1, 16))
        pnl2, to2 = make_pnl_turnover_series(dates2)
        await populate_duckdb_vault(cat, "alpha_c2", dates2, pnl2, to2)

        async with db.session() as s:
            c2 = ResearchCandidate(session_id=sess.id, alpha_id="alpha_c2", status=CandidateStatus.ACCEPTED)
            s.add(c2)
            await s.flush()
            s.add(ResearchEvaluation(candidate_id=c2.id, is_verified=True, val_composite_score=1.6, checks_passed=8, checks_total=8))

        # Candidate 2 holdout start date is earlier than c1 holdout end date -> Blocked!
        with pytest.raises(HoldoutChronologyError, match="strict walk-forward policy"):
            await engine.graduate_candidate(c2.id)

    asyncio.run(_run())


# ===========================================================================
# 6. Graduation Idempotency, Immutability & Re-evaluation Blocking
# ===========================================================================


def test_graduation_idempotency_and_immutability(tmp_path: Path):
    """Verifies that graduation is idempotent and graduated candidates cannot be re-evaluated."""
    async def _run():
        db = Database(tmp_path / "sqlite.db")
        await db.create_all()
        cat = Catalog(tmp_path / "catalog.duckdb")
        await cat.open()
        vault = AlphaVault(cat)
        engine = EvaluationEngine(db, vault)

        dates = make_calendar_days(100, start_date=date(2023, 1, 2))
        pnl, to = make_pnl_turnover_series(dates)
        await populate_duckdb_vault(cat, "alpha_idem", dates, pnl, to)

        async with db.session() as s:
            sess = ResearchSession(name="Idem Test", region="USA", universe="TOP3000")
            s.add(sess)
            await s.flush()

            c = ResearchCandidate(session_id=sess.id, alpha_id="alpha_idem", status=CandidateStatus.ACCEPTED)
            s.add(c)
            await s.flush()
            s.add(ResearchEvaluation(candidate_id=c.id, is_verified=True, val_composite_score=1.5, checks_passed=8, checks_total=8))

        # First graduation
        sc1 = await engine.graduate_candidate(c.id)
        assert sc1.holdout_sharpe is not None

        # Count assessment records
        async with db.session() as s:
            assessments = (await s.execute(select(HoldoutAssessment))).scalars().all()
            assert len(assessments) == 1

        # Second graduation call: returns existing scorecard without creating duplicates
        sc2 = await engine.graduate_candidate(c.id)
        assert sc2.holdout_sharpe == sc1.holdout_sharpe
        assert sc2.config_hash == sc1.config_hash

        async with db.session() as s:
            assessments2 = (await s.execute(select(HoldoutAssessment))).scalars().all()
            assert len(assessments2) == 1

        # Re-evaluating candidate via evaluate_candidate MUST be rejected to preserve immutability
        with pytest.raises(HoldoutImmutabilityError, match="already GRADUATED and immutable"):
            await engine.evaluate_candidate(c.id)

    asyncio.run(_run())


# ===========================================================================
# 7. Legacy Contamination Migration & Full-History Backfill
# ===========================================================================


def test_legacy_contamination_backfill_full_history(tmp_path: Path):
    """Verifies that legacy records backfill full historical span or UNKNOWN without fabricating dates."""
    async def _run():
        db = Database(tmp_path / "sqlite.db")
        await db.create_all()
        cat = Catalog(tmp_path / "catalog.duckdb")
        await cat.open()
        vault = AlphaVault(cat)
        engine = EvaluationEngine(db, vault)

        dates = make_calendar_days(100, start_date=date(2022, 1, 3))
        pnl, to = make_pnl_turnover_series(dates)
        # Store series for legacy candidate 1
        await populate_duckdb_vault(cat, "alpha_legacy_with_series", dates, pnl, to)

        async with db.session() as s:
            sess = ResearchSession(name="Legacy Session")
            s.add(sess)
            await s.flush()

            # Legacy candidate 1: has verified series in DuckDB
            c1 = ResearchCandidate(session_id=sess.id, alpha_id="alpha_legacy_with_series", status=CandidateStatus.EVALUATED)
            s.add(c1)
            await s.flush()
            s.add(ResearchEvaluation(
                candidate_id=c1.id,
                evaluation_scheme="LEGACY_2WAY",
                is_verified=True,
                full_sharpe=1.8,
            ))

            # Legacy candidate 2: lacks series in DuckDB
            c2 = ResearchCandidate(session_id=sess.id, alpha_id="alpha_legacy_missing_series", status=CandidateStatus.EVALUATED)
            s.add(c2)
            await s.flush()
            s.add(ResearchEvaluation(
                candidate_id=c2.id,
                evaluation_scheme="LEGACY_2WAY",
                is_verified=True,
                full_sharpe=2.1,
            ))

        # Run backfill
        count = await engine.backfill_legacy_holdout_contamination()
        assert count == 2

        # Verify backfill records
        async with db.session() as s:
            # Candidate 1: full history [min_date, max_date]
            stmt1 = select(HoldoutAssessment).where(HoldoutAssessment.candidate_id == c1.id)
            rec1 = (await s.execute(stmt1)).scalar_one()
            assert rec1.assessment_type == "LEGACY_CONTAMINATED"
            assert rec1.holdout_start_date == dates[0]
            assert rec1.holdout_end_date == dates[-1]
            assert rec1.observation_count == 100

            # Candidate 2: unknown exposure, zero fabricated dates
            stmt2 = select(HoldoutAssessment).where(HoldoutAssessment.candidate_id == c2.id)
            rec2 = (await s.execute(stmt2)).scalar_one()
            assert rec2.assessment_type == "LEGACY_CONTAMINATED_UNKNOWN"
            assert rec2.holdout_start_date is None
            assert rec2.holdout_end_date is None
            assert rec2.observation_count is None

        # Verify idempotency: running backfill a second time does not re-add
        count_second = await engine.backfill_legacy_holdout_contamination()
        assert count_second == 0

    asyncio.run(_run())


# ===========================================================================
# 8. Cross-Scope / Cross-Session Rejection & Ineligible Candidate Guard
# ===========================================================================


def test_cross_scope_and_cross_session_holdout_rejection(tmp_path: Path):
    """Proves that holdout protection is program-wide across different scopes and sessions."""
    async def _run():
        db = Database(tmp_path / "sqlite.db")
        await db.create_all()
        cat = Catalog(tmp_path / "catalog.duckdb")
        await cat.open()
        vault = AlphaVault(cat)
        engine = EvaluationEngine(db, vault)

        # Dates: 100 days Jan-May 2023
        dates_a = make_calendar_days(100, start_date=date(2023, 1, 2))
        pnl_a, to_a = make_pnl_turnover_series(dates_a)
        await populate_duckdb_vault(cat, "alpha_usa", dates_a, pnl_a, to_a)

        # Session 1: USA scope
        async with db.session() as s:
            sess_usa = ResearchSession(name="USA Session", region="USA", universe="TOP3000")
            s.add(sess_usa)
            await s.flush()
            cand_usa = ResearchCandidate(
                session_id=sess_usa.id,
                alpha_id="alpha_usa",
                status=CandidateStatus.ACCEPTED,
                sim_region="USA",
                sim_universe="TOP3000",
            )
            s.add(cand_usa)
            await s.flush()
            s.add(ResearchEvaluation(
                candidate_id=cand_usa.id,
                is_verified=True,
                val_composite_score=1.5,
                checks_passed=8,
                checks_total=8,
            ))

        # Graduate USA candidate
        sc_usa = await engine.graduate_candidate(cand_usa.id)
        assert sc_usa.holdout_sharpe is not None

        # Session 2: EUR scope attempting to evaluate identical/overlapping dates
        dates_eur = make_calendar_days(100, start_date=date(2023, 1, 16))
        pnl_eur, to_eur = make_pnl_turnover_series(dates_eur)
        await populate_duckdb_vault(cat, "alpha_eur", dates_eur, pnl_eur, to_eur)

        async with db.session() as s:
            sess_eur = ResearchSession(name="EUR Session", region="EUR", universe="TOP2500")
            s.add(sess_eur)
            await s.flush()
            cand_eur = ResearchCandidate(
                session_id=sess_eur.id,
                alpha_id="alpha_eur",
                status=CandidateStatus.ACCEPTED,
                sim_region="EUR",
                sim_universe="TOP2500",
            )
            s.add(cand_eur)
            await s.flush()
            s.add(ResearchEvaluation(
                candidate_id=cand_eur.id,
                is_verified=True,
                val_composite_score=1.7,
                checks_passed=8,
                checks_total=8,
            ))

        # Rejection MUST occur program-wide despite different session and scope!
        with pytest.raises((HoldoutChronologyError, HoldoutOverlapError)):
            await engine.graduate_candidate(cand_eur.id)

    asyncio.run(_run())


def test_ineligible_candidate_graduation_rejection(tmp_path: Path):
    """Proves that candidates not in ACCEPTED status or missing series cannot graduate."""
    async def _run():
        db = Database(tmp_path / "sqlite.db")
        await db.create_all()
        cat = Catalog(tmp_path / "catalog.duckdb")
        await cat.open()
        vault = AlphaVault(cat)
        engine = EvaluationEngine(db, vault)

        async with db.session() as s:
            sess = ResearchSession(name="Test Session", region="USA", universe="TOP3000")
            s.add(sess)
            await s.flush()

            # Candidate in EVALUATED (not yet ACCEPTED) status
            c1 = ResearchCandidate(session_id=sess.id, alpha_id="alpha_uneval", status=CandidateStatus.EVALUATED)
            s.add(c1)
            # Candidate with ACCEPTED status but missing series in AlphaVault
            c2 = ResearchCandidate(session_id=sess.id, alpha_id="alpha_no_series", status=CandidateStatus.ACCEPTED)
            s.add(c2)
            await s.flush()

        # Non-ACCEPTED candidate rejected
        with pytest.raises(HoldoutChronologyError, match="must be 'ACCEPTED' to graduate"):
            await engine.graduate_candidate(c1.id)

        # Missing series candidate rejected
        with pytest.raises(InsufficientDataError, match="lacks verified series in vault"):
            await engine.graduate_candidate(c2.id)

    asyncio.run(_run())


# ===========================================================================
# 9. API Masking & Self-Learning Holdout Non-Consumption
# ===========================================================================


def test_evaluation_apis_mask_holdout_and_session_summary_invariance(tmp_path: Path):
    """Verifies that API candidate listings mask ungraduated holdouts and session summary uses validation Sharpe."""
    from alpha_harness.api.research import list_candidates, list_sessions

    async def _run():
        db = Database(tmp_path / "sqlite.db")
        await db.create_all()

        async with db.session() as s:
            sess = ResearchSession(name="API Masking Test", prompt="test", region="USA", universe="TOP3000")
            s.add(sess)
            await s.flush()

            # Candidate 1: ACCEPTED with holdout metric prematurely stored in DB
            c1 = ResearchCandidate(session_id=sess.id, status=CandidateStatus.ACCEPTED, expression="rank(close)")
            s.add(c1)
            await s.flush()
            ev1 = ResearchEvaluation(
                candidate_id=c1.id,
                evaluation_scheme="THREE_WAY_CHRONOLOGICAL",
                is_verified=True,
                full_sharpe=2.0,
                val_sharpe=1.8,
                val_composite_score=1.4,
                holdout_sharpe=4.5,
                holdout_after_cost_sharpe=4.2,
                holdout_fitness=3.8,
            )
            s.add(ev1)

            # Candidate 2: GRADUATED with holdout metric
            c2 = ResearchCandidate(session_id=sess.id, status=CandidateStatus.GRADUATED, expression="ts_mean(close, 5)")
            s.add(c2)
            await s.flush()
            ev2 = ResearchEvaluation(
                candidate_id=c2.id,
                evaluation_scheme="SPLIT_3WAY_FINAL",
                is_verified=True,
                full_sharpe=2.2,
                val_sharpe=2.0,
                val_composite_score=1.6,
                holdout_sharpe=3.5,
                holdout_after_cost_sharpe=3.2,
                holdout_fitness=2.9,
            )
            s.add(ev2)

        # Mock minimal state
        class MockState:
            pass

        st = MockState()
        st.db = db

        # Test list_candidates
        cands = await list_candidates(session_id=sess.id, state=st)
        assert len(cands) == 2

        cand1_out = next(c for c in cands if c.id == c1.id)
        cand2_out = next(c for c in cands if c.id == c2.id)

        # ACCEPTED candidate: holdout metrics MUST be masked as None
        assert cand1_out.is_graduated is False
        assert cand1_out.holdout_sharpe is None
        assert cand1_out.holdout_after_cost_sharpe is None
        assert cand1_out.holdout_fitness is None
        assert cand1_out.val_sharpe == 1.8

        # GRADUATED candidate: holdout metrics are revealed
        assert cand2_out.is_graduated is True
        assert cand2_out.holdout_sharpe == 3.5
        assert cand2_out.holdout_after_cost_sharpe == 3.2
        assert cand2_out.holdout_fitness == 2.9

        # Test list_sessions: best_sharpe must be max(val_sharpe), NOT max(holdout_sharpe)
        sessions_out = await list_sessions(state=st)
        assert len(sessions_out) == 1
        # max val_sharpe is 2.0 (Candidate 2), NOT 4.5 (Candidate 1's holdout_sharpe)
        assert sessions_out[0].best_sharpe == 2.0

    asyncio.run(_run())


def test_self_learning_cycle_consumes_validation_metrics_only(tmp_path: Path):
    """Verifies that SelfLearningEngine reflection operates strictly on validation composite scores."""
    async def _run():
        db = Database(tmp_path / "sqlite.db")
        await db.create_all()

        async with db.session() as s:
            sess = ResearchSession(name="Learning Cycle Test", prompt="test", region="USA", universe="TOP3000", current_generation=1)
            s.add(sess)
            await s.flush()

            # Cand 1: Moderate val (1.5), huge holdout (5.0)
            c1 = ResearchCandidate(session_id=sess.id, status=CandidateStatus.ACCEPTED, expression="rank(volume)", generation=1)
            s.add(c1)
            await s.flush()
            s.add(ResearchEvaluation(
                candidate_id=c1.id,
                evaluation_scheme="THREE_WAY_CHRONOLOGICAL",
                is_verified=True,
                val_sharpe=1.5,
                val_composite_score=1.1,
                holdout_sharpe=5.0,
            ))

            # Cand 2: High val (2.8), moderate holdout (1.2)
            c2 = ResearchCandidate(session_id=sess.id, status=CandidateStatus.ACCEPTED, expression="ts_std(close, 20)", generation=1)
            s.add(c2)
            await s.flush()
            s.add(ResearchEvaluation(
                candidate_id=c2.id,
                evaluation_scheme="THREE_WAY_CHRONOLOGICAL",
                is_verified=True,
                val_sharpe=2.8,
                val_composite_score=2.2,
                holdout_sharpe=1.2,
            ))

        memory = ResearchMemory(db)
        engine = SelfLearningEngine(db, memory=memory)
        res = await engine.execute_learning_cycle(session_id=sess.id)

        # Best Sharpe in learning cycle MUST be max validation Sharpe (2.8), NOT holdout (5.0)
        assert res.best_sharpe == 2.8
        # Winner must be Candidate 2
        winners, _ = await memory.get_session_performance_split(session_id=sess.id, generation=1)
        assert len(winners) == 1
        assert winners[0]["candidate_id"] == c2.id

    asyncio.run(_run())


def test_concurrent_graduation_requests_atomicity(tmp_path: Path):
    """Verifies that concurrent graduation requests are safely serialized by _graduation_lock."""
    async def _run():
        db = Database(tmp_path / "sqlite.db")
        await db.create_all()
        cat = Catalog(tmp_path / "catalog.duckdb")
        await cat.open()
        vault = AlphaVault(cat)
        engine = EvaluationEngine(db, vault)

        # Two candidates on identical historical window
        dates = make_calendar_days(100, start_date=date(2023, 1, 2))
        pnl, to = make_pnl_turnover_series(dates)
        await populate_duckdb_vault(cat, "alpha_conc_1", dates, pnl, to)
        await populate_duckdb_vault(cat, "alpha_conc_2", dates, pnl, to)

        async with db.session() as s:
            sess = ResearchSession(name="Concurrent Test", region="USA", universe="TOP3000")
            s.add(sess)
            await s.flush()

            c1 = ResearchCandidate(session_id=sess.id, alpha_id="alpha_conc_1", status=CandidateStatus.ACCEPTED)
            c2 = ResearchCandidate(session_id=sess.id, alpha_id="alpha_conc_2", status=CandidateStatus.ACCEPTED)
            s.add_all([c1, c2])
            await s.flush()
            s.add_all([
                ResearchEvaluation(candidate_id=c1.id, is_verified=True, val_composite_score=1.5, checks_passed=8, checks_total=8),
                ResearchEvaluation(candidate_id=c2.id, is_verified=True, val_composite_score=1.6, checks_passed=8, checks_total=8),
            ])

        # Execute both graduations concurrently
        res = await asyncio.gather(
            engine.graduate_candidate(c1.id),
            engine.graduate_candidate(c2.id),
            return_exceptions=True,
        )

        # Exactly one MUST succeed, and the other MUST fail with Chronology or Overlap error
        successes = [r for r in res if isinstance(r, EvaluationScorecard)]
        failures = [r for r in res if isinstance(r, Exception)]

        assert len(successes) == 1
        assert len(failures) == 1
        assert isinstance(failures[0], (HoldoutChronologyError, HoldoutOverlapError))

    asyncio.run(_run())

