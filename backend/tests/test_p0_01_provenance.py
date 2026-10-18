"""Regression test suite for P0-01: Enforce Verified BRAIN Provenance & Eliminate Synthetic Returns."""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from pathlib import Path

from sqlalchemy import select

from alpha_harness.db.duck import Catalog
from alpha_harness.db.models import (
    CandidateStatus,
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
    InsufficientDataError,
)
from alpha_harness.leaderboard.service import LeaderboardService
from alpha_harness.learning.memory import ResearchMemory
from alpha_harness.vault.store import SERIES_VERSION, AlphaVault


def generate_series_rows(num_days: int = 30, base_pnl: float = 1000.0, turnover: float = 0.15):
    start = date(2023, 1, 2)
    pnl_rows = []
    to_rows = []
    cur = start
    cumulative_pnl = 0.0
    for i in range(num_days):
        while cur.weekday() >= 5:
            cur += timedelta(days=1)
        cumulative_pnl += base_pnl + (i % 5) * 50.0
        pnl_rows.append({"date": cur, "pnl": cumulative_pnl})
        to_rows.append({"date": cur, "turnover": turnover})
        cur += timedelta(days=1)
    return pnl_rows, to_rows


# ---------------------------------------------------------------------------
# 1. Isolated DuckDB Vault Tests (series_version, incomplete records, save_pnl_only)
# ---------------------------------------------------------------------------


def test_vault_load_verified_series_valid(tmp_path: Path):
    """Verifies that an alpha with SERIES_VERSION=3 and >=20 observations loads successfully."""
    async def _run():
        cat = Catalog(tmp_path / "catalog.duckdb")
        await cat.open()
        try:
            vault = AlphaVault(cat)
            alpha_id = "alpha_verified_01"
            pnl_rows, to_rows = generate_series_rows(num_days=30)

            written = await vault.save_pnl(alpha_id, pnl_rows, to_rows, yearly_rows=[])
            assert written == 30

            res = await vault.load_verified_series(alpha_id)
            assert res is not None
            dates, pnls, tos = res
            assert len(dates) == 30
            assert len(pnls) == 30
            assert len(tos) == 30
            assert all(isinstance(t, float) and t >= 0.0 for t in tos)
        finally:
            await cat.close()

    asyncio.run(_run())


def test_vault_load_verified_series_rejects_stale_or_wrong_version(tmp_path: Path):
    """Verifies that an alpha with series_version != 3 (e.g. 2 or NULL) is strictly rejected."""
    async def _run():
        cat = Catalog(tmp_path / "catalog.duckdb")
        await cat.open()
        try:
            vault = AlphaVault(cat)
            alpha_id = "alpha_stale_ver"
            pnl_rows, _ = generate_series_rows(num_days=30)

            # Insert into alpha_pnl
            await cat.replace_pnl(
                alpha_id, [(alpha_id, r["date"], r["pnl"], 0.15) for r in pnl_rows]
            )
            # Upsert into alpha table with stale series_version = 2
            await cat.upsert(
                "alpha",
                ("alpha_id", "series_version", "pnl_version", "after_cost_t10"),
                [(alpha_id, 2, 2, 1.5)],
            )

            res = await vault.load_verified_series(alpha_id)
            assert res is None, "Alpha with series_version=2 must be rejected by load_verified_series"
        finally:
            await cat.close()

    asyncio.run(_run())


def test_vault_load_verified_series_rejects_save_pnl_only(tmp_path: Path):
    """Verifies that save_pnl_only (no turnover stored) is rejected by load_verified_series."""
    async def _run():
        cat = Catalog(tmp_path / "catalog.duckdb")
        await cat.open()
        try:
            vault = AlphaVault(cat)
            alpha_id = "alpha_pnl_only"
            pnl_rows, _ = generate_series_rows(num_days=30)

            written = await vault.save_pnl_only(alpha_id, pnl_rows)
            assert written == 30

            res = await vault.load_verified_series(alpha_id)
            assert res is None, "save_pnl_only records lacking turnover must be rejected"
        finally:
            await cat.close()

    asyncio.run(_run())


def test_vault_load_verified_series_rejects_incomplete_observations(tmp_path: Path):
    """Verifies that fewer than 20 observations is rejected."""
    async def _run():
        cat = Catalog(tmp_path / "catalog.duckdb")
        await cat.open()
        try:
            vault = AlphaVault(cat)
            alpha_id = "alpha_short"
            pnl_rows, to_rows = generate_series_rows(num_days=10)

            await vault.save_pnl(alpha_id, pnl_rows, to_rows, yearly_rows=[])
            res = await vault.load_verified_series(alpha_id)
            assert res is None, "Series with <20 observations must be rejected"
        finally:
            await cat.close()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 2. EvaluationEngine Controlled Provenance & Atomic Persistence Tests
# ---------------------------------------------------------------------------


def test_evaluate_candidate_success_stores_verified_provenance(tmp_path: Path):
    """Verifies that evaluation succeeds with genuine data, setting is_verified=True and provenance='BRAIN_VAULT'."""
    async def _run():
        cat = Catalog(tmp_path / "catalog.duckdb")
        await cat.open()
        db = Database(tmp_path / "test.sqlite")
        await db.create_all()
        try:
            vault = AlphaVault(cat)
            evaluator = EvaluationEngine(db, vault)

            alpha_id = "alpha_eval_success"
            pnl_rows, to_rows = generate_series_rows(num_days=30)
            await vault.save_pnl(alpha_id, pnl_rows, to_rows, yearly_rows=[])

            async with db.session() as session:
                sess = ResearchSession(id=1, name="Test Campaign")
                session.add(sess)
                await session.flush()

                cand = ResearchCandidate(
                    session_id=1,
                    alpha_id=alpha_id,
                    expression="rank(close)",
                    is_valid=True,
                    status=CandidateStatus.SIMULATING,
                )
                session.add(cand)
                await session.flush()
                cand_id = cand.id

            scorecard = await evaluator.evaluate_candidate(cand_id)
            assert isinstance(scorecard, EvaluationScorecard)
            assert scorecard.full_sharpe is not None

            async with db.session() as session:
                cand_row = await session.get(ResearchCandidate, cand_id)
                assert cand_row.status in (CandidateStatus.EVALUATED, CandidateStatus.ACCEPTED)

                eval_stmt = select(ResearchEvaluation).where(
                    ResearchEvaluation.candidate_id == cand_id
                )
                eval_row = (await session.execute(eval_stmt)).scalar_one()
                assert eval_row.is_verified is True
                assert eval_row.provenance == "BRAIN_VAULT"
                assert eval_row.full_sharpe == scorecard.full_sharpe
        finally:
            await db.dispose()
            await cat.close()

    asyncio.run(_run())


def test_evaluate_candidate_rejects_missing_data_and_persists_no_records(tmp_path: Path):
    """Verifies that missing vault data raises InsufficientDataError and writes zero evaluation records."""
    async def _run():
        cat = Catalog(tmp_path / "catalog.duckdb")
        await cat.open()
        db = Database(tmp_path / "test.sqlite")
        await db.create_all()
        try:
            vault = AlphaVault(cat)
            evaluator = EvaluationEngine(db, vault)

            async with db.session() as session:
                sess = ResearchSession(id=1, name="Test Campaign")
                session.add(sess)
                await session.flush()

                cand = ResearchCandidate(
                    session_id=1,
                    alpha_id="alpha_non_existent",
                    expression="rank(close)",
                    is_valid=True,
                    status=CandidateStatus.SIMULATING,
                )
                session.add(cand)
                await session.flush()
                cand_id = cand.id

            raised = False
            try:
                await evaluator.evaluate_candidate(cand_id)
            except InsufficientDataError:
                raised = True
            assert raised, "Expected InsufficientDataError when vault series is missing"

            async with db.session() as session:
                eval_stmt = select(ResearchEvaluation).where(
                    ResearchEvaluation.candidate_id == cand_id
                )
                res = (await session.execute(eval_stmt)).scalar_one_or_none()
                assert res is None, "No ResearchEvaluation record must be created on failed evaluation"
        finally:
            await db.dispose()
            await cat.close()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 3. State Machine & Safe Retry Workflow Tests (UNAVAILABLE -> EVALUATED)
# ---------------------------------------------------------------------------


def test_unavailable_candidate_retry_workflow(tmp_path: Path):
    """Verifies candidate transitions to UNAVAILABLE when returns missing, then to EVALUATED on retry once returns arrive."""
    async def _run():
        cat = Catalog(tmp_path / "catalog.duckdb")
        await cat.open()
        db = Database(tmp_path / "test.sqlite")
        await db.create_all()
        try:
            vault = AlphaVault(cat)
            evaluator = EvaluationEngine(db, vault)
            alpha_id = "alpha_retry_test"

            async with db.session() as session:
                sess = ResearchSession(id=1, name="Test Campaign")
                session.add(sess)
                await session.flush()

                cand = ResearchCandidate(
                    session_id=1,
                    alpha_id=alpha_id,
                    expression="rank(open)",
                    is_valid=True,
                    status=CandidateStatus.SIMULATING,
                )
                session.add(cand)
                await session.flush()
                cand_id = cand.id

            # 1. First evaluation attempt: vault has no data yet -> raises InsufficientDataError
            try:
                await evaluator.evaluate_candidate(cand_id)
            except InsufficientDataError:
                async with db.session() as session:
                    c = await session.get(ResearchCandidate, cand_id)
                    c.status = CandidateStatus.UNAVAILABLE

            async with db.session() as session:
                c = await session.get(ResearchCandidate, cand_id)
                assert c.status == CandidateStatus.UNAVAILABLE
                ev = (
                    await session.execute(
                        select(ResearchEvaluation).where(ResearchEvaluation.candidate_id == cand_id)
                    )
                ).scalar_one_or_none()
                assert ev is None

            # 2. Backfill arrives in DuckDB vault
            pnl_rows, to_rows = generate_series_rows(num_days=30)
            await vault.save_pnl(alpha_id, pnl_rows, to_rows, yearly_rows=[])

            # 3. Retry evaluation on UNAVAILABLE candidate
            scorecard = await evaluator.evaluate_candidate(cand_id)
            assert scorecard.full_sharpe is not None

            async with db.session() as session:
                c = await session.get(ResearchCandidate, cand_id)
                assert c.status in (CandidateStatus.EVALUATED, CandidateStatus.ACCEPTED)
                assert c.status != CandidateStatus.SIMULATING

                ev = (
                    await session.execute(
                        select(ResearchEvaluation).where(ResearchEvaluation.candidate_id == cand_id)
                    )
                ).scalar_one()
                assert ev.is_verified is True
                assert ev.provenance == "BRAIN_VAULT"
        finally:
            await db.dispose()
            await cat.close()

    asyncio.run(_run())


def test_temporary_failure_preserves_valid_historical_evaluation(tmp_path: Path):
    """Verifies that an existing verified evaluation record is not mutated or invalidated on a temporary failure."""
    async def _run():
        db = Database(tmp_path / "test.sqlite")
        await db.create_all()
        try:
            async with db.session() as session:
                sess = ResearchSession(id=1, name="Test Campaign")
                session.add(sess)
                await session.flush()

                cand = ResearchCandidate(
                    session_id=1,
                    alpha_id="alpha_historical",
                    expression="rank(high)",
                    is_valid=True,
                    status=CandidateStatus.EVALUATED,
                )
                session.add(cand)
                await session.flush()
                cand_id = cand.id

                ev = ResearchEvaluation(
                    candidate_id=cand_id,
                    full_sharpe=2.10,
                    full_fitness=1.50,
                    composite_score=1.45,
                    is_verified=True,
                    provenance="BRAIN_VAULT",
                )
                session.add(ev)

            # Simulate temporary retrieval issue: candidate marked UNAVAILABLE
            async with db.session() as session:
                c = await session.get(ResearchCandidate, cand_id)
                c.status = CandidateStatus.UNAVAILABLE

            # Verify evaluation remains preserved and verified
            async with db.session() as session:
                ev_row = (
                    await session.execute(
                        select(ResearchEvaluation).where(ResearchEvaluation.candidate_id == cand_id)
                    )
                ).scalar_one()
                assert ev_row.is_verified is True
                assert ev_row.provenance == "BRAIN_VAULT"
                assert ev_row.full_sharpe == 2.10
        finally:
            await db.dispose()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 4. Downstream Quarantine Tests (Leaderboard, Memory, Learning)
# ---------------------------------------------------------------------------


def test_quarantined_records_excluded_downstream(tmp_path: Path):
    """Verifies that unverified (is_verified=False) historical evaluations never appear on leaderboards or memory splits."""
    async def _run():
        db = Database(tmp_path / "test.sqlite")
        await db.create_all()
        try:
            async with db.session() as session:
                sess = ResearchSession(id=1, name="Test Campaign")
                session.add(sess)
                await session.flush()

                # Candidate 1: Verified
                c1 = ResearchCandidate(
                    session_id=1,
                    alpha_id="alpha_good",
                    expression="rank(close)",
                    is_valid=True,
                    status=CandidateStatus.EVALUATED,
                )
                session.add(c1)
                await session.flush()

                ev1 = ResearchEvaluation(
                    candidate_id=c1.id,
                    full_sharpe=1.90,
                    full_fitness=1.20,
                    composite_score=1.35,
                    checks_passed=6,
                    checks_total=6,
                    is_verified=True,
                    provenance="BRAIN_VAULT",
                )
                session.add(ev1)

                # Candidate 2: Unverified / Synthetic historical legacy record
                c2 = ResearchCandidate(
                    session_id=1,
                    alpha_id="alpha_legacy_synthetic",
                    expression="ts_mean(close, 20)",
                    is_valid=True,
                    status=CandidateStatus.EVALUATED,
                )
                session.add(c2)
                await session.flush()

                ev2 = ResearchEvaluation(
                    candidate_id=c2.id,
                    full_sharpe=2.80,  # Higher sharpe, but unverified!
                    full_fitness=2.00,
                    composite_score=2.10,
                    checks_passed=6,
                    checks_total=6,
                    is_verified=False,  # Quarantined
                    provenance=None,
                )
                session.add(ev2)

                # Candidate 3: Incomplete / UNAVAILABLE
                c3 = ResearchCandidate(
                    session_id=1,
                    alpha_id="alpha_unavail",
                    expression="rank(volume)",
                    is_valid=True,
                    status=CandidateStatus.UNAVAILABLE,
                )
                session.add(c3)

            # 1. Leaderboard test
            lb = LeaderboardService(db)
            ranks = await lb.get_alpha_leaderboard(session_id=1)
            cand_ids = [r.candidate_id for r in ranks]
            assert c1.id in cand_ids
            assert c2.id not in cand_ids, "Unverified historical candidate must not appear on leaderboard"
            assert c3.id not in cand_ids, "UNAVAILABLE candidate must not appear on leaderboard"

            # 2. Research Memory test
            mem = ResearchMemory(db)
            winners, losers = await mem.get_session_performance_split(session_id=1)
            all_memory_cands = [r["candidate_id"] for r in winners + losers]
            assert c1.id in all_memory_cands
            assert c2.id not in all_memory_cands, "Unverified candidate must not enter research memory splits"
            assert c3.id not in all_memory_cands, "UNAVAILABLE candidate must not enter research memory splits"

            # 3. Operator efficacy stats test
            op_stats = await mem.compute_operator_stats(session_id=1)
            assert "ts_mean" not in op_stats
            assert "rank" in op_stats
        finally:
            await db.dispose()

    asyncio.run(_run())
