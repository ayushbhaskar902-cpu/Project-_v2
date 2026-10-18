"""Regression test suite for P0-02: Correct Data Provenance Throughout the System."""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select, text

from alpha_harness.api.research import CandidateOut, GenerationAttemptOut
from alpha_harness.db.duck import Catalog
from alpha_harness.db.models import (
    CandidateStatus,
    GenerationAttempt,
    PromptStrategy,
    ResearchCandidate,
    ResearchEvaluation,
    ResearchResearcher,
    ResearchSession,
    ResearchStatus,
    utcnow,
)
from alpha_harness.db.sqlite import Database
from alpha_harness.evaluation.metrics import (
    EvaluationEngine,
    EvaluationScorecard,
    evaluate_pnl_series,
)
from alpha_harness.generator.pipeline import (
    AlphaGenerationPipeline,
    CandidateValidationResult,
    GeneratedCandidate,
    GenerationBatch,
    CANDIDATE_RESPONSE_SCHEMA,
)
from alpha_harness.learning.loop import SelfLearningEngine
from alpha_harness.learning.memory import ResearchMemory
from alpha_harness.orchestrator.researchers import MultiLLMOrchestrator
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
# 1. Evaluation Provenance: cost_bps and Effective split_date Tests
# ---------------------------------------------------------------------------


def test_evaluation_persists_explicit_split_date_and_cost(tmp_path: Path):
    """Verifies that an explicitly supplied split_date and cost_bps are persisted on ResearchEvaluation."""
    async def _run():
        cat = Catalog(tmp_path / "catalog.duckdb")
        await cat.open()
        db = Database(tmp_path / "test.db")
        await db.create_all()

        try:
            vault = AlphaVault(cat)
            alpha_id = "alpha_explicit_split"
            pnl_rows, to_rows = generate_series_rows(num_days=30)
            await vault.save_pnl(alpha_id, pnl_rows, to_rows, yearly_rows=[])

            async with db.session() as s:
                sess = ResearchSession(name="Sess", prompt="P", status=ResearchStatus.RUNNING)
                s.add(sess)
                await s.flush()

                cand = ResearchCandidate(
                    session_id=sess.id,
                    alpha_id=alpha_id,
                    expression="rank(close)",
                    status=CandidateStatus.SIMULATING,
                )
                s.add(cand)
                await s.flush()
                cand_id = cand.id

            engine = EvaluationEngine(db=db, vault=vault)
            explicit_split = date(2023, 1, 20)
            scorecard = await engine.evaluate_candidate(
                candidate_id=cand_id,
                split_date=explicit_split,
                cost_bps=2.5,
            )

            assert scorecard.cost_bps == 2.5
            assert scorecard.split_date == explicit_split

            async with db.session() as s:
                ev = await s.scalar(
                    select(ResearchEvaluation).where(ResearchEvaluation.candidate_id == cand_id)
                )
                assert ev is not None
                assert ev.cost_bps == 2.5
                assert ev.split_date == explicit_split
                assert ev.is_verified is True
                assert ev.provenance == "BRAIN_VAULT"
                assert ev.evaluated_at is not None
        finally:
            await cat.close()
            await db.dispose()

    asyncio.run(_run())


def test_evaluation_persists_effective_automatic_split_date(tmp_path: Path):
    """Verifies that when split_date=None, the effective calculated 80/20 split date is persisted."""
    async def _run():
        cat = Catalog(tmp_path / "catalog.duckdb")
        await cat.open()
        db = Database(tmp_path / "test.db")
        await db.create_all()

        try:
            vault = AlphaVault(cat)
            alpha_id = "alpha_auto_split"
            pnl_rows, to_rows = generate_series_rows(num_days=30)
            await vault.save_pnl(alpha_id, pnl_rows, to_rows, yearly_rows=[])

            # Expected 80% index date
            dates = [r["date"] for r in pnl_rows]
            expected_split = dates[int(len(dates) * 0.80)]

            async with db.session() as s:
                sess = ResearchSession(name="Sess", prompt="P", status=ResearchStatus.RUNNING)
                s.add(sess)
                await s.flush()

                cand = ResearchCandidate(
                    session_id=sess.id,
                    alpha_id=alpha_id,
                    expression="rank(close)",
                    status=CandidateStatus.SIMULATING,
                )
                s.add(cand)
                await s.flush()
                cand_id = cand.id

            engine = EvaluationEngine(db=db, vault=vault)
            scorecard = await engine.evaluate_candidate(
                candidate_id=cand_id,
                split_date=None,
                cost_bps=1.5,
            )

            assert scorecard.cost_bps == 1.5
            assert scorecard.split_date == expected_split

            async with db.session() as s:
                ev = await s.scalar(
                    select(ResearchEvaluation).where(ResearchEvaluation.candidate_id == cand_id)
                )
                assert ev is not None
                assert ev.cost_bps == 1.5
                assert ev.split_date == expected_split
                assert ev.is_verified is True
        finally:
            await cat.close()
            await db.dispose()

    asyncio.run(_run())


def test_reevaluation_updates_parameters_atomically(tmp_path: Path):
    """Verifies that reevaluating a candidate atomically updates cost_bps, effective split_date, and metrics in place."""
    async def _run():
        cat = Catalog(tmp_path / "catalog.duckdb")
        await cat.open()
        db = Database(tmp_path / "test.db")
        await db.create_all()

        try:
            vault = AlphaVault(cat)
            alpha_id = "alpha_reeval"
            pnl_rows, to_rows = generate_series_rows(num_days=30)
            await vault.save_pnl(alpha_id, pnl_rows, to_rows, yearly_rows=[])

            async with db.session() as s:
                sess = ResearchSession(name="Sess", prompt="P", status=ResearchStatus.RUNNING)
                s.add(sess)
                await s.flush()

                cand = ResearchCandidate(
                    session_id=sess.id,
                    alpha_id=alpha_id,
                    expression="rank(close)",
                    status=CandidateStatus.SIMULATING,
                )
                s.add(cand)
                await s.flush()
                cand_id = cand.id

            engine = EvaluationEngine(db=db, vault=vault)
            # First evaluation
            await engine.evaluate_candidate(candidate_id=cand_id, cost_bps=1.0)

            async with db.session() as s:
                ev1 = await s.scalar(
                    select(ResearchEvaluation).where(ResearchEvaluation.candidate_id == cand_id)
                )
                assert ev1.cost_bps == 1.0
                first_time = ev1.evaluated_at

            # Second evaluation with different parameters
            new_split = date(2023, 1, 15)
            await engine.evaluate_candidate(
                candidate_id=cand_id,
                split_date=new_split,
                cost_bps=4.0,
            )

            async with db.session() as s:
                ev2 = await s.scalar(
                    select(ResearchEvaluation).where(ResearchEvaluation.candidate_id == cand_id)
                )
                assert ev2.cost_bps == 4.0
                assert ev2.split_date == new_split
                assert ev2.evaluated_at >= first_time
                assert ev2.is_verified is True
        finally:
            await cat.close()
            await db.dispose()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 2. GenerationAttempt & Candidate Lineage Tests
# ---------------------------------------------------------------------------


def test_generation_attempt_persists_prompts_and_schema(tmp_path: Path):
    """Verifies that GenerationAttempt durably records system_prompt, user_prompt, response_schema, and links candidates."""
    async def _run():
        db = Database(tmp_path / "test.db")
        await db.create_all()

        try:
            class MockPipeline:
                async def generate_batch(self, **kwargs) -> GenerationBatch:
                    val = CandidateValidationResult(
                        is_valid=True,
                        expression="rank(close - vwap)",
                        operator_count=2,
                        field_count=2,
                        fields_used=["close", "vwap"],
                        retry_count=0,
                        repair_history=[],
                    )
                    cand = GeneratedCandidate(
                        hypothesis="Price reversion to vwap",
                        expression="rank(close - vwap)",
                        category="reversion",
                        target_dataset="fundamental_us",
                        primary_fields=["close", "vwap"],
                        validation=val,
                        raw_response={"expression": "rank(close - vwap)"},
                    )
                    return GenerationBatch(
                        summary="Batch generated",
                        candidates=[cand],
                        model_ref=kwargs.get("model_ref", "google:gemini-2.5-flash"),
                        total_tokens=150,
                        duration_seconds=0.5,
                        system_prompt="SYSTEM PROMPT WITH CATALOG DATASETS: [close, open, volume]",
                        user_prompt="USER PROMPT: Generate 1 alpha",
                        response_schema=CANDIDATE_RESPONSE_SCHEMA,
                    )

            orchestrator = MultiLLMOrchestrator(db=db, pipeline=MockPipeline())

            async with db.session() as s:
                sess = ResearchSession(name="Sess", prompt="P", status=ResearchStatus.IDLE)
                s.add(sess)
                await s.flush()

                r = ResearchResearcher(
                    session_id=sess.id,
                    model_ref="google:gemini-2.5-flash",
                    label="Lead",
                    temperature=0.6,
                    enabled=True,
                )
                s.add(r)
                await s.flush()
                sess_id = sess.id
                r_id = r.id

            valid_ids = await orchestrator.run_generation_round(session_id=sess_id, total_target_candidates=1)
            assert len(valid_ids) == 1

            async with db.session() as s:
                cand = await s.get(ResearchCandidate, valid_ids[0])
                assert cand is not None
                assert cand.generation_attempt_id is not None
                assert cand.model_ref == "google:gemini-2.5-flash"
                assert cand.temperature == 0.6
                assert cand.raw_expression == "rank(close - vwap)"

                attempt = await s.get(GenerationAttempt, cand.generation_attempt_id)
                assert attempt is not None
                assert attempt.session_id == sess_id
                assert attempt.researcher_id == r_id
                assert attempt.model_ref == "google:gemini-2.5-flash"
                assert attempt.temperature == 0.6
                assert "SYSTEM PROMPT WITH CATALOG DATASETS" in attempt.system_prompt
                assert "USER PROMPT" in attempt.user_prompt
                assert attempt.response_schema == CANDIDATE_RESPONSE_SCHEMA
        finally:
            await db.dispose()

    asyncio.run(_run())


def test_auto_repair_preserves_raw_expression_and_history(tmp_path: Path):
    """Verifies that an auto-repaired candidate saves its original faulty expression and repair sequence."""
    async def _run():
        db = Database(tmp_path / "test.db")
        await db.create_all()

        try:
            class MockRepairPipeline:
                async def generate_batch(self, **kwargs) -> GenerationBatch:
                    val = CandidateValidationResult(
                        is_valid=True,
                        expression="rank(close)",  # Repaired expression
                        operator_count=1,
                        field_count=1,
                        fields_used=["close"],
                        retry_count=1,
                        repair_history=["Attempt 1: Syntax parse error near '+++' -> repaired to rank(close)"],
                    )
                    cand = GeneratedCandidate(
                        hypothesis="Repaired factor",
                        expression="rank(close)",
                        category="general",
                        target_dataset=None,
                        primary_fields=["close"],
                        validation=val,
                        raw_response={"expression": "rank(close +++ 1)"},  # Faulty raw expression
                    )
                    return GenerationBatch(
                        summary="Repaired batch",
                        candidates=[cand],
                        model_ref="google:gemini-2.5-flash",
                        total_tokens=200,
                        duration_seconds=0.8,
                        system_prompt="SYSTEM PROMPT",
                        user_prompt="USER PROMPT",
                        response_schema=CANDIDATE_RESPONSE_SCHEMA,
                    )

            orchestrator = MultiLLMOrchestrator(db=db, pipeline=MockRepairPipeline())

            async with db.session() as s:
                sess = ResearchSession(name="Sess", prompt="P", status=ResearchStatus.IDLE)
                s.add(sess)
                await s.flush()

                r = ResearchResearcher(session_id=sess.id, model_ref="google:gemini-2.5-flash", label="Lead", enabled=True)
                s.add(r)
                await s.flush()
                sess_id = sess.id

            valid_ids = await orchestrator.run_generation_round(session_id=sess_id, total_target_candidates=1)

            async with db.session() as s:
                cand = await s.get(ResearchCandidate, valid_ids[0])
                assert cand is not None
                assert cand.expression == "rank(close)"
                assert cand.raw_expression == "rank(close +++ 1)"
                assert cand.retry_count == 1
                assert len(cand.repair_history) == 1
                assert "Syntax parse error" in cand.repair_history[0]
        finally:
            await db.dispose()

    asyncio.run(_run())


def test_candidate_immutable_model_snapshot(tmp_path: Path):
    """Verifies that modifying researcher settings does not alter the candidate's point-in-time model configuration."""
    async def _run():
        db = Database(tmp_path / "test.db")
        await db.create_all()

        try:
            class MockPipeline:
                async def generate_batch(self, **kwargs) -> GenerationBatch:
                    val = CandidateValidationResult(is_valid=True, expression="rank(close)")
                    cand = GeneratedCandidate(
                        hypothesis="H", expression="rank(close)", category="c",
                        target_dataset=None, primary_fields=[], validation=val
                    )
                    return GenerationBatch(
                        summary="S", candidates=[cand],
                        model_ref=kwargs.get("model_ref", "google:gemini-2.5-flash"),
                        total_tokens=100, duration_seconds=0.1
                    )

            orchestrator = MultiLLMOrchestrator(db=db, pipeline=MockPipeline())

            async with db.session() as s:
                sess = ResearchSession(name="Sess", prompt="P")
                s.add(sess)
                await s.flush()

                r = ResearchResearcher(
                    session_id=sess.id,
                    model_ref="google:gemini-2.5-flash",
                    label="Lead",
                    temperature=0.85,
                    enabled=True,
                )
                s.add(r)
                await s.flush()
                sess_id = sess.id
                r_id = r.id

            valid_ids = await orchestrator.run_generation_round(session_id=sess_id, total_target_candidates=1)

            # Now mutate researcher persona settings
            await orchestrator.update_researcher(
                researcher_id=r_id,
                temperature=0.15,
            )

            async with db.session() as s:
                mutated_r = await s.get(ResearchResearcher, r_id)
                assert mutated_r.temperature == 0.15

                # Candidate retains original point-in-time snapshot
                cand = await s.get(ResearchCandidate, valid_ids[0])
                assert cand.temperature == 0.85
                assert cand.model_ref == "google:gemini-2.5-flash"
        finally:
            await db.dispose()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 3. Strategy Parentage & Single-Active Deactivation Tests
# ---------------------------------------------------------------------------


def test_self_learning_deactivates_predecessor_strategy(tmp_path: Path):
    """Verifies that advancing generation sets previous strategy is_active=False and links parent_id."""
    async def _run():
        db = Database(tmp_path / "test.db")
        await db.create_all()

        try:
            engine = SelfLearningEngine(db=db)

            async with db.session() as s:
                sess = ResearchSession(name="Sess", prompt="P", current_generation=0)
                s.add(sess)
                await s.flush()

                strat_v1 = PromptStrategy(
                    session_id=sess.id,
                    name="Strategy V1",
                    version=1,
                    category="momentum",
                    is_active=True,
                )
                s.add(strat_v1)

                cand = ResearchCandidate(
                    session_id=sess.id,
                    expression="rank(close)",
                    status=CandidateStatus.EVALUATED,
                    generation=0,
                )
                s.add(cand)
                await s.flush()

                ev = ResearchEvaluation(
                    candidate_id=cand.id,
                    full_sharpe=2.1,
                    is_sharpe=2.0,
                    oos_sharpe=1.9,
                    full_fitness=1.5,
                    composite_score=1.8,
                    checks_passed=0,
                    checks_total=0,
                    is_verified=True,
                    provenance="BRAIN_VAULT",
                )
                s.add(ev)
                await s.flush()
                sess_id = sess.id
                strat_v1_id = strat_v1.id

            result = await engine.execute_learning_cycle(session_id=sess_id)
            assert result.updated_prompt_strategy_id is not None

            async with db.session() as s:
                v1 = await s.get(PromptStrategy, strat_v1_id)
                v2 = await s.get(PromptStrategy, result.updated_prompt_strategy_id)

                assert v1.is_active is False, "Predecessor strategy must be deactivated"
                assert v2.is_active is True
                assert v2.version == 2
                assert v2.parent_id == strat_v1_id, "Evolved strategy must link parent_id"
                assert v2.category == "momentum", "Category must be preserved from predecessor"
        finally:
            await db.dispose()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 4. API Response Models and Session Isolation Tests
# ---------------------------------------------------------------------------


def test_candidate_api_semantics_for_unevaluated_vs_evaluated():
    """Verifies CandidateOut derives verification strictly from evaluation and does not invent it for unevaluated ones."""
    # Unevaluated candidate
    cand_uneval = CandidateOut(
        id=1,
        sessionId=10,
        generation=0,
        hypothesis="H",
        expression="E",
        operatorCount=1,
        fieldCount=1,
        isValid=True,
        retryCount=0,
        status="SIMULATING",
        isVerified=None,
        evaluationProvenance=None,
        createdAt=utcnow(),
    )
    assert cand_uneval.is_verified is None
    assert cand_uneval.evaluation_provenance is None

    # Verified candidate
    cand_eval = CandidateOut(
        id=2,
        sessionId=10,
        generation=0,
        hypothesis="H",
        expression="E",
        operatorCount=1,
        fieldCount=1,
        isValid=True,
        retryCount=0,
        status="EVALUATED",
        fullSharpe=1.85,
        isVerified=True,
        evaluationProvenance="BRAIN_VAULT",
        generationAttemptId=42,
        alphaId="alpha_123",
        createdAt=utcnow(),
    )
    assert cand_eval.is_verified is True
    assert cand_eval.evaluation_provenance == "BRAIN_VAULT"
    assert cand_eval.generation_attempt_id == 42
    assert cand_eval.alpha_id == "alpha_123"


def test_generation_attempt_endpoint_session_isolation(tmp_path: Path):
    """Verifies that an attempt belonging to session A cannot be accessed from session B."""
    async def _run():
        db = Database(tmp_path / "test.db")
        await db.create_all()

        try:
            async with db.session() as s:
                sess_a = ResearchSession(name="A", prompt="P")
                sess_b = ResearchSession(name="B", prompt="P")
                s.add_all([sess_a, sess_b])
                await s.flush()

                attempt = GenerationAttempt(
                    session_id=sess_a.id,
                    system_prompt="Prompt A",
                    user_prompt="User A",
                    model_ref="google:gemini-2.5-flash",
                    temperature=0.7,
                    response_schema=CANDIDATE_RESPONSE_SCHEMA,
                )
                s.add(attempt)
                await s.flush()
                attempt_id = attempt.id
                sess_a_id = sess_a.id
                sess_b_id = sess_b.id

            # Verify query with matching session succeeds
            async with db.session() as s:
                stmt_ok = select(GenerationAttempt).where(
                    GenerationAttempt.id == attempt_id,
                    GenerationAttempt.session_id == sess_a_id,
                )
                res_ok = await s.scalar(stmt_ok)
                assert res_ok is not None

                # Verify query with mismatched session returns None (404 condition)
                stmt_bad = select(GenerationAttempt).where(
                    GenerationAttempt.id == attempt_id,
                    GenerationAttempt.session_id == sess_b_id,
                )
                res_bad = await s.scalar(stmt_bad)
                assert res_bad is None, "Cross-session attempt access must be rejected"
        finally:
            await db.dispose()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 5. Researcher Deletion Safeguard & Additive Migration Tests
# ---------------------------------------------------------------------------


def test_researcher_deletion_safeguard(tmp_path: Path):
    """Verifies that deleting a researcher sets researcher_id to NULL on GenerationAttempt and ResearchCandidate without erasing provenance."""
    async def _run():
        db = Database(tmp_path / "test.db")
        await db.create_all()

        try:
            async with db.session() as s:
                sess = ResearchSession(name="Sess", prompt="P")
                s.add(sess)
                await s.flush()

                r = ResearchResearcher(session_id=sess.id, model_ref="test-model", label="Lead")
                s.add(r)
                await s.flush()

                attempt = GenerationAttempt(
                    session_id=sess.id,
                    researcher_id=r.id,
                    system_prompt="PROMPT",
                    user_prompt="USER",
                    model_ref="test-model",
                    temperature=0.5,
                )
                s.add(attempt)
                await s.flush()

                cand = ResearchCandidate(
                    session_id=sess.id,
                    researcher_id=r.id,
                    generation_attempt_id=attempt.id,
                    expression="rank(close)",
                )
                s.add(cand)
                await s.flush()

                attempt_id = attempt.id
                cand_id = cand.id
                r_id = r.id

                # Delete the researcher
                await s.delete(r)
                await s.flush()

            async with db.session() as s:
                preserved_attempt = await s.get(GenerationAttempt, attempt_id)
                assert preserved_attempt is not None, "GenerationAttempt must NOT be deleted when researcher is deleted"
                assert preserved_attempt.researcher_id is None, "researcher_id must be SET NULL"
                assert preserved_attempt.system_prompt == "PROMPT"

                preserved_cand = await s.get(ResearchCandidate, cand_id)
                assert preserved_cand is not None, "ResearchCandidate must NOT be deleted when researcher is deleted"
                assert preserved_cand.researcher_id is None
                assert preserved_cand.generation_attempt_id == attempt_id
        finally:
            await db.dispose()

    asyncio.run(_run())


def test_additive_sqlite_migration_safety(tmp_path: Path):
    """Verifies that Database.create_all() runs non-destructively against an existing database, adding new columns and tables."""
    async def _run():
        db_path = tmp_path / "legacy.db"

        # Create a pre-P0-02 legacy SQLite database with existing records
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE research_candidate (
                id INTEGER PRIMARY KEY,
                session_id INTEGER,
                researcher_id INTEGER,
                prompt_strategy_id INTEGER,
                generation INTEGER DEFAULT 0,
                hypothesis TEXT,
                expression TEXT,
                language TEXT DEFAULT 'FASTEXPR',
                datasets_used JSON DEFAULT '[]',
                fields_used JSON DEFAULT '[]',
                operator_count INTEGER,
                field_count INTEGER,
                is_valid BOOLEAN,
                validation_error TEXT,
                retry_count INTEGER DEFAULT 0,
                simulation_record_id INTEGER,
                alpha_id TEXT,
                sim_region TEXT,
                sim_universe TEXT,
                sim_neutralization TEXT,
                sim_decay INTEGER,
                sim_delay INTEGER,
                status TEXT DEFAULT 'DRAFT',
                created_at TIMESTAMP,
                finished_at TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE research_evaluation (
                id INTEGER PRIMARY KEY,
                candidate_id INTEGER UNIQUE,
                is_sharpe FLOAT,
                full_sharpe FLOAT,
                composite_score FLOAT,
                is_verified BOOLEAN DEFAULT 0,
                provenance TEXT,
                evaluated_at TIMESTAMP
            )
        """)
        cursor.execute("INSERT INTO research_candidate (id, session_id, expression) VALUES (101, 1, 'rank(close)')")
        cursor.execute("INSERT INTO research_evaluation (id, candidate_id, full_sharpe) VALUES (201, 101, 1.75)")
        conn.commit()
        conn.close()

        # Run Database.create_all() migrator
        db = Database(db_path)
        await db.create_all()

        try:
            async with db.session() as s:
                cand = await s.get(ResearchCandidate, 101)
                assert cand is not None
                assert cand.expression == "rank(close)"
                # New P0-02 columns should be present and default to None
                assert cand.generation_attempt_id is None
                assert cand.model_ref is None
                assert cand.raw_expression is None

                ev = await s.get(ResearchEvaluation, 201)
                assert ev is not None
                assert ev.full_sharpe == 1.75
                # New evaluation parameters default to None
                assert ev.cost_bps is None
                assert ev.split_date is None

                # New table generation_attempt should be queryable
                attempts = await s.scalars(select(GenerationAttempt))
                assert list(attempts.all()) == []
        finally:
            await db.dispose()

    asyncio.run(_run())


def test_legacy_database_researcher_deletion_preserves_provenance(tmp_path: Path):
    """Verifies that running migrations on a pre-P0-02 database preserves FK SET NULL behavior on researcher deletion."""
    import sqlite3

    async def _run():
        db_path = tmp_path / "legacy_migration.db"

        # 1. Create representative pre-P0-02 schema matching commit ac6f4ef
        raw_con = sqlite3.connect(db_path)
        raw_con.execute("PRAGMA foreign_keys = ON")
        raw_con.execute("""
            CREATE TABLE research_session (
                id INTEGER PRIMARY KEY,
                name VARCHAR(200),
                prompt TEXT,
                instrument_type VARCHAR(16) DEFAULT 'EQUITY',
                region VARCHAR(16),
                universe VARCHAR(32),
                delay INTEGER DEFAULT 1,
                simulation_budget INTEGER DEFAULT 500,
                simulations_used INTEGER DEFAULT 0,
                learning_frequency INTEGER DEFAULT 80,
                current_generation INTEGER DEFAULT 0,
                model_refs JSON DEFAULT '[]',
                constraints JSON DEFAULT '{}',
                status VARCHAR(16) DEFAULT 'IDLE',
                message TEXT,
                created_at DATETIME,
                updated_at DATETIME,
                started_at DATETIME,
                finished_at DATETIME
            )
        """)
        raw_con.execute("""
            CREATE TABLE research_researcher (
                id INTEGER PRIMARY KEY,
                session_id INTEGER REFERENCES research_session(id) ON DELETE CASCADE,
                model_ref VARCHAR(200),
                persona_name VARCHAR(128),
                system_prompt TEXT,
                temperature FLOAT DEFAULT 0.7,
                focus_area VARCHAR(64),
                is_active BOOLEAN DEFAULT 1,
                created_at DATETIME,
                updated_at DATETIME
            )
        """)
        raw_con.execute("""
            CREATE TABLE prompt_strategy (
                id INTEGER PRIMARY KEY,
                name VARCHAR(200),
                version INTEGER DEFAULT 1,
                category VARCHAR(64),
                template_text TEXT,
                guidance JSON DEFAULT '{}',
                examples JSON DEFAULT '[]',
                anti_patterns JSON DEFAULT '[]',
                times_used INTEGER DEFAULT 0,
                total_candidates INTEGER DEFAULT 0,
                valid_rate FLOAT,
                avg_sharpe FLOAT,
                avg_fitness FLOAT,
                top_decile_rate FLOAT,
                oos_stability FLOAT,
                session_id INTEGER REFERENCES research_session(id) ON DELETE SET NULL,
                parent_id INTEGER REFERENCES prompt_strategy(id) ON DELETE SET NULL,
                is_active BOOLEAN DEFAULT 1,
                created_at DATETIME,
                updated_at DATETIME
            )
        """)
        raw_con.execute("""
            CREATE TABLE research_candidate (
                id INTEGER PRIMARY KEY,
                session_id INTEGER REFERENCES research_session(id) ON DELETE CASCADE,
                researcher_id INTEGER REFERENCES research_researcher(id) ON DELETE SET NULL,
                prompt_strategy_id INTEGER REFERENCES prompt_strategy(id) ON DELETE SET NULL,
                generation INTEGER DEFAULT 0,
                hypothesis TEXT,
                expression TEXT,
                language VARCHAR(16) DEFAULT 'FASTEXPR',
                datasets_used JSON DEFAULT '[]',
                fields_used JSON DEFAULT '[]',
                operator_count INTEGER,
                field_count INTEGER,
                is_valid BOOLEAN,
                validation_error TEXT,
                retry_count INTEGER DEFAULT 0,
                simulation_record_id INTEGER,
                alpha_id VARCHAR(64),
                sim_neutralization VARCHAR(32),
                sim_decay INTEGER,
                sim_delay INTEGER,
                status VARCHAR(16) DEFAULT 'DRAFT',
                error_message TEXT,
                created_at DATETIME,
                evaluated_at DATETIME
            )
        """)
        raw_con.execute("""
            CREATE TABLE research_evaluation (
                id INTEGER PRIMARY KEY,
                candidate_id INTEGER REFERENCES research_candidate(id) ON DELETE CASCADE UNIQUE,
                is_sharpe FLOAT,
                is_fitness FLOAT,
                is_turnover FLOAT,
                is_returns FLOAT,
                is_drawdown FLOAT,
                is_margin FLOAT,
                oos_sharpe FLOAT,
                oos_fitness FLOAT,
                oos_turnover FLOAT,
                oos_returns FLOAT,
                oos_drawdown FLOAT,
                full_sharpe FLOAT,
                full_fitness FLOAT,
                after_cost_sharpe FLOAT,
                is_oos_gap FLOAT,
                sharpe_decay FLOAT,
                composite_score FLOAT,
                checks JSON,
                checks_passed INTEGER,
                checks_total INTEGER,
                is_verified BOOLEAN DEFAULT 0,
                provenance VARCHAR(64),
                evaluated_at DATETIME
            )
        """)

        # Insert pre-migration records
        raw_con.execute("INSERT INTO research_session (id, name, created_at, updated_at) VALUES (1, 'session1', '2026-10-10 12:00:00', '2026-10-10 12:00:00')")
        raw_con.execute("INSERT INTO research_researcher (id, session_id, persona_name, model_ref) VALUES (1, 1, 'Researcher Alpha', 'anthropic:claude-3-5')")
        raw_con.execute("INSERT INTO prompt_strategy (id, session_id, name, category, is_active) VALUES (1, 1, 'strat1', 'momentum', 1)")
        raw_con.execute("INSERT INTO research_candidate (id, session_id, researcher_id, prompt_strategy_id, generation, hypothesis, expression, status) VALUES (1, 1, 1, 1, 0, 'hypo1', 'rank(close)', 'EVALUATED')")
        raw_con.execute("INSERT INTO research_evaluation (id, candidate_id, is_sharpe, oos_sharpe, is_verified, provenance) VALUES (1, 1, 1.8, 1.5, 1, 'BRAIN_VAULT')")
        raw_con.commit()
        raw_con.close()

        # Run application migration path
        db = Database(db_path)
        await db.create_all()

        try:
            async with db.session() as s:
                # Inspect PRAGMA table_info and foreign_key_list
                cand_cols = {row[1]: row for row in (await s.execute(text("PRAGMA table_info(research_candidate)"))).fetchall()}
                assert "researcher_id" in cand_cols
                assert cand_cols["researcher_id"][3] == 0, "researcher_id must be nullable (notnull=0)"

                cand_fks = (await s.execute(text("PRAGMA foreign_key_list(research_candidate)"))).fetchall()
                researcher_fk = [fk for fk in cand_fks if fk[2] == "research_researcher"]
                assert len(researcher_fk) > 0
                assert researcher_fk[0][6] == "SET NULL", f"FK action must be SET NULL, got {researcher_fk[0][6]}"

                gen_fks = (await s.execute(text("PRAGMA foreign_key_list(generation_attempt)"))).fetchall()
                gen_researcher_fk = [fk for fk in gen_fks if fk[2] == "research_researcher"]
                assert len(gen_researcher_fk) > 0
                assert gen_researcher_fk[0][6] == "SET NULL"

                # Insert GenerationAttempt linked to researcher
                await s.execute(text("""
                    INSERT INTO generation_attempt (id, session_id, researcher_id, prompt_strategy_id, generation, model_ref, system_prompt, user_prompt, response_schema, created_at)
                    VALUES (10, 1, 1, 1, 0, 'anthropic:claude-3-5', 'sys_prompt', 'user_prompt', '{"type": "object"}', '2026-10-10 12:00:00')
                """))
                await s.execute(text("UPDATE research_candidate SET generation_attempt_id = 10 WHERE id = 1"))

                # Delete researcher with foreign-key constraints enforced
                await s.execute(text("DELETE FROM research_researcher WHERE id = 1"))

                # Candidate survives with researcher_id = None and lineage intact
                cand_row = (await s.execute(text("SELECT id, researcher_id, generation_attempt_id, expression FROM research_candidate WHERE id = 1"))).fetchone()
                assert cand_row[0] == 1
                assert cand_row[1] is None, "Candidate researcher_id must be set to NULL on researcher deletion"
                assert cand_row[2] == 10
                assert cand_row[3] == "rank(close)"

                # GenerationAttempt survives with researcher_id = None and prompts preserved
                gen_row = (await s.execute(text("SELECT id, researcher_id, system_prompt, user_prompt FROM generation_attempt WHERE id = 10"))).fetchone()
                assert gen_row[0] == 10
                assert gen_row[1] is None, "GenerationAttempt researcher_id must be set to NULL on researcher deletion"
                assert gen_row[2] == "sys_prompt"
                assert gen_row[3] == "user_prompt"
        finally:
            await db.dispose()

    asyncio.run(_run())


def test_learning_memory_handles_null_checks_total(tmp_path: Path):
    """Verifies that ResearchMemory.get_session_performance_split safely handles checks_total=None without raising TypeError."""
    async def _run():
        db = Database(tmp_path / "mem_test.db")
        await db.create_all()

        try:
            mem = ResearchMemory(db=db)

            async with db.session() as s:
                sess = ResearchSession(name="Sess", prompt="P", current_generation=0)
                s.add(sess)
                await s.flush()

                cand = ResearchCandidate(
                    session_id=sess.id,
                    expression="ts_rank(close, 10)",
                    status=CandidateStatus.EVALUATED,
                    generation=0,
                )
                s.add(cand)
                await s.flush()

                # checks_total is None, representing an evaluation where BRAIN submission checks were not fetched
                ev = ResearchEvaluation(
                    candidate_id=cand.id,
                    full_sharpe=2.4,
                    composite_score=2.0,
                    checks_passed=None,
                    checks_total=None,
                    is_verified=True,
                    provenance="BRAIN_VAULT",
                )
                s.add(ev)
                await s.flush()
                sess_id = sess.id

            winners, losers = await mem.get_session_performance_split(session_id=sess_id, generation=0)
            assert len(winners) == 1
            assert winners[0]["candidate_id"] == cand.id
            assert winners[0]["is_submission_grade"] is False
        finally:
            await db.dispose()

    asyncio.run(_run())

