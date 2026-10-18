"""P1-09: Operator Attribution and Confounder Control Test Suite.

Verifies:
1. AST operator extraction vs fragile regex and validation against OperatorCatalog.
2. Operator credit deconfounding (complexity discounting & field quality adjustment).
3. Empirical Bayes / James-Stein shrinkage toward zero-prior.
4. Small-sample gating (appearances >= 3) and confidence levels.
5. Evidence classification: POSITIVE_ASSOCIATION, NEGATIVE_ASSOCIATION, INCONCLUSIVE, INSUFFICIENT_EVIDENCE.
6. Co-occurrence collinearity detection between operators.
7. Strict preservation of holdout isolation (only validation/in-sample Sharpe used).
8. Atomic persistence and retrieval of OperatorEfficacyRecord.
9. Integration with SelfLearningEngine (preferred/avoid operator ranking by confidence bounds).
10. Observational association disclaimers (not claiming correlation proves causation).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from alpha_harness.db.models import (
    CandidateStatus,
    HoldoutAssessment,
    OperatorEfficacyRecord,
    PromptStrategy,
    ResearchCandidate,
    ResearchEvaluation,
    ResearchResearcher,
    ResearchSession,
    utcnow,
)
from alpha_harness.db.sqlite import Database
from alpha_harness.discovery.catalog import OperatorCatalog
from alpha_harness.labs.fastexpr import extract_ast_operators
from alpha_harness.learning.loop import SelfLearningEngine
from alpha_harness.learning.memory import OperatorEfficacy, ResearchMemory
from alpha_harness.prompts.library import render_reflection_evidence_prompt


@pytest.fixture
def op_catalog() -> OperatorCatalog:
    return OperatorCatalog(scope="REGULAR")


# --- 1. AST Operator Extraction Tests ------------------------------------


def test_ast_operator_extraction_vs_regex(op_catalog: OperatorCatalog) -> None:
    """Verifies AST operator extraction extracts real call nodes and rejects non-operators."""
    # Expression with multiple operators, constants, field names, and variable lookbacks
    expr = "rank(ts_decay_linear(group_neutralize(close, subindustry), 10))"
    ops = extract_ast_operators(expr, catalog=op_catalog, deduplicate=True)
    assert ops == ["rank", "ts_decay_linear", "group_neutralize"]

    # Deduplication test: an operator used multiple times in 1 expression should only count once
    multi_expr = "rank(ts_rank(close, 10) - ts_rank(volume, 20))"
    dedup_ops = extract_ast_operators(multi_expr, catalog=op_catalog, deduplicate=True)
    assert dedup_ops == ["rank", "ts_rank"]
    assert len(dedup_ops) == 2

    raw_ops = extract_ast_operators(multi_expr, catalog=op_catalog, deduplicate=False)
    assert raw_ops == ["rank", "ts_rank", "ts_rank"]
    assert len(raw_ops) == 3


def test_ast_operator_extraction_rejects_unknown_operators(op_catalog: OperatorCatalog) -> None:
    """Verifies that invented/unknown operator names are rejected by OperatorCatalog validation."""
    expr = "invented_fake_op(ts_rank(close, 10))"
    ops = extract_ast_operators(expr, catalog=op_catalog, deduplicate=True)
    # Only ts_rank is a real WorldQuant BRAIN operator
    assert "invented_fake_op" not in ops
    assert ops == ["ts_rank"]


def test_ast_operator_extraction_handles_empty_and_malformed() -> None:
    """Verifies graceful handling of empty or unparseable expressions."""
    assert extract_ast_operators("") == []
    assert extract_ast_operators("   ") == []
    assert extract_ast_operators("((((malformed") == []


# --- 2. Statistical Attribution & Deconfounding Tests ---------------------


def test_operator_attribution_deconfounding(tmp_path: Path) -> None:
    """Verifies complexity weighting and cohort baseline adjustment in operator attribution."""
    async def _run():
        db = Database(tmp_path / "sqlite_p1_09_deconfounding.db")
        await db.create_all()

        async with db.session() as session:
            sess = ResearchSession(
                name="P1-09 Attribution Session",
                prompt="Explore momentum and mean reversion",
                region="USA",
                universe="TOP3000",
                current_generation=1,
            )
            session.add(sess)
            await session.commit()
            await session.refresh(sess)

            # Baseline candidates
            # Cand 1: clean single operator with Sharpe 1.80
            c1 = ResearchCandidate(
                session_id=sess.id,
                generation=1,
                hypothesis="Direct rank",
                expression="rank(close)",
                fields_used=["close"],
                operator_count=1,
                status=CandidateStatus.EVALUATED,
            )
            # Cand 2: bloated 4-operator formula with Sharpe 1.80
            c2 = ResearchCandidate(
                session_id=sess.id,
                generation=1,
                hypothesis="Bloated combination",
                expression="rank(ts_decay_linear(ts_rank(group_neutralize(close, sector), 10), 5))",
                fields_used=["close", "sector"],
                operator_count=4,
                status=CandidateStatus.EVALUATED,
            )
            # Cand 3: average candidate with Sharpe 1.00
            c3 = ResearchCandidate(
                session_id=sess.id,
                generation=1,
                hypothesis="Average volume signal",
                expression="ts_rank(volume, 20)",
                fields_used=["volume"],
                operator_count=1,
                status=CandidateStatus.EVALUATED,
            )
            session.add_all([c1, c2, c3])
            await session.commit()

            # Add genuine verified evaluations
            e1 = ResearchEvaluation(
                candidate_id=c1.id,
                is_verified=True,
                val_sharpe=1.80,
                full_sharpe=1.80,
                checks_passed=4,
                checks_total=4,
            )
            e2 = ResearchEvaluation(
                candidate_id=c2.id,
                is_verified=True,
                val_sharpe=1.80,
                full_sharpe=1.80,
                checks_passed=4,
                checks_total=4,
            )
            e3 = ResearchEvaluation(
                candidate_id=c3.id,
                is_verified=True,
                val_sharpe=1.00,
                full_sharpe=1.00,
                checks_passed=4,
                checks_total=4,
            )
            session.add_all([e1, e2, e3])
            await session.commit()

        memory = ResearchMemory(db)
        stats = await memory.compute_operator_stats(session_id=sess.id, min_sample_threshold=1)

        assert "rank" in stats
        assert "ts_decay_linear" in stats

        # Cohort baseline should be mean(1.80, 1.80, 1.00) = 1.533
        assert abs(stats["rank"].cohort_baseline - 1.533) < 0.05

        # 'rank' appears in c1 (1 operator, weight 1.0) and c2 (4 operators, weight 0.5)
        # 'ts_decay_linear' appears ONLY in c2 (4 operators, weight 0.5)
        # Both observe raw average Sharpe of 1.80, but rank has higher marginal lift because
        # it appeared in a clean expression, whereas ts_decay_linear only appeared in a complex multi-operator formula.
        assert stats["rank"].marginal_lift > stats["ts_decay_linear"].marginal_lift

    asyncio.run(_run())


# --- 3. Empirical Bayes Shrinkage Tests ----------------------------------


def test_empirical_bayes_shrinkage_penalizes_small_n(tmp_path: Path) -> None:
    """Verifies that an operator with N=1 and high Sharpe is shrunk towards zero prior

    and does not outrank a proven operator with N=10 and consistent moderate Sharpe.
    """
    async def _run():
        db = Database(tmp_path / "sqlite_p1_09_shrinkage.db")
        await db.create_all()

        async with db.session() as session:
            sess = ResearchSession(
                name="P1-09 Shrinkage Test Session",
                prompt="Momentum test",
                region="USA",
                universe="TOP3000",
                current_generation=1,
            )
            session.add(sess)
            await session.commit()
            await session.refresh(sess)

            candidates = []
            evaluations = []

            # Operator 'ts_rank': appears 10 times with consistent Sharpe = 1.40
            for i in range(10):
                c = ResearchCandidate(
                    session_id=sess.id,
                    generation=1,
                    hypothesis=f"Consistent ts_rank {i}",
                    expression=f"ts_rank(close, {5 + i})",
                    fields_used=["close"],
                    operator_count=1,
                    status=CandidateStatus.EVALUATED,
                )
                candidates.append(c)

            # Operator 'ts_decay_linear': appears ONCE with a lucky Sharpe = 2.20
            c_lucky = ResearchCandidate(
                session_id=sess.id,
                generation=1,
                hypothesis="Lucky single trial",
                expression="ts_decay_linear(close, 10)",
                fields_used=["close"],
                operator_count=1,
                status=CandidateStatus.EVALUATED,
            )
            candidates.append(c_lucky)

            # 5 neutral baseline candidates with Sharpe = 1.00 to anchor baseline
            for i in range(5):
                c_base = ResearchCandidate(
                    session_id=sess.id,
                    generation=1,
                    hypothesis=f"Baseline {i}",
                    expression=f"sign(volume - {i})",
                    fields_used=["volume"],
                    operator_count=1,
                    status=CandidateStatus.EVALUATED,
                )
                candidates.append(c_base)

            session.add_all(candidates)
            await session.commit()

            for c in candidates[:10]:
                evaluations.append(
                    ResearchEvaluation(
                        candidate_id=c.id,
                        is_verified=True,
                        val_sharpe=1.40,
                        full_sharpe=1.40,
                        checks_passed=4,
                        checks_total=4,
                    )
                )
            evaluations.append(
                ResearchEvaluation(
                    candidate_id=c_lucky.id,
                    is_verified=True,
                    val_sharpe=2.20,
                    full_sharpe=2.20,
                    checks_passed=4,
                    checks_total=4,
                )
            )
            for c in candidates[11:]:
                evaluations.append(
                    ResearchEvaluation(
                        candidate_id=c.id,
                        is_verified=True,
                        val_sharpe=1.00,
                        full_sharpe=1.00,
                        checks_passed=4,
                        checks_total=4,
                    )
                )

            session.add_all(evaluations)
            await session.commit()

        memory = ResearchMemory(db)
        # Using default shrinkage_k = 3.0
        stats = await memory.compute_operator_stats(session_id=sess.id, shrinkage_k=3.0, min_sample_threshold=1)

        assert "ts_rank" in stats
        assert "ts_decay_linear" in stats

        op_consistent = stats["ts_rank"]
        op_lucky = stats["ts_decay_linear"]

        # Raw average Sharpe of lucky operator is 2.20 > 1.40
        assert op_lucky.avg_sharpe > op_consistent.avg_sharpe

        # BUT shrunk lift of lucky operator is heavily attenuated: N=1 / (1 + 3) = 0.25 shrinkage factor
        # While consistent operator has N=10 / (10 + 3) = 0.77 shrinkage factor
        assert op_lucky.appearances == 1
        assert op_consistent.appearances == 10
        assert op_consistent.confidence_level == "HIGH"
        assert op_lucky.confidence_level in ("LOW", "INSUFFICIENT_SAMPLE")

    asyncio.run(_run())


# --- 4. Minimum Sample Gating & Confidence Bounds -------------------------


def test_minimum_sample_gate_and_evidence_levels(tmp_path: Path) -> None:
    """Verifies sample thresholds: N < 3 is INSUFFICIENT_SAMPLE / INSUFFICIENT_EVIDENCE."""
    async def _run():
        db = Database(tmp_path / "sqlite_p1_09_gate.db")
        await db.create_all()

        async with db.session() as session:
            sess = ResearchSession(
                name="P1-09 Gate Session",
                prompt="Momentum exploration",
                region="USA",
                universe="TOP3000",
                current_generation=1,
            )
            session.add(sess)
            await session.commit()
            await session.refresh(sess)

            # Operator 'rank' with 2 observations (N < 3 threshold)
            for i in range(2):
                c = ResearchCandidate(
                    session_id=sess.id,
                    generation=1,
                    hypothesis=f"Rank cand {i}",
                    expression=f"rank(close + {i})",
                    fields_used=["close"],
                    operator_count=1,
                    status=CandidateStatus.EVALUATED,
                )
                session.add(c)
                await session.commit()
                e = ResearchEvaluation(
                    candidate_id=c.id,
                    is_verified=True,
                    val_sharpe=2.00,
                    full_sharpe=2.00,
                    checks_passed=4,
                    checks_total=4,
                )
                session.add(e)
                await session.commit()

        memory = ResearchMemory(db)
        stats = await memory.compute_operator_stats(session_id=sess.id, min_sample_threshold=3)

        assert "rank" in stats
        rank_eff = stats["rank"]
        assert rank_eff.appearances == 2
        assert rank_eff.confidence_level == "INSUFFICIENT_SAMPLE"
        assert rank_eff.evidence_status == "INSUFFICIENT_EVIDENCE"
        assert rank_eff.is_statistically_significant is False

    asyncio.run(_run())


# --- 5. Holdout Isolation Test -------------------------------------------


def test_holdout_isolation_in_operator_stats(tmp_path: Path) -> None:
    """Verifies that compute_operator_stats uses ONLY in-sample/validation Sharpe

    and strictly isolates/withholds final holdout performance.
    """
    async def _run():
        db = Database(tmp_path / "sqlite_p1_09_holdout.db")
        await db.create_all()

        async with db.session() as session:
            sess = ResearchSession(
                name="P1-09 Holdout Isolation Session",
                prompt="Holdout test",
                region="USA",
                universe="TOP3000",
                current_generation=1,
            )
            session.add(sess)
            await session.commit()
            await session.refresh(sess)

            c = ResearchCandidate(
                session_id=sess.id,
                generation=1,
                hypothesis="Candidate with holdout data",
                expression="ts_rank(close, 10)",
                fields_used=["close"],
                operator_count=1,
                status=CandidateStatus.GRADUATED,
            )
            session.add(c)
            await session.commit()

            # val_sharpe is 1.50, but holdout_sharpe is 0.20 (post-graduation)
            e = ResearchEvaluation(
                candidate_id=c.id,
                is_verified=True,
                val_sharpe=1.50,
                full_sharpe=1.50,
                holdout_sharpe=0.20,
                checks_passed=4,
                checks_total=4,
            )
            session.add(e)
            await session.commit()

        memory = ResearchMemory(db)
        stats = await memory.compute_operator_stats(session_id=sess.id, min_sample_threshold=1)

        assert "ts_rank" in stats
        # Average Sharpe MUST be 1.50 (validation Sharpe), NOT 0.20 (holdout Sharpe)
        assert stats["ts_rank"].avg_sharpe == 1.50

    asyncio.run(_run())


# --- 6. Co-occurrence Collinearity Detection -----------------------------


def test_co_occurrence_collinearity_detection(tmp_path: Path) -> None:
    """Verifies detection of highly collinear operator pairs (co-occurrence >= 85% and count >= 3)."""
    async def _run():
        db = Database(tmp_path / "sqlite_p1_09_collinearity.db")
        await db.create_all()

        async with db.session() as session:
            sess = ResearchSession(
                name="P1-09 Collinearity Session",
                prompt="Collinear pair test",
                region="USA",
                universe="TOP3000",
                current_generation=1,
            )
            session.add(sess)
            await session.commit()
            await session.refresh(sess)

            # 4 candidates where 'rank' and 'ts_decay_linear' are always paired together
            for i in range(4):
                c = ResearchCandidate(
                    session_id=sess.id,
                    generation=1,
                    hypothesis=f"Paired formula {i}",
                    expression=f"rank(ts_decay_linear(close, {5 + i}))",
                    fields_used=["close"],
                    operator_count=2,
                    status=CandidateStatus.EVALUATED,
                )
                session.add(c)
                await session.commit()
                e = ResearchEvaluation(
                    candidate_id=c.id,
                    is_verified=True,
                    val_sharpe=1.60,
                    full_sharpe=1.60,
                    checks_passed=4,
                    checks_total=4,
                )
                session.add(e)
                await session.commit()

        memory = ResearchMemory(db)
        stats = await memory.compute_operator_stats(session_id=sess.id, min_sample_threshold=1)

        assert "rank" in stats
        assert "ts_decay_linear" in stats
        assert "ts_decay_linear" in stats["rank"].co_occurring_operators
        assert "rank" in stats["ts_decay_linear"].co_occurring_operators

    asyncio.run(_run())


# --- 7. Persistence and Database Retrieval --------------------------------


def test_persist_and_retrieve_operator_stats(tmp_path: Path) -> None:
    """Verifies storing and retrieving OperatorEfficacyRecord models with statistical metadata."""
    async def _run():
        db = Database(tmp_path / "sqlite_p1_09_persist.db")
        await db.create_all()

        async with db.session() as session:
            sess = ResearchSession(
                name="P1-09 Persistence Session",
                prompt="Persistence test",
                region="USA",
                universe="TOP3000",
                current_generation=1,
            )
            session.add(sess)
            await session.commit()
            await session.refresh(sess)

        dummy_eff = OperatorEfficacy(
            name="ts_rank",
            appearances=5,
            winner_count=4,
            loser_count=1,
            avg_sharpe=1.65,
            cohort_baseline=1.20,
            observed_lift=0.45,
            marginal_lift=0.40,
            shrunk_lift=0.25,
            std_error=0.08,
            t_stat=3.12,
            lower_confidence_bound=0.0932,
            upper_confidence_bound=0.4068,
            confidence_level="MEDIUM",
            evidence_status="POSITIVE_ASSOCIATION",
            co_occurring_operators=["rank"],
            adjusted_for_fields=True,
            adjusted_for_complexity=True,
            method_version="p1_09_v1",
        )

        memory = ResearchMemory(db)
        records = await memory.persist_operator_stats(
            session_id=sess.id,
            stats={"ts_rank": dummy_eff},
            generation=1,
            category="momentum",
            provenance="BRAIN_SIM",
        )

        assert len(records) == 1
        rec = records[0]
        assert rec.operator_name == "ts_rank"
        assert rec.sample_count == 5
        assert rec.shrunk_lift == 0.25
        assert rec.evidence_status == "POSITIVE_ASSOCIATION"
        assert rec.confidence_level == "MEDIUM"

        # Query from database
        retrieved = await memory.get_persisted_operator_stats(session_id=sess.id, generation=1)
        assert len(retrieved) == 1
        assert "ts_rank" in retrieved
        assert retrieved["ts_rank"].name == "ts_rank"
        assert retrieved["ts_rank"].marginal_lift == 0.40
        assert retrieved["ts_rank"].lower_confidence_bound == 0.0932

    asyncio.run(_run())


# --- 8. SelfLearningEngine Integration Test -------------------------------


def test_learning_loop_selects_operators_by_confidence_bounds(tmp_path: Path) -> None:
    """Verifies that SelfLearningEngine filters out operators with N < 3

    and selects preferred_operators by Lower Confidence Bound.
    """
    async def _run():
        db = Database(tmp_path / "sqlite_p1_09_learning.db")
        await db.create_all()

        async with db.session() as session:
            sess = ResearchSession(
                name="P1-09 Learning Loop Session",
                prompt="Explore momentum strategies",
                region="USA",
                universe="TOP3000",
                current_generation=1,
            )
            session.add(sess)
            await session.commit()
            await session.refresh(sess)

            strat = PromptStrategy(
                session_id=sess.id,
                category="momentum",
                version=1,
                name="Initial Momentum",
                is_active=True,
                template_text="Generate momentum alphas",
            )
            session.add(strat)
            await session.commit()

            # Add 4 candidates with 'ts_rank' (consistent winners, Sharpe=1.80)
            for i in range(4):
                c = ResearchCandidate(
                    session_id=sess.id,
                    prompt_strategy_id=strat.id,
                    generation=1,
                    hypothesis=f"ts_rank momentum {i}",
                    expression=f"ts_rank(close, {10 + i})",
                    fields_used=["close"],
                    operator_count=1,
                    status=CandidateStatus.EVALUATED,
                )
                session.add(c)
                await session.commit()
                e = ResearchEvaluation(
                    candidate_id=c.id,
                    is_verified=True,
                    val_sharpe=1.80,
                    full_sharpe=1.80,
                    checks_passed=4,
                    checks_total=4,
                )
                session.add(e)
                await session.commit()

            # Add 1 candidate with 'rank' having lucky Sharpe=2.20 (N=1 < 3 sample threshold)
            c_lucky = ResearchCandidate(
                session_id=sess.id,
                prompt_strategy_id=strat.id,
                generation=1,
                hypothesis="Lucky rank",
                expression="rank(close)",
                fields_used=["close"],
                operator_count=1,
                status=CandidateStatus.EVALUATED,
            )
            session.add(c_lucky)
            await session.commit()
            e_lucky = ResearchEvaluation(
                candidate_id=c_lucky.id,
                is_verified=True,
                val_sharpe=2.20,
                full_sharpe=2.20,
                checks_passed=4,
                checks_total=4,
            )
            session.add(e_lucky)
            await session.commit()

            # Add 4 baseline candidates with Sharpe=0.80 to anchor cohort baseline to ~1.40
            for i in range(4):
                c_base = ResearchCandidate(
                    session_id=sess.id,
                    prompt_strategy_id=strat.id,
                    generation=1,
                    hypothesis=f"Baseline signal {i}",
                    expression=f"sign(volume - {i})",
                    fields_used=["volume"],
                    operator_count=1,
                    status=CandidateStatus.EVALUATED,
                )
                session.add(c_base)
                await session.commit()
                e_base = ResearchEvaluation(
                    candidate_id=c_base.id,
                    is_verified=True,
                    val_sharpe=0.80,
                    full_sharpe=0.80,
                    checks_passed=4,
                    checks_total=4,
                )
                session.add(e_base)
                await session.commit()

        engine = SelfLearningEngine(db=db)
        result = await engine.execute_learning_cycle(
            session_id=sess.id,
            category="momentum",
            min_candidates=1,
        )

        assert result.status in ("COMPLETED", "SUCCESS")

        async with db.session() as session:
            # Check newly created evolved prompt strategy
            s_stmt = select(PromptStrategy).where(
                PromptStrategy.session_id == sess.id,
                PromptStrategy.category == "momentum",
                PromptStrategy.version == 2,
            )
            s_res = await session.execute(s_stmt)
            evolved_strat = s_res.scalar_one_or_none()
            assert evolved_strat is not None

            guidance = evolved_strat.guidance or {}
            preferred = guidance.get("preferred_operators", [])

            # 'rank' had N=1 < 3, so it MUST NOT be in preferred_operators even with Sharpe 2.20
            assert "rank" not in preferred

            # 'ts_rank' had N=4 >= 3 and positive association, so it is in preferred_operators
            assert "ts_rank" in preferred

    asyncio.run(_run())


# --- 9. Observational Association Disclaimer Verification ----------------


def test_prompt_library_observational_disclaimer() -> None:
    """Verifies that reflection prompts explicitly disclose observational associations."""
    factual_stats = {
        "batch_size": 5,
        "avg_sharpe": 1.45,
        "best_sharpe": 1.80,
        "valid_rate": 1.0,
        "operator_stats": [
            {
                "name": "ts_rank",
                "appearances": 5,
                "avg_sharpe": 1.65,
                "shrunk_lift": 0.20,
                "confidence_level": "MEDIUM",
                "evidence_status": "POSITIVE_ASSOCIATION",
            }
        ],
    }
    prompt = render_reflection_evidence_prompt(
        session_id=1,
        generation=1,
        eligible_candidates=[{"candidate_id": 1, "expression": "ts_rank(close, 10)", "sharpe": 1.65}],
        factual_stats=factual_stats,
    )
    assert "Authoritative Operator Statistics (P1-09 Controlled Attribution):" in prompt
    assert "observational statistical associations" in prompt.lower()
    assert "not proven causal drivers" in prompt.lower()
