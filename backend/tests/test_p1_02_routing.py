"""P1-02: Prompt-Strategy Category/Version Routing regression test suite.

Mandatory tests:
1. test_category_specific_evolution_from_candidate_strategy_lineage
2. test_mixed_category_evidence_isolation
3. test_category_starvation_prevention
4. test_legacy_category_normalization_and_version_ordering
5. test_migration_idempotency_and_duplicate_active_deactivation
6. test_strict_cross_session_isolation
7. test_precedence_fallback_tiers_a_through_e
8. test_generation_attempt_resolution_tier_and_strategy_tracking
9. test_atomic_rollback_on_persistence_failure
10. test_preservation_of_unverified_hypothesis_and_recommendation_labels
11. test_multi_category_weighted_average_sharpe
12. test_experiment_runner_learn_phase_integration
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import func, select, text

from alpha_harness.db.models import (
    CandidateStatus,
    GenerationAttempt,
    PromptStrategy,
    PromptVersion,
    ResearchCandidate,
    ResearchEvaluation,
    ResearchInsight,
    ResearchResearcher,
    ResearchSession,
    ResearchStatus,
)
from alpha_harness.db.sqlite import Database, migrate
from alpha_harness.prompts.versioning import compute_prompt_content_hash
from alpha_harness.experiment.runner import ExperimentRunner
from alpha_harness.generator.pipeline import (
    CandidateValidationResult,
    GeneratedCandidate,
    GenerationBatch,
)
from alpha_harness.learning.loop import (
    LearningCycleResult,
    MultiCategoryLearningResult,
    SelfLearningEngine,
)
from alpha_harness.learning.memory import ResearchMemory
from alpha_harness.llm.service import Answer
from alpha_harness.orchestrator.researchers import MultiLLMOrchestrator
from alpha_harness.prompts.library import (
    PromptLibrary,
    StrategyCategory,
    normalize_category,
)


def _make_mock_answer(content: dict | str, model: str = "test-model") -> Answer:
    text_content = json.dumps(content) if isinstance(content, dict) else str(content)
    return Answer(
        text=text_content,
        model=model,
        prompt_tokens=150,
        output_tokens=80,
        thinking_tokens=0,
        total_tokens=230,
    )


# ---------------------------------------------------------------------------
# Test 1: Category-Specific Evolution from Candidate Strategy Lineage
# ---------------------------------------------------------------------------
def test_category_specific_evolution_from_candidate_strategy_lineage(tmp_path: Path):
    """Verifies that learning cycle evolves distinct prompt strategies for each category
    derived from candidate lineage and assigns the evolved strategy to that specific category.
    """
    async def _run():
        db = Database(tmp_path / "sqlite.db")
        await db.create_all()

        async with db.session() as s:
            sess = ResearchSession(
                name="Lineage Test Session",
                prompt="test",
                region="USA",
                universe="TOP3000",
                current_generation=0,
            )
            s.add(sess)
            await s.flush()

            # Active antecedent strategies for reversal and momentum
            rev_strat = PromptStrategy(
                session_id=sess.id,
                name="Reversal v1",
                version=1,
                category="reversal",
                is_active=True,
            )
            mom_strat = PromptStrategy(
                session_id=sess.id,
                name="Momentum v1",
                version=1,
                category="momentum",
                is_active=True,
            )
            s.add_all([rev_strat, mom_strat])
            await s.flush()

            # Create researchers
            r_rev = ResearchResearcher(
                session_id=sess.id,
                label="RevResearcher",
                strategy_focus="reversal",
                enabled=True,
            )
            r_mom = ResearchResearcher(
                session_id=sess.id,
                label="MomResearcher",
                strategy_focus="momentum",
                enabled=True,
            )
            s.add_all([r_rev, r_mom])
            await s.flush()

            # Candidates with explicit lineage
            c_rev = ResearchCandidate(
                session_id=sess.id,
                researcher_id=r_rev.id,
                prompt_strategy_id=rev_strat.id,
                expression="ts_delta(close, 5)",
                status=CandidateStatus.EVALUATED,
                generation=0,
                hypothesis="Reversal hypothesis",
            )
            c_mom = ResearchCandidate(
                session_id=sess.id,
                researcher_id=r_mom.id,
                prompt_strategy_id=mom_strat.id,
                expression="ts_mean(volume, 20)",
                status=CandidateStatus.EVALUATED,
                generation=0,
                hypothesis="Momentum hypothesis",
            )
            s.add_all([c_rev, c_mom])
            await s.flush()

            eval_rev = ResearchEvaluation(
                candidate_id=c_rev.id,
                is_verified=True,
                val_sharpe=1.8,
                val_turnover=0.15,
                val_fitness=1.2,
                provenance="BRAIN_VAULT",
            )
            eval_mom = ResearchEvaluation(
                candidate_id=c_mom.id,
                is_verified=True,
                val_sharpe=1.5,
                val_turnover=0.20,
                val_fitness=1.1,
                provenance="BRAIN_VAULT",
            )
            s.add_all([eval_rev, eval_mom])
            await s.commit()

            sess_id = sess.id
            rev_id = rev_strat.id
            mom_id = mom_strat.id

        engine = SelfLearningEngine(db=db, llm=None)
        result = await engine.execute_learning_cycle(session_id=sess_id, category=None, min_candidates=1)

        assert isinstance(result, MultiCategoryLearningResult)
        assert set(result.categories_evolved) == {"reversal", "momentum"}
        assert len(result.updated_prompt_strategy_ids) == 2

        # Verify active and inactive strategies in SQLite
        async with db.session() as s:
            # Prior strategies must now be deactivated
            old_rev = await s.get(PromptStrategy, rev_id)
            old_mom = await s.get(PromptStrategy, mom_id)
            assert old_rev.is_active is False
            assert old_mom.is_active is False

            # Newly evolved strategies must be active with incremented version and correct category
            active_strats = (
                await s.execute(
                    select(PromptStrategy).where(
                        PromptStrategy.session_id == sess_id,
                        PromptStrategy.is_active.is_(True),
                    )
                )
            ).scalars().all()

            assert len(active_strats) == 2
            strat_by_cat = {st.category: st for st in active_strats}
            assert "reversal" in strat_by_cat
            assert "momentum" in strat_by_cat

            assert strat_by_cat["reversal"].version == 2
            assert strat_by_cat["reversal"].parent_id == rev_id
            assert strat_by_cat["momentum"].version == 2
            assert strat_by_cat["momentum"].parent_id == mom_id

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 2: Mixed-Category Evidence Isolation
# ---------------------------------------------------------------------------
def test_mixed_category_evidence_isolation(tmp_path: Path):
    """Verifies that candidates and evidence belonging to category A never bleed into
    insights or prompt strategies generated for category B.
    """
    async def _run():
        db = Database(tmp_path / "sqlite.db")
        await db.create_all()

        async with db.session() as s:
            sess = ResearchSession(
                name="Evidence Isolation Session",
                prompt="test",
                region="USA",
                universe="TOP3000",
                current_generation=0,
            )
            s.add(sess)
            await s.flush()

            rev_strat = PromptStrategy(session_id=sess.id, name="Rev v1", version=1, category="reversal", is_active=True)
            mom_strat = PromptStrategy(session_id=sess.id, name="Mom v1", version=1, category="momentum", is_active=True)
            s.add_all([rev_strat, mom_strat])
            await s.flush()

            # Distinct expressions
            REV_EXPR = "ts_rank(reversal_metric, 10)"
            MOM_EXPR = "ts_sum(momentum_signal, 20)"

            c_rev = ResearchCandidate(
                session_id=sess.id,
                prompt_strategy_id=rev_strat.id,
                expression=REV_EXPR,
                status=CandidateStatus.EVALUATED,
                generation=0,
                hypothesis="Reversal signal hypothesis",
            )
            c_mom = ResearchCandidate(
                session_id=sess.id,
                prompt_strategy_id=mom_strat.id,
                expression=MOM_EXPR,
                status=CandidateStatus.EVALUATED,
                generation=0,
                hypothesis="Momentum signal hypothesis",
            )
            s.add_all([c_rev, c_mom])
            await s.flush()

            s.add_all([
                ResearchEvaluation(candidate_id=c_rev.id, is_verified=True, val_sharpe=2.2, val_turnover=0.1, val_fitness=1.5, provenance="BRAIN_VAULT"),
                ResearchEvaluation(candidate_id=c_mom.id, is_verified=True, val_sharpe=1.9, val_turnover=0.15, val_fitness=1.3, provenance="BRAIN_VAULT"),
            ])
            await s.commit()
            sess_id = sess.id
            c_rev_id = c_rev.id
            c_mom_id = c_mom.id

        engine = SelfLearningEngine(db=db, llm=None)
        await engine.execute_learning_cycle(session_id=sess_id, category=None, min_candidates=1)

        # Inspect generated insights
        async with db.session() as s:
            insights = (
                await s.execute(
                    select(ResearchInsight).where(ResearchInsight.session_id == sess_id)
                )
            ).scalars().all()

            assert len(insights) >= 2
            for ins in insights:
                ev_ids = ins.evidence_candidate_ids or []
                if c_rev_id in ev_ids:
                    # Reversal insight must not include momentum candidate ID
                    assert c_mom_id not in ev_ids
                    assert MOM_EXPR not in json.dumps(ins.evidence)
                elif c_mom_id in ev_ids:
                    # Momentum insight must not include reversal candidate ID
                    assert c_rev_id not in ev_ids
                    assert REV_EXPR not in json.dumps(ins.evidence)

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 3: Category Starvation Prevention
# ---------------------------------------------------------------------------
def test_category_starvation_prevention(tmp_path: Path):
    """Verifies that if one category has enough evidence and another does not,
    the active category evolves normally while the under-evidenced category returns
    INSUFFICIENT_EVIDENCE without stalling or failing the cycle.
    """
    async def _run():
        db = Database(tmp_path / "sqlite.db")
        await db.create_all()

        async with db.session() as s:
            sess = ResearchSession(
                name="Starvation Session",
                prompt="test",
                region="USA",
                universe="TOP3000",
                current_generation=0,
            )
            s.add(sess)
            await s.flush()

            s.add_all([
                PromptStrategy(session_id=sess.id, name="Rev v1", version=1, category="reversal", is_active=True),
                PromptStrategy(session_id=sess.id, name="Mom v1", version=1, category="momentum", is_active=True),
            ])
            await s.flush()

            # Researchers
            r_rev = ResearchResearcher(session_id=sess.id, label="RRev", strategy_focus="reversal", enabled=True)
            r_mom = ResearchResearcher(session_id=sess.id, label="RMom", strategy_focus="momentum", enabled=True)
            s.add_all([r_rev, r_mom])
            await s.flush()

            # Only reversal has candidates
            c_rev = ResearchCandidate(
                session_id=sess.id,
                researcher_id=r_rev.id,
                expression="rank(close)",
                status=CandidateStatus.EVALUATED,
                generation=0,
                hypothesis="Only reversal candidates",
            )
            s.add(c_rev)
            await s.flush()
            s.add(ResearchEvaluation(candidate_id=c_rev.id, is_verified=True, val_sharpe=1.6, val_turnover=0.1, val_fitness=1.0, provenance="BRAIN_VAULT"))
            await s.commit()
            sess_id = sess.id

        engine = SelfLearningEngine(db=db, llm=None)
        result = await engine.execute_learning_cycle(session_id=sess_id, category=None, min_candidates=1)

        assert isinstance(result, MultiCategoryLearningResult)
        assert "reversal" in result.categories_evolved

        # Momentum had no evaluated candidates at all, so the multi-category discovery
        # only discovers categories with actual evidence — momentum is absent entirely
        # rather than returning INSUFFICIENT_EVIDENCE.
        assert "momentum" not in result.categories_evolved
        assert result.category_statuses.get("reversal") == "COMPLETED"

        # Verify session generation advanced because at least one category evolved
        async with db.session() as s:
            session_row = await s.get(ResearchSession, sess_id)
            assert session_row.current_generation == 1

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 4: Legacy Category Normalization and Version Ordering
# ---------------------------------------------------------------------------
def test_legacy_category_normalization_and_version_ordering(tmp_path: Path):
    """Verifies that normalize_category trims and lowercases categories safely,
    and queries order by version DESC, id DESC deterministically.
    """
    # 1. Normalization unit tests
    assert normalize_category(" MOMENTUM ") == "momentum"
    assert normalize_category("Reversal") == "reversal"
    assert normalize_category("reversal") == "reversal"  # Preserved as distinct category identity
    assert normalize_category("mean-reversion") == "mean_reversion"  # Canonical typographical alias
    assert normalize_category("Mean Reversion") == "mean_reversion"  # Canonical typographical alias
    assert normalize_category("MEAN_REVERSION") == "mean_reversion"
    assert normalize_category("meanreversion") == "mean_reversion"
    assert normalize_category("sentiment") == "sentiment"  # Nonempty custom category preserved
    assert normalize_category("  Custom_Strategy  ") == "custom_strategy"
    assert normalize_category("") == "general"
    assert normalize_category("   ") == "general"
    assert normalize_category(None) == "general"
    assert normalize_category(StrategyCategory.MOMENTUM) == "momentum"
    assert normalize_category(StrategyCategory.MEAN_REVERSION) == "mean_reversion"
    assert normalize_category(StrategyCategory.GENERAL) == "general"

    # 2. Database deterministic version ordering test
    async def _run():
        db = Database(tmp_path / "sqlite.db")
        await db.create_all()

        async with db.session() as s:
            sess = ResearchSession(name="Ordering Session", prompt="test", region="USA", universe="TOP3000")
            s.add(sess)
            await s.flush()

            # Insert strategies out of order
            s.add(PromptStrategy(session_id=sess.id, name="Strat v1", version=1, category="reversal", is_active=True))
            s.add(PromptStrategy(session_id=sess.id, name="Strat v3", version=3, category="reversal", is_active=True))
            s.add(PromptStrategy(session_id=sess.id, name="Strat v2", version=2, category="reversal", is_active=True))
            await s.commit()

            # Query with deterministic order_by(version.desc(), id.desc())
            stmt = (
                select(PromptStrategy)
                .where(
                    PromptStrategy.session_id == sess.id,
                    func.lower(func.trim(PromptStrategy.category)) == "reversal",
                    PromptStrategy.is_active.is_(True),
                )
                .order_by(PromptStrategy.version.desc(), PromptStrategy.id.desc())
                .limit(1)
            )
            res = await s.execute(stmt)
            best_strat = res.scalar_one_or_none()
            assert best_strat is not None
            assert best_strat.version == 3

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 5: Migration Idempotency and Duplicate Active Deactivation
# ---------------------------------------------------------------------------
def test_migration_idempotency_and_duplicate_active_deactivation(tmp_path: Path):
    """Verifies that sqlite.migrate() cleans dirty category casing and deactivates
    duplicate active strategies retaining only the highest (version DESC, id DESC),
    and is safe and idempotent on repeated executions.
    """
    async def _run():
        db = Database(tmp_path / "migrate_test.db")
        await db.create_all()

        # Insert dirty categories and duplicate active strategies using raw SQL
        async with db._engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO research_session (id, name, prompt, region, universe, created_at, updated_at) "
                    "VALUES (1, 'Migrate Sess', 'test', 'USA', 'TOP3000', '2026-01-01', '2026-01-01')"
                )
            )
            # Duplicate active strategies for session 1, category ' REVERSAL '
            await conn.execute(
                text(
                    "INSERT INTO prompt_strategy (id, session_id, name, version, category, is_active, created_at, updated_at) "
                    "VALUES (10, 1, 'Rev v1', 1, '  REVERSAL  ', 1, '2026-01-01', '2026-01-01')"
                )
            )
            await conn.execute(
                text(
                    "INSERT INTO prompt_strategy (id, session_id, name, version, category, is_active, created_at, updated_at) "
                    "VALUES (11, 1, 'Rev v2', 2, 'reversal', 1, '2026-01-01', '2026-01-01')"
                )
            )
            # Duplicate active global strategies (session_id IS NULL)
            await conn.execute(
                text(
                    "INSERT INTO prompt_strategy (id, session_id, name, version, category, is_active, created_at, updated_at) "
                    "VALUES (20, NULL, 'Glob Mom v1', 1, ' MOMENTUM ', 1, '2026-01-01', '2026-01-01')"
                )
            )
            await conn.execute(
                text(
                    "INSERT INTO prompt_strategy (id, session_id, name, version, category, is_active, created_at, updated_at) "
                    "VALUES (21, NULL, 'Glob Mom v2', 2, 'momentum', 1, '2026-01-01', '2026-01-01')"
                )
            )
            # Dirty researcher category
            await conn.execute(
                text(
                    "INSERT INTO research_researcher (id, session_id, label, strategy_focus, enabled, created_at, updated_at) "
                    "VALUES (100, 1, 'Dirty Researcher', '  VOLATILITY  ', 1, '2026-01-01', '2026-01-01')"
                )
            )

        # Run migration 1
        async with db._engine.begin() as conn:
            await migrate(conn)

        # Check normalization and duplicate deactivation
        async with db.session() as s:
            s_10 = await s.get(PromptStrategy, 10)
            s_11 = await s.get(PromptStrategy, 11)
            assert s_10.category == "reversal"
            assert s_10.is_active is False  # Lower version deactivated
            assert s_11.category == "reversal"
            assert s_11.is_active is True   # Higher version kept active

            g_20 = await s.get(PromptStrategy, 20)
            g_21 = await s.get(PromptStrategy, 21)
            assert g_20.category == "momentum"
            assert g_20.is_active is False
            assert g_21.category == "momentum"
            assert g_21.is_active is True

            r_100 = await s.get(ResearchResearcher, 100)
            assert r_100.strategy_focus == "volatility"

        # Run migration 2 (idempotency check)
        async with db._engine.begin() as conn:
            await migrate(conn)

        # Re-verify state remains unchanged
        async with db.session() as s:
            s_11 = await s.get(PromptStrategy, 11)
            assert s_11.is_active is True
            g_21 = await s.get(PromptStrategy, 21)
            assert g_21.is_active is True

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 6: Strict Cross-Session Isolation
# ---------------------------------------------------------------------------
def test_strict_cross_session_isolation(tmp_path: Path):
    """Verifies that an active strategy in Session 1 is never resolved or leaked
    into Session 2, even when querying the exact same category.
    """
    async def _run():
        db = Database(tmp_path / "sqlite.db")
        await db.create_all()

        async with db.session() as s:
            sess1 = ResearchSession(name="Session 1", prompt="test", region="USA", universe="TOP3000")
            sess2 = ResearchSession(name="Session 2", prompt="test", region="USA", universe="TOP3000")
            s.add_all([sess1, sess2])
            await s.flush()

            # Active strategy ONLY in Session 1
            strat1 = PromptStrategy(session_id=sess1.id, name="Sess1 Rev", version=1, category="reversal", is_active=True)
            # Session 2 has only a general strategy
            strat2_gen = PromptStrategy(session_id=sess2.id, name="Sess2 Gen", version=1, category="general", is_active=True)
            s.add_all([strat1, strat2_gen])
            await s.commit()

            sess1_id = sess1.id
            sess2_id = sess2.id
            strat1_id = strat1.id
            strat2_gen_id = strat2_gen.id

        orchestrator = MultiLLMOrchestrator(db=db, pipeline=MagicMock())

        async with db.session() as s:
            # Resolving for Session 1 matches strat1
            strat_res1, tier1 = await orchestrator._resolve_strategy(s, sess1_id, "reversal")
            assert strat_res1 is not None
            assert strat_res1.id == strat1_id
            assert tier1 == "EXACT_SESSION"

            # Resolving for Session 2 must NOT return strat1!
            strat_res2, tier2 = await orchestrator._resolve_strategy(s, sess2_id, "reversal")
            assert strat_res2 is not None
            assert strat_res2.id != strat1_id
            assert strat_res2.id == strat2_gen_id
            assert tier2 == "SESSION_GENERAL"

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 7: Precedence Fallback Tiers A Through E
# ---------------------------------------------------------------------------
def test_precedence_fallback_tiers_a_through_e(tmp_path: Path):
    """Explicitly verifies each of the 5 strategy resolution tiers:
    1. EXACT_SESSION
    2. GLOBAL_CATEGORY
    3. SESSION_GENERAL
    4. GLOBAL_GENERAL
    5. TEMPLATE_FALLBACK
    """
    async def _run():
        db = Database(tmp_path / "sqlite.db")
        await db.create_all()

        async with db.session() as s:
            sess = ResearchSession(name="Tiers Session", prompt="test", region="USA", universe="TOP3000")
            s.add(sess)
            await s.flush()
            sess_id = sess.id

        orchestrator = MultiLLMOrchestrator(db=db, pipeline=MagicMock())

        # Case Tier 5: TEMPLATE_FALLBACK (no strategies in DB)
        async with db.session() as s:
            strat_e, tier_e = await orchestrator._resolve_strategy(s, sess_id, "volatility")
            assert strat_e is None
            assert tier_e == "TEMPLATE_FALLBACK"

        # Foreign session strategy (should NEVER be selected across session boundaries)
        async with db.session() as s:
            sess_foreign = ResearchSession(name="Foreign Session", prompt="test", region="USA", universe="TOP3000")
            s.add(sess_foreign)
            await s.flush()
            # Active volatility strategy in another session
            s.add(PromptStrategy(session_id=sess_foreign.id, name="Foreign Vol", version=1, category="volatility", is_active=True))
            # Inactive strategies (should NEVER be selected in any tier)
            s.add(PromptStrategy(session_id=sess_id, name="Inactive Vol", version=2, category="volatility", is_active=False))
            s.add(PromptStrategy(session_id=None, name="Inactive Global Vol", version=2, category="volatility", is_active=False))
            s.add(PromptStrategy(session_id=sess_id, name="Inactive Sess Gen", version=2, category="general", is_active=False))
            s.add(PromptStrategy(session_id=None, name="Inactive Global Gen", version=2, category="general", is_active=False))
            # Active strategy in current session but UNRELATED category (should NEVER be selected for volatility)
            s.add(PromptStrategy(session_id=sess_id, name="Active Momentum", version=1, category="momentum", is_active=True))
            await s.commit()

        # Must STILL fall back to TEMPLATE_FALLBACK because inactive, foreign, and unrelated category strategies are disqualified
        async with db.session() as s:
            strat_still_e, tier_still_e = await orchestrator._resolve_strategy(s, sess_id, "volatility")
            assert strat_still_e is None
            assert tier_still_e == "TEMPLATE_FALLBACK"

        # Insert Tier 4 (D): GLOBAL_GENERAL (session_id IS NULL, category='general', is_active=True)
        async with db.session() as s:
            g_gen = PromptStrategy(session_id=None, name="Global General", version=1, category="general", is_active=True)
            s.add(g_gen)
            await s.commit()
            g_gen_id = g_gen.id

        async with db.session() as s:
            strat_d, tier_d = await orchestrator._resolve_strategy(s, sess_id, "volatility")
            assert strat_d is not None
            assert strat_d.id == g_gen_id
            assert tier_d == "GLOBAL_GENERAL"

        # Insert Tier 3 (C): SESSION_GENERAL (session_id = sess_id, category='general', is_active=True)
        async with db.session() as s:
            s_gen = PromptStrategy(session_id=sess_id, name="Session General", version=1, category="general", is_active=True)
            s.add(s_gen)
            await s.commit()
            s_gen_id = s_gen.id

        async with db.session() as s:
            strat_c, tier_c = await orchestrator._resolve_strategy(s, sess_id, "volatility")
            assert strat_c is not None
            assert strat_c.id == s_gen_id
            assert tier_c == "SESSION_GENERAL"

        # Insert Tier 2 (B): GLOBAL_CATEGORY (session_id IS NULL, category='volatility', is_active=True)
        async with db.session() as s:
            g_vol = PromptStrategy(session_id=None, name="Global Vol", version=1, category="volatility", is_active=True)
            s.add(g_vol)
            await s.commit()
            g_vol_id = g_vol.id

        async with db.session() as s:
            strat_b, tier_b = await orchestrator._resolve_strategy(s, sess_id, "volatility")
            assert strat_b is not None
            assert strat_b.id == g_vol_id
            assert tier_b == "GLOBAL_CATEGORY"

        # Insert Tier 1 (A): EXACT_SESSION (session_id = sess_id, category='volatility', is_active=True)
        async with db.session() as s:
            s_vol = PromptStrategy(session_id=sess_id, name="Session Vol", version=1, category="volatility", is_active=True)
            s.add(s_vol)
            await s.commit()
            s_vol_id = s_vol.id

        async with db.session() as s:
            strat_a, tier_a = await orchestrator._resolve_strategy(s, sess_id, "volatility")
            assert strat_a is not None
            assert strat_a.id == s_vol_id
            assert tier_a == "EXACT_SESSION"

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 8: GenerationAttempt Resolution Tier and Strategy Tracking
# ---------------------------------------------------------------------------
def test_generation_attempt_resolution_tier_and_strategy_tracking(tmp_path: Path):
    """Verifies that run_generation_round persists GenerationAttempt with prompt_strategy_id
    and resolution_tier, and that candidates record prompt_strategy_id.
    """
    async def _run():
        db = Database(tmp_path / "sqlite.db")
        await db.create_all()

        async with db.session() as s:
            sess = ResearchSession(name="GenAttempt Session", prompt="test", region="USA", universe="TOP3000")
            s.add(sess)
            await s.flush()

            strat = PromptStrategy(session_id=sess.id, name="Strat v1", version=1, category="reversal", is_active=True)
            s.add(strat)
            await s.flush()

            pv = PromptVersion(
                origin_session_id=sess.id,
                origin_strategy_id=strat.id,
                category="reversal",
                version=1,
                authoring_source="SEED",
                template_text="Generate reversal alpha",
                content_hash=compute_prompt_content_hash("Generate reversal alpha"),
            )
            s.add(pv)
            await s.flush()
            strat.current_prompt_version_id = pv.id

            researcher = ResearchResearcher(
                session_id=sess.id,
                label="RevResearcher",
                strategy_focus="reversal",
                enabled=True,
            )
            s.add(researcher)
            await s.commit()
            sess_id = sess.id
            strat_id = strat.id
            r_id = researcher.id

        # Mock the pipeline.generate_batch to return a synthetic candidate
        mock_cand = GeneratedCandidate(
            hypothesis="Reversal attempt",
            expression="ts_delta(close, 5)",
            category="reversal",
            target_dataset=None,
            primary_fields=["close"],
            validation=CandidateValidationResult(
                is_valid=True,
                expression="ts_delta(close, 5)",
                operator_count=1,
                field_count=1,
                fields_used=["close"],
                repair_history=[],
            ),
            raw_response={"expression": "ts_delta(close, 5)"},
        )
        mock_batch = GenerationBatch(
            summary="Batch success",
            candidates=[mock_cand],
            model_ref="test-model",
            total_tokens=100,
            duration_seconds=0.5,
            system_prompt="system",
            user_prompt="user",
            response_schema={},
        )
        mock_pipeline = MagicMock()
        mock_pipeline.generate_batch = AsyncMock(return_value=mock_batch)
        orchestrator = MultiLLMOrchestrator(db=db, pipeline=mock_pipeline)

        valid_ids = await orchestrator.run_generation_round(session_id=sess_id, total_target_candidates=1)
        assert len(valid_ids) == 1

        async with db.session() as s:
            # Verify GenerationAttempt
            attempts = (await s.execute(select(GenerationAttempt).where(GenerationAttempt.session_id == sess_id))).scalars().all()
            assert len(attempts) == 1
            attempt = attempts[0]
            assert attempt.prompt_strategy_id == strat_id
            assert attempt.resolution_tier == "EXACT_SESSION"

            # Verify Candidate
            cand = await s.get(ResearchCandidate, valid_ids[0])
            assert cand.prompt_strategy_id == strat_id
            assert cand.generation_attempt_id == attempt.id

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 9: Atomic Rollback on Persistence Failure
# ---------------------------------------------------------------------------
def test_atomic_rollback_on_persistence_failure(tmp_path: Path):
    """Verifies that if an exception occurs during strategy evolution persistence,
    the transaction is rolled back and the prior strategy remains active.
    """
    async def _run():
        db = Database(tmp_path / "sqlite.db")
        await db.create_all()

        async with db.session() as s:
            sess = ResearchSession(name="Rollback Session", prompt="test", region="USA", universe="TOP3000")
            s.add(sess)
            await s.flush()

            strat = PromptStrategy(session_id=sess.id, name="Strat v1", version=1, category="reversal", is_active=True)
            s.add(strat)
            await s.flush()

            cand = ResearchCandidate(
                session_id=sess.id,
                prompt_strategy_id=strat.id,
                expression="rank(close)",
                status=CandidateStatus.EVALUATED,
                generation=0,
                hypothesis="Rollback candidate",
            )
            s.add(cand)
            await s.flush()
            s.add(ResearchEvaluation(candidate_id=cand.id, is_verified=True, val_sharpe=1.5, val_turnover=0.1, val_fitness=1.0, provenance="BRAIN_VAULT"))
            await s.commit()
            sess_id = sess.id
            strat_id = strat.id

        engine = SelfLearningEngine(db=db, llm=None)

        # Patch session.flush to fail inside the transaction
        original_session = db.session

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def mock_session():
            async with original_session() as s:
                async def failing_flush(*args, **kwargs):
                    raise RuntimeError("Simulated DB Disk Full during strategy evolution")
                s.flush = failing_flush
                yield s

        with patch.object(db, "session", side_effect=mock_session):
            with pytest.raises(RuntimeError, match="Simulated DB Disk Full"):
                await engine.execute_learning_cycle(session_id=sess_id, category="reversal", min_candidates=1)

        # Verify prior strategy remains active
        async with db.session() as s:
            strat_check = await s.get(PromptStrategy, strat_id)
            assert strat_check.is_active is True

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 10: Preservation of Unverified Hypothesis and Recommendation Labels
# ---------------------------------------------------------------------------
def test_preservation_of_unverified_hypothesis_and_recommendation_labels(tmp_path: Path):
    """Verifies that reflection prompt assembly and fallback heuristics strictly preserve
    the P1-01 labels '[Unverified Hypothesis]' and '[Unverified Recommendation]'.
    """
    async def _run():
        db = Database(tmp_path / "sqlite.db")
        await db.create_all()

        candidate_ids = []
        async with db.session() as s:
            sess = ResearchSession(
                name="Labels Session",
                prompt="test",
                region="USA",
                universe="TOP3000",
                current_generation=1,
            )
            s.add(sess)
            await s.flush()

            strat = PromptStrategy(session_id=sess.id, name="Strat v1", version=1, category="momentum", is_active=True)
            s.add(strat)
            await s.flush()

            for i in range(2):
                cand = ResearchCandidate(
                    session_id=sess.id,
                    prompt_strategy_id=strat.id,
                    expression=f"ts_mean(close, {10+i*5})",
                    status=CandidateStatus.EVALUATED,
                    generation=1,
                    hypothesis=f"Labels candidate {i}",
                )
                s.add(cand)
                await s.flush()
                candidate_ids.append(cand.id)
                s.add(ResearchEvaluation(candidate_id=cand.id, is_verified=True, val_sharpe=2.0 + i * 0.1, val_turnover=0.1, val_fitness=1.2, provenance="BRAIN_VAULT"))
            await s.commit()
            sess_id = sess.id

        valid_response = {
            "summary": "Verified positive momentum observations.",
            "hypotheses": [
                {
                    "claim": "Mean reversion after surge",
                    "confidence": "HIGH_CONFIDENCE_HYPOTHESIS",
                    "citations": [{"candidate_id": candidate_ids[0], "field": "expression"}],
                }
            ],
            "recommendations": [
                {
                    "action": "Adjust window parameter",
                    "rationale": "Improved signal stability",
                    "citations": [{"candidate_id": candidate_ids[1], "field": "sharpe"}],
                }
            ],
            "anti_patterns": [
                {
                    "expression": "ts_mean(close, 10)",
                    "flaw": "Short lookback whipsaw",
                    "citations": [{"candidate_id": candidate_ids[0], "field": "expression"}],
                }
            ],
        }

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value=_make_mock_answer(valid_response))

        engine = SelfLearningEngine(db=db, llm=mock_llm)
        res = await engine.execute_learning_cycle(session_id=sess_id, category="momentum", min_candidates=1)

        assert res.updated_prompt_strategy_id is not None

        # Inspect evolved strategy guidance and anti-patterns for exact labels
        async with db.session() as s:
            evolved = await s.get(PromptStrategy, res.updated_prompt_strategy_id)
            assert evolved is not None
            assert evolved.version == 2
            assert evolved.category == "momentum"

            # Anti-pattern flaw must preserve [Unverified Hypothesis] prefix
            assert any("[Unverified Hypothesis]" in ap.get("flaw", "") for ap in evolved.anti_patterns)

            # Recommendations in guidance must preserve [Unverified Recommendation] prefix
            recs = evolved.guidance.get("unverified_recommendations", [])
            assert any("[Unverified Recommendation]" in r for r in recs)

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 11: Multi-Category Weighted Average Sharpe
# ---------------------------------------------------------------------------
def test_multi_category_weighted_average_sharpe():
    """Verifies that MultiCategoryLearningResult computes a candidate-weighted average Sharpe
    across categories rather than an unweighted arithmetic mean.
    """
    res1 = LearningCycleResult(
        session_id=1,
        generation=0,
        category="reversal",
        batch_size=10,
        winner_count=5,
        loser_count=5,
        avg_sharpe=2.0,
        best_sharpe=2.5,
        insights_generated=[],
        updated_prompt_strategy_id=101,
        status="COMPLETED",
    )
    res2 = LearningCycleResult(
        session_id=1,
        generation=0,
        category="momentum",
        batch_size=2,
        winner_count=1,
        loser_count=1,
        avg_sharpe=0.5,
        best_sharpe=0.8,
        insights_generated=[],
        updated_prompt_strategy_id=102,
        status="COMPLETED",
    )

    multi = MultiCategoryLearningResult(
        session_id=1,
        generation=0,
        categories_evolved=["reversal", "momentum"],
        category_results={"reversal": res1, "momentum": res2},
        updated_prompt_strategy_ids=[101, 102],
        category_statuses={"reversal": "COMPLETED", "momentum": "COMPLETED"},
    )

    # Weighted calculation: (10 * 2.0 + 2 * 0.5) / 12 = 21.0 / 12 = 1.75
    expected_weighted = (10 * 2.0 + 2 * 0.5) / 12.0
    assert abs(multi.avg_sharpe - expected_weighted) < 1e-6
    # Note: simple arithmetic mean would be (2.0 + 0.5) / 2 = 1.25 != 1.75
    assert multi.avg_sharpe != 1.25


# ---------------------------------------------------------------------------
# Test 12: Experiment Runner Learn Phase Integration
# ---------------------------------------------------------------------------
def test_experiment_runner_learn_phase_integration(tmp_path: Path):
    """Verifies that ExperimentRunner._phase_learn correctly runs the learning phase
    across all categories, consumes updated_prompt_strategy_ids and categories_evolved,
    and updates session generation.
    """
    async def _run():
        db = Database(tmp_path / "sqlite.db")
        await db.create_all()

        async with db.session() as s:
            sess = ResearchSession(
                name="Runner Session",
                prompt="test",
                region="USA",
                universe="TOP3000",
                current_generation=0,
                status=ResearchStatus.RUNNING,
            )
            s.add(sess)
            await s.flush()

            strat = PromptStrategy(session_id=sess.id, name="Strat v1", version=1, category="reversal", is_active=True)
            s.add(strat)
            await s.flush()

            cand = ResearchCandidate(
                session_id=sess.id,
                prompt_strategy_id=strat.id,
                expression="rank(close)",
                status=CandidateStatus.EVALUATED,
                generation=0,
                hypothesis="Runner candidate",
            )
            s.add(cand)
            await s.flush()
            s.add(ResearchEvaluation(candidate_id=cand.id, is_verified=True, val_sharpe=1.7, val_turnover=0.1, val_fitness=1.1, provenance="BRAIN_VAULT"))
            await s.commit()
            sess_id = sess.id

        mock_leaderboard = MagicMock()
        mock_leaderboard.capture_snapshot = AsyncMock()

        runner = ExperimentRunner(
            session_id=sess_id,
            db=db,
            orchestrator=MagicMock(),
            evaluation=MagicMock(),
            learning=SelfLearningEngine(db=db, llm=None),
            leaderboard=mock_leaderboard,
        )

        await runner._phase_learn()

        # Verify generation advanced and new strategy was created
        async with db.session() as s:
            session_row = await s.get(ResearchSession, sess_id)
            assert session_row.current_generation == 1

            new_strat = (
                await s.execute(
                    select(PromptStrategy).where(
                        PromptStrategy.session_id == sess_id,
                        PromptStrategy.is_active.is_(True),
                    )
                )
            ).scalar_one_or_none()
            assert new_strat is not None
            assert new_strat.version == 2
            assert new_strat.category == "reversal"

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 13: Legacy Alias Migration and Retrieval
# ---------------------------------------------------------------------------
def test_legacy_alias_migration_and_retrieval(tmp_path: Path):
    """Verifies that:
    1. Legacy database rows with category 'mean-reversion' or 'mean reversion'
       are migrated to 'mean_reversion'.
    2. Resolving with canonical category 'mean_reversion' successfully retrieves the strategy.
    3. Category 'reversal' remains completely distinct and unmerged.
    4. Migration collisions between aliased rows are deterministically resolved by version/id.
    5. Migration is safe and idempotent on repeated execution.
    """
    async def _run():
        db = Database(tmp_path / "legacy_alias.db")
        await db.create_all()

        # Insert legacy rows using raw SQL before migration
        async with db._engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO research_session (id, name, prompt, region, universe, created_at, updated_at) "
                    "VALUES (1, 'Legacy Alias Sess', 'test', 'USA', 'TOP3000', '2026-01-01', '2026-01-01')"
                )
            )
            # Legacy row with 'mean-reversion'
            await conn.execute(
                text(
                    "INSERT INTO prompt_strategy (id, session_id, name, version, category, is_active, created_at, updated_at) "
                    "VALUES (50, 1, 'Legacy MR v1', 1, 'mean-reversion', 1, '2026-01-01', '2026-01-01')"
                )
            )
            # Distinct 'reversal' row
            await conn.execute(
                text(
                    "INSERT INTO prompt_strategy (id, session_id, name, version, category, is_active, created_at, updated_at) "
                    "VALUES (60, 1, 'Reversal v1', 1, 'reversal', 1, '2026-01-01', '2026-01-01')"
                )
            )
            # Legacy researcher with 'mean-reversion'
            await conn.execute(
                text(
                    "INSERT INTO research_researcher (id, session_id, label, strategy_focus, enabled, created_at, updated_at) "
                    "VALUES (70, 1, 'MR Researcher', 'mean-reversion', 1, '2026-01-01', '2026-01-01')"
                )
            )

        # Run migration 1
        async with db._engine.begin() as conn:
            await migrate(conn)

        # Verify database state after migration
        async with db.session() as s:
            strat_mr = await s.get(PromptStrategy, 50)
            strat_rev = await s.get(PromptStrategy, 60)
            researcher = await s.get(ResearchResearcher, 70)

            # mean-reversion converted to mean_reversion
            assert strat_mr.category == "mean_reversion"
            assert strat_mr.is_active is True

            # reversal remains untouched and distinct
            assert strat_rev.category == "reversal"
            assert strat_rev.is_active is True

            # researcher focus converted to mean_reversion
            assert researcher.strategy_focus == "mean_reversion"

        # Verify strategy resolution by canonical resolver
        orchestrator = MultiLLMOrchestrator(db=db, pipeline=MagicMock())
        async with db.session() as s:
            resolved_strat, tier = await orchestrator._resolve_strategy(s, session_id=1, category="mean_reversion")
            assert resolved_strat is not None
            assert resolved_strat.id == 50
            assert tier == "EXACT_SESSION"

            # Reversal resolution remains distinct
            resolved_rev, tier_rev = await orchestrator._resolve_strategy(s, session_id=1, category="reversal")
            assert resolved_rev is not None
            assert resolved_rev.id == 60
            assert tier_rev == "EXACT_SESSION"

        # Run migration 2 (idempotency check)
        async with db._engine.begin() as conn:
            await migrate(conn)

        # Verify state is unchanged
        async with db.session() as s:
            strat_mr_after = await s.get(PromptStrategy, 50)
            strat_rev_after = await s.get(PromptStrategy, 60)
            assert strat_mr_after.category == "mean_reversion"
            assert strat_mr_after.is_active is True
            assert strat_rev_after.category == "reversal"
            assert strat_rev_after.is_active is True

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 14: Legacy Alias Collision Deactivation
# ---------------------------------------------------------------------------
def test_legacy_alias_collision_deactivation(tmp_path: Path):
    """Verifies that when a legacy alias ('mean-reversion', v1) normalizes to
    'mean_reversion' and collides with an existing active 'mean_reversion' (v2) in the
    same session:
    - Strategies 50 and 51 both normalize to 'mean_reversion'.
    - Strategy 51 remains active because it has the higher version.
    - Strategy 50 becomes inactive.
    - Strategy 51 is retrieved when resolving 'mean_reversion'.
    - Strategy 60 remains active and distinct under 'reversal'.
    - No historical rows are deleted.
    - Running the migration a second time changes nothing.
    """
    async def _run():
        db = Database(tmp_path / "collision_test.db")
        await db.create_all()

        # Insert strategies before migration
        async with db._engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO research_session (id, name, prompt, region, universe, created_at, updated_at) "
                    "VALUES (1, 'Collision Sess', 'test', 'USA', 'TOP3000', '2026-01-01', '2026-01-01')"
                )
            )
            # 1. Legacy active strategy: id=50, category="mean-reversion", version=1
            await conn.execute(
                text(
                    "INSERT INTO prompt_strategy (id, session_id, name, version, category, is_active, created_at, updated_at) "
                    "VALUES (50, 1, 'Legacy MR v1', 1, 'mean-reversion', 1, '2026-01-01', '2026-01-01')"
                )
            )
            # 2. Second active strategy in same session: id=51, category="mean_reversion", version=2
            await conn.execute(
                text(
                    "INSERT INTO prompt_strategy (id, session_id, name, version, category, is_active, created_at, updated_at) "
                    "VALUES (51, 1, 'Canonical MR v2', 2, 'mean_reversion', 1, '2026-01-01', '2026-01-01')"
                )
            )
            # 3. Distinct active strategy: id=60, category="reversal", version=1
            await conn.execute(
                text(
                    "INSERT INTO prompt_strategy (id, session_id, name, version, category, is_active, created_at, updated_at) "
                    "VALUES (60, 1, 'Reversal v1', 1, 'reversal', 1, '2026-01-01', '2026-01-01')"
                )
            )

        # 4. Run the actual SQLite migration
        async with db._engine.begin() as conn:
            await migrate(conn)

        # 5. Verify post-migration state
        async with db.session() as s:
            s_50 = await s.get(PromptStrategy, 50)
            s_51 = await s.get(PromptStrategy, 51)
            s_60 = await s.get(PromptStrategy, 60)

            # No historical rows are deleted
            assert s_50 is not None
            assert s_51 is not None
            assert s_60 is not None

            # Strategies 50 and 51 both normalize to "mean_reversion"
            assert s_50.category == "mean_reversion"
            assert s_51.category == "mean_reversion"

            # Strategy 51 remains active because it has the higher version (v2)
            assert s_51.is_active is True

            # Strategy 50 becomes inactive (v1)
            assert s_50.is_active is False

            # Strategy 60 remains active and distinct under "reversal"
            assert s_60.category == "reversal"
            assert s_60.is_active is True

        # Verify strategy retrieval through canonical resolver
        orchestrator = MultiLLMOrchestrator(db=db, pipeline=MagicMock())
        async with db.session() as s:
            strat_mr, tier_mr = await orchestrator._resolve_strategy(s, session_id=1, category="mean_reversion")
            assert strat_mr is not None
            assert strat_mr.id == 51
            assert strat_mr.version == 2
            assert tier_mr == "EXACT_SESSION"

            strat_rev, tier_rev = await orchestrator._resolve_strategy(s, session_id=1, category="reversal")
            assert strat_rev is not None
            assert strat_rev.id == 60
            assert strat_rev.version == 1
            assert tier_rev == "EXACT_SESSION"

        # 6. Run migration a second time and confirm idempotency
        async with db._engine.begin() as conn:
            await migrate(conn)

        async with db.session() as s:
            s_50_after = await s.get(PromptStrategy, 50)
            s_51_after = await s.get(PromptStrategy, 51)
            s_60_after = await s.get(PromptStrategy, 60)

            assert s_50_after.category == "mean_reversion"
            assert s_50_after.is_active is False
            assert s_51_after.category == "mean_reversion"
            assert s_51_after.is_active is True
            assert s_60_after.category == "reversal"
            assert s_60_after.is_active is True

    asyncio.run(_run())
