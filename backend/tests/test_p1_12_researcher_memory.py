"""Tests for Issue P1-12: Researcher-specific persistent episodic memory.

Validates:
1. Persistent episodic memory survives DB/session restarts.
2. Strict private memory isolation: Researcher A cannot read Researcher B's private episodes.
3. Cross-researcher writes and deletes are rejected.
4. Private episodes never leak into another researcher's prompt context.
5. Shared knowledge layer is accessible when requested without leaking private episodes.
6. Missing or invalid researcher identity raises explicit error (no silent fallback or merging).
7. Structured filters: time, category, version, outcome, validated discoveries, failures.
8. Duplicate episodes are handled consistently (repeated executions are not fabricated as discoveries).
9. Holdout isolation: P0-03 holdout metrics are strictly forbidden and cannot leak into memory.
10. Batch candidate recording maps P1-10 multiplicity splits into episodic memory.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from alpha_harness.db.models import (
    CandidateStatus,
    PromptStrategy,
    PromptVersion,
    ResearchCandidate,
    ResearchEpisode,
    ResearchEvaluation,
    ResearchResearcher,
    ResearchSession,
    utcnow,
)
from alpha_harness.db.sqlite import Database
from alpha_harness.learning.memory import ResearchMemory


@pytest.fixture
def test_db(tmp_path: Path) -> Database:
    db_path = tmp_path / "test_p1_12_memory.db"
    db = Database(db_path)
    asyncio.run(db.create_all())
    return db


def test_researcher_memory_persistence_and_restart(tmp_path: Path):
    """Proves that a researcher can record an episode and retrieve it across DB restarts."""
    async def _run():
        db_file = tmp_path / "restart_test.db"
        db1 = Database(db_file)
        await db1.create_all()

        # 1. Create session and researcher in DB 1
        async with db1.session() as s:
            sess = ResearchSession(name="Session 1", region="USA", universe="TOP3000")
            s.add(sess)
            await s.flush()

            r1 = ResearchResearcher(
                session_id=sess.id,
                label="Momentum Researcher",
                strategy_focus="momentum",
                model_ref="test:model-1",
            )
            s.add(r1)
            await s.flush()
            sess_id = sess.id
            r1_id = r1.id

        mem1 = ResearchMemory(db1)
        episode = await mem1.record_research_episode(
            session_id=sess_id,
            researcher_id=r1_id,
            generation=1,
            strategy_focus="momentum",
            dataset_id="model_data_us",
            region="USA",
            universe="TOP3000",
            hypothesis="Cross-sectional momentum over 20-day horizon",
            rationale="Captures trend continuation in large caps",
            expression="ts_decay_linear(returns, 20)",
            funnel_stage="EVALUATED",
            outcome="STATISTICAL_WINNER",
            selection_tier="SIGNIFICANT_WINNER",
            validation_sharpe=1.85,
            validation_fitness=1.42,
            validation_turnover=0.15,
            t_years=4.0,
            sharpe_se=0.25,
            t_stat=7.4,
            p_value=0.0001,
            lcb95=1.36,
            is_statistically_significant=True,
            lessons_learned="20-day linear decay provides smooth trend signal without excess turnover.",
        )

        assert episode.id is not None
        ep_id = episode.id

        # Dispose db1 to simulate shutdown
        await db1.dispose()

        # 2. Re-open database with new Database instance
        db2 = Database(db_file)
        mem2 = ResearchMemory(db2)

        retrieved = await mem2.get_researcher_episodes(researcher_id=r1_id)
        assert len(retrieved) == 1
        ep = retrieved[0]
        assert ep.id == ep_id
        assert ep.researcher_id == r1_id
        assert ep.hypothesis == "Cross-sectional momentum over 20-day horizon"
        assert ep.expression == "ts_decay_linear(returns, 20)"
        assert ep.outcome == "STATISTICAL_WINNER"
        assert ep.validation_sharpe == 1.85
        assert ep.t_stat == 7.4
        assert ep.is_statistically_significant is True

        await db2.dispose()

    asyncio.run(_run())


def test_strict_private_memory_isolation(test_db: Database):
    """Proves that Researcher A cannot read Researcher B's private episodes."""
    async def _run():
        async with test_db.session() as s:
            sess = ResearchSession(name="Session 1", region="USA", universe="TOP3000")
            s.add(sess)
            await s.flush()

            ra = ResearchResearcher(session_id=sess.id, label="Researcher A", strategy_focus="momentum")
            rb = ResearchResearcher(session_id=sess.id, label="Researcher B", strategy_focus="reversal")
            s.add_all([ra, rb])
            await s.flush()
            sess_id = sess.id
            ra_id = ra.id
            rb_id = rb.id

        mem = ResearchMemory(test_db)

        # Researcher A records a private episode
        await mem.record_research_episode(
            session_id=sess_id,
            researcher_id=ra_id,
            generation=1,
            strategy_focus="momentum",
            hypothesis="Secret momentum alpha for Researcher A",
            expression="rank(ts_mean(close, 10))",
            outcome="STATISTICAL_WINNER",
            validation_sharpe=2.1,
            is_shared_knowledge=False,  # Private
        )

        # Researcher B records a private episode
        await mem.record_research_episode(
            session_id=sess_id,
            researcher_id=rb_id,
            generation=1,
            strategy_focus="reversal",
            hypothesis="Confidential reversal alpha for Researcher B",
            expression="rank(-ts_delta(close, 3))",
            outcome="STATISTICAL_LOSER",
            validation_sharpe=-0.4,
            is_shared_knowledge=False,  # Private
        )

        # Researcher A queries: MUST see only Researcher A's episode
        eps_a = await mem.get_researcher_episodes(researcher_id=ra_id, include_shared=False)
        assert len(eps_a) == 1
        assert eps_a[0].researcher_id == ra_id
        assert "Secret momentum alpha" in eps_a[0].hypothesis
        assert "Confidential reversal alpha" not in str([e.hypothesis for e in eps_a])

        # Researcher B queries: MUST see only Researcher B's episode
        eps_b = await mem.get_researcher_episodes(researcher_id=rb_id, include_shared=False)
        assert len(eps_b) == 1
        assert eps_b[0].researcher_id == rb_id
        assert "Confidential reversal alpha" in eps_b[0].hypothesis
        assert "Secret momentum alpha" not in str([e.hypothesis for e in eps_b])

        # Even with include_shared=True, Researcher A's private episode is NOT visible to Researcher B
        eps_b_with_shared = await mem.get_researcher_episodes(researcher_id=rb_id, include_shared=True)
        assert len(eps_b_with_shared) == 1
        assert eps_b_with_shared[0].researcher_id == rb_id

    asyncio.run(_run())


def test_cross_researcher_writes_and_deletes_rejected(test_db: Database):
    """Proves that cross-researcher updates and deletes are rejected without corrupting data."""
    async def _run():
        async with test_db.session() as s:
            sess = ResearchSession(name="Session 1")
            s.add(sess)
            await s.flush()
            ra = ResearchResearcher(session_id=sess.id, label="Researcher A")
            rb = ResearchResearcher(session_id=sess.id, label="Researcher B")
            s.add_all([ra, rb])
            await s.flush()
            sess_id = sess.id
            ra_id = ra.id
            rb_id = rb.id

        mem = ResearchMemory(test_db)

        # Researcher A records an episode
        ep_a = await mem.record_research_episode(
            session_id=sess_id,
            researcher_id=ra_id,
            hypothesis="Original Hypothesis",
            expression="close / ts_mean(close, 20)",
            lessons_learned="Initial observation",
        )
        ep_id = ep_a.id

        # Researcher B attempts to update Researcher A's episode: MUST return False (rejected)
        update_success = await mem.update_researcher_episode(
            researcher_id=rb_id,
            episode_id=ep_id,
            lessons_learned="Hacked by Researcher B!",
        )
        assert update_success is False

        # Verify episode was NOT modified
        ep_check = await mem.get_researcher_episodes(researcher_id=ra_id)
        assert ep_check[0].lessons_learned == "Initial observation"

        # Researcher B attempts to delete Researcher A's episode: MUST return False (rejected)
        delete_success = await mem.delete_researcher_episode(
            researcher_id=rb_id,
            episode_id=ep_id,
        )
        assert delete_success is False

        # Verify episode still exists
        ep_check_after_del = await mem.get_researcher_episodes(researcher_id=ra_id)
        assert len(ep_check_after_del) == 1
        assert ep_check_after_del[0].id == ep_id

        # Researcher A deletes their own episode: MUST succeed
        own_delete = await mem.delete_researcher_episode(
            researcher_id=ra_id,
            episode_id=ep_id,
        )
        assert own_delete is True
        assert len(await mem.get_researcher_episodes(researcher_id=ra_id)) == 0

    asyncio.run(_run())


def test_private_episodes_never_appear_in_peer_prompt_context(test_db: Database):
    """Proves that private episodes never leak into another researcher's prompt context."""
    async def _run():
        async with test_db.session() as s:
            sess = ResearchSession(name="Session 1")
            s.add(sess)
            await s.flush()
            ra = ResearchResearcher(session_id=sess.id, label="Researcher A")
            rb = ResearchResearcher(session_id=sess.id, label="Researcher B")
            s.add_all([ra, rb])
            await s.flush()
            sess_id = sess.id
            ra_id = ra.id
            rb_id = rb.id

        mem = ResearchMemory(test_db)

        # Researcher A records a validated discovery and a failure
        await mem.record_research_episode(
            session_id=sess_id,
            researcher_id=ra_id,
            generation=1,
            hypothesis="Super Secret Trend Alpha",
            expression="ts_regression(close, 30)",
            outcome="STATISTICAL_WINNER",
            is_statistically_significant=True,
            validation_sharpe=2.45,
            t_stat=8.1,
            is_shared_knowledge=False,
        )
        await mem.record_research_episode(
            session_id=sess_id,
            researcher_id=ra_id,
            generation=1,
            hypothesis="Secret Failed Volatility Hypothesis",
            expression="stddev(returns, 5)",
            outcome="STATISTICAL_LOSER",
            failure_category="HIGH_TURNOVER",
            lessons_learned="5-day window generates unviable turnover.",
            is_shared_knowledge=False,
        )

        # Researcher A builds memory context: MUST contain the secrets
        ctx_a = await mem.build_researcher_memory_context(researcher_id=ra_id)
        assert "Super Secret Trend Alpha" in ctx_a
        assert "ts_regression(close, 30)" in ctx_a
        assert "HIGH_TURNOVER" in ctx_a

        # Researcher B builds memory context: MUST NOT contain any of Researcher A's private secrets
        ctx_b = await mem.build_researcher_memory_context(researcher_id=rb_id)
        assert "Super Secret Trend Alpha" not in ctx_b
        assert "ts_regression(close, 30)" not in ctx_b
        assert "Secret Failed Volatility Hypothesis" not in ctx_b
        assert "stddev(returns, 5)" not in ctx_b
        assert ctx_b == ""  # Empty because Researcher B has zero episodes

    asyncio.run(_run())


def test_shared_knowledge_layer_sharing_and_promotion(test_db: Database):
    """Proves that explicitly shared knowledge is available to peer researchers while unshared remains private."""
    async def _run():
        async with test_db.session() as s:
            sess = ResearchSession(name="Session 1")
            s.add(sess)
            await s.flush()
            ra = ResearchResearcher(session_id=sess.id, label="Researcher A")
            rb = ResearchResearcher(session_id=sess.id, label="Researcher B")
            s.add_all([ra, rb])
            await s.flush()
            sess_id = sess.id
            ra_id = ra.id
            rb_id = rb.id

        mem = ResearchMemory(test_db)

        # Researcher A records a discovery initially private
        ep_a = await mem.record_research_episode(
            session_id=sess_id,
            researcher_id=ra_id,
            generation=1,
            hypothesis="Shared Operator Fact: TS Rank Stability",
            expression="ts_rank(volume, 60)",
            outcome="STATISTICAL_WINNER",
            is_statistically_significant=True,
            validation_sharpe=1.9,
            t_stat=6.2,
            is_shared_knowledge=False,
        )

        # Researcher B with include_shared=True cannot see it yet
        eps_b_pre = await mem.get_researcher_episodes(researcher_id=rb_id, include_shared=True)
        assert len(eps_b_pre) == 0

        # Researcher B attempts to promote Researcher A's episode: MUST return False (cannot promote peer's episode)
        illegal_promote = await mem.promote_to_shared_knowledge(researcher_id=rb_id, episode_id=ep_a.id)
        assert illegal_promote is False

        # Researcher A promotes their own episode to shared knowledge
        promote_success = await mem.promote_to_shared_knowledge(researcher_id=ra_id, episode_id=ep_a.id)
        assert promote_success is True

        # Now Researcher B with include_shared=True can see the shared knowledge
        eps_b_post = await mem.get_researcher_episodes(researcher_id=rb_id, include_shared=True)
        assert len(eps_b_post) == 1
        assert eps_b_post[0].expression == "ts_rank(volume, 60)"
        assert eps_b_post[0].is_shared_knowledge is True

        # But Researcher B with include_shared=False cannot see it (strictly private view)
        eps_b_private_only = await mem.get_researcher_episodes(researcher_id=rb_id, include_shared=False)
        assert len(eps_b_private_only) == 0

    asyncio.run(_run())


def test_missing_or_invalid_researcher_identity_handling(test_db: Database):
    """Proves that missing or invalid researcher identity raises explicit ValueError."""
    async def _run():
        mem = ResearchMemory(test_db)

        # 1. Recording with invalid researcher_id
        with pytest.raises(ValueError, match="Valid positive researcher_id is required"):
            await mem.record_research_episode(session_id=1, researcher_id=0, expression="close")

        with pytest.raises(ValueError, match="Valid positive researcher_id is required"):
            await mem.record_research_episode(session_id=1, researcher_id=-5, expression="close")

        # 2. Retrieval with invalid researcher_id
        with pytest.raises(ValueError, match="Valid positive researcher_id is required"):
            await mem.get_researcher_episodes(researcher_id=0)

        # 3. Update with invalid researcher_id
        with pytest.raises(ValueError, match="Valid positive researcher_id is required"):
            await mem.update_researcher_episode(researcher_id=0, episode_id=1)

        # 4. Delete with invalid researcher_id
        with pytest.raises(ValueError, match="Valid positive researcher_id is required"):
            await mem.delete_researcher_episode(researcher_id=-1, episode_id=1)

    asyncio.run(_run())


def test_structured_retrieval_filters(test_db: Database):
    """Proves that deterministic filters narrow down episodic memory accurately."""
    async def _run():
        async with test_db.session() as s:
            sess = ResearchSession(name="Session 1", region="USA", universe="TOP3000")
            s.add(sess)
            await s.flush()
            r = ResearchResearcher(session_id=sess.id, label="Lead Quant", strategy_focus="momentum")
            s.add(r)
            await s.flush()
            sess_id = sess.id
            r_id = r.id

        mem = ResearchMemory(test_db)

        # Record 4 different episodes with varying metadata
        # Episode 1: Momentum, USA, Gen 1, Winner
        await mem.record_research_episode(
            session_id=sess_id,
            researcher_id=r_id,
            generation=1,
            strategy_focus="momentum",
            dataset_id="ds_usa",
            region="USA",
            universe="TOP3000",
            hypothesis_family="trend_following",
            expression="ts_mean(close, 10)",
            outcome="STATISTICAL_WINNER",
            is_statistically_significant=True,
        )
        # Episode 2: Mean Reversion, EUR, Gen 1, Loser
        await mem.record_research_episode(
            session_id=sess_id,
            researcher_id=r_id,
            generation=1,
            strategy_focus="mean_reversion",
            dataset_id="ds_eur",
            region="EUR",
            universe="TOP1000",
            hypothesis_family="oscillator",
            expression="rsi(close, 14)",
            outcome="STATISTICAL_LOSER",
            failure_category="NEGATIVE_SHARPE",
        )
        # Episode 3: Momentum, USA, Gen 2, Winner
        await mem.record_research_episode(
            session_id=sess_id,
            researcher_id=r_id,
            generation=2,
            strategy_focus="momentum",
            dataset_id="ds_usa",
            region="USA",
            universe="TOP3000",
            hypothesis_family="trend_following",
            expression="ts_mean(close, 20)",
            outcome="STATISTICAL_WINNER",
            is_statistically_significant=True,
        )
        # Episode 4: Momentum, USA, Gen 2, Syntax Error
        await mem.record_research_episode(
            session_id=sess_id,
            researcher_id=r_id,
            generation=2,
            strategy_focus="momentum",
            dataset_id="ds_usa",
            region="USA",
            universe="TOP3000",
            hypothesis_family="breakout",
            expression="broken_func(close))",
            outcome="SYNTAX_ERROR",
            failure_category="SYNTAX",
        )

        # 1. Filter by generation
        gen1_eps = await mem.get_researcher_episodes(researcher_id=r_id, min_generation=1, max_generation=1)
        assert len(gen1_eps) == 2
        assert {e.generation for e in gen1_eps} == {1}

        # 2. Filter by region
        eur_eps = await mem.get_researcher_episodes(researcher_id=r_id, region="EUR")
        assert len(eur_eps) == 1
        assert eur_eps[0].region == "EUR"

        # 3. Filter by hypothesis family
        trend_eps = await mem.get_researcher_episodes(researcher_id=r_id, hypothesis_family="trend_following")
        assert len(trend_eps) == 2
        assert all(e.hypothesis_family == "trend_following" for e in trend_eps)

        # 4. Filter only validated discoveries
        winners = await mem.get_researcher_episodes(researcher_id=r_id, only_validated_discoveries=True)
        assert len(winners) == 2
        assert all(e.outcome == "STATISTICAL_WINNER" for e in winners)

        # 5. Filter only failures
        failures = await mem.get_researcher_episodes(researcher_id=r_id, only_failures=True)
        assert len(failures) == 2
        assert {e.outcome for e in failures} == {"STATISTICAL_LOSER", "SYNTAX_ERROR"}

    asyncio.run(_run())


def test_duplicate_episode_handling(test_db: Database):
    """Proves that repeated executions of the same expression are tracked and deduplicated."""
    async def _run():
        async with test_db.session() as s:
            sess = ResearchSession(name="Session 1")
            s.add(sess)
            await s.flush()
            r = ResearchResearcher(session_id=sess.id, label="Lead Quant")
            s.add(r)
            await s.flush()
            sess_id = sess.id
            r_id = r.id

        mem = ResearchMemory(test_db)

        expr = "rank(ts_decay_linear(volume, 10))"

        # Run 1
        ep1 = await mem.record_research_episode(
            session_id=sess_id,
            researcher_id=r_id,
            generation=1,
            expression=expr,
            outcome="STATISTICAL_WINNER",
        )
        assert ep1.error_details is None or not ep1.error_details.get("is_repeated_execution")

        # Run 2 (exact same expression executed again)
        ep2 = await mem.record_research_episode(
            session_id=sess_id,
            researcher_id=r_id,
            generation=2,
            expression=expr,
            outcome="STATISTICAL_WINNER",
        )
        assert ep2.error_details is not None
        assert ep2.error_details.get("is_repeated_execution") is True
        assert ep2.error_details.get("prior_execution_count") == 1

        # Default retrieval deduplicates by expression: only 1 returned (the latest from gen 2)
        deduped = await mem.get_researcher_episodes(researcher_id=r_id, deduplicate_by_expression=True)
        assert len(deduped) == 1
        assert deduped[0].id == ep2.id
        assert deduped[0].generation == 2

        # Retrieval without deduplication returns both runs
        all_runs = await mem.get_researcher_episodes(researcher_id=r_id, deduplicate_by_expression=False)
        assert len(all_runs) == 2

    asyncio.run(_run())


def test_holdout_isolation_preservation(test_db: Database):
    """Proves that P0-03 holdout metrics are strictly forbidden from entering episodic memory or context."""
    async def _run():
        async with test_db.session() as s:
            sess = ResearchSession(name="Session 1")
            s.add(sess)
            await s.flush()
            r = ResearchResearcher(session_id=sess.id, label="Lead Quant")
            s.add(r)
            await s.flush()
            cand = ResearchCandidate(
                session_id=sess.id,
                researcher_id=r.id,
                expression="rank(close)",
                hypothesis="Test",
            )
            s.add(cand)
            await s.flush()

            # Create evaluation with non-leaked holdout metric
            ev = ResearchEvaluation(
                candidate_id=cand.id,
                is_verified=True,
                val_sharpe=1.5,
                holdout_sharpe=3.99,  # Genuine holdout metric (quarantined)
                holdout_returns=0.08,
            )
            s.add(ev)
            await s.flush()
            sess_id = sess.id
            r_id = r.id
            cand_id = cand.id
            ev_id = ev.id

        mem = ResearchMemory(test_db)

        # 1. Attempting to record with holdout_sharpe argument directly: MUST raise ValueError
        with pytest.raises(ValueError, match="Holdout metric 'holdout_sharpe' is strictly prohibited"):
            await mem.record_research_episode(
                session_id=sess_id,
                researcher_id=r_id,
                expression="rank(close)",
                holdout_sharpe=3.99,
            )

        # 2. Record legitimate episode using validation Sharpe only
        ep = await mem.record_research_episode(
            session_id=sess_id,
            researcher_id=r_id,
            candidate_id=cand_id,
            expression="rank(close)",
            validation_sharpe=1.5,
            outcome="STATISTICAL_WINNER",
            is_statistically_significant=True,
        )

        # 3. Verify that mutating holdout_sharpe in the DB does NOT change the episode or context
        async with test_db.session() as s:
            ev_row = await s.get(ResearchEvaluation, ev_id)
            ev_row.holdout_sharpe = 99.99  # Extreme change to holdout partition
            await s.flush()

        retrieved_ep = (await mem.get_researcher_episodes(researcher_id=r_id))[0]
        assert retrieved_ep.validation_sharpe == 1.5
        assert not hasattr(retrieved_ep, "holdout_sharpe")

        # 4. Context check: holdout numbers never appear
        ctx = await mem.build_researcher_memory_context(researcher_id=r_id)
        assert "1.50" in ctx
        assert "99.99" not in ctx
        assert "3.99" not in ctx
        assert "holdout" not in ctx.lower()

    asyncio.run(_run())


def test_batch_recording_from_candidates(test_db: Database):
    """Proves that batch recording correctly captures P1-10 multiplicity splits into episodic memory."""
    async def _run():
        async with test_db.session() as s:
            sess = ResearchSession(name="Session 1", region="USA", universe="TOP3000")
            s.add(sess)
            await s.flush()

            r_mom = ResearchResearcher(session_id=sess.id, label="Mom Researcher", strategy_focus="momentum")
            r_val = ResearchResearcher(session_id=sess.id, label="Val Researcher", strategy_focus="value")
            s.add_all([r_mom, r_val])
            await s.flush()

            c1 = ResearchCandidate(
                session_id=sess.id,
                researcher_id=r_mom.id,
                expression="ts_decay_linear(close, 20)",
                hypothesis="Momentum Alpha",
            )
            c2 = ResearchCandidate(
                session_id=sess.id,
                researcher_id=r_val.id,
                expression="book_to_price / close",
                hypothesis="Value Alpha",
            )
            s.add_all([c1, c2])
            await s.commit()

            sess_id = sess.id
            mom_id = r_mom.id
            val_id = r_val.id
            c1_id = c1.id
            c2_id = c2.id

        mem = ResearchMemory(test_db)

        # Simulated candidate evaluations
        candidates = [
            {
                "candidate_id": c1_id,
                "researcher_id": mom_id,
                "category": "momentum",
                "expression": "ts_decay_linear(close, 20)",
                "hypothesis": "Momentum Alpha",
                "sharpe": 2.1,
                "fitness": 1.5,
                "turnover": 0.2,
                "sharpe_decay": 0.05,
                "t_years": 4.0,
                "sharpe_se": 0.25,
                "t_vs_zero": 8.4,
                "p_value": 0.0001,
                "lcb95": 1.61,
                "z_score": 2.8,
                "fdr_q_value": 0.001,
            },
            {
                "candidate_id": c2_id,
                "researcher_id": val_id,
                "category": "value",
                "expression": "book_to_price / close",
                "hypothesis": "Value Alpha",
                "sharpe": -0.8,
                "fitness": -0.5,
                "turnover": 0.1,
                "sharpe_decay": -0.1,
                "t_years": 4.0,
                "sharpe_se": 0.28,
                "t_vs_zero": -2.8,
                "p_value": 0.99,
                "lcb95": -1.35,
                "z_score": -3.2,
                "fdr_q_value": 0.85,
            },
        ]

        # Mock P1-10 multiplicity split
        winners_split = [
            {
                "candidate_id": c1_id,
                "selection_tier": "SIGNIFICANT_WINNER",
                "sharpe_se": 0.25,
                "t_vs_zero": 8.4,
                "p_value": 0.0001,
                "lcb95": 1.61,
                "z_score": 2.8,
                "fdr_q_value": 0.001,
            }
        ]
        losers_split = [
            {
                "candidate_id": c2_id,
                "selection_tier": "SIGNIFICANT_LOSER",
                "sharpe_se": 0.28,
                "t_vs_zero": -2.8,
                "p_value": 0.99,
                "lcb95": -1.35,
                "z_score": -3.2,
                "fdr_q_value": 0.85,
            }
        ]

        recorded_episodes = await mem.record_batch_episodes_from_candidates(
            session_id=sess_id,
            generation=1,
            candidates_with_evals=candidates,
            selection_split=(winners_split, losers_split),
        )

        assert len(recorded_episodes) == 2

        # Check Momentum Researcher's episode
        mom_eps = await mem.get_researcher_episodes(researcher_id=mom_id)
        assert len(mom_eps) == 1
        assert mom_eps[0].outcome == "STATISTICAL_WINNER"
        assert mom_eps[0].selection_tier == "SIGNIFICANT_WINNER"
        assert mom_eps[0].is_statistically_significant is True
        assert mom_eps[0].validation_sharpe == 2.1

        # Check Value Researcher's episode
        val_eps = await mem.get_researcher_episodes(researcher_id=val_id)
        assert len(val_eps) == 1
        assert val_eps[0].outcome == "STATISTICAL_LOSER"
        assert val_eps[0].selection_tier == "SIGNIFICANT_LOSER"
        assert val_eps[0].is_statistically_significant is False
        assert val_eps[0].validation_sharpe == -0.8
        assert val_eps[0].failure_category == "STATISTICAL_UNDERPERFORMANCE"

    asyncio.run(_run())


def test_researcher_scoped_prompt_update_cannot_mutate_peer(test_db: Database):
    """P1-12 Acceptance Criterion:
    A second researcher cannot have its prompt or persona settings silently changed by a scoped update."""
    async def _run():
        from alpha_harness.orchestrator.researchers import MultiLLMOrchestrator
        from unittest.mock import MagicMock

        async with test_db.session() as s:
            sess = ResearchSession(name="Session 1")
            s.add(sess)
            await s.flush()

            ra = ResearchResearcher(
                session_id=sess.id,
                label="Researcher A",
                strategy_focus="momentum",
                system_prompt="You are a pure momentum researcher.",
                temperature=0.7,
            )
            rb = ResearchResearcher(
                session_id=sess.id,
                label="Researcher B",
                strategy_focus="value",
                system_prompt="You are a deep value researcher.",
                temperature=0.5,
            )
            s.add_all([ra, rb])
            await s.flush()
            ra_id = ra.id
            rb_id = rb.id

        orch = MultiLLMOrchestrator(db=test_db, pipeline=MagicMock())

        # Scoped update targeting Researcher A only
        updated = await orch.update_researcher(
            researcher_id=ra_id,
            system_prompt="You are an EVOLVED momentum expert focusing on 60-day breakouts.",
            temperature=0.85,
            strategy_focus="momentum",
        )
        assert updated is True

        # Verify Researcher A changed
        async with test_db.session() as s:
            ra_fresh = await s.get(ResearchResearcher, ra_id)
            assert ra_fresh.system_prompt == "You are an EVOLVED momentum expert focusing on 60-day breakouts."
            assert ra_fresh.temperature == 0.85

            # Verify Researcher B was NOT mutated or contaminated
            rb_fresh = await s.get(ResearchResearcher, rb_id)
            assert rb_fresh.system_prompt == "You are a deep value researcher."
            assert rb_fresh.temperature == 0.5
            assert rb_fresh.strategy_focus == "value"

    asyncio.run(_run())
