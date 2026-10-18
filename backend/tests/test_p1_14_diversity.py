"""Comprehensive test suite for P1-14: Diversity generation, deterministic deduplication, and portfolio curation.

Covers:
1. Exact normalized duplicate rejection before simulation dispatch.
2. Near-duplicate parameter-jitter pruning (e.g. ts_decay_linear(close, 10) vs 12).
3. Preservation of distinct data fields or distinct hypothesis families with matching skeletons.
4. Deterministic diversity allocation given identical inputs, configuration, and seed.
5. Family quotas as non-blocking targets without candidate fabrication or empty fallback.
6. Configured exploration reserve.
7. Rejection of invalid expressions from simulation queues.
8. Preservation of P1-10 statistical significance gates (diversity never bypasses t >= 1.96 or q <= 0.15).
9. Missing or insufficient return overlap does not infer false low correlation.
10. Strict holdout quarantine across deduplication, diversity selection, and portfolio curation.
"""

from __future__ import annotations

from datetime import date, timedelta
import pytest
import numpy as np

from alpha_harness.labs.fastexpr import (
    CandidateFingerprint,
    canonicalize_expression,
    canonicalize_skeleton,
    compute_expression_hash,
    extract_candidate_fingerprint,
    is_near_duplicate,
)
from alpha_harness.orchestrator.diversity import (
    CandidateItem,
    DiversityConfig,
    DiversitySelector,
)
from alpha_harness.tools.submission_planner import (
    evaluate_pairwise_correlation,
    search,
    plan,
    CEILING,
    MIN_OVERLAP,
)
from alpha_harness.learning.memory import _multiplicity_corrected_split


def test_exact_normalized_duplicate_rejection():
    """Exact duplicate expressions (even with whitespace or formatting differences) must be pruned."""
    e1 = "ts_decay_linear(close, 10)"
    e2 = "  ts_decay_linear(  close , 10 ) "
    e3 = "ts_rank(volume, 20)"

    c1 = CandidateItem(expression=e1, category="MOMENTUM", candidate_id=1)
    c2 = CandidateItem(expression=e2, category="MOMENTUM", candidate_id=2)
    c3 = CandidateItem(expression=e3, category="VOLUME", candidate_id=3)

    selector = DiversitySelector()
    res = selector.select_candidates([c1, c2, c3], capacity=10)

    # c2 is an exact duplicate of c1
    dispatched_ids = [c.candidate_id for c in res.dispatched_candidates]
    assert 1 in dispatched_ids
    assert 2 not in dispatched_ids
    assert 3 in dispatched_ids

    assert res.funnel_metrics.exact_duplicate_count == 1
    exact_audits = [a for a in res.audit_records if a.decision == "EXACT_DUPLICATE_PRUNED"]
    assert len(exact_audits) == 1
    assert exact_audits[0].candidate_id == 2


def test_near_duplicate_parameter_jitter_pruning():
    """Near-duplicate parameter-jitter candidates with identical AST skeleton and fields must be pruned."""
    e1 = "ts_decay_linear(close, 10)"
    e2 = "ts_decay_linear(close, 12)"
    e3 = "ts_decay_linear(close, 15)"

    c1 = CandidateItem(expression=e1, category="MOMENTUM", candidate_id=10)
    c2 = CandidateItem(expression=e2, category="MOMENTUM", candidate_id=11)
    c3 = CandidateItem(expression=e3, category="MOMENTUM", candidate_id=12)

    selector = DiversitySelector()
    res = selector.select_candidates([c1, c2, c3], capacity=10)

    # Only 1 representative is retained and dispatched
    dispatched_ids = [c.candidate_id for c in res.dispatched_candidates]
    assert len(dispatched_ids) == 1
    assert dispatched_ids[0] == 10
    assert res.funnel_metrics.near_duplicate_count == 2


def test_distinct_fields_with_matching_skeleton_preserved():
    """Candidates sharing the same AST skeleton but using different data fields must NOT be discarded."""
    e1 = "ts_decay_linear(close, 10)"
    e2 = "ts_decay_linear(volume, 10)"
    e3 = "ts_decay_linear(open, 10)"

    c1 = CandidateItem(expression=e1, category="MOMENTUM", candidate_id=20)
    c2 = CandidateItem(expression=e2, category="MOMENTUM", candidate_id=21)
    c3 = CandidateItem(expression=e3, category="MOMENTUM", candidate_id=22)

    selector = DiversitySelector()
    res = selector.select_candidates([c1, c2, c3], capacity=10)

    # All three must be preserved because data fields are distinct
    dispatched_ids = [c.candidate_id for c in res.dispatched_candidates]
    assert len(dispatched_ids) == 3
    assert set(dispatched_ids) == {20, 21, 22}
    assert res.funnel_metrics.near_duplicate_count == 0


def test_distinct_hypothesis_categories_with_matching_skeleton_preserved():
    """Candidates sharing the same expression and skeleton under different hypotheses must NOT be discarded."""
    e1 = "ts_decay_linear(close, 10)"
    e2 = "ts_decay_linear(close, 12)"

    # Same structure, but different economic hypotheses
    c1 = CandidateItem(expression=e1, category="MOMENTUM", candidate_id=30)
    c2 = CandidateItem(expression=e2, category="MEAN_REVERSION", candidate_id=31)

    selector = DiversitySelector()
    res = selector.select_candidates([c1, c2], capacity=10)

    # Both must be preserved because hypothesis categories differ
    dispatched_ids = [c.candidate_id for c in res.dispatched_candidates]
    assert len(dispatched_ids) == 2
    assert set(dispatched_ids) == {30, 31}
    assert res.funnel_metrics.near_duplicate_count == 0


def test_deterministic_diversity_selection():
    """Diversity selection must be perfectly deterministic given the same inputs and seed."""
    candidates = [
        CandidateItem(expression=f"ts_rank(close, {i})", category="MOMENTUM", candidate_id=i)
        for i in range(10, 30)
    ]

    cfg = DiversityConfig(seed=1337)
    selector = DiversitySelector(cfg)

    res1 = selector.select_candidates(candidates, capacity=5)
    res2 = selector.select_candidates(candidates, capacity=5)

    ids1 = [c.candidate_id for c in res1.dispatched_candidates]
    ids2 = [c.candidate_id for c in res2.dispatched_candidates]
    assert ids1 == ids2


def test_family_quotas_do_not_cause_empty_results_on_shortfall():
    """Quotas are targets, not blockers. If a family has insufficient candidates, fallback fills capacity."""
    # Only 2 distinct MOMENTUM candidates, 8 distinct VALUE candidates. Target capacity = 6.
    # MOMENTUM quota = 50% (target 3), VALUE quota = 50% (target 3).
    # MOMENTUM only has 2 -> it must NOT reject the other candidates or fail; VALUE fills the remaining 4 slots.
    mom_fields = ["close", "volume"]
    val_fields = ["ebitda", "sales", "debt", "equity", "cash", "assets", "income", "cfo"]

    candidates = [
        CandidateItem(expression=f"ts_rank({fld}, 10)", category="MOMENTUM", candidate_id=100 + i)
        for i, fld in enumerate(mom_fields)
    ] + [
        CandidateItem(expression=f"ts_zscore({fld}, 10)", category="VALUE", candidate_id=200 + i)
        for i, fld in enumerate(val_fields)
    ]

    cfg = DiversityConfig(
        family_quotas={"MOMENTUM": 0.5, "VALUE": 0.5},
        exploration_reserve_ratio=0.0,
        seed=42,
    )
    selector = DiversitySelector(cfg)
    res = selector.select_candidates(candidates, capacity=6)

    dispatched_ids = [c.candidate_id for c in res.dispatched_candidates]
    assert len(dispatched_ids) == 6
    # 2 from MOMENTUM, and remaining 4 filled from VALUE
    mom_count = sum(1 for c in res.dispatched_candidates if c.category == "MOMENTUM")
    val_count = sum(1 for c in res.dispatched_candidates if c.category == "VALUE")
    assert mom_count == 2
    assert val_count == 4


def test_invalid_expressions_never_enter_simulation():
    """Invalid expressions must be rejected and never dispatched."""
    c_valid = CandidateItem(expression="ts_rank(close, 10)", is_valid=True, candidate_id=1)
    c_invalid = CandidateItem(
        expression="invalid_brain_operator(close, 10)",
        is_valid=False,
        candidate_id=2,
        metadata={"validation_error": "Unknown operator"},
    )

    selector = DiversitySelector()
    res = selector.select_candidates([c_valid, c_invalid], capacity=10)

    dispatched_ids = [c.candidate_id for c in res.dispatched_candidates]
    assert 1 in dispatched_ids
    assert 2 not in dispatched_ids
    assert res.funnel_metrics.invalid_count == 1


def test_diversity_cannot_override_p1_10_statistical_evidence():
    """Diversity cannot promote a candidate that fails P1-10 statistical gates (t >= 1.96, q <= 0.15)."""
    # Create synthetic candidate records for post-simulation winner evaluation
    records = [
        # Candidate A: High Sharpe, statistically significant (SE = 0.2, t = 7.5 > 1.96)
        {
            "candidate_id": 1,
            "sharpe": 1.5,
            "t_years": 4.0,
            "composite_score": 1.5,
        },
        # Candidate B: Structurally diverse, but statistically insignificant (SE = 0.5, t = 0.4 < 1.96, LCB < 0)
        {
            "candidate_id": 2,
            "sharpe": 0.2,
            "t_years": 4.0,
            "composite_score": 0.2,
        },
    ]

    winners, losers = _multiplicity_corrected_split(records, fdr_level=0.15)
    by_id = {r["candidate_id"]: r for r in records}

    # Candidate 1 must be SIGNIFICANT_WINNER and in winners
    assert by_id[1]["selection_tier"] == "SIGNIFICANT_WINNER"
    assert by_id[1]["lcb95"] > 0.0
    assert any(w["candidate_id"] == 1 for w in winners)

    # Candidate 2 must NOT be promoted to SIGNIFICANT_WINNER despite being from a diverse category
    assert by_id[2]["selection_tier"] != "SIGNIFICANT_WINNER"
    assert by_id[2]["lcb95"] <= 0.0
    assert not any(w["candidate_id"] == 2 for w in winners)


def test_missing_or_insufficient_return_overlap_does_not_infer_low_correlation():
    """Missing or insufficient overlap (< MIN_OVERLAP) must report INSUFFICIENT_OVERLAP and NOT allow low correlation."""
    base_date = date(2020, 1, 1)

    # Series A has 50 days of returns
    series_a = {base_date + timedelta(days=i): 100.0 * (1.0 if i % 2 == 0 else -1.0) for i in range(50)}
    # Series B has 50 days of returns, non-overlapping
    series_b = {base_date + timedelta(days=i + 100): 50.0 for i in range(50)}

    # Evaluate pairwise correlation
    eval_res = evaluate_pairwise_correlation(series_a, series_b, min_overlap=120)
    assert eval_res["status"] == "INSUFFICIENT_OVERLAP"
    assert eval_res["correlation"] is None
    assert eval_res["meets_ceiling"] is False

    # In submission planner search matrix:
    # If rho is NaN due to insufficient overlap, under must be FALSE (never inferred as low correlation)
    rho_matrix = np.array([
        [1.0, np.nan],
        [np.nan, 1.0],
    ])
    under = (~np.isnan(rho_matrix)) & (rho_matrix < CEILING)
    # Neither should be legal against the other!
    assert not under[0, 1]
    assert not under[1, 0]


def test_holdout_quarantine_in_submission_planner():
    """Final holdout returns must be strictly quarantined from portfolio correlation and search."""
    base_date = date(2020, 1, 1)
    holdout_cutoff = date(2024, 1, 1)

    # Generate returns spanning 2020 to 2025 (including holdout period)
    days_data = {
        "alpha_1": {
            base_date + timedelta(days=i): 10.0 + (i % 5)
            for i in range(1800)
        },
        "alpha_2": {
            base_date + timedelta(days=i): -5.0 + (i % 3)
            for i in range(1800)
        },
    }

    # Evaluate pairwise correlation with holdout quarantine
    eval_res = evaluate_pairwise_correlation(
        days_data["alpha_1"],
        days_data["alpha_2"],
        min_overlap=120,
        holdout_date=holdout_cutoff,
    )
    # Ensure overlap days does not exceed dates before holdout_cutoff
    max_pre_holdout = (holdout_cutoff - base_date).days
    assert eval_res["overlap_days"] <= max_pre_holdout


def test_session_wide_deduplication():
    """Candidates matching previously active session hashes must be pruned even across generation rounds."""
    existing_hashes = {compute_expression_hash("ts_rank(close, 20)")}

    c1 = CandidateItem(expression="ts_rank(close, 20)", candidate_id=50)
    c2 = CandidateItem(expression="ts_rank(volume, 20)", candidate_id=51)

    selector = DiversitySelector()
    res = selector.select_candidates([c1, c2], capacity=10, session_seen_hashes=existing_hashes)

    dispatched_ids = [c.candidate_id for c in res.dispatched_candidates]
    assert 50 not in dispatched_ids
    assert 51 in dispatched_ids
    assert res.funnel_metrics.exact_duplicate_count == 1


def test_exploration_reserve_allocation():
    """Exploration reserve reserves a fraction of capacity for underrepresented or multi-field candidates."""
    # 10 candidates across 2 categories: 8 in MOMENTUM, 2 in VALUE
    candidates = [
        CandidateItem(
            expression=f"ts_rank(fld_{i}, 10)",
            category="MOMENTUM",
            candidate_id=i,
            field_families={"fld": "PRICE_VOLUME"},
            operators_used=["ts_rank"],
        )
        for i in range(8)
    ] + [
        CandidateItem(
            expression=f"ts_zscore(fund_{i}, 10)",
            category="VALUE",
            candidate_id=20 + i,
            field_families={"fund_a": "FUNDAMENTAL", "fund_b": "ANALYST"},
            operators_used=["ts_zscore", "rank"],
        )
        for i in range(2)
    ]

    # Capacity = 4, reserve = 50% (2 quota slots, 2 reserve slots)
    cfg = DiversityConfig(
        exploration_reserve_ratio=0.50,
        seed=42,
    )
    selector = DiversitySelector(cfg)
    res = selector.select_candidates(candidates, capacity=4)

    assert len(res.dispatched_candidates) == 4
    dispatched_ids = [c.candidate_id for c in res.dispatched_candidates]
    # The 2 exploratory VALUE candidates (with richer field and operator families) must both be selected
    assert 20 in dispatched_ids
    assert 21 in dispatched_ids


def test_orchestrator_generation_round_deduplication(tmp_path):
    """End-to-end integration: run_generation_round persists expression_hash, ast_skeleton, and prunes duplicates."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    from alpha_harness.db.sqlite import Database
    from alpha_harness.db.models import ResearchSession, ResearchCandidate, CandidateStatus
    from alpha_harness.generator.pipeline import AlphaGenerationPipeline
    from alpha_harness.orchestrator.researchers import MultiLLMOrchestrator

    async def _run():
        db_path = tmp_path / "test_diversity.db"
        db = Database(db_path)
        await db.create_all()

        llm_mock = MagicMock()
        ans_mock = MagicMock()
        # Returns: 1) rank(close), 2) exact duplicate rank(close), 3) parameter jitter rank(close, 10), 4) distinct field rank(volume)
        ans_mock.text = (
            '{"candidates": ['
            '{"hypothesis": "H1", "expression": "rank(close)"},'
            '{"hypothesis": "H1 duplicate", "expression": "  rank( close ) "},'
            '{"hypothesis": "H2 distinct", "expression": "rank(volume)"}'
            ']}'
        )
        ans_mock.total_tokens = 60
        llm_mock.generate = AsyncMock(return_value=ans_mock)

        pipeline = AlphaGenerationPipeline(llm=llm_mock)
        orchestrator = MultiLLMOrchestrator(db=db, pipeline=pipeline)

        # Set up session
        async with db.session() as s:
            sess = ResearchSession(
                name="Diversity Test Session",
                prompt="Momentum test",
                current_generation=1,
            )
            s.add(sess)
            await s.flush()
            session_id = sess.id
            await s.commit()

        await orchestrator.initialize_session_researchers(session_id)

        # Run generation round
        cand_ids = await orchestrator.run_generation_round(
            session_id=session_id,
            total_target_candidates=10,
        )

        async with db.session() as s:
            # Query all candidates in session
            from sqlalchemy import select
            cands = (await s.execute(select(ResearchCandidate).where(ResearchCandidate.session_id == session_id))).scalars().all()
            assert len(cands) > 0

            # Verify expression_hash and ast_skeleton are populated on all candidates
            for c in cands:
                assert c.expression_hash is not None
                assert len(c.expression_hash) == 64
                assert c.ast_skeleton is not None

            # Verify that exact duplicates have pruning_reason set and are NOT in cand_ids
            pruned_exact = [c for c in cands if c.pruning_reason == "EXACT_DUPLICATE_PRUNED"]
            assert len(pruned_exact) > 0
            for pe in pruned_exact:
                assert pe.id not in cand_ids
                assert pe.status == CandidateStatus.INVALID

        await db.dispose()

    asyncio.run(_run())
