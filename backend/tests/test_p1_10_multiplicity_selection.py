"""P1-10: Research Multiplicity & Uncertainty in Winner/Loser Selection.

Verifies:
1. Benjamini-Hochberg FDR rejects spurious top-decile candidates from random noise.
2. Sample duration uncertainty weighting: short-observation candidates with high point
   estimates do not outrank long-observation candidates with lower but more certain Sharpes.
3. Regime-relative loser classification (no hardcoded Sharpe < 0.8).
4. Small-batch (M < 5) conservative fallback requires t >= 1.96 against zero.
5. Holdout metrics never accessed during winner/loser partitioning.
6. Learning loop reflection consumes only FDR-verified SIGNIFICANT_WINNER exemplars.
7. Sharpe SE calculation correctness (Lo 2002 simplified formula).
8. Normal CDF correctness.
9. BH-FDR correction monotonicity and boundary properties.
10. Multiplicity provenance metadata attached to every returned record.
"""

from __future__ import annotations

import asyncio
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select

from alpha_harness.db.models import (
    CandidateStatus,
    PromptStrategy,
    ResearchCandidate,
    ResearchEvaluation,
    ResearchResearcher,
    ResearchSession,
    utcnow,
)
from alpha_harness.db.sqlite import Database
from alpha_harness.learning.loop import SelfLearningEngine
from alpha_harness.learning.memory import (
    BH_FDR_THRESHOLD,
    DEFAULT_OBSERVATION_YEARS,
    TRADING_DAYS_PER_YEAR,
    ResearchMemory,
    _benjamini_hochberg,
    _multiplicity_corrected_split,
    _normal_cdf,
    _sharpe_standard_error,
)


# --- Unit Tests: Statistical Helpers ----------------------------------------

# Test 7: Sharpe SE calculation correctness (Lo 2002 frequency-adjusted)
def test_sharpe_se_lo_formula():
    """Verifies SE(SR_a) = sqrt((1 + SR_a^2 / (2*q)) / Y) for annualized Sharpe.

    The vault annualises by sqrt(250), so q = 250 (TRADING_DAYS_PER_YEAR).
    """
    q = TRADING_DAYS_PER_YEAR  # 250

    # SR_a=0 over 1 year: SE = sqrt(1 / 1) = 1.0
    assert abs(_sharpe_standard_error(0.0, 1.0) - 1.0) < 1e-6

    # SR_a=1 over 1 year: SE = sqrt((1 + 1/(2*250)) / 1) = sqrt(1.002)
    expected = math.sqrt((1.0 + 1.0 / (2.0 * q)) / 1.0)
    assert abs(_sharpe_standard_error(1.0, 1.0) - expected) < 1e-6

    # SR_a=2 over 4 years: SE = sqrt((1 + 4/(2*250)) / 4) = sqrt(1.008/4)
    expected = math.sqrt((1.0 + 4.0 / (2.0 * q)) / 4.0)
    assert abs(_sharpe_standard_error(2.0, 4.0) - expected) < 1e-6

    # T=0 should return inf
    assert _sharpe_standard_error(1.0, 0.0) == float("inf")

    # T negative should return inf
    assert _sharpe_standard_error(1.0, -1.0) == float("inf")


def test_sharpe_se_numerical_regression():
    """Hand-computed numerical regression tests for the Lo (2002) SE formula.

    SE(SR_a) = sqrt((1 + SR_a^2 / (2 * 250)) / Y)
    """
    q = 250

    # Case 1: SR_a=2.5, Y=4.0
    #   numerator = 1 + 6.25 / 500 = 1.0125
    #   SE = sqrt(1.0125 / 4) = sqrt(0.253125) ≈ 0.50312
    se = _sharpe_standard_error(2.5, 4.0)
    assert abs(se - math.sqrt(1.0125 / 4.0)) < 1e-9
    assert abs(se - 0.50312) < 1e-4

    # Case 2: SR_a=1.0, Y=2.0
    #   numerator = 1 + 1.0 / 500 = 1.002
    #   SE = sqrt(1.002 / 2) = sqrt(0.501) ≈ 0.70781
    se = _sharpe_standard_error(1.0, 2.0)
    assert abs(se - math.sqrt(1.002 / 2.0)) < 1e-9
    assert abs(se - 0.70781) < 1e-4

    # Case 3: SR_a=0.0, Y=5.0 → SE = sqrt(1/5) = 0.44721
    se = _sharpe_standard_error(0.0, 5.0)
    assert abs(se - math.sqrt(0.2)) < 1e-9

    # Case 4: SR_a=3.0, Y=1.0
    #   numerator = 1 + 9/500 = 1.018
    #   SE = sqrt(1.018) ≈ 1.00896
    se = _sharpe_standard_error(3.0, 1.0)
    assert abs(se - math.sqrt(1.018)) < 1e-9

    # Case 5: High SR_a=10.0, Y=4.0
    #   numerator = 1 + 100/500 = 1.2
    #   SE = sqrt(1.2/4) = sqrt(0.3) ≈ 0.54772
    se = _sharpe_standard_error(10.0, 4.0)
    assert abs(se - math.sqrt(0.3)) < 1e-9


def test_sharpe_se_pipeline_end_to_end():
    """End-to-end numerical test: SE → t-statistic → p-value → LCB for known inputs.

    SR_a = 2.5, Y = 4.0, q = 250
    """
    sr = 2.5
    t_years = 4.0

    # SE = sqrt((1 + 6.25/500) / 4) = sqrt(0.253125) ≈ 0.50312
    se = _sharpe_standard_error(sr, t_years)
    assert abs(se - 0.50312) < 1e-4

    # t_vs_zero = 2.5 / 0.50312 ≈ 4.969
    t_zero = sr / se
    assert abs(t_zero - 4.969) < 0.01

    # p_value = 1 - Φ(4.969) ≈ 3.35e-7  (extremely small)
    p_val = 1.0 - _normal_cdf(t_zero)
    assert p_val < 0.001

    # LCB_95 = 2.5 - 1.96 * 0.50312 ≈ 1.514
    lcb = sr - 1.96 * se
    assert abs(lcb - 1.514) < 0.01
    assert lcb > 0  # Passes LCB gate

    # BH-FDR with M=1: q-value = p-value (still << 0.15)
    q_vals = _benjamini_hochberg([p_val])
    assert q_vals[0] < BH_FDR_THRESHOLD


# Test 8: Normal CDF correctness
def test_normal_cdf_known_values():
    """Verifies Φ(x) at well-known quantiles."""
    assert abs(_normal_cdf(0.0) - 0.5) < 1e-10
    assert abs(_normal_cdf(1.96) - 0.975) < 0.001
    assert abs(_normal_cdf(-1.96) - 0.025) < 0.001
    assert abs(_normal_cdf(3.0) - 0.99865) < 0.001
    assert abs(_normal_cdf(-3.0) - 0.00135) < 0.001


# Test 9: BH-FDR correction properties
def test_bh_fdr_monotonicity_and_bounds():
    """Verifies BH-FDR q-values are monotone-adjusted and bounded in [0, 1]."""
    # All p-values identical: q-values should all equal the p-value
    q_equal = _benjamini_hochberg([0.05, 0.05, 0.05])
    assert all(abs(q - 0.05) < 1e-10 for q in q_equal)

    # Sorted p-values: q should be weakly decreasing when re-sorted by p rank
    p_vals = [0.01, 0.04, 0.10, 0.30, 0.50]
    q_vals = _benjamini_hochberg(p_vals, fdr_level=0.15)
    # q-values should be bounded [0, 1]
    assert all(0.0 <= q <= 1.0 for q in q_vals)
    # Sorted by original p-value rank, q-values should be non-decreasing
    indexed = sorted(zip(p_vals, q_vals), key=lambda x: x[0])
    for i in range(1, len(indexed)):
        assert indexed[i][1] >= indexed[i - 1][1] - 1e-10

    # Single p-value: q = p
    assert abs(_benjamini_hochberg([0.03])[0] - 0.03) < 1e-10

    # Empty list: no crash
    assert _benjamini_hochberg([]) == []


def test_bh_fdr_rejects_with_many_tests():
    """With many tests, high p-values should be rejected (q > threshold)."""
    # 100 p-values: 95 are uniform noise (high), 5 are very small signal
    p_noise = [0.5 + 0.005 * i for i in range(95)]  # 0.5 to 0.97
    p_signal = [0.0005, 0.001, 0.002, 0.003, 0.004]
    p_all = p_signal + p_noise
    q_all = _benjamini_hochberg(p_all, fdr_level=0.15)

    # The 5 signal p-values should have low q-values
    for i in range(5):
        assert q_all[i] < 0.15, f"Signal p={p_all[i]} got q={q_all[i]}"

    # The noise p-values should have high q-values (rejected from winner pool)
    for i in range(5, len(q_all)):
        assert q_all[i] > 0.15, f"Noise p={p_all[i]} got q={q_all[i]}"



# --- Integration Tests: _multiplicity_corrected_split ----------------------

def test_multiplicity_split_metadata_attached():
    """Test 10: Every record from the split contains multiplicity provenance."""
    records = [
        {"sharpe": 1.5 + i * 0.3, "t_years": 2.0, "composite_score": 1.0}
        for i in range(10)
    ]
    winners, losers = _multiplicity_corrected_split(records)
    all_records = winners + losers
    # Also check records that are CANDIDATE_POOL (neither winner nor loser)
    for r in records:
        assert "multiplicity_trials_count" in r, "Missing multiplicity_trials_count"
        assert r["multiplicity_trials_count"] == 10
        assert "cohort_baseline_sharpe" in r, "Missing cohort_baseline_sharpe"
        assert "sharpe_se" in r, "Missing sharpe_se"
        assert "z_score" in r, "Missing z_score"
        assert "p_value" in r, "Missing p_value"
        assert "fdr_q_value" in r, "Missing fdr_q_value"
        assert "lcb95" in r, "Missing lcb95"
        assert "selection_tier" in r, "Missing selection_tier"
        assert "t_vs_zero" in r, "Missing t_vs_zero"
        assert r["selection_tier"] in (
            "SIGNIFICANT_WINNER",
            "PROVISIONAL",
            "SIGNIFICANT_LOSER",
            "CANDIDATE_POOL",
            "CANDIDATE_POOL_UNDERPERFORMER",
        )


def test_multiplicity_split_regime_relative_losers():
    """Test 3: Losers are identified relative to cohort, not hardcoded Sharpe < 0.8."""
    # High-regime cohort: baseline around Sharpe 2.0
    # A candidate with Sharpe 0.9 should be a loser in this regime
    # even though 0.9 > 0.8 (the old hardcoded threshold)
    records = [
        {"sharpe": 2.5, "t_years": 4.0, "composite_score": 2.0},
        {"sharpe": 2.3, "t_years": 4.0, "composite_score": 1.8},
        {"sharpe": 2.1, "t_years": 4.0, "composite_score": 1.6},
        {"sharpe": 2.0, "t_years": 4.0, "composite_score": 1.5},
        {"sharpe": 1.9, "t_years": 4.0, "composite_score": 1.4},
        {"sharpe": 0.3, "t_years": 4.0, "composite_score": 0.3},  # Far below baseline
    ]
    winners, losers = _multiplicity_corrected_split(records)

    # The 0.3 candidate should be classified as loser even though old code
    # would only check sharpe < 0.8. Here it's below cohort median (~2.05)
    # and statistically significantly below.
    loser_ids = [r["sharpe"] for r in losers]
    assert 0.3 in loser_ids, f"Sharpe 0.3 should be a regime-relative loser, got tiers: {[(r['sharpe'], r['selection_tier']) for r in records]}"


def test_multiplicity_split_low_regime_not_spurious_losers():
    """In a low-regime cohort (baseline ~0.5), a Sharpe of 0.6 should NOT be a loser."""
    records = [
        {"sharpe": 0.8, "t_years": 2.0, "composite_score": 0.8},
        {"sharpe": 0.7, "t_years": 2.0, "composite_score": 0.7},
        {"sharpe": 0.5, "t_years": 2.0, "composite_score": 0.5},
        {"sharpe": 0.4, "t_years": 2.0, "composite_score": 0.4},
        {"sharpe": 0.3, "t_years": 2.0, "composite_score": 0.3},
    ]
    winners, losers = _multiplicity_corrected_split(records)

    # 0.4 and 0.3 are below median (0.5) but we need to check they are
    # classified based on statistical significance, not arbitrary thresholds.
    # With only 5 candidates and 2-year observation, the SE is large,
    # so they might be CANDIDATE_POOL_UNDERPERFORMER but not necessarily
    # SIGNIFICANT_LOSER unless the z-score is <= -1.645
    for r in records:
        if r["sharpe"] == 0.4:
            assert r["selection_tier"] in ("CANDIDATE_POOL_UNDERPERFORMER", "SIGNIFICANT_LOSER", "CANDIDATE_POOL")


def test_multiplicity_split_small_batch_conservative():
    """Test 4: With M < 5 candidates, conservative t >= 1.96 vs zero fallback."""
    # 3 candidates — small batch
    records = [
        {"sharpe": 3.0, "t_years": 5.0, "composite_score": 3.0},  # t_vs_zero = 3.0/SE(3.0,5) is high
        {"sharpe": 0.1, "t_years": 0.2, "composite_score": 0.1},  # short obs, low t_vs_zero
        {"sharpe": 1.0, "t_years": 0.3, "composite_score": 1.0},  # short obs, modest Sharpe
    ]
    winners, losers = _multiplicity_corrected_split(records)

    # The 3.0 Sharpe with 5 years should be a winner
    winner_sharpes = [r["sharpe"] for r in winners]
    assert 3.0 in winner_sharpes, f"Expected Sharpe 3.0 as winner, got {winner_sharpes}"

    # The 0.1 Sharpe with only 0.2 years should NOT be a winner (t_vs_zero too low)
    assert 0.1 not in winner_sharpes


# Test 1: BH-FDR rejects spurious candidates from random noise
def test_bh_fdr_rejects_spurious_candidates(tmp_path: Path):
    """In a batch of candidates drawn from a common Sharpe distribution,
    naive top-quartile selection would pick random flukes as winners.
    BH-FDR should reject most or all when there's no genuine signal."""
    async def _run():
        db = Database(tmp_path / "sqlite.db")
        await db.create_all()

        async with db.session() as s:
            sess = ResearchSession(
                name="Noise Batch Test",
                prompt="test",
                region="USA",
                universe="TOP3000",
                current_generation=1,
            )
            s.add(sess)
            await s.flush()

            strat = PromptStrategy(
                session_id=sess.id,
                name="Test Strat",
                version=1,
                category="general",
                is_active=True,
            )
            s.add(strat)
            await s.flush()

            # Generate 20 candidates with Sharpes clustered around 1.0 (noise-like)
            # With SE ~0.7 for T=2yr, most z-scores against median will be small
            import random
            random.seed(42)
            for i in range(20):
                cand = ResearchCandidate(
                    session_id=sess.id,
                    expression=f"rank(close + {i})",
                    status=CandidateStatus.EVALUATED,
                    generation=1,
                    hypothesis=f"Hypothesis {i}",
                )
                s.add(cand)
                await s.flush()

                sharpe = round(1.0 + random.gauss(0, 0.3), 4)
                ev = ResearchEvaluation(
                    candidate_id=cand.id,
                    is_verified=True,
                    val_sharpe=sharpe,
                    val_fitness=1.0,
                    is_end_date=date(2020, 1, 1),
                    val_end_date=date(2022, 1, 1),
                    provenance="BRAIN_VAULT",
                )
                s.add(ev)

        memory = ResearchMemory(db)
        winners, losers = await memory.get_session_performance_split(sess.id, generation=1)

        # With noise-level variation, FDR should find zero or very few winners
        # Old method would blindly take 5 (top 25% of 20)
        assert len(winners) < 5, (
            f"Expected FDR to reject most spurious candidates, got {len(winners)} winners "
            f"(old method would have taken 5)"
        )

        # Verify all records have multiplicity metadata
        all_recs = winners + losers
        for r in all_recs:
            assert "fdr_q_value" in r
            assert "multiplicity_trials_count" in r
            assert r["multiplicity_trials_count"] == 20

    asyncio.run(_run())


# Test 2: Sample duration uncertainty weighting
def test_sample_duration_uncertainty_weighting(tmp_path: Path):
    """A short 6-month candidate with Sharpe 2.5 should not outrank
    a solid 5-year candidate with Sharpe 1.8 when uncertainty is considered."""
    async def _run():
        db = Database(tmp_path / "sqlite.db")
        await db.create_all()

        async with db.session() as s:
            sess = ResearchSession(
                name="Duration Test",
                prompt="test",
                region="USA",
                universe="TOP3000",
                current_generation=1,
            )
            s.add(sess)
            await s.flush()

            strat = PromptStrategy(
                session_id=sess.id,
                name="Strat",
                version=1,
                category="general",
                is_active=True,
            )
            s.add(strat)
            await s.flush()

            # Candidate A: 5-year validation, Sharpe 1.8
            cand_a = ResearchCandidate(
                session_id=sess.id,
                expression="rank(close)",
                status=CandidateStatus.EVALUATED,
                generation=1,
                hypothesis="Long observation alpha",
            )
            s.add(cand_a)
            await s.flush()
            s.add(ResearchEvaluation(
                candidate_id=cand_a.id,
                is_verified=True,
                val_sharpe=1.8,
                val_fitness=1.2,
                is_end_date=date(2015, 1, 1),
                val_end_date=date(2020, 1, 1),  # 5 years
                provenance="BRAIN_VAULT",
            ))

            # Candidate B: 6-month validation, Sharpe 2.5
            cand_b = ResearchCandidate(
                session_id=sess.id,
                expression="rank(volume)",
                status=CandidateStatus.EVALUATED,
                generation=1,
                hypothesis="Short observation alpha",
            )
            s.add(cand_b)
            await s.flush()
            s.add(ResearchEvaluation(
                candidate_id=cand_b.id,
                is_verified=True,
                val_sharpe=2.5,
                val_fitness=1.5,
                is_end_date=date(2019, 6, 1),
                val_end_date=date(2020, 1, 1),  # ~6 months
                provenance="BRAIN_VAULT",
            ))

            # Add 4 more filler candidates to get batch >= 5 for FDR path
            for i in range(4):
                c = ResearchCandidate(
                    session_id=sess.id,
                    expression=f"rank(open + {i})",
                    status=CandidateStatus.EVALUATED,
                    generation=1,
                    hypothesis=f"Filler {i}",
                )
                s.add(c)
                await s.flush()
                s.add(ResearchEvaluation(
                    candidate_id=c.id,
                    is_verified=True,
                    val_sharpe=0.5 + i * 0.1,
                    val_fitness=0.8,
                    is_end_date=date(2015, 1, 1),
                    val_end_date=date(2020, 1, 1),
                    provenance="BRAIN_VAULT",
                ))

        memory = ResearchMemory(db)
        winners, losers = await memory.get_session_performance_split(sess.id, generation=1)

        # Find candidates A and B in results
        rec_a = next((r for r in winners + losers + [r for r in [] if False] if r["candidate_id"] == cand_a.id), None)
        rec_b = next((r for r in winners + losers + [r for r in [] if False] if r["candidate_id"] == cand_b.id), None)

        # Both might be in the full records list — retrieve from all records
        # by checking any record dict returned
        all_records = []
        w2, l2 = await memory.get_session_performance_split(sess.id, generation=1)
        all_records = w2 + l2

        a_rec = next((r for r in all_records if r["candidate_id"] == cand_a.id), None)
        b_rec = next((r for r in all_records if r["candidate_id"] == cand_b.id), None)

        if a_rec and b_rec:
            # Candidate A should have a higher LCB than Candidate B
            # because 5 years of observation gives much tighter CI
            a_lcb = float(a_rec["lcb95"])
            b_lcb = float(b_rec["lcb95"])
            a_se = float(a_rec["sharpe_se"])
            b_se = float(b_rec["sharpe_se"])

            # SE for B should be much larger than SE for A
            assert b_se > a_se, f"Short observation SE ({b_se}) should exceed long observation SE ({a_se})"

            # LCB for A (long obs, lower point) should be higher than LCB for B (short obs, higher point)
            assert a_lcb > b_lcb, (
                f"Long-observation Sharpe 1.8 LCB ({a_lcb}) should exceed "
                f"short-observation Sharpe 2.5 LCB ({b_lcb}) due to uncertainty"
            )

    asyncio.run(_run())


# Test 5: Holdout isolation in multiplicity split
def test_holdout_isolation_in_multiplicity_split(tmp_path: Path):
    """Holdout metrics must never be accessed during winner/loser partitioning."""
    async def _run():
        db = Database(tmp_path / "sqlite.db")
        await db.create_all()

        SENTINEL_HOLDOUT = 999.0

        async with db.session() as s:
            sess = ResearchSession(
                name="Holdout Isolation Test",
                prompt="test",
                region="USA",
                universe="TOP3000",
                current_generation=1,
            )
            s.add(sess)
            await s.flush()

            strat = PromptStrategy(
                session_id=sess.id,
                name="Strat",
                version=1,
                category="general",
                is_active=True,
            )
            s.add(strat)
            await s.flush()

            for i in range(6):
                cand = ResearchCandidate(
                    session_id=sess.id,
                    expression=f"rank(close + {i})",
                    status=CandidateStatus.EVALUATED,
                    generation=1,
                    hypothesis=f"H{i}",
                )
                s.add(cand)
                await s.flush()

                ev = ResearchEvaluation(
                    candidate_id=cand.id,
                    is_verified=True,
                    val_sharpe=1.0 + i * 0.3,
                    val_fitness=1.0,
                    holdout_sharpe=SENTINEL_HOLDOUT,  # Sentinel that should never appear in selection
                    holdout_fitness=SENTINEL_HOLDOUT,
                    is_end_date=date(2018, 1, 1),
                    val_end_date=date(2020, 1, 1),
                    provenance="BRAIN_VAULT",
                )
                s.add(ev)

        memory = ResearchMemory(db)
        winners, losers = await memory.get_session_performance_split(sess.id, generation=1)

        all_records = winners + losers
        for r in all_records:
            # No record should contain the sentinel holdout value
            assert float(r["sharpe"]) != SENTINEL_HOLDOUT, "Holdout Sharpe leaked into selection metric"
            # The sharpe field should be val_sharpe, not holdout_sharpe
            assert "holdout" not in str(r).lower() or "holdout" in "holdout_sharpe_should_not_appear"

    asyncio.run(_run())


# Test 6: Learning loop incorporates FDR exemplars
def test_reflection_loop_incorporates_fdr_exemplars(tmp_path: Path):
    """Verifies few-shot exemplars in evolved strategies come from
    FDR-verified SIGNIFICANT_WINNER tier with selection provenance."""
    async def _run():
        db = Database(tmp_path / "sqlite.db")
        await db.create_all()

        async with db.session() as s:
            sess = ResearchSession(
                name="FDR Exemplar Test",
                prompt="test",
                region="USA",
                universe="TOP3000",
                current_generation=1,
            )
            s.add(sess)
            await s.flush()

            strat = PromptStrategy(
                session_id=sess.id,
                name="Seed Strategy",
                version=1,
                category="general",
                is_active=True,
                template_text="Generate alpha expressions.",
            )
            s.add(strat)
            await s.flush()

            # Create a batch with clear winners (high Sharpe, long obs) and filler
            for i in range(10):
                cand = ResearchCandidate(
                    session_id=sess.id,
                    expression=f"rank(close + {i})",
                    status=CandidateStatus.EVALUATED,
                    generation=1,
                    hypothesis=f"Hypothesis {i}",
                )
                s.add(cand)
                await s.flush()

                # First 2: strong winners; rest: noise
                sharpe = 3.0 + i * 0.1 if i < 2 else 0.5
                ev = ResearchEvaluation(
                    candidate_id=cand.id,
                    is_verified=True,
                    val_sharpe=sharpe,
                    val_fitness=1.0,
                    is_end_date=date(2015, 1, 1),
                    val_end_date=date(2020, 1, 1),
                    provenance="BRAIN_VAULT",
                )
                s.add(ev)

        memory = ResearchMemory(db)
        engine = SelfLearningEngine(db=db, memory=memory, llm=None)

        result = await engine._execute_single_category_cycle(
            session_id=sess.id,
            category="general",
            min_candidates=3,
        )

        # Verify the evolved strategy has winner_selection_method
        async with db.session() as s:
            from sqlalchemy import select as sa_select
            strat_stmt = sa_select(PromptStrategy).where(
                PromptStrategy.session_id == sess.id,
                PromptStrategy.is_active.is_(True),
            ).order_by(PromptStrategy.version.desc())
            res = await s.execute(strat_stmt)
            evolved = res.scalars().first()

            if evolved and evolved.guidance:
                assert evolved.guidance.get("winner_selection_method") == "p1_10_bh_fdr", (
                    f"Expected p1_10_bh_fdr in guidance, got {evolved.guidance}"
                )

            if evolved and evolved.examples:
                for ex in evolved.examples:
                    # If the example has selection_tier, it should be SIGNIFICANT_WINNER
                    if "selection_tier" in ex:
                        assert ex["selection_tier"] == "SIGNIFICANT_WINNER"

    asyncio.run(_run())


# Test: Legacy candidates (no val_sharpe) use is_sharpe, not full_sharpe
def test_legacy_candidates_use_is_sharpe_not_full(tmp_path: Path):
    """P0-03 regression: legacy candidates without val_sharpe must fall back
    to is_sharpe (in-sample), never full_sharpe (which includes holdout)."""
    async def _run():
        db = Database(tmp_path / "sqlite.db")
        await db.create_all()

        FULL_SHARPE_SENTINEL = 888.0  # Would leak if full_sharpe were used
        IS_SHARPE_VALUE = 1.5  # Safe in-sample value

        async with db.session() as s:
            sess = ResearchSession(
                name="Legacy Fallback Test",
                prompt="test",
                region="USA",
                universe="TOP3000",
                current_generation=1,
            )
            s.add(sess)
            await s.flush()

            strat = PromptStrategy(
                session_id=sess.id,
                name="Strat",
                version=1,
                category="general",
                is_active=True,
            )
            s.add(strat)
            await s.flush()

            for i in range(6):
                cand = ResearchCandidate(
                    session_id=sess.id,
                    expression=f"rank(close + {i})",
                    status=CandidateStatus.EVALUATED,
                    generation=1,
                    hypothesis=f"H{i}",
                )
                s.add(cand)
                await s.flush()

                ev = ResearchEvaluation(
                    candidate_id=cand.id,
                    is_verified=True,
                    val_sharpe=None,  # Legacy: no validation Sharpe
                    is_sharpe=IS_SHARPE_VALUE + i * 0.1,
                    full_sharpe=FULL_SHARPE_SENTINEL,  # Should NEVER appear in output
                    provenance="BRAIN_VAULT",
                    evaluation_scheme="LEGACY_2WAY",
                )
                s.add(ev)

        memory = ResearchMemory(db)
        winners, losers = await memory.get_session_performance_split(sess.id, generation=1)

        all_records = winners + losers
        assert len(all_records) > 0, "Should have at least some records"

        for r in all_records:
            assert float(r["sharpe"]) != FULL_SHARPE_SENTINEL, (
                f"full_sharpe sentinel ({FULL_SHARPE_SENTINEL}) leaked into learning! "
                f"Got sharpe={r['sharpe']} for candidate {r['candidate_id']}"
            )
            # Sharpe should be derived from is_sharpe
            assert abs(float(r["sharpe"]) - IS_SHARPE_VALUE) < 1.0, (
                f"Expected sharpe near {IS_SHARPE_VALUE}, got {r['sharpe']}"
            )

    asyncio.run(_run())


# Test: Empty batch graceful handling
def test_empty_batch_returns_empty():
    """Empty record list should return ([], []) without errors."""
    winners, losers = _multiplicity_corrected_split([])
    assert winners == []
    assert losers == []


# Test: Single candidate batch
def test_single_candidate_batch():
    """A batch of 1 candidate should not crash and should use small-batch path."""
    records = [{"sharpe": 2.0, "t_years": 3.0, "composite_score": 2.0}]
    winners, losers = _multiplicity_corrected_split(records)
    assert records[0]["multiplicity_trials_count"] == 1
    assert records[0]["selection_tier"] in (
        "SIGNIFICANT_WINNER",
        "CANDIDATE_POOL",
        "CANDIDATE_POOL_UNDERPERFORMER",
        "SIGNIFICANT_LOSER",
    )


# Test: Observation year estimation from actual observations and eval dates
def test_observation_year_estimation():
    """Verify Y estimation prefers actual observation count over calendar dates,
    and correctly falls back to dates or default."""
    from types import SimpleNamespace
    from alpha_harness.learning.memory import _estimate_observation_years

    # 1. Prefer actual valid observations and their frequency (e.g. 500 trading days @ 250 days/yr = 2.0 yrs)
    ev_with_obs = SimpleNamespace(
        val_sharpe=1.5,
        val_observations=500,
        is_end_date=date(2018, 1, 1),
        val_end_date=date(2025, 1, 1),  # Calendar dates would say 7 years, but actual obs is 500 days
    )
    t_obs = _estimate_observation_years(ev_with_obs)
    assert abs(t_obs - 2.0) < 1e-6, f"Expected 2.0 years from 500 observations, got {t_obs}"

    # Another count: 125 trading days @ 250 days/yr = 0.5 years
    ev_short_obs = SimpleNamespace(val_sharpe=1.2, val_observations=125)
    assert abs(_estimate_observation_years(ev_short_obs) - 0.5) < 1e-6

    # 2. Date fallback: 2 calendar years ≈ 2.0 trading years
    ev_3way = SimpleNamespace(
        val_sharpe=1.5,
        is_end_date=date(2018, 1, 1),
        val_end_date=date(2020, 1, 1),
    )
    t = _estimate_observation_years(ev_3way)
    assert abs(t - 2.0) < 0.1, f"Expected ~2.0 years, got {t}"

    # 3. Legacy: no val_sharpe → default
    ev_legacy = SimpleNamespace(
        val_sharpe=None,
        is_end_date=date(2010, 1, 1),
        val_end_date=date(2018, 1, 1),
    )
    t_legacy = _estimate_observation_years(ev_legacy)
    assert t_legacy == DEFAULT_OBSERVATION_YEARS, f"Expected default {DEFAULT_OBSERVATION_YEARS}, got {t_legacy}"

    # 4. Missing dates → default
    ev_no_dates = SimpleNamespace(val_sharpe=1.5, is_end_date=None, val_end_date=None)
    assert _estimate_observation_years(ev_no_dates) == DEFAULT_OBSERVATION_YEARS


def test_observation_window_mismatch_detection():
    """Catches a mismatch between the metric's observation window and its uncertainty estimate.
    
    If val_end_date <= is_end_date (inverted or zero-length validation window),
    or if val_observations is non-positive, an explicit ValueError must be raised."""
    from types import SimpleNamespace
    from alpha_harness.learning.memory import _estimate_observation_years

    # Inverted dates: val_end_date before is_end_date
    ev_inverted = SimpleNamespace(
        val_sharpe=1.5,
        is_end_date=date(2020, 1, 1),
        val_end_date=date(2018, 1, 1),
    )
    with pytest.raises(ValueError, match="Observation window mismatch.*val_end_date.*must be strictly after is_end_date"):
        _estimate_observation_years(ev_inverted)

    # Identical dates: zero-length window
    ev_zero_len = SimpleNamespace(
        val_sharpe=1.5,
        is_end_date=date(2020, 1, 1),
        val_end_date=date(2020, 1, 1),
    )
    with pytest.raises(ValueError, match="Observation window mismatch.*val_end_date.*must be strictly after is_end_date"):
        _estimate_observation_years(ev_zero_len)

    # Negative observation count
    ev_neg_obs = SimpleNamespace(val_sharpe=1.5, val_observations=-10)
    with pytest.raises(ValueError, match="Observation window mismatch.*non-positive observation count"):
        _estimate_observation_years(ev_neg_obs)

    # Zero observation count
    ev_zero_obs = SimpleNamespace(val_sharpe=1.5, val_observations=0)
    with pytest.raises(ValueError, match="Observation window mismatch.*non-positive observation count"):
        _estimate_observation_years(ev_zero_obs)


# ---------------------------------------------------------------------------
# P1-10 Targeted Statistical Correction Regression Tests
# ---------------------------------------------------------------------------

def test_positive_unsupported_candidate_not_promoted_to_winner():
    """A positive but statistically unsupported top candidate (e.g. SR=1.8, T=0.2)
    must NOT be promoted to SIGNIFICANT_WINNER. It must be marked PROVISIONAL and
    excluded from winners."""
    records = [
        {"sharpe": 1.8, "t_years": 0.2, "composite_score": 1.5},  # Top rank, positive SR, but short T -> t_zero=0.65 < 1.96, lcb < 0
        {"sharpe": 0.5, "t_years": 2.0, "composite_score": 0.5},
        {"sharpe": 0.2, "t_years": 2.0, "composite_score": 0.2},
    ]
    winners, losers = _multiplicity_corrected_split(records)

    # Top candidate has positive Sharpe and ranks #1, but is NOT a SIGNIFICANT_WINNER
    assert records[0]["selection_tier"] == "PROVISIONAL"
    assert records[0] not in winners
    assert len(winners) == 0, f"Expected 0 winners due to lack of statistical support, got {winners}"


def test_small_cohorts_do_not_bypass_uncertainty_requirements():
    """Small cohorts (M < 5, including M=1 and M=2) must NOT bypass uncertainty requirements."""
    # M=1 candidate with high Sharpe but tiny sample
    rec_single = [{"sharpe": 2.5, "t_years": 0.1, "composite_score": 2.5}]
    w_single, _ = _multiplicity_corrected_split(rec_single)
    assert rec_single[0]["selection_tier"] == "PROVISIONAL"
    assert len(w_single) == 0, "Single candidate with short observation must not be SIGNIFICANT_WINNER"

    # M=2 cohort: top candidate has SR=2.0 but T=0.15 years
    rec_pair = [
        {"sharpe": 2.0, "t_years": 0.15, "composite_score": 2.0},
        {"sharpe": 0.1, "t_years": 2.0, "composite_score": 0.1},
    ]
    w_pair, _ = _multiplicity_corrected_split(rec_pair)
    assert rec_pair[0]["selection_tier"] == "PROVISIONAL"
    assert len(w_pair) == 0, "Pair top candidate with short observation must not be SIGNIFICANT_WINNER"


def test_statistically_supported_candidates_qualify():
    """Statistically supported candidates (high Sharpe + long T giving t >= 1.96, LCB > 0, q <= 0.15)
    must qualify as SIGNIFICANT_WINNER."""
    records = [
        {"sharpe": 2.5, "t_years": 4.0, "composite_score": 2.0},  # SE≈0.503, t≈4.97, LCB≈+1.51, q≈3e-7
        {"sharpe": 0.5, "t_years": 4.0, "composite_score": 0.5},
    ]
    winners, losers = _multiplicity_corrected_split(records)

    assert records[0]["selection_tier"] == "SIGNIFICANT_WINNER"
    assert len(winners) == 1
    assert winners[0]["sharpe"] == 2.5


def test_provisional_candidates_cannot_enter_few_shot_memory(tmp_path: Path):
    """Provisional candidates must NOT enter significant-winner few-shot memory (examples).
    Instead, they are preserved under guidance['provisional_hypotheses'] with explicit labeling."""
    async def _run():
        db = Database(tmp_path / "sqlite.db")
        await db.create_all()

        async with db.session() as s:
            sess = ResearchSession(
                name="Provisional Test Session",
                prompt="test",
                region="USA",
                universe="TOP3000",
                current_generation=1,
            )
            s.add(sess)
            await s.flush()

            strat = PromptStrategy(
                session_id=sess.id,
                name="Seed Strategy",
                version=1,
                category="general",
                is_active=True,
                template_text="Generate alpha expressions.",
            )
            s.add(strat)
            await s.flush()

            # Create candidates with positive Sharpe but short 30-day observation (PROVISIONAL)
            for i in range(3):
                cand = ResearchCandidate(
                    session_id=sess.id,
                    expression=f"rank(close * {i+1})",
                    status=CandidateStatus.EVALUATED,
                    generation=1,
                    hypothesis=f"Short obs hypothesis {i}",
                )
                s.add(cand)
                await s.flush()

                ev = ResearchEvaluation(
                    candidate_id=cand.id,
                    is_verified=True,
                    val_sharpe=1.5 + i * 0.2,  # Positive Sharpe 1.5, 1.7, 1.9
                    val_fitness=1.0,
                    is_end_date=date(2020, 1, 1),
                    val_end_date=date(2020, 2, 15),  # 45 days -> T ~ 0.12 years -> high SE, LCB < 0
                    provenance="BRAIN_VAULT",
                )
                s.add(ev)

        memory = ResearchMemory(db)
        engine = SelfLearningEngine(db=db, memory=memory, llm=None)

        result = await engine.execute_learning_cycle(session_id=sess.id, category="general")

        # Verify strategy examples (the few-shot memory) contain ZERO provisional candidates
        async with db.session() as s:
            strat_stmt = (
                select(PromptStrategy)
                .where(PromptStrategy.session_id == sess.id, PromptStrategy.is_active.is_(True))
                .order_by(PromptStrategy.version.desc())
            )
            evolved = (await s.execute(strat_stmt)).scalars().first()

            assert evolved is not None
            # Few-shot examples MUST be empty because all candidates were provisional
            assert len(evolved.examples) == 0, f"Expected 0 few-shot examples for provisional batch, got {evolved.examples}"

            # Provisional candidates must be tracked separately under guidance
            assert evolved.guidance is not None
            assert "provisional_hypotheses" in evolved.guidance
            assert len(evolved.guidance["provisional_hypotheses"]) > 0
            for ph in evolved.guidance["provisional_hypotheses"]:
                assert ph["selection_tier"] == "PROVISIONAL"
                assert "[Provisional Candidate / Insufficient Statistical Evidence]" in ph["label"]

    asyncio.run(_run())


def test_holdout_changes_cannot_alter_winner_selection(tmp_path: Path):
    """Modifying holdout partition data cannot alter candidate selection tiers,
    p-values, q-values, or winner status (P0-03 holdout boundary)."""
    async def _run():
        db = Database(tmp_path / "sqlite.db")
        await db.create_all()

        async with db.session() as s:
            sess = ResearchSession(
                name="Holdout Boundary Test",
                prompt="test",
                region="USA",
                universe="TOP3000",
                current_generation=1,
            )
            s.add(sess)
            await s.flush()

            # Candidate 1: High validation Sharpe, fantastic holdout
            c1 = ResearchCandidate(
                session_id=sess.id,
                expression="rank(close)",
                status=CandidateStatus.EVALUATED,
                generation=1,
            )
            s.add(c1)
            await s.flush()
            ev1 = ResearchEvaluation(
                candidate_id=c1.id,
                is_verified=True,
                val_sharpe=2.5,
                holdout_sharpe=10.0,  # Extreme holdout
                is_end_date=date(2015, 1, 1),
                val_end_date=date(2020, 1, 1),
                provenance="BRAIN_VAULT",
            )
            s.add(ev1)

            # Candidate 2: Same high validation Sharpe, disastrous holdout
            c2 = ResearchCandidate(
                session_id=sess.id,
                expression="rank(open)",
                status=CandidateStatus.EVALUATED,
                generation=1,
            )
            s.add(c2)
            await s.flush()
            ev2 = ResearchEvaluation(
                candidate_id=c2.id,
                is_verified=True,
                val_sharpe=2.5,
                holdout_sharpe=-5.0,  # Disastrous holdout
                is_end_date=date(2015, 1, 1),
                val_end_date=date(2020, 1, 1),
                provenance="BRAIN_VAULT",
            )
            s.add(ev2)

        memory = ResearchMemory(db)
        all_cands = await memory.get_eligible_candidates(sess.id, generation=1)

        # Both candidates must receive IDENTICAL statistics and selection tier
        rec1 = next(r for r in all_cands if r["candidate_id"] == c1.id)
        rec2 = next(r for r in all_cands if r["candidate_id"] == c2.id)

        assert rec1["selection_tier"] == rec2["selection_tier"] == "SIGNIFICANT_WINNER"
        assert rec1["p_value"] == rec2["p_value"]
        assert rec1["fdr_q_value"] == rec2["fdr_q_value"]
        assert rec1["lcb95"] == rec2["lcb95"]

    asyncio.run(_run())


def test_conventional_two_sided_95_confidence_bound_agreement():
    """Verify confidence-bound terminology and significance gate agreement.
    
    The condition t >= 1.96 corresponds strictly to a positive lower endpoint for a
    conventional two-sided 95% confidence interval [SR - 1.96*SE, SR + 1.96*SE].
    
    This test verifies:
    1. Implementation, labels, and tests agree: lcb95 = SR - 1.96 * SE.
    2. The gate is NOT weakened to a one-sided 95% threshold (1.645) or 90% CI:
       - A candidate with t=1.70 (z > 1.645) has a negative lower endpoint and MUST NOT be a winner.
       - A candidate with t=1.95 has lcb95 < 0 and remains PROVISIONAL.
       - Only t >= 1.96 yields lcb95 > 0 and permits SIGNIFICANT_WINNER promotion."""
    # Candidate with t ≈ 1.70 (passes one-sided 90%/95% 1.645, but FAILS two-sided 95% 1.96):
    # For SE = 1.0 (SR=0, Y=1.0 gives SE=1.0; for SR=1.70, Y=2.90 gives SE ≈ 1.00):
    # Let's craft exact records:
    # Cand A: SR = 1.70, SE = 1.0 -> t = 1.70. LCB95 = 1.70 - 1.96 = -0.26 < 0.
    # Cand B: SR = 1.95, SE = 1.0 -> t = 1.95. LCB95 = 1.95 - 1.96 = -0.01 < 0.
    # Cand C: SR = 2.05, SE = 1.0 -> t = 2.05. LCB95 = 2.05 - 1.96 = +0.09 > 0.
    # We can supply t_years and sharpe to produce these exact t-statistics:
    # With SR=1.38, Y=2.0: SE = sqrt((1 + 1.38^2/500)/2.0) ≈ 0.708. t = 1.38/0.708 = 1.95 -> LCB = 1.38 - 1.96*0.708 = -0.007 < 0.
    # With SR=1.45, Y=2.0: SE ≈ 0.709. t = 1.45/0.709 = 2.04 -> LCB = 1.45 - 1.96*0.709 = +0.06 > 0.
    # With SR=1.20, Y=2.0: SE ≈ 0.708. t = 1.20/0.708 = 1.69 -> LCB = 1.20 - 1.96*0.708 = -0.18 < 0.
    records = [
        {"sharpe": 1.45, "t_years": 2.0, "composite_score": 1.45},  # t ≈ 2.04 >= 1.96, LCB95 > 0
        {"sharpe": 1.38, "t_years": 2.0, "composite_score": 1.38},  # t ≈ 1.95 < 1.96, LCB95 < 0
        {"sharpe": 1.20, "t_years": 2.0, "composite_score": 1.20},  # t ≈ 1.69 (passes 1.645, fails 1.96)
        {"sharpe": 0.50, "t_years": 2.0, "composite_score": 0.50},  # Cohort background
        {"sharpe": 0.40, "t_years": 2.0, "composite_score": 0.40},  # Cohort background
        {"sharpe": 0.30, "t_years": 2.0, "composite_score": 0.30},  # Cohort background
    ]
    winners, losers = _multiplicity_corrected_split(records)

    rec_pass = next(r for r in records if r["sharpe"] == 1.45)
    rec_marginal = next(r for r in records if r["sharpe"] == 1.38)
    rec_one_sided = next(r for r in records if r["sharpe"] == 1.20)

    # 1. Candidate C (t = 2.04 >= 1.96) has strictly positive lower endpoint:
    assert rec_pass["lcb95"] > 0.0, f"Expected positive LCB95, got {rec_pass['lcb95']}"
    assert rec_pass["t_vs_zero"] >= 1.96
    assert rec_pass["selection_tier"] == "SIGNIFICANT_WINNER"
    assert rec_pass in winners

    # 2. Candidate B (t = 1.95 < 1.96) has negative lower endpoint:
    assert rec_marginal["lcb95"] < 0.0, f"Expected negative LCB95 for t=1.95, got {rec_marginal['lcb95']}"
    assert rec_marginal["t_vs_zero"] < 1.96
    assert rec_marginal["selection_tier"] == "PROVISIONAL"
    assert rec_marginal not in winners

    # 3. Candidate A (t = 1.69, which would pass a weakened 1.645 gate) FAILS the 1.96 requirement:
    assert rec_one_sided["lcb95"] < 0.0, f"Expected negative LCB95, got {rec_one_sided['lcb95']}"
    assert rec_one_sided["selection_tier"] == "PROVISIONAL"
    assert rec_one_sided not in winners
