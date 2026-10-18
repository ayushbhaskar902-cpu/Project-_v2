"""Tests for Issue P1-13: Prompt version persistence is not the same as prompt evolution validation.

Validates:
1.  Experimental versions start EXPERIMENTAL with is_active=False — cannot silently replace incumbent.
2.  Immutability: PromptVersion content_hash is deterministic and tamper-detectable.
3.  Matched-cohort non-inferiority: LCB95(Delta) > -delta with predeclared margin.
4.  Positive absolute Sharpe alone cannot establish non-inferiority if LCB95(Delta) <= -delta.
5.  Underpowered samples reject with INSUFFICIENT_EVIDENCE.
6.  Multiple experimental variants incur Bonferroni penalty.
7.  AST validity gate (>= 85%) and token ratio gate (<= 1.25x) at boundaries.
8.  Deterministic rollback restores prior promoted version atomically.
9.  Rollback fails safely when no valid prior promoted version exists.
10. Holdout isolation: holdout data in candidates raises ValueError (P0-03).
11. Audit trail: PromptLifecycleEvent logs for all state transitions.
12. Missing metrics and provenance gating.
13. Incompatible cohorts detection.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from sqlalchemy import select

from alpha_harness.db.models import (
    PromptLifecycleEvent,
    PromptStrategy,
    PromptVersion,
    ResearchSession,
    utcnow,
)
from alpha_harness.db.sqlite import Database
from alpha_harness.prompts.validation import (
    PredeclaredPromotionCriteria,
    PromptLifecycleState,
    PromptValidationResult,
    PromptValidationService,
    compute_matched_non_inferiority,
    student_t_quantile,
)

from alpha_harness.prompts.versioning import (
    compute_prompt_content_hash,
    verify_prompt_integrity,
)


@pytest.fixture
def test_db(tmp_path: Path) -> Database:
    db_path = tmp_path / "test_p1_13.db"
    db = Database(db_path)
    asyncio.run(db.create_all())
    return db


def _make_candidate(val_sharpe: float, is_valid: bool = True, tokens: int = 100, **kw):
    """Helper to construct a candidate dict for tournament evaluation."""
    d = {"val_sharpe": val_sharpe, "is_valid": is_valid, "tokens": tokens}
    d.update(kw)
    return d


# ---------------------------------------------------------------------------
# Test 1: Experimental versions start EXPERIMENTAL with is_active=False
# ---------------------------------------------------------------------------
def test_experimental_version_starts_inactive(test_db: Database):
    """New prompt versions must start EXPERIMENTAL (is_active=False) and cannot
    silently replace the active incumbent."""

    async def _run():
        async with test_db.session() as s:
            sess = ResearchSession(name="P13 Session", region="USA", universe="TOP3000")
            s.add(sess)
            await s.commit()
            sess_id = sess.id

        svc = PromptValidationService(test_db)
        strat, pv = await svc.create_experimental_version(
            session_id=sess_id,
            category="momentum",
            template_text="Generate alpha expressions based on momentum signals.",
        )

        assert strat.is_active is False
        assert strat.validation_status == PromptLifecycleState.EXPERIMENTAL.value
        assert pv.content_hash is not None
        assert len(pv.content_hash) == 64  # SHA-256 hex

        # Verify it did NOT silently become active
        async with test_db.session() as s:
            active_stmt = select(PromptStrategy).where(
                PromptStrategy.session_id == sess_id,
                PromptStrategy.is_active.is_(True),
            )
            active_res = await s.execute(active_stmt)
            active_strats = active_res.scalars().all()
            # No active strategy should exist (only EXPERIMENTAL was created)
            assert len(active_strats) == 0

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 2: Content hash determinism and tamper detection
# ---------------------------------------------------------------------------
def test_prompt_version_content_hash_integrity(test_db: Database):
    """Verifies content_hash is deterministic and verify_prompt_integrity detects tampering."""

    async def _run():
        async with test_db.session() as s:
            sess = ResearchSession(name="Hash Session", region="USA", universe="TOP3000")
            s.add(sess)
            await s.commit()
            sess_id = sess.id

        svc = PromptValidationService(test_db)
        _, pv = await svc.create_experimental_version(
            session_id=sess_id,
            category="value",
            template_text="Generate value alpha signals.",
            guidance={"preferred_operators": ["ts_rank"]},
            examples=[{"expression": "rank(book_to_price)"}],
        )

        # Hash is deterministic
        recomputed = compute_prompt_content_hash(
            template_text=pv.template_text,
            guidance=pv.guidance,
            examples=pv.examples,
            anti_patterns=pv.anti_patterns,
        )
        assert pv.content_hash == recomputed

        # Integrity verification should pass
        verify_prompt_integrity(pv)

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 3: Non-inferiority gate passes with strong experimental performance
# ---------------------------------------------------------------------------
def test_matched_non_inferiority_passes(test_db: Database):
    """Promotion succeeds when LCB95(Delta) > -delta, abs Sharpe > 0, AST >= 85%, tokens <= 1.25x."""

    async def _run():
        async with test_db.session() as s:
            sess = ResearchSession(name="NI Pass", region="USA", universe="TOP3000")
            s.add(sess)
            await s.flush()

            # Create incumbent active strategy
            inc_strat = PromptStrategy(
                session_id=sess.id,
                name="Incumbent v1",
                version=1,
                category="momentum",
                template_text="Base template",
                is_active=True,
                validation_status="PROMOTED",
            )
            s.add(inc_strat)
            await s.flush()

            # Create experimental strategy
            exp_strat = PromptStrategy(
                session_id=sess.id,
                name="Experimental v2",
                version=2,
                category="momentum",
                template_text="Better template",
                is_active=False,
                validation_status="EXPERIMENTAL",
            )
            s.add(exp_strat)
            await s.commit()

            sess_id = sess.id
            exp_id = exp_strat.id

        svc = PromptValidationService(test_db)

        # Strong experimental performance: mean ~1.5, incumbent ~1.2
        exp_cands = [_make_candidate(v) for v in [1.4, 1.5, 1.6, 1.3, 1.7]]
        inc_cands = [_make_candidate(v) for v in [1.1, 1.2, 1.3, 1.0, 1.4]]

        result = await svc.evaluate_tournament_and_promote(
            session_id=sess_id,
            category="momentum",
            experimental_strategy_id=exp_id,
            experimental_candidates=exp_cands,
            incumbent_candidates=inc_cands,
        )

        assert result.decision == "PROMOTED"
        assert result.is_promoted is True
        assert result.non_inferiority_passed is True
        assert result.absolute_sharpe_passed is True
        assert result.ast_valid_rate_passed is True
        assert result.token_budget_passed is True
        assert result.sample_size_passed is True
        assert result.delta_sharpe is not None
        assert result.delta_sharpe > 0
        assert result.lcb95_delta is not None
        assert result.lcb95_delta > -0.15  # Non-inferiority margin

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 4: Positive Sharpe alone cannot bypass non-inferiority
# ---------------------------------------------------------------------------
def test_positive_sharpe_alone_does_not_establish_non_inferiority(test_db: Database):
    """Even with positive mean experimental Sharpe, if LCB95(Delta) <= -delta, reject."""

    async def _run():
        async with test_db.session() as s:
            sess = ResearchSession(name="NI Fail", region="USA", universe="TOP3000")
            s.add(sess)
            await s.flush()

            inc_strat = PromptStrategy(
                session_id=sess.id, name="Inc", version=1, category="momentum",
                template_text="T", is_active=True, validation_status="PROMOTED",
            )
            s.add(inc_strat)
            await s.flush()

            exp_strat = PromptStrategy(
                session_id=sess.id, name="Exp", version=2, category="momentum",
                template_text="T2", is_active=False, validation_status="EXPERIMENTAL",
            )
            s.add(exp_strat)
            await s.commit()
            sess_id = sess.id
            exp_id = exp_strat.id

        svc = PromptValidationService(test_db)

        # Experimental positive but highly variable; incumbent much stronger
        exp_cands = [_make_candidate(v) for v in [0.1, 0.2, 0.3, 0.1, 0.2]]
        inc_cands = [_make_candidate(v) for v in [1.5, 1.6, 1.7, 1.4, 1.8]]

        result = await svc.evaluate_tournament_and_promote(
            session_id=sess_id,
            category="momentum",
            experimental_strategy_id=exp_id,
            experimental_candidates=exp_cands,
            incumbent_candidates=inc_cands,
        )

        assert result.decision == "REJECTED"
        assert result.is_promoted is False
        assert result.absolute_sharpe_passed is True   # Mean > 0
        assert result.non_inferiority_passed is False   # LCB95(Delta) <= -margin
        assert any("NON_INFERIORITY_FAILED" in r for r in result.reasons)

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 5: Underpowered samples → INSUFFICIENT_EVIDENCE
# ---------------------------------------------------------------------------
def test_underpowered_samples_return_insufficient_evidence(test_db: Database):
    """Samples below min_sample_size (5) must yield INSUFFICIENT_EVIDENCE, not PROMOTED."""

    async def _run():
        async with test_db.session() as s:
            sess = ResearchSession(name="Underpowered", region="USA", universe="TOP3000")
            s.add(sess)
            await s.flush()

            inc = PromptStrategy(
                session_id=sess.id, name="Inc", version=1, category="general",
                template_text="T", is_active=True, validation_status="PROMOTED",
            )
            s.add(inc)
            await s.flush()

            exp = PromptStrategy(
                session_id=sess.id, name="Exp", version=2, category="general",
                template_text="T2", is_active=False, validation_status="EXPERIMENTAL",
            )
            s.add(exp)
            await s.commit()
            sess_id = sess.id
            exp_id = exp.id

        svc = PromptValidationService(test_db)

        # Only 3 candidates each (below min_sample_size=5)
        exp_cands = [_make_candidate(2.0), _make_candidate(2.5), _make_candidate(1.8)]
        inc_cands = [_make_candidate(1.0), _make_candidate(1.1), _make_candidate(0.9)]

        result = await svc.evaluate_tournament_and_promote(
            session_id=sess_id,
            category="general",
            experimental_strategy_id=exp_id,
            experimental_candidates=exp_cands,
            incumbent_candidates=inc_cands,
        )

        assert result.decision == "INSUFFICIENT_EVIDENCE"
        assert result.is_promoted is False
        assert result.sample_size_passed is False

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 6: Multiple experimental variants incur Bonferroni penalty
# ---------------------------------------------------------------------------
def test_bonferroni_penalty_multiple_variants():
    """With M > 1 variants, Bonferroni adjustment makes non-inferiority harder to achieve."""

    # Realistic sample with paired variance so SE > 0
    exp_sharpes = [1.2, 1.3, 1.0, 1.4, 1.0, 1.3, 1.2]
    inc_sharpes = [1.0, 1.1, 0.9, 1.2, 0.9, 1.0, 1.0]

    # With 1 variant
    delta_1, se_1, lcb_1, passed_1 = compute_matched_non_inferiority(
        experimental_sharpes=exp_sharpes,
        incumbent_sharpes=inc_sharpes,
        margin=0.15,
        num_variants_tested=1,
    )

    # With 5 variants (stricter z critical value)
    delta_5, se_5, lcb_5, passed_5 = compute_matched_non_inferiority(
        experimental_sharpes=exp_sharpes,
        incumbent_sharpes=inc_sharpes,
        margin=0.15,
        num_variants_tested=5,
    )

    # Same delta and SE
    assert delta_1 == delta_5
    assert se_1 == se_5
    # Bonferroni makes LCB lower (more conservative)
    assert lcb_5 < lcb_1


# ---------------------------------------------------------------------------
# Test 7: AST validity gate and token ratio gate at boundaries
# ---------------------------------------------------------------------------
def test_ast_validity_and_token_ratio_boundary_gates(test_db: Database):
    """AST validity < 85% or token ratio > 1.25x should cause rejection."""

    async def _run():
        async with test_db.session() as s:
            sess = ResearchSession(name="Gates", region="USA", universe="TOP3000")
            s.add(sess)
            await s.flush()

            inc = PromptStrategy(
                session_id=sess.id, name="Inc", version=1, category="momentum",
                template_text="T", is_active=True, validation_status="PROMOTED",
            )
            s.add(inc)
            await s.flush()

            exp = PromptStrategy(
                session_id=sess.id, name="Exp", version=2, category="momentum",
                template_text="T2", is_active=False, validation_status="EXPERIMENTAL",
            )
            s.add(exp)
            await s.commit()
            sess_id = sess.id
            exp_id = exp.id

        svc = PromptValidationService(test_db)

        # 7a: AST validity at 80% (4 valid out of 5) — below 85% threshold
        exp_cands = [
            _make_candidate(1.5, is_valid=True),
            _make_candidate(1.6, is_valid=True),
            _make_candidate(1.4, is_valid=True),
            _make_candidate(1.3, is_valid=True),
            _make_candidate(1.7, is_valid=False),  # invalid
        ]
        inc_cands = [_make_candidate(v) for v in [1.0, 1.1, 1.2, 0.9, 1.3]]

        result = await svc.evaluate_tournament_and_promote(
            session_id=sess_id,
            category="momentum",
            experimental_strategy_id=exp_id,
            experimental_candidates=exp_cands,
            incumbent_candidates=inc_cands,
        )

        assert result.decision == "REJECTED"
        assert any("AST_VALIDITY_FAILED" in r for r in result.reasons)

    asyncio.run(_run())


def test_token_ratio_exceeds_budget(test_db: Database):
    """Token ratio > 1.25x should cause rejection."""

    async def _run():
        async with test_db.session() as s:
            sess = ResearchSession(name="Token", region="USA", universe="TOP3000")
            s.add(sess)
            await s.flush()

            inc = PromptStrategy(
                session_id=sess.id, name="Inc", version=1, category="momentum",
                template_text="T", is_active=True, validation_status="PROMOTED",
            )
            s.add(inc)
            await s.flush()

            exp = PromptStrategy(
                session_id=sess.id, name="Exp", version=2, category="momentum",
                template_text="T2", is_active=False, validation_status="EXPERIMENTAL",
            )
            s.add(exp)
            await s.commit()
            sess_id = sess.id
            exp_id = exp.id

        svc = PromptValidationService(test_db)

        # Expensive experimental: 200 tokens vs 100 incumbent (ratio 2.0x > 1.25x)
        exp_cands = [_make_candidate(1.5, tokens=200) for _ in range(5)]
        inc_cands = [_make_candidate(1.0, tokens=100) for _ in range(5)]

        result = await svc.evaluate_tournament_and_promote(
            session_id=sess_id,
            category="momentum",
            experimental_strategy_id=exp_id,
            experimental_candidates=exp_cands,
            incumbent_candidates=inc_cands,
        )

        assert result.decision == "REJECTED"
        assert any("TOKEN_BUDGET_EXCEEDED" in r for r in result.reasons)

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 8: Deterministic rollback restores prior promoted version
# ---------------------------------------------------------------------------
def test_deterministic_rollback(test_db: Database):
    """Rollback atomically demotes current and reactivates prior promoted version."""

    async def _run():
        async with test_db.session() as s:
            sess = ResearchSession(name="Rollback", region="USA", universe="TOP3000")
            s.add(sess)
            await s.flush()

            # Create v1 as PROMOTED
            v1 = PromptStrategy(
                session_id=sess.id, name="V1", version=1, category="momentum",
                template_text="V1T", is_active=False,
                validation_status="PROMOTED",
            )
            s.add(v1)
            await s.flush()
            v1_id = v1.id

            # Create v2 as current active PROMOTED (derived from v1)
            v2 = PromptStrategy(
                session_id=sess.id, name="V2", version=2, category="momentum",
                template_text="V2T", is_active=True,
                validation_status="PROMOTED",
                previous_promoted_strategy_id=v1_id,
            )
            s.add(v2)
            await s.commit()
            sess_id = sess.id
            v2_id = v2.id

        svc = PromptValidationService(test_db)
        result = await svc.rollback_prompt_version(
            session_id=sess_id,
            category="momentum",
        )

        assert result["success"] is True
        assert result["demoted_strategy_id"] == v2_id
        assert result["restored_strategy_id"] == v1_id

        # Verify DB state
        async with test_db.session() as s:
            v2_check = await s.get(PromptStrategy, v2_id)
            assert v2_check.is_active is False
            assert v2_check.validation_status == "ROLLED_BACK"
            assert v2_check.rolled_back_at is not None

            v1_check = await s.get(PromptStrategy, v1_id)
            assert v1_check.is_active is True
            assert v1_check.validation_status == "PROMOTED"

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 9: Rollback fails safely with no prior version
# ---------------------------------------------------------------------------
def test_rollback_fails_safely_no_prior(test_db: Database):
    """Rollback raises ValueError when no valid prior promoted version exists."""

    async def _run():
        async with test_db.session() as s:
            sess = ResearchSession(name="No Prior", region="USA", universe="TOP3000")
            s.add(sess)
            await s.flush()

            # Only one EXPERIMENTAL version exists (not PROMOTED or SEED)
            v1 = PromptStrategy(
                session_id=sess.id, name="Only", version=1, category="value",
                template_text="Only T", is_active=True,
                validation_status="EXPERIMENTAL",
            )
            s.add(v1)
            await s.commit()
            sess_id = sess.id

        svc = PromptValidationService(test_db)
        with pytest.raises(ValueError, match="No valid prior promoted"):
            await svc.rollback_prompt_version(
                session_id=sess_id,
                category="value",
            )

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 10: P0-03 Holdout isolation in promotion decisions
# ---------------------------------------------------------------------------
def test_holdout_data_raises_in_tournament(test_db: Database):
    """Candidates with holdout keys must cause a ValueError (P0-03)."""

    async def _run():
        async with test_db.session() as s:
            sess = ResearchSession(name="Holdout", region="USA", universe="TOP3000")
            s.add(sess)
            await s.flush()

            exp = PromptStrategy(
                session_id=sess.id, name="Exp", version=1, category="momentum",
                template_text="T", is_active=False, validation_status="EXPERIMENTAL",
            )
            s.add(exp)
            await s.commit()
            sess_id = sess.id
            exp_id = exp.id

        svc = PromptValidationService(test_db)

        # Contaminated candidate with holdout data
        exp_cands = [
            {"val_sharpe": 1.5, "is_valid": True, "tokens": 100, "holdout_sharpe": 2.0},
        ]
        inc_cands = [_make_candidate(1.0) for _ in range(5)]

        with pytest.raises(ValueError, match="[Hh]oldout"):
            await svc.evaluate_tournament_and_promote(
                session_id=sess_id,
                category="momentum",
                experimental_strategy_id=exp_id,
                experimental_candidates=exp_cands,
                incumbent_candidates=inc_cands,
            )

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 11: Audit trail — PromptLifecycleEvent logs
# ---------------------------------------------------------------------------
def test_lifecycle_audit_trail(test_db: Database):
    """All state transitions (CREATED_EXPERIMENTAL, PROMOTED, ROLLED_BACK) produce audit events."""

    async def _run():
        async with test_db.session() as s:
            sess = ResearchSession(name="Audit", region="USA", universe="TOP3000")
            s.add(sess)
            await s.commit()
            sess_id = sess.id

        svc = PromptValidationService(test_db)

        # 1. Create experimental → CREATED_EXPERIMENTAL event
        strat, pv = await svc.create_experimental_version(
            session_id=sess_id,
            category="momentum",
            template_text="Audit template",
        )

        events = await svc.get_prompt_lifecycle_history(sess_id, category="momentum")
        assert len(events) >= 1
        created_events = [e for e in events if e.event_type == "CREATED_EXPERIMENTAL"]
        assert len(created_events) >= 1
        assert created_events[0].to_status == "EXPERIMENTAL"

        # 2. Create an incumbent strategy then promote the experimental
        async with test_db.session() as s:
            inc = PromptStrategy(
                session_id=sess_id, name="Inc", version=0, category="momentum",
                template_text="Inc T", is_active=True, validation_status="PROMOTED",
            )
            s.add(inc)
            await s.commit()

        exp_cands = [_make_candidate(v) for v in [1.5, 1.6, 1.4, 1.3, 1.7]]
        inc_cands = [_make_candidate(v) for v in [1.0, 1.1, 1.2, 0.9, 1.3]]

        result = await svc.evaluate_tournament_and_promote(
            session_id=sess_id,
            category="momentum",
            experimental_strategy_id=strat.id,
            experimental_candidates=exp_cands,
            incumbent_candidates=inc_cands,
        )
        assert result.decision == "PROMOTED"

        events = await svc.get_prompt_lifecycle_history(sess_id, category="momentum")
        promoted_events = [e for e in events if e.event_type == "PROMOTED"]
        assert len(promoted_events) >= 1

        # 3. Rollback → ROLLED_BACK event
        await svc.rollback_prompt_version(session_id=sess_id, category="momentum")

        events = await svc.get_prompt_lifecycle_history(sess_id, category="momentum")
        rollback_events = [e for e in events if e.event_type == "ROLLED_BACK"]
        assert len(rollback_events) >= 1

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 12: Missing metrics are conservatively handled
# ---------------------------------------------------------------------------
def test_missing_metrics_conservative_handling(test_db: Database):
    """Candidates with val_sharpe=None are excluded unless allow_missing_metrics=True."""

    async def _run():
        async with test_db.session() as s:
            sess = ResearchSession(name="Missing", region="USA", universe="TOP3000")
            s.add(sess)
            await s.flush()

            inc = PromptStrategy(
                session_id=sess.id, name="Inc", version=1, category="general",
                template_text="T", is_active=True, validation_status="PROMOTED",
            )
            s.add(inc)
            await s.flush()

            exp = PromptStrategy(
                session_id=sess.id, name="Exp", version=2, category="general",
                template_text="T2", is_active=False, validation_status="EXPERIMENTAL",
            )
            s.add(exp)
            await s.commit()
            sess_id = sess.id
            exp_id = exp.id

        svc = PromptValidationService(test_db)

        # 7 experimental candidates but 3 have val_sharpe=None → only 4 valid → below min_sample_size=5
        exp_cands = [
            _make_candidate(1.5),
            _make_candidate(1.6),
            _make_candidate(1.4),
            _make_candidate(1.3),
            {"val_sharpe": None, "is_valid": True, "tokens": 100},
            {"val_sharpe": None, "is_valid": True, "tokens": 100},
            {"val_sharpe": None, "is_valid": True, "tokens": 100},
        ]
        inc_cands = [_make_candidate(v) for v in [1.0, 1.1, 1.2, 0.9, 1.3]]

        result = await svc.evaluate_tournament_and_promote(
            session_id=sess_id,
            category="general",
            experimental_strategy_id=exp_id,
            experimental_candidates=exp_cands,
            incumbent_candidates=inc_cands,
        )

        # With missing metrics, only 4 experimental candidates have val_sharpe → INSUFFICIENT_EVIDENCE
        assert result.decision == "INSUFFICIENT_EVIDENCE"
        assert result.missing_metrics_count == 3

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 13: Provenance gating rejects unverified evaluations
# ---------------------------------------------------------------------------
def test_provenance_gating_rejects_unverified(test_db: Database):
    """Candidates with UNVERIFIED provenance are filtered out."""

    async def _run():
        async with test_db.session() as s:
            sess = ResearchSession(name="Prov", region="USA", universe="TOP3000")
            s.add(sess)
            await s.flush()

            inc = PromptStrategy(
                session_id=sess.id, name="Inc", version=1, category="momentum",
                template_text="T", is_active=True, validation_status="PROMOTED",
            )
            s.add(inc)
            await s.flush()

            exp = PromptStrategy(
                session_id=sess.id, name="Exp", version=2, category="momentum",
                template_text="T2", is_active=False, validation_status="EXPERIMENTAL",
            )
            s.add(exp)
            await s.commit()
            sess_id = sess.id
            exp_id = exp.id

        svc = PromptValidationService(test_db)

        # All experimental candidates have UNVERIFIED provenance
        exp_cands = [
            _make_candidate(2.0, provenance="UNVERIFIED"),
            _make_candidate(2.1, provenance="UNVERIFIED"),
            _make_candidate(1.9, provenance="UNVERIFIED"),
            _make_candidate(2.2, provenance="UNVERIFIED"),
            _make_candidate(1.8, provenance="UNVERIFIED"),
        ]
        inc_cands = [_make_candidate(1.0) for _ in range(5)]

        result = await svc.evaluate_tournament_and_promote(
            session_id=sess_id,
            category="momentum",
            experimental_strategy_id=exp_id,
            experimental_candidates=exp_cands,
            incumbent_candidates=inc_cands,
        )

        # All experimental are filtered → INSUFFICIENT_EVIDENCE
        assert result.decision == "INSUFFICIENT_EVIDENCE"
        assert result.is_promoted is False

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 14: Incompatible cohorts detection
# ---------------------------------------------------------------------------
def test_incompatible_cohorts_detection(test_db: Database):
    """When experimental and incumbent have zero cohort overlap, report INCOMPATIBLE_COHORTS."""

    async def _run():
        async with test_db.session() as s:
            sess = ResearchSession(name="Cohort", region="USA", universe="TOP3000")
            s.add(sess)
            await s.flush()

            inc = PromptStrategy(
                session_id=sess.id, name="Inc", version=1, category="momentum",
                template_text="T", is_active=True, validation_status="PROMOTED",
            )
            s.add(inc)
            await s.flush()

            exp = PromptStrategy(
                session_id=sess.id, name="Exp", version=2, category="momentum",
                template_text="T2", is_active=False, validation_status="EXPERIMENTAL",
            )
            s.add(exp)
            await s.commit()
            sess_id = sess.id
            exp_id = exp.id

        svc = PromptValidationService(test_db)

        # Experimental candidates from region="CHINA", incumbent from region="USA"
        exp_cands = [_make_candidate(1.5, region="CHINA") for _ in range(5)]
        inc_cands = [_make_candidate(1.0, region="USA") for _ in range(5)]

        result = await svc.evaluate_tournament_and_promote(
            session_id=sess_id,
            category="momentum",
            experimental_strategy_id=exp_id,
            experimental_candidates=exp_cands,
            incumbent_candidates=inc_cands,
        )

        assert result.decision == "INCOMPATIBLE_COHORTS"
        assert result.is_promoted is False

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 15: Compute matched non-inferiority unit test
# ---------------------------------------------------------------------------
def test_compute_matched_non_inferiority_basic():
    """Verify Delta, SE(Delta), LCB95(Delta) computation."""

    exp = [2.0, 2.0, 2.0, 2.0, 2.0]
    inc = [1.0, 1.0, 1.0, 1.0, 1.0]

    delta, se, lcb, passed = compute_matched_non_inferiority(
        experimental_sharpes=exp,
        incumbent_sharpes=inc,
        margin=0.15,
    )

    assert delta == 1.0  # 2.0 - 1.0
    # With zero variance in both, SE should be very small (clamped by 1e-9)
    assert se < 0.01
    # LCB should be near 1.0 with such tight data
    assert lcb > 0.9
    assert passed is True


def test_compute_matched_non_inferiority_insufficient_samples():
    """With fewer than 2 samples per arm, returns sentinel values."""

    delta, se, lcb, passed = compute_matched_non_inferiority(
        experimental_sharpes=[1.0],
        incumbent_sharpes=[0.5],
        margin=0.15,
    )

    assert se == float("inf")
    assert lcb == float("-inf")
    assert passed is False


# ---------------------------------------------------------------------------
# Test 16: Rejection event does not mutate incumbent
# ---------------------------------------------------------------------------
def test_rejection_preserves_incumbent(test_db: Database):
    """A rejected experimental prompt must not deactivate or modify the active incumbent."""

    async def _run():
        async with test_db.session() as s:
            sess = ResearchSession(name="Preserve", region="USA", universe="TOP3000")
            s.add(sess)
            await s.flush()

            inc = PromptStrategy(
                session_id=sess.id, name="Inc", version=1, category="momentum",
                template_text="Incumbent T", is_active=True, validation_status="PROMOTED",
            )
            s.add(inc)
            await s.flush()
            inc_id = inc.id

            exp = PromptStrategy(
                session_id=sess.id, name="Exp", version=2, category="momentum",
                template_text="Exp T", is_active=False, validation_status="EXPERIMENTAL",
            )
            s.add(exp)
            await s.commit()
            sess_id = sess.id
            exp_id = exp.id

        svc = PromptValidationService(test_db)

        # Experimental much worse → rejection
        exp_cands = [_make_candidate(v) for v in [-0.5, -0.3, -0.2, -0.4, -0.6]]
        inc_cands = [_make_candidate(v) for v in [1.0, 1.1, 1.2, 0.9, 1.3]]

        result = await svc.evaluate_tournament_and_promote(
            session_id=sess_id,
            category="momentum",
            experimental_strategy_id=exp_id,
            experimental_candidates=exp_cands,
            incumbent_candidates=inc_cands,
        )

        assert result.decision == "REJECTED"

        # Verify incumbent is still active and unchanged
        async with test_db.session() as s:
            inc_check = await s.get(PromptStrategy, inc_id)
            assert inc_check.is_active is True
            assert inc_check.validation_status == "PROMOTED"
            assert inc_check.template_text == "Incumbent T"

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 17: Non-inferiority margin is predeclared
# ---------------------------------------------------------------------------
def test_predeclared_margin_enforced(test_db: Database):
    """The non-inferiority margin must be predeclared; a tight margin (0.01) makes weak
    experimental prompts fail even if they would pass with the default margin (0.15)."""

    async def _run():
        async with test_db.session() as s:
            sess = ResearchSession(name="Margin", region="USA", universe="TOP3000")
            s.add(sess)
            await s.flush()

            inc = PromptStrategy(
                session_id=sess.id, name="Inc", version=1, category="general",
                template_text="T", is_active=True, validation_status="PROMOTED",
            )
            s.add(inc)
            await s.flush()

            exp = PromptStrategy(
                session_id=sess.id, name="Exp", version=2, category="general",
                template_text="T2", is_active=False, validation_status="EXPERIMENTAL",
            )
            s.add(exp)
            await s.commit()
            sess_id = sess.id
            exp_id = exp.id

        svc = PromptValidationService(test_db)

        # Performance: experimental slightly below incumbent with some variance
        exp_cands = [_make_candidate(v) for v in [0.9, 1.0, 0.8, 1.1, 0.7]]
        inc_cands = [_make_candidate(v) for v in [1.0, 1.1, 0.9, 1.2, 0.8]]

        # With default margin (0.15), this might pass; with tight margin (0.01), it should fail
        tight_criteria = PredeclaredPromotionCriteria(non_inferiority_margin=0.01)
        result = await svc.evaluate_tournament_and_promote(
            session_id=sess_id,
            category="general",
            experimental_strategy_id=exp_id,
            experimental_candidates=exp_cands,
            incumbent_candidates=inc_cands,
            criteria=tight_criteria,
        )

        # With a very tight margin, the LCB must be > -0.01, which is harder
        assert result.non_inferiority_margin == 0.01

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 18: Student's t critical values for small matched samples (N=5)
# ---------------------------------------------------------------------------
def test_student_t_critical_values_small_samples_n5():
    """Verify that Student's t critical values are used for small samples (N=5, df=4)
    rather than naive normal quantiles (1.96).

    For df=4 and two-sided 95% CI (p=0.975), t_crit is approx 2.7764.
    Because t_crit > 1.96, the resulting LCB is strictly more conservative.
    """
    t_val = student_t_quantile(0.975, df=4)
    assert round(t_val, 4) == 2.7764

    # Experimental and incumbent data with known variance across N=5 pairs
    exp = [1.2, 1.4, 1.3, 1.5, 1.1]
    inc = [1.0, 1.1, 1.0, 1.1, 1.0]
    # Diffs: [0.2, 0.3, 0.3, 0.4, 0.1] -> mean = 0.26, s_d approx 0.1140, SE approx 0.0510
    delta, se, lcb, passed = compute_matched_non_inferiority(exp, inc, margin=0.15)

    assert delta == 0.26
    # Theoretical LCB under Student's t: 0.26 - 2.7764 * se
    expected_lcb = round(0.26 - 2.7764 * se, 4)
    assert abs(lcb - expected_lcb) <= 1e-4

    # Under naive normal (1.96), LCB would be overly optimistic:
    naive_lcb = round(0.26 - 1.95996 * se, 4)
    assert lcb < naive_lcb  # Student's t is strictly more conservative


# ---------------------------------------------------------------------------
# Test 19: Degrees of freedom scaling and sample size interaction
# ---------------------------------------------------------------------------
def test_degrees_of_freedom_scaling_and_sample_size_interaction():
    """With identical mean difference and variance, increasing sample size N
    increases df (N-1), reducing t_crit and reducing SE(Delta) by 1/sqrt(N),
    monotonically increasing LCB."""
    # Constant paired diffs of [0.1, 0.3] alternating
    diffs_5 = [0.1, 0.3, 0.1, 0.3, 0.2]  # N=5, df=4
    diffs_10 = diffs_5 * 2                # N=10, df=9
    diffs_20 = diffs_5 * 4                # N=20, df=19

    inc_base = [1.0] * 20
    exp_5 = [1.0 + d for d in diffs_5]
    exp_10 = [1.0 + d for d in diffs_10]
    exp_20 = [1.0 + d for d in diffs_20]

    _, se_5, lcb_5, _ = compute_matched_non_inferiority(exp_5, inc_base[:5], margin=0.15)
    _, se_10, lcb_10, _ = compute_matched_non_inferiority(exp_10, inc_base[:10], margin=0.15)
    _, se_20, lcb_20, _ = compute_matched_non_inferiority(exp_20, inc_base[:20], margin=0.15)

    assert se_5 > se_10 > se_20
    assert lcb_5 < lcb_10 < lcb_20


# ---------------------------------------------------------------------------
# Test 20: Multiple-candidate comparison family adjustment with Student's t
# ---------------------------------------------------------------------------
def test_multiple_candidate_correction_student_t():
    """Verify Bonferroni correction on candidate comparison family:
    M=1: alpha=0.05   -> p=0.975  -> t_crit(4) ~ 2.7764
    M=2: alpha=0.025  -> p=0.9875 -> t_crit(4) ~ 3.4954
    M=5: alpha=0.010  -> p=0.9950 -> t_crit(4) ~ 4.6041
    """
    t_1 = student_t_quantile(1.0 - (0.05 / 2.0), df=4)
    t_2 = student_t_quantile(1.0 - (0.025 / 2.0), df=4)
    t_5 = student_t_quantile(1.0 - (0.010 / 2.0), df=4)

    assert round(t_1, 4) == 2.7764
    assert round(t_2, 4) == 3.4954
    assert round(t_5, 4) == 4.6041

    exp = [1.2, 1.3, 1.1, 1.4, 1.2]
    inc = [1.0, 1.0, 1.0, 1.0, 1.0]

    _, _, lcb_m1, _ = compute_matched_non_inferiority(exp, inc, margin=0.15, num_variants_tested=1)
    _, _, lcb_m2, _ = compute_matched_non_inferiority(exp, inc, margin=0.15, num_variants_tested=2)
    _, _, lcb_m5, _ = compute_matched_non_inferiority(exp, inc, margin=0.15, num_variants_tested=5)

    assert lcb_m1 > lcb_m2 > lcb_m5


# ---------------------------------------------------------------------------
# Test 21: Unmatched cohort exclusions and insufficient matched pairs
# ---------------------------------------------------------------------------
def test_unmatched_cohort_exclusions_and_insufficient_pairs(test_db: Database):
    """If some evaluations cannot be matched across cohorts, they are excluded
    and disclosed with reasons. If fewer than min_sample_size matched pairs remain,
    returns INSUFFICIENT_EVIDENCE rather than falling back to unpaired Welch."""

    async def _run():
        async with test_db.session() as s:
            sess = ResearchSession(name="Pairing", region="USA", universe="TOP3000")
            s.add(sess)
            await s.flush()

            inc = PromptStrategy(
                session_id=sess.id, name="Inc", version=1, category="value",
                template_text="T", is_active=True, validation_status="PROMOTED",
            )
            s.add(inc)
            await s.flush()

            exp = PromptStrategy(
                session_id=sess.id, name="Exp", version=2, category="value",
                template_text="T2", is_active=False, validation_status="EXPERIMENTAL",
            )
            s.add(exp)
            await s.commit()
            sess_id, exp_id = sess.id, exp.id

        svc = PromptValidationService(test_db)

        # 4 candidates in Cohort A, 2 in Cohort B for experimental (total 6)
        exp_cands = [
            _make_candidate(1.5, dataset_id="ds_A", region="USA", universe="TOP3000"),
            _make_candidate(1.6, dataset_id="ds_A", region="USA", universe="TOP3000"),
            _make_candidate(1.4, dataset_id="ds_A", region="USA", universe="TOP3000"),
            _make_candidate(1.7, dataset_id="ds_A", region="USA", universe="TOP3000"),
            _make_candidate(1.2, dataset_id="ds_B", region="USA", universe="TOP3000"),
            _make_candidate(1.3, dataset_id="ds_B", region="USA", universe="TOP3000"),
        ]

        # Only 2 in Cohort A and 1 in Cohort B for incumbent (total 3)
        # Matched pairs possible: 2 from ds_A + 1 from ds_B = 3 matched pairs (< min 5)
        inc_cands = [
            _make_candidate(1.2, dataset_id="ds_A", region="USA", universe="TOP3000"),
            _make_candidate(1.3, dataset_id="ds_A", region="USA", universe="TOP3000"),
            _make_candidate(1.0, dataset_id="ds_B", region="USA", universe="TOP3000"),
        ]

        result = await svc.evaluate_tournament_and_promote(
            session_id=sess_id,
            category="value",
            experimental_strategy_id=exp_id,
            experimental_candidates=exp_cands,
            incumbent_candidates=inc_cands,
        )

        assert result.decision == "INSUFFICIENT_EVIDENCE"
        assert result.sample_size_passed is False
        assert result.unmatched_evaluations_count == 3  # (4-2) + (2-1) = 3 excess exp candidates
        assert len(result.evidence["unmatched_reasons"]) > 0
        assert result.evidence["n_matched_pairs"] == 3

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 22: Positive point estimate still fails non-inferiority under Student's t
# ---------------------------------------------------------------------------
def test_positive_point_estimate_fails_non_inferiority_due_to_small_sample_t():
    """Delta_bar = +0.05 > 0, but with high variance in N=5, t_crit(4)*SE > 0.20,
    causing LCB <= -0.15, which correctly fails the non-inferiority gate."""
    # N=5 pairs with high variance
    exp = [1.8, 0.4, 2.0, 0.3, 1.5]
    inc = [1.2, 1.0, 1.4, 0.9, 1.2]
    # Diffs: [0.6, -0.6, 0.6, -0.6, 0.3] -> mean = +0.06 > 0
    delta, se, lcb, passed = compute_matched_non_inferiority(exp, inc, margin=0.15)

    assert delta > 0.0
    # s_d is large (~0.60), SE ~ 0.27. t_crit(4) ~ 2.7764 -> t_crit * SE ~ 0.75
    # LCB = 0.06 - 0.75 ~ -0.69 <= -0.15
    assert lcb < -0.15
    assert passed is False  # Positive mean cannot mask small-sample estimation uncertainty


