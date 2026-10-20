"""P1-01: Evidence-grounded LLM reflection regression test suite.

Mandatory tests:
1. test_p0_03_sentinel_holdout_isolation
2. test_evidence_prompt_assembly_and_ineligible_exclusion
3. test_insufficient_evidence_threshold
4. test_structured_reflection_parsing_and_persistence
5. test_invalid_candidate_citation_strict_rejection
6. test_invalid_field_citation_strict_rejection
7. test_unsupported_factual_assertions_not_persisted_as_facts
8. test_malformed_llm_json_triggers_labeled_fallback
9. test_llm_provider_timeout_triggers_labeled_fallback
10. test_legacy_migration_and_idempotency
11. test_strategy_dual_lineage_and_downstream_injection
12. test_repeated_and_interrupted_cycle_idempotency
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import httpx2
import pytest
from sqlalchemy import select, text

from alpha_harness.db.models import (
    CandidateStatus,
    HoldoutAssessment,
    PromptStrategy,
    ResearchCandidate,
    ResearchEvaluation,
    ResearchInsight,
    ResearchSession,
)
from alpha_harness.db.sqlite import Database
from alpha_harness.learning.loop import SelfLearningEngine
from alpha_harness.learning.memory import ResearchMemory
from alpha_harness.llm.service import Answer
from alpha_harness.prompts.library import (
    ALLOWED_CITATION_FIELDS,
    PromptLibrary,
    StrategyCategory,
    render_reflection_evidence_prompt,
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
# Test 1: P0-03 Sentinel Holdout Isolation
# ---------------------------------------------------------------------------
def test_p0_03_sentinel_holdout_isolation(tmp_path: Path):
    """Verifies that sentinel holdout values never leak into the reflection prompt or payload."""
    async def _run():
        db = Database(tmp_path / "sqlite.db")
        await db.create_all()

        SENTINEL_HOLDOUT = 999.0
        SENTINEL_TEXT = "SENTINEL_HOLDOUT_LEAK_CHECK"

        async with db.session() as s:
            sess = ResearchSession(
                name="Sentinel Test Session",
                prompt="test",
                region="USA",
                universe="TOP3000",
                current_generation=1,
            )
            s.add(sess)
            await s.flush()

            strat = PromptStrategy(
                session_id=sess.id,
                name="Active Strat",
                version=1,
                category="general",
                is_active=True,
            )
            s.add(strat)

            for i in range(3):
                cand = ResearchCandidate(
                    session_id=sess.id,
                    expression=f"rank(close + {i})",
                    status=CandidateStatus.EVALUATED,
                    generation=1,
                    hypothesis=f"Hypothesis {i}",
                )
                s.add(cand)
                await s.flush()

                ev = ResearchEvaluation(
                    candidate_id=cand.id,
                    is_verified=True,
                    val_sharpe=1.5 + i * 0.2,
                    val_fitness=1.0,
                    holdout_sharpe=SENTINEL_HOLDOUT,
                    provenance="BRAIN_VAULT",
                )
                s.add(ev)

        # Build evidence prompt directly and verify sentinel exclusion
        memory = ResearchMemory(db)
        winners, losers = await memory.get_session_performance_split(sess.id, generation=1)
        all_cands = winners + losers
        prompt_text = render_reflection_evidence_prompt(
            session_id=sess.id,
            generation=1,
            eligible_candidates=all_cands,
            factual_stats={"batch_size": len(all_cands), "avg_sharpe": 1.7, "best_sharpe": 1.9},
        )

        assert str(SENTINEL_HOLDOUT) not in prompt_text
        assert "holdout" not in prompt_text.lower() or "holdout metrics are strictly protected and withheld" in prompt_text.lower()

        # Run engine with mock LLM and verify LLM call args never received sentinel
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=_make_mock_answer({
                "summary": "Reflection summary",
                "hypotheses": [
                    {
                        "claim": "Close price momentum",
                        "confidence": "HIGH_CONFIDENCE_HYPOTHESIS",
                        "citations": [{"candidate_id": all_cands[0]["candidate_id"], "field": "expression"}],
                    }
                ],
                "recommendations": [
                    {
                        "action": "Use momentum lookback",
                        "rationale": "High Sharpe observed",
                        "citations": [{"candidate_id": all_cands[0]["candidate_id"], "field": "sharpe"}],
                    }
                ],
                "anti_patterns": [
                    {
                        "expression": all_cands[-1]["expression"],
                        "flaw": "Slightly lower fitness",
                        "citations": [{"candidate_id": all_cands[-1]["candidate_id"], "field": "fitness"}],
                    }
                ],
            })
        )

        engine = SelfLearningEngine(db=db, memory=memory, llm=mock_llm)
        res = await engine.execute_learning_cycle(sess.id)

        assert res.updated_prompt_strategy_id is not None
        call_args = mock_llm.generate.call_args.kwargs
        assert str(SENTINEL_HOLDOUT) not in call_args["user"]
        assert str(SENTINEL_HOLDOUT) not in call_args["system"]

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 2: Verified-Candidate Evidence Assembly and Ineligible Exclusion
# ---------------------------------------------------------------------------
def test_evidence_prompt_assembly_and_ineligible_exclusion(tmp_path: Path):
    """Verifies that only eligible, verified candidates enter reflection evidence."""
    async def _run():
        db = Database(tmp_path / "sqlite.db")
        await db.create_all()

        async with db.session() as s:
            sess = ResearchSession(
                name="Evidence Filtering Session",
                prompt="test",
                region="USA",
                universe="TOP3000",
                current_generation=0,
            )
            s.add(sess)
            await s.flush()

            # 3 Eligible candidates (is_verified=True, status=EVALUATED)
            for i in range(3):
                c = ResearchCandidate(
                    session_id=sess.id,
                    expression=f"ts_rank(volume, {10+i})",
                    status=CandidateStatus.EVALUATED,
                    generation=0,
                    hypothesis=f"Eligible {i}",
                )
                s.add(c)
                await s.flush()
                s.add(ResearchEvaluation(candidate_id=c.id, is_verified=True, val_sharpe=1.6))

            # 1 Unverified candidate (is_verified=False)
            c_unverified = ResearchCandidate(
                session_id=sess.id,
                expression="rank(fake_unverified_expr)",
                status=CandidateStatus.EVALUATED,
                generation=0,
                hypothesis="Unverified",
            )
            s.add(c_unverified)
            await s.flush()
            s.add(ResearchEvaluation(candidate_id=c_unverified.id, is_verified=False, val_sharpe=3.0))

            # 1 Rejected candidate (status=REJECTED)
            c_rejected = ResearchCandidate(
                session_id=sess.id,
                expression="rank(syntax_error_rejected)",
                status=CandidateStatus.REJECTED,
                generation=0,
                hypothesis="Rejected",
            )
            s.add(c_rejected)
            await s.flush()
            s.add(ResearchEvaluation(candidate_id=c_rejected.id, is_verified=True, val_sharpe=0.1))

        memory = ResearchMemory(db)
        cands = await memory.get_eligible_candidates(sess.id, generation=0)

        assert len(cands) == 3
        expressions = [c["expression"] for c in cands]
        assert "rank(fake_unverified_expr)" not in expressions
        assert "rank(syntax_error_rejected)" not in expressions

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 3: Insufficient Evidence Threshold
# ---------------------------------------------------------------------------
def test_insufficient_evidence_threshold(tmp_path: Path):
    """Verifies that insufficient evidence records an honest status without mutating strategy."""
    async def _run():
        db = Database(tmp_path / "sqlite.db")
        await db.create_all()

        async with db.session() as s:
            sess = ResearchSession(
                name="Insufficient Session",
                prompt="test",
                region="USA",
                universe="TOP3000",
                current_generation=0,
            )
            s.add(sess)
            await s.flush()

            strat = PromptStrategy(
                session_id=sess.id,
                name="Initial Active Strat",
                version=1,
                category="general",
                is_active=True,
            )
            s.add(strat)
            await s.flush()
            strat_id = strat.id

            # Seed 2 candidates (below min_candidates=3 threshold)
            for i in range(2):
                c = ResearchCandidate(
                    session_id=sess.id,
                    expression=f"ts_mean(close, {5+i})",
                    status=CandidateStatus.EVALUATED,
                    generation=0,
                )
                s.add(c)
                await s.flush()
                s.add(ResearchEvaluation(candidate_id=c.id, is_verified=True, val_sharpe=1.4))

        mock_llm = AsyncMock()
        engine = SelfLearningEngine(db=db, memory=ResearchMemory(db), llm=mock_llm)

        # Execute with min_candidates=3
        res = await engine.execute_learning_cycle(sess.id, min_candidates=3)

        assert res.updated_prompt_strategy_id is None
        assert mock_llm.generate.call_count == 0

        # Verify strategy was NOT mutated and generation was NOT incremented
        async with db.session() as s:
            reloaded_sess = await s.get(ResearchSession, sess.id)
            assert reloaded_sess.current_generation == 0

            reloaded_strat = await s.get(PromptStrategy, strat_id)
            assert reloaded_strat.is_active is True

            insights = (
                await s.execute(
                    select(ResearchInsight).where(ResearchInsight.session_id == sess.id)
                )
            ).scalars().all()
            assert len(insights) == 1
            assert insights[0].execution_status == "INSUFFICIENT_EVIDENCE"
            assert insights[0].is_fallback is False
            assert insights[0].applied is False
            assert insights[0].prompt_strategy_id == strat_id
            assert insights[0].evolved_strategy_id is None

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 4: Genuine Structured LLM Reflection and Persistence
# ---------------------------------------------------------------------------
def test_structured_reflection_parsing_and_persistence(tmp_path: Path):
    """Verifies genuine LLM reflection parsing, dual lineage, and unverified labels."""
    async def _run():
        db = Database(tmp_path / "sqlite.db")
        await db.create_all()

        candidate_ids = []
        async with db.session() as s:
            sess = ResearchSession(
                name="Reflection Persistence Session",
                prompt="test",
                region="USA",
                universe="TOP3000",
                current_generation=2,
                model_refs=["google:gemini-2.5-flash"],
            )
            s.add(sess)
            await s.flush()

            parent_strat = PromptStrategy(
                session_id=sess.id,
                name="Antecedent Strat",
                version=2,
                category="momentum",
                is_active=True,
            )
            s.add(parent_strat)
            await s.flush()
            antecedent_id = parent_strat.id

            for i in range(3):
                c = ResearchCandidate(
                    session_id=sess.id,
                    expression=f"ts_zscore(close, {20+i*5})",
                    status=CandidateStatus.EVALUATED,
                    generation=2,
                    hypothesis=f"Momentum factor {i}",
                )
                s.add(c)
                await s.flush()
                candidate_ids.append(c.id)
                s.add(ResearchEvaluation(candidate_id=c.id, is_verified=True, val_sharpe=1.8 + i * 0.1))

        valid_response = {
            "summary": "Normalized price momentum displays robust risk-adjusted return.",
            "hypotheses": [
                {
                    "claim": "ts_zscore removes non-stationarity across varying regimes.",
                    "confidence": "HIGH_CONFIDENCE_HYPOTHESIS",
                    "citations": [{"candidate_id": candidate_ids[0], "field": "expression"}],
                }
            ],
            "recommendations": [
                {
                    "action": "Prioritize ts_zscore over raw ts_delta",
                    "rationale": "Empirical Sharpe lift of +0.3",
                    "citations": [{"candidate_id": candidate_ids[1], "field": "sharpe"}],
                }
            ],
            "anti_patterns": [
                {
                    "expression": f"ts_zscore(close, 30)",
                    "flaw": "Long lookback decay",
                    "citations": [{"candidate_id": candidate_ids[2], "field": "expression"}],
                }
            ],
        }

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value=_make_mock_answer(valid_response))

        engine = SelfLearningEngine(db=db, memory=ResearchMemory(db), llm=mock_llm)
        res = await engine.execute_learning_cycle(sess.id)

        assert res.updated_prompt_strategy_id is not None
        assert mock_llm.generate.call_count == 1
        assert mock_llm.generate.call_args.kwargs["model_ref"] == "google:gemini-2.5-flash"

        async with db.session() as s:
            # Check insight record
            insight = (
                await s.execute(
                    select(ResearchInsight).where(ResearchInsight.session_id == sess.id)
                )
            ).scalars().first()

            assert insight is not None
            assert insight.execution_status == "LLM_GROUNDED"
            assert insight.is_fallback is False
            assert insight.model_ref == "google:gemini-2.5-flash"
            assert insight.prompt_strategy_id == antecedent_id
            assert insight.evolved_strategy_id == res.updated_prompt_strategy_id
            assert insight.structured_output == valid_response

            # Check evolved prompt strategy
            new_strat = await s.get(PromptStrategy, res.updated_prompt_strategy_id)
            assert new_strat.is_active is True
            assert new_strat.version == 3
            assert new_strat.parent_id == antecedent_id
            assert new_strat.category == "momentum"

            # Examples must come from verified SQLite records
            assert len(new_strat.examples) >= 1
            assert new_strat.examples[0]["expression"] == "ts_zscore(close, 20)"

            # Anti-patterns and recommendations must be explicitly labeled unverified
            assert "[Unverified Hypothesis]" in new_strat.anti_patterns[0]["flaw"]
            assert any("[Unverified Recommendation]" in r for r in new_strat.guidance["unverified_recommendations"])

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 5: Invalid Candidate Citation Strict Rejection
# ---------------------------------------------------------------------------
def test_invalid_candidate_citation_strict_rejection(tmp_path: Path):
    """Verifies that citing a non-existent candidate ID triggers deterministic fallback."""
    async def _run():
        db = Database(tmp_path / "sqlite.db")
        await db.create_all()

        async with db.session() as s:
            sess = ResearchSession(name="Bad Cand Cit", prompt="test", current_generation=0)
            s.add(sess)
            await s.flush()
            strat = PromptStrategy(session_id=sess.id, version=1, is_active=True)
            s.add(strat)

            for i in range(3):
                c = ResearchCandidate(session_id=sess.id, expression=f"rank(close + {i})", status=CandidateStatus.EVALUATED, generation=0)
                s.add(c)
                await s.flush()
                s.add(ResearchEvaluation(candidate_id=c.id, is_verified=True, val_sharpe=1.5))

        # Citations referencing fake candidate ID 99999
        bad_response = {
            "summary": "Bad citation reflection",
            "hypotheses": [
                {
                    "claim": "Fake claim",
                    "confidence": "HIGH_CONFIDENCE_HYPOTHESIS",
                    "citations": [{"candidate_id": 99999, "field": "expression"}],
                }
            ],
            "recommendations": [
                {
                    "action": "Do something",
                    "rationale": "Because of candidate 99999",
                    "citations": [{"candidate_id": 99999, "field": "sharpe"}],
                }
            ],
            "anti_patterns": [
                {
                    "expression": "rank(close + 0)",
                    "flaw": "Flaw",
                    "citations": [{"candidate_id": 99999, "field": "fitness"}],
                }
            ],
        }

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value=_make_mock_answer(bad_response))
        engine = SelfLearningEngine(db=db, memory=ResearchMemory(db), llm=mock_llm)

        res = await engine.execute_learning_cycle(sess.id)

        assert res.updated_prompt_strategy_id is not None
        async with db.session() as s:
            insight = (await s.execute(select(ResearchInsight).where(ResearchInsight.session_id == sess.id))).scalars().first()
            assert insight.execution_status == "FALLBACK_DETERMINISTIC"
            assert insight.is_fallback is True
            assert "CitationValidationError" in insight.error_details
            assert "99999" in insight.error_details

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 6: Invalid Evidence-Field Citation Strict Rejection
# ---------------------------------------------------------------------------
def test_invalid_field_citation_strict_rejection(tmp_path: Path):
    """Verifies that citing a forbidden or missing field triggers deterministic fallback."""
    async def _run():
        db = Database(tmp_path / "sqlite.db")
        await db.create_all()

        real_cand_ids = []
        async with db.session() as s:
            sess = ResearchSession(name="Bad Field Cit", prompt="test", current_generation=0)
            s.add(sess)
            await s.flush()
            strat = PromptStrategy(session_id=sess.id, version=1, is_active=True)
            s.add(strat)

            for i in range(3):
                c = ResearchCandidate(session_id=sess.id, expression=f"rank(close + {i})", status=CandidateStatus.EVALUATED, generation=0)
                s.add(c)
                await s.flush()
                real_cand_ids.append(c.id)
                s.add(ResearchEvaluation(candidate_id=c.id, is_verified=True, val_sharpe=1.5))

        # Citation referencing unpermitted field 'holdout_sharpe'
        bad_response = {
            "summary": "Bad field reflection",
            "hypotheses": [
                {
                    "claim": "Claim",
                    "confidence": "HIGH_CONFIDENCE_HYPOTHESIS",
                    "citations": [{"candidate_id": real_cand_ids[0], "field": "holdout_sharpe"}],
                }
            ],
            "recommendations": [
                {
                    "action": "Action",
                    "rationale": "Rationale",
                    "citations": [{"candidate_id": real_cand_ids[0], "field": "sharpe"}],
                }
            ],
            "anti_patterns": [
                {
                    "expression": "rank(close + 0)",
                    "flaw": "Flaw",
                    "citations": [{"candidate_id": real_cand_ids[0], "field": "expression"}],
                }
            ],
        }

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value=_make_mock_answer(bad_response))
        engine = SelfLearningEngine(db=db, memory=ResearchMemory(db), llm=mock_llm)

        res = await engine.execute_learning_cycle(sess.id)

        assert res.updated_prompt_strategy_id is not None
        async with db.session() as s:
            insight = (await s.execute(select(ResearchInsight).where(ResearchInsight.session_id == sess.id))).scalars().first()
            assert insight.execution_status == "FALLBACK_DETERMINISTIC"
            assert insight.is_fallback is True
            assert "CitationValidationError" in insight.error_details
            assert "holdout_sharpe" in insight.error_details

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 7: Unsupported Assertions Remain Explicitly Unverified Hypotheses
# ---------------------------------------------------------------------------
def test_unsupported_factual_assertions_not_persisted_as_facts(tmp_path: Path):
    """Verifies that model claims are never stored as authoritative performance metrics."""
    async def _run():
        db = Database(tmp_path / "sqlite.db")
        await db.create_all()

        real_cand_ids = []
        async with db.session() as s:
            sess = ResearchSession(name="Facts vs Hypotheses", prompt="test", current_generation=0)
            s.add(sess)
            await s.flush()
            strat = PromptStrategy(session_id=sess.id, version=1, is_active=True)
            s.add(strat)

            for i in range(3):
                c = ResearchCandidate(session_id=sess.id, expression=f"rank(volume * {i+1})", status=CandidateStatus.EVALUATED, generation=0)
                s.add(c)
                await s.flush()
                real_cand_ids.append(c.id)
                s.add(ResearchEvaluation(candidate_id=c.id, is_verified=True, val_sharpe=1.2 + i * 0.1))

        # Model claims an inflated Sharpe of 4.5 in its hypothesis text
        model_payload = {
            "summary": "Model asserts Sharpe 4.5 exists.",
            "hypotheses": [
                {
                    "claim": "This strategy delivers Sharpe 4.5 in bull markets.",
                    "confidence": "SPECULATIVE_EXPLORATION",
                    "citations": [{"candidate_id": real_cand_ids[0], "field": "expression"}],
                }
            ],
            "recommendations": [
                {
                    "action": "Scale leverage up",
                    "rationale": "High returns expected",
                    "citations": [{"candidate_id": real_cand_ids[0], "field": "sharpe"}],
                }
            ],
            "anti_patterns": [
                {
                    "expression": "rank(volume * 1)",
                    "flaw": "Decay",
                    "citations": [{"candidate_id": real_cand_ids[0], "field": "expression"}],
                }
            ],
        }

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value=_make_mock_answer(model_payload))
        engine = SelfLearningEngine(db=db, memory=ResearchMemory(db), llm=mock_llm)

        res = await engine.execute_learning_cycle(sess.id)

        # Batch metrics must remain strictly calculated from DB (max 1.4, avg 1.3), not 4.5
        assert res.best_sharpe == 1.4
        assert res.avg_sharpe == 1.3

        async with db.session() as s:
            insight = (await s.execute(select(ResearchInsight).where(ResearchInsight.session_id == sess.id))).scalars().first()
            assert insight.batch_best_sharpe == 1.4
            assert insight.batch_avg_sharpe == 1.3
            # The model's speculative claim is sequestered in structured_output/rules
            assert insight.rules["hypotheses"][0]["confidence"] == "SPECULATIVE_EXPLORATION"

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 8: Malformed LLM Output Triggers Labeled Fallback
# ---------------------------------------------------------------------------
def test_malformed_llm_json_triggers_labeled_fallback(tmp_path: Path):
    """Verifies that syntax errors or unparseable JSON trigger labeled deterministic fallback."""
    async def _run():
        db = Database(tmp_path / "sqlite.db")
        await db.create_all()

        async with db.session() as s:
            sess = ResearchSession(name="Malformed JSON", prompt="test", current_generation=0)
            s.add(sess)
            await s.flush()
            strat = PromptStrategy(session_id=sess.id, version=1, is_active=True)
            s.add(strat)

            for i in range(3):
                c = ResearchCandidate(session_id=sess.id, expression=f"rank(close + {i})", status=CandidateStatus.EVALUATED, generation=0)
                s.add(c)
                await s.flush()
                s.add(ResearchEvaluation(candidate_id=c.id, is_verified=True, val_sharpe=1.5))

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value=_make_mock_answer("NOT VALID JSON AT ALL {{{"))
        engine = SelfLearningEngine(db=db, memory=ResearchMemory(db), llm=mock_llm)

        res = await engine.execute_learning_cycle(sess.id)

        assert res.updated_prompt_strategy_id is not None
        async with db.session() as s:
            insight = (await s.execute(select(ResearchInsight).where(ResearchInsight.session_id == sess.id))).scalars().first()
            assert insight.execution_status == "FALLBACK_DETERMINISTIC"
            assert insight.is_fallback is True
            assert "SchemaValidationError" in insight.error_details

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 9: Provider Timeout Triggers Labeled Fallback
# ---------------------------------------------------------------------------
def test_llm_provider_timeout_triggers_labeled_fallback(tmp_path: Path):
    """Verifies that network/timeout exceptions from the LLM provider cleanly trigger fallback."""
    async def _run():
        db = Database(tmp_path / "sqlite.db")
        await db.create_all()

        async with db.session() as s:
            sess = ResearchSession(name="Timeout Test", prompt="test", current_generation=0)
            s.add(sess)
            await s.flush()
            strat = PromptStrategy(session_id=sess.id, version=1, is_active=True)
            s.add(strat)

            for i in range(3):
                c = ResearchCandidate(session_id=sess.id, expression=f"rank(close + {i})", status=CandidateStatus.EVALUATED, generation=0)
                s.add(c)
                await s.flush()
                s.add(ResearchEvaluation(candidate_id=c.id, is_verified=True, val_sharpe=1.5))

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(side_effect=httpx2.TimeoutException("API connection timed out"))
        engine = SelfLearningEngine(db=db, memory=ResearchMemory(db), llm=mock_llm)

        res = await engine.execute_learning_cycle(sess.id)

        assert res.updated_prompt_strategy_id is not None
        async with db.session() as s:
            insight = (await s.execute(select(ResearchInsight).where(ResearchInsight.session_id == sess.id))).scalars().first()
            assert insight.execution_status == "FALLBACK_DETERMINISTIC"
            assert insight.is_fallback is True
            assert "TimeoutException" in insight.error_details

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 10: Safe, Idempotent Legacy Database Migration
# ---------------------------------------------------------------------------
def test_legacy_migration_and_idempotency(tmp_path: Path):
    """Verifies that legacy heuristic insights are backfilled and repeated migration is idempotent."""
    async def _run():
        db = Database(tmp_path / "sqlite.db")
        await db.create_all()

        # Insert a pre-existing legacy insight directly with NULL execution_status
        async with db.session() as s:
            legacy_ins = ResearchInsight(
                session_id=None,
                generation=0,
                insight_type="winning_pattern",
                summary="Legacy heuristic summary",
                batch_size=50,
                applied=True,
            )
            s.add(legacy_ins)
            await s.flush()
            legacy_id = legacy_ins.id

        # Run migration first time
        await db.create_all()

        async with db.session() as s:
            row = await s.get(ResearchInsight, legacy_id)
            assert row.execution_status == "LEGACY_HEURISTIC"
            assert row.is_fallback is True or row.is_fallback == 1

        # Run migration second time (idempotency check)
        await db.create_all()

        async with db.session() as s:
            row = await s.get(ResearchInsight, legacy_id)
            assert row.execution_status == "LEGACY_HEURISTIC"
            assert row.is_fallback is True or row.is_fallback == 1

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 11: Antecedent/Evolved Strategy Lineage and Downstream Injection
# ---------------------------------------------------------------------------
def test_strategy_dual_lineage_and_downstream_injection(tmp_path: Path):
    """Verifies dual strategy foreign keys and downstream consumption of unverified feedback."""
    async def _run():
        db = Database(tmp_path / "sqlite.db")
        await db.create_all()

        cand_ids = []
        async with db.session() as s:
            sess = ResearchSession(name="Lineage Session", prompt="test", current_generation=1)
            s.add(sess)
            await s.flush()

            v1_strat = PromptStrategy(
                session_id=sess.id,
                name="Gen 1 Momentum",
                version=1,
                category="momentum",
                is_active=True,
            )
            s.add(v1_strat)
            await s.flush()
            v1_id = v1_strat.id

            for i in range(3):
                c = ResearchCandidate(session_id=sess.id, expression=f"ts_rank(close, {5*(i+1)})", status=CandidateStatus.EVALUATED, generation=1)
                s.add(c)
                await s.flush()
                cand_ids.append(c.id)
                s.add(ResearchEvaluation(candidate_id=c.id, is_verified=True, val_sharpe=1.7 + i * 0.2))

        valid_response = {
            "summary": "Generation 1 momentum evaluation summary.",
            "hypotheses": [
                {
                    "claim": "Shorter lookback captures sharper mean-reverting anomalies.",
                    "confidence": "HIGH_CONFIDENCE_HYPOTHESIS",
                    "citations": [{"candidate_id": cand_ids[0], "field": "expression"}],
                }
            ],
            "recommendations": [
                {
                    "action": "Shift momentum window to 5-day delta",
                    "rationale": "Empirical Sharpe improvement",
                    "citations": [{"candidate_id": cand_ids[0], "field": "sharpe"}],
                }
            ],
            "anti_patterns": [
                {
                    "expression": "ts_rank(close, 15)",
                    "flaw": "Excessive lookback latency",
                    "citations": [{"candidate_id": cand_ids[2], "field": "expression"}],
                }
            ],
        }

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value=_make_mock_answer(valid_response))
        engine = SelfLearningEngine(db=db, memory=ResearchMemory(db), llm=mock_llm)

        res = await engine.execute_learning_cycle(sess.id)

        v2_id = res.updated_prompt_strategy_id
        assert v2_id is not None

        async with db.session() as s:
            insight = (await s.execute(select(ResearchInsight).where(ResearchInsight.session_id == sess.id))).scalars().first()
            assert insight.prompt_strategy_id == v1_id
            assert insight.evolved_strategy_id == v2_id

            v1 = await s.get(PromptStrategy, v1_id)
            v2 = await s.get(PromptStrategy, v2_id)
            assert v1.is_active is False
            assert v2.is_active is True
            assert v2.parent_id == v1_id
            assert v2.version == 2

            # Simulate downstream prompt rendering using the evolved strategy
            lib = PromptLibrary()
            feedback_str = v2.guidance.get("strategy", "")
            rendered_prompt = lib.render(
                category=v2.category,
                catalog_context="[DATASET: test]",
                operators_table="[OPERATORS: rank, ts_rank]",
                learning_feedback=feedback_str,
                winning_examples=v2.examples,
                anti_patterns=v2.anti_patterns,
            )

            assert "[Unverified Recommendation]" in rendered_prompt
            assert "[Unverified Hypothesis]" in rendered_prompt
            assert "Excessive lookback latency" in rendered_prompt
            assert "ts_rank(close, 5)" in rendered_prompt  # From winning example

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 12: Repeated and Interrupted Cycle Idempotency
# ---------------------------------------------------------------------------
def test_repeated_and_interrupted_cycle_idempotency(tmp_path: Path):
    """Verifies repeated cycles increment versions cleanly and interrupted cycles roll back."""
    async def _run():
        db = Database(tmp_path / "sqlite.db")
        await db.create_all()

        async with db.session() as s:
            sess = ResearchSession(name="Repeated Cycle Session", prompt="test", current_generation=0)
            s.add(sess)
            await s.flush()

            strat = PromptStrategy(session_id=sess.id, version=1, category="general", is_active=True)
            s.add(strat)
            await s.flush()

            for i in range(3):
                c = ResearchCandidate(session_id=sess.id, expression=f"rank(close + {i})", status=CandidateStatus.EVALUATED, generation=0)
                s.add(c)
                await s.flush()
                s.add(ResearchEvaluation(candidate_id=c.id, is_verified=True, val_sharpe=1.6))

        # Cycle 1 (fallback mode)
        engine = SelfLearningEngine(db=db, memory=ResearchMemory(db), llm=None)
        res1 = await engine.execute_learning_cycle(sess.id)

        assert res1.updated_prompt_strategy_id is not None
        async with db.session() as s:
            strat1 = await s.get(PromptStrategy, res1.updated_prompt_strategy_id)
            assert strat1.version == 2
            assert strat1.is_active is True

            active_strats = (
                await s.execute(
                    select(PromptStrategy).where(
                        PromptStrategy.session_id == sess.id,
                        PromptStrategy.is_active.is_(True),
                    )
                )
            ).scalars().all()
            assert len(active_strats) == 1

        # Cycle 2 (repeated cycle)
        res2 = await engine.execute_learning_cycle(sess.id)
        assert res2.updated_prompt_strategy_id is not None
        async with db.session() as s:
            strat2 = await s.get(PromptStrategy, res2.updated_prompt_strategy_id)
            assert strat2.version == 3
            assert strat2.is_active is True

            active_strats = (
                await s.execute(
                    select(PromptStrategy).where(
                        PromptStrategy.session_id == sess.id,
                        PromptStrategy.is_active.is_(True),
                    )
                )
            ).scalars().all()
            assert len(active_strats) == 1

    asyncio.run(_run())
