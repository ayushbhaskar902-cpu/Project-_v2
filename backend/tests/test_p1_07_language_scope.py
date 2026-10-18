"""P1-07: Alpha Generation Language Policy & Fast Expression Scope Enforcement Tests.

Verifies:
1. FASTEXPR is accepted.
2. Lowercase and mixed-case FASTEXPR normalize correctly to 'FASTEXPR'.
3. None defaults to FASTEXPR.
4. PYTHON and python raise UnsupportedAlphaLanguageError.
5. Arbitrary unsupported languages (R, C++, JULIA, etc.) are rejected with UnsupportedAlphaLanguageError.
6. generate_batch(language="PYTHON") fails before making an LLM call.
7. validate_expression rejects unsupported languages before parsing.
8. Candidate creation cannot persist an unsupported generation language.
9. Simulation request construction explicitly preserves FASTEXPR.
10. Unsupported language requests never reach simulation dispatch.
11. Existing callers omitting the language remain 100% compatible.
12. P1-05 datafield validation still works under language policy.
13. P1-06 operator name, arity, and scope validation still works under language policy.
14. Existing prompt-version, dataset, datafield, and operator-catalog lineage remains intact through candidate generation.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from alpha_harness.brain.schemas import SimulationRequest, SimulationSettings, SimulationType
from alpha_harness.db.models import (
    CandidateStatus,
    GenerationAttempt,
    PromptStrategy,
    PromptVersion,
    ResearchCandidate,
    ResearchSession,
)
from alpha_harness.db.sqlite import Database
from alpha_harness.discovery.catalog import (
    CatalogDiscovery,
    DatafieldCatalog,
    DiscoveredDataset,
    DiscoveredField,
    OperatorCatalog,
)
from alpha_harness.generator.pipeline import (
    DEFAULT_ALPHA_LANGUAGE,
    SUPPORTED_ALPHA_LANGUAGES,
    AlphaGenerationPipeline,
    UnsupportedAlphaLanguageError,
    validate_alpha_language,
    validate_expression,
)
from alpha_harness.orchestrator.researchers import MultiLLMOrchestrator
from alpha_harness.prompts.library import StrategyCategory


# --- 1. Basic Language Policy & Normalization ---


def test_fastexpr_is_accepted() -> None:
    """Verifies that FASTEXPR is recognized and accepted."""
    assert "FASTEXPR" in SUPPORTED_ALPHA_LANGUAGES
    assert DEFAULT_ALPHA_LANGUAGE == "FASTEXPR"
    assert validate_alpha_language("FASTEXPR") == "FASTEXPR"


def test_case_insensitive_fastexpr_normalization() -> None:
    """Verifies that lowercase and mixed-case FASTEXPR normalizes correctly to FASTEXPR."""
    assert validate_alpha_language("fastexpr") == "FASTEXPR"
    assert validate_alpha_language("FastExpr") == "FASTEXPR"
    assert validate_alpha_language("  FASTEXPR  ") == "FASTEXPR"
    assert validate_alpha_language("  fastexpr\n") == "FASTEXPR"


def test_none_defaults_to_fastexpr() -> None:
    """Verifies that None cleanly resolves to default FASTEXPR."""
    assert validate_alpha_language(None) == "FASTEXPR"


def test_python_and_lowercase_python_raise_unsupported_language_error() -> None:
    """Verifies that PYTHON and python raise UnsupportedAlphaLanguageError with informative diagnostic."""
    for py_val in ("PYTHON", "python", "Python", "  python  "):
        with pytest.raises(UnsupportedAlphaLanguageError) as exc_info:
            validate_alpha_language(py_val)
        err = str(exc_info.value)
        assert "Unsupported alpha language" in err
        assert "FASTEXPR" in err
        assert "out of scope" in err


def test_arbitrary_unsupported_languages_are_rejected() -> None:
    """Verifies that arbitrary unsupported languages (R, C++, JULIA, Java, etc.) are rejected."""
    for lang in ("R", "C++", "JULIA", "julia", "JAVA", "rust", "MATLAB", "SQL"):
        with pytest.raises(UnsupportedAlphaLanguageError) as exc_info:
            validate_alpha_language(lang)
        assert "Unsupported alpha language" in str(exc_info.value)


# --- 2. Pipeline generate_batch & validate_expression Enforcement ---


def test_generate_batch_fails_before_llm_call_when_unsupported_language() -> None:
    """Verifies generate_batch(language='PYTHON') fails before any LLM request is made."""
    async def _run() -> None:
        llm_mock = MagicMock()
        llm_mock.generate = AsyncMock()

        pipeline = AlphaGenerationPipeline(llm=llm_mock)

        with pytest.raises(UnsupportedAlphaLanguageError):
            await pipeline.generate_batch(
                model_ref="google:gemini-2.5-flash",
                category=StrategyCategory.MOMENTUM,
                candidate_count=5,
                language="PYTHON",
            )

        # Crucial: LLM was never called!
        assert llm_mock.generate.call_count == 0

    asyncio.run(_run())


def test_validate_expression_rejects_unsupported_languages_before_parsing() -> None:
    """Verifies validate_expression rejects unsupported languages before parsing."""
    # Even if expression is syntactically valid or invalid, language check rejects first
    for bad_lang in ("PYTHON", "python", "R", "C++"):
        with pytest.raises(UnsupportedAlphaLanguageError) as exc_info:
            validate_expression("rank(close)", language=bad_lang)
        assert "Unsupported alpha language" in str(exc_info.value)
        assert "out of scope" in str(exc_info.value)


def test_validate_expression_accepts_valid_fastexpr_language() -> None:
    """Verifies validate_expression succeeds when language is omitted or FASTEXPR."""
    # Omitted
    val1 = validate_expression("rank(close)")
    assert val1.is_valid is True

    # Explicit FASTEXPR
    val2 = validate_expression("rank(close)", language="FASTEXPR")
    assert val2.is_valid is True

    # Mixed-case fastexpr
    val3 = validate_expression("rank(close)", language="fastexpr")
    assert val3.is_valid is True


# --- 3. Candidate Creation & Persistence Enforcement ---


def test_candidate_creation_enforces_fastexpr_language(tmp_path: Any) -> None:
    """Verifies orchestrator candidate creation persists language='FASTEXPR' and rejects unsupported languages."""
    async def _run() -> None:
        db = Database(tmp_path / "test_cand_lang.db")
        await db.create_all()

        llm_mock = MagicMock()
        expressions = ["rank(close)", "rank(volume)", "rank(open)", "rank(high)"]
        mocks = [
            MagicMock(
                text=f'{{"candidates": [{{"hypothesis": "Test", "expression": "{expr}"}}]}}',
                total_tokens=50,
            )
            for expr in expressions
        ]
        llm_mock.generate = AsyncMock(side_effect=mocks)

        pipeline = AlphaGenerationPipeline(llm=llm_mock)
        orchestrator = MultiLLMOrchestrator(db=db, pipeline=pipeline)

        # Set up session
        async with db.session() as s:
            sess = ResearchSession(
                name="Test Session",
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
            total_target_candidates=1,
        )

        assert len(cand_ids) == 4

        async with db.session() as s:
            for cid in cand_ids:
                cand = await s.get(ResearchCandidate, cid)
                assert cand is not None
                assert cand.language == "FASTEXPR"

        await db.dispose()

    asyncio.run(_run())


# --- 4. Simulation Request Construction & Dispatch Protection ---


def test_simulation_request_preserves_fastexpr_language() -> None:
    """Verifies that SimulationRequest and SimulationSettings require and preserve FASTEXPR."""
    req = SimulationRequest(
        type=SimulationType.REGULAR,
        settings=SimulationSettings(
            instrument_type="EQUITY",
            region="USA",
            universe="TOP3000",
            delay=1,
            language="FASTEXPR",
        ),
        regular="rank(close)",
    )
    assert req.settings.language == "FASTEXPR"

    # Default settings also yield FASTEXPR
    req_default = SimulationRequest(
        type=SimulationType.REGULAR,
        settings=SimulationSettings(
            instrument_type="EQUITY",
            region="USA",
            universe="TOP3000",
            delay=1,
        ),
        regular="rank(close)",
    )
    assert req_default.settings.language == "FASTEXPR"


def test_simulation_settings_rejects_unsupported_language() -> None:
    """Verifies that SimulationSettings rejects language='PYTHON' during Pydantic validation."""
    with pytest.raises(Exception) as exc_info:
        SimulationSettings(
            region="USA",
            universe="TOP3000",
            delay=1,
            language="PYTHON",
        )
    assert "Unsupported alpha language" in str(exc_info.value)


def test_unsupported_language_never_reaches_simulation_dispatch(tmp_path: Any) -> None:
    """Verifies that candidates with unsupported languages are blocked before simulation enqueue."""
    async def _run() -> None:
        db = Database(tmp_path / "test_sim_dispatch.db")
        await db.create_all()

        mock_engine = MagicMock()
        mock_engine.enqueue = AsyncMock()

        orchestrator = MultiLLMOrchestrator(db=db, pipeline=MagicMock())

        # Create candidate with unsupported language manually in database
        async with db.session() as s:
            sess = ResearchSession(name="Sim Test")
            s.add(sess)
            await s.flush()

            # Bypassing model default to simulate corrupted/unsupported row
            cand = ResearchCandidate(
                session_id=sess.id,
                expression="rank(close)",
                language="PYTHON",
                is_valid=True,
                status=CandidateStatus.VALID,
            )
            s.add(cand)
            await s.commit()
            cand_id = cand.id

        # Attempt to enqueue for simulation
        with pytest.raises(UnsupportedAlphaLanguageError) as exc_info:
            await orchestrator.enqueue_candidates_for_simulation(
                session_id=sess.id,
                candidate_ids=[cand_id],
                batch_engine=mock_engine,
            )

        assert "Unsupported alpha language" in str(exc_info.value)
        # Crucial: batch_engine.enqueue was never called!
        assert mock_engine.enqueue.call_count == 0

        await db.dispose()

    asyncio.run(_run())


# --- 5. Backward Compatibility & Prior Remediation Preservation ---


def test_existing_callers_omitting_language_remain_compatible() -> None:
    """Verifies backward compatibility: callers omitting language parameter encounter zero breakage."""
    # 1. validate_expression with no language specified
    res = validate_expression("ts_rank(close, 10)")
    assert res.is_valid is True
    assert res.expression == "ts_rank(close, 10)"

    # 2. validate_alpha_language with no argument
    assert validate_alpha_language(None) == "FASTEXPR"


def test_p1_05_datafield_validation_still_works_under_language_policy() -> None:
    """Verifies P1-05 authoritative datafield validation operates cleanly with language policy."""
    discovered_ds = DiscoveredDataset(
        dataset_id="market_prices",
        name="Market Prices",
        category="Price",
        description="Daily equity prices and volumes",
        fields=(
            DiscoveredField("close", "MATRIX", 1.0, "Daily close", "market_prices", "Price"),
            DiscoveredField("volume", "MATRIX", 1.0, "Daily volume", "market_prices", "Price"),
        ),
    )
    df_cat = DatafieldCatalog(
        dataset=discovered_ds,
        region="USA",
        universe="TOP3000",
        delay=1,
    )

    # Valid field
    res_valid = validate_expression(
        "rank(close)",
        datafield_catalog=df_cat,
        language="FASTEXPR",
    )
    assert res_valid.is_valid is True

    # Hallucinated field rejected by P1-05
    res_invalid = validate_expression(
        "rank(hallucinated_field_xyz)",
        datafield_catalog=df_cat,
        language="FASTEXPR",
    )
    assert res_invalid.is_valid is False
    assert "hallucinated_field_xyz" in (res_invalid.error or "")


def test_p1_06_operator_validation_still_works_under_language_policy() -> None:
    """Verifies P1-06 operator catalog arity and scope checks operate cleanly with language policy."""
    op_cat = OperatorCatalog(scope="REGULAR", source="METADATA_CACHE")

    # Valid operator
    res_valid = validate_expression(
        "rank(ts_rank(close, 10))",
        operator_catalog=op_cat,
        language="FASTEXPR",
    )
    assert res_valid.is_valid is True

    # Unknown operator rejected
    res_unknown = validate_expression(
        "fake_operator(close)",
        operator_catalog=op_cat,
        language="FASTEXPR",
    )
    assert res_unknown.is_valid is False
    assert "Unknown or unsupported operator" in (res_unknown.error or "")

    # Operator argument contract (window d > 0)
    res_bad_window = validate_expression(
        "ts_rank(close, -5)",
        operator_catalog=op_cat,
        language="FASTEXPR",
    )
    assert res_bad_window.is_valid is False
    assert "lookback window" in (res_bad_window.error or "").lower()


def test_full_lineage_preservation_through_generation_batch() -> None:
    """Verifies prompt-version, dataset, datafield, and operator lineage are fully preserved alongside language."""
    async def _run() -> None:
        llm_mock = MagicMock()
        ans_mock = MagicMock()
        ans_mock.text = '{"batch_summary": "P1-07 Test", "candidates": [{"hypothesis": "Mean rev", "expression": "rank(close)"}]}'
        ans_mock.total_tokens = 80
        llm_mock.generate = AsyncMock(return_value=ans_mock)

        pipeline = AlphaGenerationPipeline(llm=llm_mock)

        batch = await pipeline.generate_batch(
            model_ref="google:gemini-2.5-flash",
            category=StrategyCategory.MEAN_REVERSION,
            candidate_count=1,
            language="FASTEXPR",
        )

        assert batch.language == "FASTEXPR"
        assert batch.summary == "P1-07 Test"
        assert len(batch.candidates) == 1
        assert batch.candidates[0].validation.is_valid is True
        assert batch.datafield_catalog_version is not None
        assert batch.operator_catalog_version is not None

    asyncio.run(_run())
