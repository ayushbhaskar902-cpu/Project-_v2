"""P1-06: Authoritative and Versioned Operator Signatures, Arity, and Metadata Validation Tests.

Verifies:
1. Deterministic operator catalog SHA-256 fingerprinting (op_v1_<sha256[:16]>).
2. Different operator definitions/scopes produce different version fingerprints.
3. Order-independent version calculation for operator catalogs.
4. Valid operator acceptance across Time-Series, Cross-Sectional, Group, Math, Logic, and Vector categories.
5. Unknown/unsupported operator rejection with authoritative diagnostic: "Unknown or unsupported operator: <name>".
6. Operator arity validation (under-arity and over-arity rejection).
7. Scope filtering (COMBO-only operator rejection in REGULAR scope).
8. to_operator_table() compatibility with fastexpr.validate().
9. from_brain_operators() construction from Pydantic models and dictionaries.
10. from_metadata_cache() construction from cached operator definitions.
11. validate_expression() integration with OperatorCatalog and operators_used extraction.
12. format_prompt_summary() includes version, scope, and operator categories.
13. GenerationBatch retains operator_catalog_version.
14. Pipeline repair loop targeted hints for operator errors (unknown, scope, arity).
15. Orchestrator persistence of operator_catalog_version and operators_used on GenerationAttempt and ResearchCandidate.
16. Idempotent SQLite migrations with honest NULL defaults for legacy records.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select, text

from alpha_harness.account import PlatformMetadata
from alpha_harness.brain.errors import BrainError
from alpha_harness.brain.schemas import Operator
from alpha_harness.db.models import (
    CandidateStatus,
    GenerationAttempt,
    MetadataCache,
    ResearchCandidate,
    ResearchResearcher,
    ResearchSession,
)
from alpha_harness.db.sqlite import Database, migrate
from alpha_harness.discovery.catalog import (
    BUILTIN_OPERATOR_DEFS,
    CatalogDiscovery,
    OperatorArityError,
    OperatorCatalog,
    OperatorCatalogError,
    OperatorEntry,
    OperatorScopeError,
    UnknownOperatorError,
)
from alpha_harness.generator.pipeline import (
    AlphaGenerationPipeline,
    CandidateValidationResult,
    GeneratedCandidate,
    GenerationBatch,
    validate_expression,
)
from alpha_harness.labs.fastexpr import parse, validate
from alpha_harness.orchestrator.researchers import MultiLLMOrchestrator, ResearcherConfig
from alpha_harness.prompts.library import StrategyCategory


# =============================================================================
# 1. Deterministic Fingerprinting & Scope Tests
# =============================================================================


def test_deterministic_operator_catalog_fingerprinting() -> None:
    """Verifies identical operator definitions produce the exact same version string."""
    cat1 = OperatorCatalog(scope="REGULAR")
    cat2 = OperatorCatalog(scope="REGULAR")

    assert cat1.catalog_version.startswith("op_v1_")
    assert len(cat1.catalog_version) == 22  # "op_v1_" (6) + 16 hex chars
    assert cat1.catalog_version == cat2.catalog_version
    assert cat1.version == cat1.catalog_version


def test_order_independent_fingerprinting() -> None:
    """Verifies operator list order does not affect the deterministic SHA-256 fingerprint."""
    defs1 = [
        {"name": "rank", "definition": "rank(x)", "category": "Cross-Sectional", "scope": ["REGULAR"]},
        {"name": "abs", "definition": "abs(x)", "category": "Math", "scope": ["REGULAR"]},
    ]
    defs2 = [
        {"name": "abs", "definition": "abs(x)", "category": "Math", "scope": ["REGULAR"]},
        {"name": "rank", "definition": "rank(x)", "category": "Cross-Sectional", "scope": ["REGULAR"]},
    ]

    cat1 = OperatorCatalog(operator_defs=defs1, scope="REGULAR")
    cat2 = OperatorCatalog(operator_defs=defs2, scope="REGULAR")

    assert cat1.catalog_version == cat2.catalog_version


def test_different_operator_sets_produce_different_fingerprints() -> None:
    """Verifies adding, changing or removing operators changes the catalog version."""
    base_defs = [
        {"name": "rank", "definition": "rank(x)", "category": "Cross-Sectional", "scope": ["REGULAR"]},
    ]
    cat_base = OperatorCatalog(operator_defs=base_defs, scope="REGULAR")

    # Adding an operator
    more_defs = [
        {"name": "rank", "definition": "rank(x)", "category": "Cross-Sectional", "scope": ["REGULAR"]},
        {"name": "abs", "definition": "abs(x)", "category": "Math", "scope": ["REGULAR"]},
    ]
    cat_more = OperatorCatalog(operator_defs=more_defs, scope="REGULAR")

    assert cat_base.catalog_version != cat_more.catalog_version

    # Altering an arity / definition
    diff_arity = [
        {"name": "rank", "definition": "rank(x, driver=y)", "category": "Cross-Sectional", "scope": ["REGULAR"]},
    ]
    cat_diff = OperatorCatalog(operator_defs=diff_arity, scope="REGULAR")

    assert cat_base.catalog_version != cat_diff.catalog_version


def test_scope_filtering_produces_distinct_fingerprints() -> None:
    """Verifies different scopes (REGULAR vs COMBO) produce distinct catalog versions."""
    defs = [
        {"name": "rank", "definition": "rank(x)", "category": "Cross-Sectional", "scope": ["REGULAR", "COMBO"]},
        {"name": "combo_op", "definition": "combo_op(alpha1, alpha2)", "category": "Combo", "scope": ["COMBO"]},
    ]
    reg_cat = OperatorCatalog(operator_defs=defs, scope="REGULAR")
    combo_cat = OperatorCatalog(operator_defs=defs, scope="COMBO")

    assert reg_cat.catalog_version != combo_cat.catalog_version
    assert "combo_op" not in reg_cat.get_operator_names()
    assert "combo_op" in combo_cat.get_operator_names()


# =============================================================================
# 2. Operator Validation & Arity Diagnostics Tests
# =============================================================================


def test_valid_operators_accepted_in_catalog() -> None:
    """Verifies standard BRAIN operators across categories pass validate_operator."""
    cat = OperatorCatalog(scope="REGULAR")

    # Time series
    valid, diag = cat.validate_operator("ts_rank", 2)
    assert valid is True
    assert diag is None

    valid, diag = cat.validate_operator("ts_delta", 2)
    assert valid is True

    valid, diag = cat.validate_operator("ts_corr", 3)
    assert valid is True

    # Cross-sectional
    valid, diag = cat.validate_operator("rank", 1)
    assert valid is True

    # Group
    valid, diag = cat.validate_operator("group_rank", 2)
    assert valid is True

    valid, diag = cat.validate_operator("group_neutralize", 2)
    assert valid is True

    # Math
    valid, diag = cat.validate_operator("abs", 1)
    assert valid is True

    valid, diag = cat.validate_operator("max", 2)
    assert valid is True

    # Logic
    valid, diag = cat.validate_operator("trade_when", 3)
    assert valid is True


def test_unknown_operator_rejection_diagnostic() -> None:
    """Verifies unknown operators are rejected with the authoritative diagnostic."""
    cat = OperatorCatalog(scope="REGULAR")

    valid, diag = cat.validate_operator("hallucinated_operator", 2)
    assert valid is False
    assert diag == "Unknown or unsupported operator: hallucinated_operator"

    valid, diag = cat.validate_operator("np_sin", 1)
    assert valid is False
    assert diag == "Unknown or unsupported operator: np_sin"


def test_operator_under_arity_rejection() -> None:
    """Verifies calling an operator with fewer arguments than required yields specific diagnostic."""
    cat = OperatorCatalog(scope="REGULAR")

    # ts_rank requires 2 inputs (x, d)
    valid, diag = cat.validate_operator("ts_rank", 1)
    assert valid is False
    assert diag is not None
    assert "Invalid operator arity: ts_rank requires at least 2 inputs (got 1)" in diag

    # ts_corr requires 3 inputs (x, y, d)
    valid, diag = cat.validate_operator("ts_corr", 2)
    assert valid is False
    assert diag is not None
    assert "Invalid operator arity: ts_corr requires at least 3 inputs (got 2)" in diag


def test_operator_over_arity_rejection() -> None:
    """Verifies calling an operator with more arguments than accepted yields specific diagnostic."""
    cat = OperatorCatalog(scope="REGULAR")

    # rank accepts 1 input (x)
    valid, diag = cat.validate_operator("rank", 2)
    assert valid is False
    assert diag is not None
    assert "Invalid operator arity: rank accepts at most 1 inputs (got 2)" in diag

    # abs accepts 1 input
    valid, diag = cat.validate_operator("abs", 3)
    assert valid is False
    assert diag is not None
    assert "Invalid operator arity: abs accepts at most 1 inputs (got 3)" in diag


def test_scope_exclusion_diagnostic() -> None:
    """Verifies operator available in COMBO but not REGULAR gives scope-specific diagnostic."""
    defs = [
        {"name": "rank", "definition": "rank(x)", "category": "Cross-Sectional", "scope": ["REGULAR", "COMBO"]},
        {"name": "combo_weight", "definition": "combo_weight(a, b)", "category": "Combo", "scope": ["COMBO"]},
    ]
    reg_cat = OperatorCatalog(operator_defs=defs, scope="REGULAR")

    valid, diag = reg_cat.validate_operator("combo_weight", 2)
    assert valid is False
    assert diag is not None
    assert "Operator 'combo_weight' is only available in scope COMBO, not in REGULAR" in diag


# =============================================================================
# 3. fastexpr Compatibility & Classmethods Tests
# =============================================================================


def test_to_operator_table_fastexpr_validate_compatibility() -> None:
    """Verifies to_operator_table() returns a dictionary compatible with fastexpr.validate()."""
    cat = OperatorCatalog(scope="REGULAR")
    table = cat.to_operator_table()

    assert "rank" in table
    assert "ts_rank" in table
    assert "group_neutralize" in table

    # Valid expression
    tree = parse("rank(ts_rank(close, 10))")
    problems = validate(tree, table, {"close"})
    assert problems == []

    # Operator with wrong arity
    bad_tree = parse("rank(close, 10)")
    bad_problems = validate(bad_tree, table, {"close"})
    assert len(bad_problems) > 0
    assert "rank does not take 2 inputs" in bad_problems[0]


def test_from_brain_operators_construction() -> None:
    """Verifies construction from GET /operators Pydantic models and dictionaries."""
    brain_ops = [
        Operator(
            name="ts_rank",
            category="Time Series",
            scope=["REGULAR", "COMBO"],
            definition="ts_rank(x, d)",
            description="Percentile rank",
        ),
        Operator(
            name="combo_only",
            category="Combo",
            scope=["COMBO"],
            definition="combo_only(x, y)",
            description="Combo operator",
        ),
    ]

    cat_pydantic = OperatorCatalog.from_brain_operators(brain_ops, scope="REGULAR")
    assert "ts_rank" in cat_pydantic.get_operator_names()
    assert "combo_only" not in cat_pydantic.get_operator_names()

    # Raw dicts
    raw_dicts = [
        {"name": "rank", "definition": "rank(x)", "category": "Cross-Sectional", "scope": ["REGULAR"]},
    ]
    cat_dict = OperatorCatalog.from_brain_operators(raw_dicts, scope="REGULAR")
    assert "rank" in cat_dict.get_operator_names()


def test_from_metadata_cache_construction() -> None:
    """Verifies construction from MetadataCache cached JSON records."""
    cached = [
        {"name": "ts_delta", "definition": "ts_delta(x, d)", "category": "Time Series", "scope": ["REGULAR"]},
    ]
    cat = OperatorCatalog.from_metadata_cache(cached, scope="REGULAR")
    assert "ts_delta" in cat.get_operator_names()
    assert cat.get_operator("ts_delta") is not None


# =============================================================================
# 4. validate_expression Integration & operators_used Extraction Tests
# =============================================================================


def test_validate_expression_with_operator_catalog() -> None:
    """Verifies validate_expression enforces OperatorCatalog rules and extracts operators_used."""
    cat = OperatorCatalog(scope="REGULAR")

    # Valid expression
    res_valid = validate_expression("rank(ts_rank(close, 10))", operator_catalog=cat)
    assert res_valid.is_valid is True
    assert res_valid.operator_count == 2
    assert sorted(res_valid.operators_used) == ["rank", "ts_rank"]

    # Unknown operator
    res_unknown = validate_expression("magic_op(close, 10)", operator_catalog=cat)
    assert res_unknown.is_valid is False
    assert "Unknown or unsupported operator: magic_op" in (res_unknown.error or "")
    assert res_unknown.operators_used == ["magic_op"]

    # Wrong arity
    res_arity = validate_expression("ts_rank(close)", operator_catalog=cat)
    assert res_arity.is_valid is False
    assert "ts_rank does not take 1 inputs" in (res_arity.error or "")


def test_format_prompt_summary_includes_version_and_constraints() -> None:
    """Verifies format_prompt_summary includes catalog version and strict constraint text."""
    cat = OperatorCatalog(scope="REGULAR")
    summary = cat.format_prompt_summary()

    assert cat.catalog_version in summary
    assert "Scope: REGULAR" in summary
    assert "STRICT OPERATOR CONSTRAINTS" in summary
    assert "Time Series:" in summary
    assert "Cross-Sectional:" in summary


# =============================================================================
# 5. Pipeline Repair & GenerationBatch Tests
# =============================================================================


def test_targeted_hint_for_operator_errors() -> None:
    """Verifies pipeline targeted error feedback properly identifies operator issues."""
    from alpha_harness.generator.pipeline import _targeted_hint

    # Unknown operator error
    hint_unknown = _targeted_hint("Unknown or unsupported operator: foobar")
    assert "Unknown or unsupported operator: foobar" in hint_unknown
    assert "Only use operators from the authoritative operator catalog" in hint_unknown

    # Out of scope error
    hint_scope = _targeted_hint("Operator 'combo_op' is only available in scope COMBO, not in REGULAR")
    assert "Use only REGULAR-scope operators" in hint_scope

    # Arity error
    hint_arity = _targeted_hint("Invalid operator arity: ts_rank requires at least 2 inputs")
    assert "Verify operator argument counts" in hint_arity


def test_generation_batch_persists_operator_catalog_version() -> None:
    """Verifies generate_batch sets operator_catalog_version on the resulting GenerationBatch."""
    async def _run() -> None:
        import json
        ans_mock = MagicMock()
        ans_mock.text = json.dumps({
            "batch_summary": "Test Batch",
            "candidates": [
                {
                    "hypothesis": "Test momentum",
                    "expression": "rank(ts_rank(close, 10))",
                    "category": "momentum",
                    "target_dataset": "market_prices",
                    "primary_fields": ["close"],
                }
            ],
        })
        ans_mock.total_tokens = 100
        llm_mock = MagicMock()
        llm_mock.generate = AsyncMock(return_value=ans_mock)

        pipeline = AlphaGenerationPipeline(llm=llm_mock)
        custom_cat = OperatorCatalog(scope="REGULAR")

        batch = await pipeline.generate_batch(
            model_ref="google:gemini-2.5-flash",
            category=StrategyCategory.MOMENTUM,
            candidate_count=1,
            operator_catalog=custom_cat,
        )

        assert batch.operator_catalog_version == custom_cat.catalog_version
        assert len(batch.candidates) == 1
        assert batch.candidates[0].validation.is_valid is True
        assert batch.candidates[0].validation.operators_used == ["rank", "ts_rank"]

    asyncio.run(_run())


# =============================================================================
# 6. Researcher Orchestration Persistence Tests
# =============================================================================


def test_researchers_persists_operator_catalog_version_and_operators_used(tmp_path: Any) -> None:
    """Verifies MultiLLMOrchestrator records operator_catalog_version and operators_used into SQLite."""
    async def _run() -> None:
        import json
        db_path = tmp_path / "test_p1_06_orchestrator.db"
        db = Database(db_path)
        await db.create_all()

        ans_mock = MagicMock()
        ans_mock.text = json.dumps({
            "batch_summary": "Batch with operator lineage",
            "candidates": [
                {
                    "hypothesis": "Price momentum signal",
                    "expression": "rank(ts_rank(close, 10))",
                    "category": "momentum",
                    "target_dataset": "market_prices",
                    "primary_fields": ["close"],
                }
            ],
        })
        ans_mock.total_tokens = 100
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value=ans_mock)

        discovery = CatalogDiscovery()
        pipeline = AlphaGenerationPipeline(llm=mock_llm, discovery=discovery)
        orchestrator = MultiLLMOrchestrator(db=db, pipeline=pipeline)

        async with db.session() as session:
            sess = ResearchSession(
                name="Operator Lineage Session",
                region="USA",
                universe="TOP3000",
                delay=1,
            )
            session.add(sess)
            await session.flush()
            sess_id = sess.id

        await orchestrator.initialize_session_researchers(
            session_id=sess_id,
            custom_roster=[
                ResearcherConfig(
                    model_ref="test-model",
                    label="Momentum Specialist",
                    strategy_focus="momentum",
                )
            ],
        )

        await orchestrator.run_generation_round(
            session_id=sess_id,
            total_target_candidates=1,
            auto_repair=False,
        )

        async with db.session() as session:
            # Check GenerationAttempt
            attempt = (await session.execute(select(GenerationAttempt))).scalars().first()
            assert attempt is not None
            assert attempt.operator_catalog_version is not None
            assert attempt.operator_catalog_version.startswith("op_v1_")

            # Check ResearchCandidate
            cand_row = (await session.execute(select(ResearchCandidate))).scalars().first()
            assert cand_row is not None
            assert cand_row.operator_catalog_version == attempt.operator_catalog_version
            assert cand_row.operators_used == ["rank", "ts_rank"]

        await db.dispose()

    asyncio.run(_run())


# =============================================================================
# 7. SQLite Migration Idempotency & Legacy Record Compatibility Tests
# =============================================================================


def test_sqlite_migrations_remain_idempotent_and_legacy_compatible(tmp_path: Any) -> None:
    """Verifies schema migrations add operator columns idempotently and legacy records have honest NULL defaults."""
    async def _run() -> None:
        db_path = tmp_path / "test_p1_06_migration.db"
        db = Database(db_path)

        # 1. Initial schema creation
        await db.create_all()

        # 2. Insert legacy records lacking P1-06 operator lineage
        async with db.session() as session:
            sess = ResearchSession(name="Legacy Operator Session")
            session.add(sess)
            await session.flush()

            # Insert legacy candidate without P1-06 fields
            await session.execute(
                text(
                    """
                    INSERT INTO research_candidate (session_id, hypothesis, expression, is_valid, status, created_at)
                    VALUES (:sess_id, 'Legacy Alpha', 'close', 1, 'VALID', datetime('now'))
                    """
                ),
                {"sess_id": sess.id},
            )
            # Insert legacy generation attempt without P1-06 fields
            await session.execute(
                text(
                    """
                    INSERT INTO generation_attempt (session_id, model_ref, created_at)
                    VALUES (:sess_id, 'legacy-model', datetime('now'))
                    """
                ),
                {"sess_id": sess.id},
            )

        # 3. Run migration twice (idempotency check)
        async with db._engine.begin() as conn:
            await migrate(conn)
        async with db._engine.begin() as conn:
            await migrate(conn)

        # 4. Verify legacy records have honest NULL values, not fabricated versions
        async with db.session() as session:
            cand_row = (await session.execute(select(ResearchCandidate))).scalars().first()
            assert cand_row is not None
            assert cand_row.operator_catalog_version is None
            assert cand_row.operators_used is None

            att_row = (await session.execute(select(GenerationAttempt))).scalars().first()
            assert att_row is not None
            assert att_row.operator_catalog_version is None

        await db.dispose()

    asyncio.run(_run())


# =============================================================================
# 8. Operator Argument Contract Verification Tests
# =============================================================================


def test_operator_argument_contract_window_bounds() -> None:
    """Verifies that lookback window constraints (d > 0 and integer) are validated."""
    cat = OperatorCatalog(scope="REGULAR")

    # Valid window d = 10
    res_valid = validate_expression("ts_rank(close, 10)", operator_catalog=cat)
    assert res_valid.is_valid is True

    # Lookback window d <= 0 (e.g. 0)
    res_zero = validate_expression("ts_rank(close, 0)", operator_catalog=cat)
    assert res_zero.is_valid is False
    assert "lookback window 'd' must be greater than 0" in (res_zero.error or "")

    # Lookback window d < 0 (e.g. -5)
    res_neg = validate_expression("ts_rank(close, -5)", operator_catalog=cat)
    assert res_neg.is_valid is False
    assert "lookback window 'd' must be greater than 0" in (res_neg.error or "")

    # Lookback window non-integer float (e.g. 5.5)
    res_float = validate_expression("ts_rank(close, 5.5)", operator_catalog=cat)
    assert res_float.is_valid is False
    assert "must be an integer number of days" in (res_float.error or "")


def test_operator_argument_contract_grouping_positions() -> None:
    """Verifies that grouping operators reject numeric constants in the group argument slot."""
    cat = OperatorCatalog(scope="REGULAR")

    # Valid grouping field
    res_valid = validate_expression("group_rank(close, sector)", operator_catalog=cat)
    assert res_valid.is_valid is True

    # Numeric constant in grouping argument position
    res_num = validate_expression("group_rank(close, 10)", operator_catalog=cat)
    assert res_num.is_valid is False
    assert "requires a grouping field (e.g. sector, industry) as argument 2, not a number" in (res_num.error or "")

    res_num_neg = validate_expression("group_neutralize(close, -1)", operator_catalog=cat)
    assert res_num_neg.is_valid is False
    assert "requires a grouping field (e.g. sector, industry) as argument 2, not a number" in (res_num_neg.error or "")


# =============================================================================
# 9. Live BRAIN Metadata Retrieval & MetadataCache Integration Tests
# =============================================================================


def test_platform_metadata_live_sync_and_catalog_construction(tmp_path: Any) -> None:
    """Verifies live operator metadata retrieval from BRAIN endpoints and OperatorCatalog construction."""
    async def _run() -> None:
        db = Database(tmp_path / "test_live_meta.db")
        await db.create_all()

        mock_endpoints = MagicMock()
        mock_endpoints.list_operators = AsyncMock(
            return_value=[
                Operator(
                    name="ts_rank",
                    category="Time Series",
                    scope=["REGULAR", "COMBO"],
                    definition="ts_rank(x, d)",
                    description="Percentile rank",
                ),
                Operator(
                    name="rank",
                    category="Cross-Sectional",
                    scope=["REGULAR", "COMBO"],
                    definition="rank(x)",
                    description="Universe rank",
                ),
            ]
        )

        metadata = PlatformMetadata(db, mock_endpoints)
        catalog = await metadata.get_operator_catalog(scope="REGULAR", force_refresh=True)

        assert catalog.source == "BRAIN_LIVE"
        assert catalog.is_verified is True
        assert catalog.is_fallback is False
        assert catalog.catalog_version.startswith("op_v1_")
        assert "ts_rank" in catalog.get_operator_names()
        assert "rank" in catalog.get_operator_names()

        # Verify cached in SQLite
        async with db.session() as s:
            cached_row = await s.get(MetadataCache, "operators")
            assert cached_row is not None
            assert len(cached_row.value["items"]) == 2

        await db.dispose()

    asyncio.run(_run())


def test_platform_metadata_cached_reuse_within_ttl(tmp_path: Any) -> None:
    """Verifies cached operator metadata is reused without network calls when within TTL."""
    async def _run() -> None:
        db = Database(tmp_path / "test_cached_meta.db")
        await db.create_all()

        mock_endpoints = MagicMock()
        mock_endpoints.list_operators = AsyncMock(
            return_value=[
                Operator(
                    name="ts_rank",
                    category="Time Series",
                    scope=["REGULAR"],
                    definition="ts_rank(x, d)",
                )
            ]
        )

        metadata = PlatformMetadata(db, mock_endpoints)
        # First call populates cache
        cat1 = await metadata.get_operator_catalog(scope="REGULAR")
        assert cat1.source == "BRAIN_LIVE"
        assert mock_endpoints.list_operators.call_count == 1

        # Second call reuses cache within TTL; endpoint must NOT be called again
        cat2 = await metadata.get_operator_catalog(scope="REGULAR", force_refresh=False)
        assert cat2.source == "METADATA_CACHE"
        assert cat2.is_verified is True
        assert cat2.is_fallback is False
        assert cat2.catalog_version == cat1.catalog_version
        assert mock_endpoints.list_operators.call_count == 1

        await db.dispose()

    asyncio.run(_run())


def test_platform_metadata_force_refresh_updates_cache_and_fingerprint(tmp_path: Any) -> None:
    """Verifies force_refresh re-reads from endpoints, updating cache and catalog fingerprint."""
    async def _run() -> None:
        db = Database(tmp_path / "test_force_refresh.db")
        await db.create_all()

        mock_endpoints = MagicMock()
        mock_endpoints.list_operators = AsyncMock(
            return_value=[
                Operator(
                    name="ts_rank",
                    category="Time Series",
                    scope=["REGULAR"],
                    definition="ts_rank(x, d)",
                )
            ]
        )

        metadata = PlatformMetadata(db, mock_endpoints)
        cat1 = await metadata.get_operator_catalog(scope="REGULAR")

        # Now BRAIN API adds a new operator
        mock_endpoints.list_operators = AsyncMock(
            return_value=[
                Operator(
                    name="ts_rank",
                    category="Time Series",
                    scope=["REGULAR"],
                    definition="ts_rank(x, d)",
                ),
                Operator(
                    name="ts_zscore",
                    category="Time Series",
                    scope=["REGULAR"],
                    definition="ts_zscore(x, d)",
                ),
            ]
        )

        cat2 = await metadata.get_operator_catalog(scope="REGULAR", force_refresh=True)
        assert cat2.source == "BRAIN_LIVE"
        assert cat2.catalog_version != cat1.catalog_version
        assert "ts_zscore" in cat2.get_operator_names()

        await db.dispose()

    asyncio.run(_run())


def test_platform_metadata_failed_refresh_preserves_valid_cache(tmp_path: Any) -> None:
    """Verifies that when a live refresh fails, the last known valid cache is preserved and reused."""
    async def _run() -> None:
        db = Database(tmp_path / "test_preserve_cache.db")
        await db.create_all()

        mock_endpoints = MagicMock()
        mock_endpoints.list_operators = AsyncMock(
            return_value=[
                Operator(
                    name="ts_rank",
                    category="Time Series",
                    scope=["REGULAR"],
                    definition="ts_rank(x, d)",
                )
            ]
        )

        metadata = PlatformMetadata(db, mock_endpoints)
        cat_initial = await metadata.get_operator_catalog(scope="REGULAR")
        assert cat_initial.source == "BRAIN_LIVE"

        # Now endpoints fail (network down or BrainError)
        mock_endpoints.list_operators = AsyncMock(side_effect=BrainError("BRAIN API connection timed out"))

        # Forced refresh fails, but existing valid cache is preserved!
        cat_fallback = await metadata.get_operator_catalog(scope="REGULAR", force_refresh=True)
        assert cat_fallback.source == "METADATA_CACHE"
        assert cat_fallback.is_verified is True
        assert cat_fallback.is_fallback is False
        assert cat_fallback.catalog_version == cat_initial.catalog_version
        assert "ts_rank" in cat_fallback.get_operator_names()

        await db.dispose()

    asyncio.run(_run())


def test_platform_metadata_fallback_when_no_cache_and_remote_fails(tmp_path: Any) -> None:
    """Verifies that built-in fallback is used when no cache exists and remote is down, labeled BUILTIN_FALLBACK."""
    async def _run() -> None:
        db = Database(tmp_path / "test_builtin_fallback.db")
        await db.create_all()

        mock_endpoints = MagicMock()
        mock_endpoints.list_operators = AsyncMock(side_effect=BrainError("BRAIN offline"))

        metadata = PlatformMetadata(db, mock_endpoints)
        cat = await metadata.get_operator_catalog(scope="REGULAR")

        # Must be labeled BUILTIN_FALLBACK, never verified
        assert cat.source == "BUILTIN_FALLBACK"
        assert cat.is_fallback is True
        assert cat.is_verified is False
        assert cat.catalog_version.startswith("op_v1_")
        assert "rank" in cat.get_operator_names()

        await db.dispose()

    asyncio.run(_run())


def test_platform_metadata_rejects_malformed_brain_response(tmp_path: Any) -> None:
    """Verifies that malformed or empty metadata from BRAIN is rejected without corrupting the catalog."""
    async def _run() -> None:
        db = Database(tmp_path / "test_malformed.db")
        await db.create_all()

        mock_endpoints = MagicMock()
        # Returns empty list or operator with missing definition
        mock_endpoints.list_operators = AsyncMock(
            return_value=[
                Operator(name="", definition="")
            ]
        )

        metadata = PlatformMetadata(db, mock_endpoints)
        cat = await metadata.get_operator_catalog(scope="REGULAR")

        # Since response was invalid, fallback is used and invalid data is rejected
        assert cat.source == "BUILTIN_FALLBACK"
        assert cat.is_fallback is True

        await db.dispose()

    asyncio.run(_run())


def test_pipeline_generation_batch_uses_resolved_platform_metadata_catalog(tmp_path: Any) -> None:
    """Verifies AlphaGenerationPipeline resolves operator catalog via PlatformMetadata at runtime."""
    async def _run() -> None:
        import json
        db = Database(tmp_path / "test_pipeline_meta.db")
        await db.create_all()

        mock_endpoints = MagicMock()
        mock_endpoints.list_operators = AsyncMock(
            return_value=[
                Operator(
                    name="ts_rank",
                    category="Time Series",
                    scope=["REGULAR"],
                    definition="ts_rank(x, d)",
                ),
                Operator(
                    name="rank",
                    category="Cross-Sectional",
                    scope=["REGULAR"],
                    definition="rank(x)",
                ),
            ]
        )

        metadata = PlatformMetadata(db, mock_endpoints)
        resolved_cat = await metadata.get_operator_catalog(scope="REGULAR")

        ans_mock = MagicMock()
        ans_mock.text = json.dumps({
            "batch_summary": "Test Batch with Live Metadata",
            "candidates": [
                {
                    "hypothesis": "Test momentum",
                    "expression": "rank(ts_rank(close, 10))",
                    "category": "momentum",
                    "target_dataset": "market_prices",
                    "primary_fields": ["close"],
                }
            ],
        })
        ans_mock.total_tokens = 100
        llm_mock = MagicMock()
        llm_mock.generate = AsyncMock(return_value=ans_mock)

        discovery = CatalogDiscovery(metadata=metadata)
        pipeline = AlphaGenerationPipeline(llm=llm_mock, discovery=discovery, metadata=metadata)

        batch = await pipeline.generate_batch(
            model_ref="google:gemini-2.5-flash",
            category=StrategyCategory.MOMENTUM,
            candidate_count=1,
        )

        assert batch.operator_catalog_version == resolved_cat.catalog_version
        assert len(batch.candidates) == 1
        assert batch.candidates[0].validation.is_valid is True

        await db.dispose()

    asyncio.run(_run())
