"""P1-05: Authoritative Versioned Datafield Catalog Regression Tests.

Verifies:
1. Deterministic catalog fingerprinting for identical definitions.
2. Different material catalog definitions produce different versions.
3. Dataset- and scope-specific allowed-field resolution.
4. A valid catalog field is accepted.
5. A valid core market field is accepted under the documented policy.
6. A valid grouping field is accepted under the documented policy.
7. A hallucinated field is rejected.
8. A known field outside the selected dataset is rejected.
9. A field with established region incompatibility is rejected.
10. A field with established delay incompatibility is rejected.
11. Referenced expression fields never expand the allowed-field set.
12. generate_batch passes intended field context to validation.
13. Invalid expressions are rejected before simulation dispatch.
14. The LLM repair loop receives useful field-validation hints.
15. Repaired expressions are fully revalidated.
16. GenerationAttempt and ResearchCandidate retain catalog-version lineage.
17. Candidate field/family lineage is persisted accurately where available.
18. Built-in fallback behavior is deterministic when external catalog data is unavailable.
19. Missing or incomplete catalog metadata does not silently make an unknown field valid.
20. SQLite migrations remain idempotent and existing records remain compatible.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select, text

from alpha_harness.db.models import (
    CandidateStatus,
    GenerationAttempt,
    ResearchCandidate,
    ResearchResearcher,
    ResearchSession,
)
from alpha_harness.db.sqlite import Database, migrate
from alpha_harness.discovery.catalog import (
    CORE_MARKET_FIELD_NAMES,
    STANDARD_GROUPING_FIELD_NAMES,
    CatalogDiscovery,
    DatafieldCatalog,
    DatasetAllocationPolicy,
    DatasetRegistry,
    DiscoveredDataset,
    DiscoveredField,
)
from alpha_harness.generator.pipeline import (
    AlphaGenerationPipeline,
    CandidateValidationResult,
    GeneratedCandidate,
    GenerationBatch,
    validate_expression,
)
from alpha_harness.orchestrator.researchers import (
    MultiLLMOrchestrator,
    ResearcherConfig,
)
from alpha_harness.prompts.library import StrategyCategory


# =============================================================================
# 1. Deterministic Fingerprinting & Version Sensitivity
# =============================================================================


def test_deterministic_catalog_fingerprinting_for_identical_definitions() -> None:
    """Verifies identical catalog definitions and scopes produce identical SHA-256 versions."""
    registry = DatasetRegistry()
    ds1 = registry.get_dataset("short_interest_us")
    ds2 = registry.get_dataset("short_interest_us")
    assert ds1 is not None and ds2 is not None

    cat1 = registry.get_datafield_catalog(dataset=ds1, region="USA", delay=1)
    cat2 = registry.get_datafield_catalog(dataset=ds2, region="USA", delay=1)

    assert cat1.catalog_version.startswith("df_v1_")
    assert cat1.catalog_version == cat2.catalog_version


def test_different_material_catalog_definitions_produce_different_versions() -> None:
    """Verifies that changes to dataset, fields, region, or delay alter the fingerprint."""
    registry = DatasetRegistry()
    ds_si = registry.get_dataset("short_interest_us")
    ds_fund = registry.get_dataset("fundamental_us")
    assert ds_si is not None and ds_fund is not None

    cat_si = registry.get_datafield_catalog(dataset=ds_si, region="USA", delay=1)
    cat_fund = registry.get_datafield_catalog(dataset=ds_fund, region="USA", delay=1)
    assert cat_si.catalog_version != cat_fund.catalog_version

    # Altering delay
    cat_si_delay2 = registry.get_datafield_catalog(dataset=ds_si, region="USA", delay=2)
    assert cat_si.catalog_version != cat_si_delay2.catalog_version

    # Altering region
    cat_si_glb = registry.get_datafield_catalog(dataset=ds_si, region="GLB", delay=1)
    assert cat_si.catalog_version != cat_si_glb.catalog_version


# =============================================================================
# 2. Scope-Specific Allowed-Field Resolution
# =============================================================================


def test_dataset_and_scope_specific_allowed_field_resolution() -> None:
    """Verifies allowed fields include core market, grouping, and selected dataset fields only."""
    registry = DatasetRegistry()
    ds = registry.get_dataset("short_interest_us")
    assert ds is not None

    catalog = registry.get_datafield_catalog(dataset=ds, region="USA", delay=1)
    allowed = catalog.get_allowed_field_names()

    # Core market fields must be present
    for cmf in CORE_MARKET_FIELD_NAMES:
        assert cmf in allowed

    # Grouping fields must be present
    for gf in STANDARD_GROUPING_FIELD_NAMES:
        assert gf in allowed

    # Selected dataset fields must be present
    assert "borrow_cost" in allowed
    assert "short_interest" in allowed
    assert "shares_on_loan" in allowed

    # Fields from other datasets must NOT be present
    assert "pe" not in allowed
    assert "news_sentiment" not in allowed



# =============================================================================
# 3. Field Acceptance & Rejection Rules
# =============================================================================


def test_valid_catalog_field_is_accepted() -> None:
    """Verifies that an expression referencing an eligible catalog field is accepted."""
    registry = DatasetRegistry()
    ds = registry.get_dataset("short_interest_us")
    cat = registry.get_datafield_catalog(dataset=ds, region="USA", delay=1)

    result = validate_expression("ts_rank(borrow_cost, 20)", datafield_catalog=cat)
    assert result.is_valid is True
    assert "borrow_cost" in result.fields_used


def test_valid_core_market_field_is_accepted_under_policy() -> None:
    """Verifies core market fields are universally permitted across equity scopes."""
    registry = DatasetRegistry()
    cat = registry.get_datafield_catalog(dataset=None, region="USA", delay=1)

    result = validate_expression("rank(close / ts_mean(open, 10))", datafield_catalog=cat)
    assert result.is_valid is True
    assert set(result.fields_used) == {"close", "open"}


def test_valid_grouping_field_is_accepted_under_policy() -> None:
    """Verifies grouping fields are accepted in operators without counting as data fields."""
    registry = DatasetRegistry()
    cat = registry.get_datafield_catalog(dataset=None, region="USA", delay=1)

    result = validate_expression("group_rank(close, sector)", datafield_catalog=cat)
    assert result.is_valid is True
    assert result.field_count == 1  # close only; sector is uncounted
    assert result.fields_used == ["close"]


def test_hallucinated_field_is_rejected() -> None:
    """Verifies expressions containing non-existent hallucinated fields are strictly rejected."""
    registry = DatasetRegistry()
    cat = registry.get_datafield_catalog(dataset=None, region="USA", delay=1)

    result = validate_expression("ts_rank(fantasy_momentum_metric, 20)", datafield_catalog=cat)
    assert result.is_valid is False
    assert "Unknown or ineligible data field: fantasy_momentum_metric" in (result.error or "")


def test_known_field_outside_selected_dataset_is_rejected() -> None:
    """Verifies that a valid field belonging to an unselected dataset is rejected."""
    registry = DatasetRegistry()
    ds_fund = registry.get_dataset("fundamental_us")
    cat_fund = registry.get_datafield_catalog(dataset=ds_fund, region="USA", delay=1)

    # borrow_cost belongs to short_interest_us, not fundamental_us
    result = validate_expression("ts_rank(borrow_cost, 20)", datafield_catalog=cat_fund)
    assert result.is_valid is False
    assert "belongs to dataset 'short_interest_us'" in (result.error or "")


def test_field_with_established_region_incompatibility_is_rejected() -> None:
    """Verifies a field from an incompatible region is rejected."""
    registry = DatasetRegistry()
    # Create custom dataset in EUR region
    eu_field = DiscoveredField(
        field_id="eu_macro_indicator",
        field_type="MATRIX",
        coverage=0.9,
        description="European macro indicator",
        dataset_id="macro_eu",
        category="Macro",
    )
    eu_dataset = DiscoveredDataset(
        dataset_id="macro_eu",
        name="EU Macro Indicators",
        category="Macro",
        description="European economic indicators",
        fields=(eu_field,),
        region="EUR",
        delay=1,
    )
    custom_reg = DatasetRegistry(
        datasets_by_region={
            "USA": list(registry._datasets_by_region.get("USA", [])),
            "EUR": [eu_dataset],
        }
    )
    cat_usa = custom_reg.get_datafield_catalog(dataset=eu_dataset, region="USA", delay=1)

    # Dataset is in EUR, target is USA -> fields should not be allowed
    result = validate_expression("ts_rank(eu_macro_indicator, 10)", datafield_catalog=cat_usa)
    assert result.is_valid is False
    assert "incompatible with target region 'USA'" in (result.error or "")


def test_field_with_established_delay_incompatibility_is_rejected() -> None:
    """Verifies a field requiring higher delay than requested is rejected."""
    delay2_field = DiscoveredField(
        field_id="quarterly_slow_metric",
        field_type="MATRIX",
        coverage=0.85,
        description="Slow accounting metric",
        dataset_id="slow_fund_us",
        category="Fundamental",
    )
    delay2_ds = DiscoveredDataset(
        dataset_id="slow_fund_us",
        name="Slow US Fundamentals",
        category="Fundamental",
        description="Delayed accounting data",
        fields=(delay2_field,),
        region="USA",
        delay=2,  # requires delay 2
    )
    custom_reg = DatasetRegistry(datasets_by_region={"USA": [delay2_ds]})
    cat_delay1 = custom_reg.get_datafield_catalog(dataset=delay2_ds, region="USA", delay=1)

    # Requested delay is 1, dataset requires delay 2
    result = validate_expression("ts_rank(quarterly_slow_metric, 10)", datafield_catalog=cat_delay1)
    assert result.is_valid is False
    assert "exceeding target delay 1" in (result.error or "")


# =============================================================================
# 4. Fix Authoritative Field Validation Bug
# =============================================================================


def test_referenced_expression_fields_never_expand_allowed_field_set() -> None:
    """Verifies that referenced fields in candidate expressions cannot self-authorize."""
    allowed = {"close", "open", "volume"}

    # Expression uses hallucinated field 'mysterious_alpha'
    result = validate_expression(
        "ts_rank(mysterious_alpha, 20)",
        allowed_field_names=allowed,
        check_operators=True,
    )
    assert result.is_valid is False
    assert "Unknown or ineligible data field: mysterious_alpha" in (result.error or "")


# =============================================================================
# 5. Pipeline & Pre-Simulation Gating
# =============================================================================


def test_generate_batch_passes_intended_field_context_to_validation() -> None:
    """Verifies pipeline passes datafield catalog into prompt generation and candidate validation."""
    async def _run() -> None:
        captured_system: str | None = None

        async def mock_generate(system: str, **kwargs: Any) -> Any:
            nonlocal captured_system
            captured_system = system
            resp = MagicMock()
            # Model returns 1 candidate using valid dataset field and 1 candidate using hallucinated field
            resp.text = (
                '{"candidates": ['
                '{"hypothesis": "Valid signal", "expression": "ts_rank(borrow_cost, 20)"},'
                '{"hypothesis": "Invalid signal", "expression": "ts_rank(hallucinated_metric, 10)"}'
                ']}'
            )
            resp.total_tokens = 300
            return resp

        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(side_effect=mock_generate)

        discovery = CatalogDiscovery()
        pipeline = AlphaGenerationPipeline(llm=mock_llm, discovery=discovery)

        batch = await pipeline.generate_batch(
            model_ref="google:gemini-2.5-flash",
            target_dataset_id="short_interest_us",
            region="USA",
            auto_repair=False,
        )

        assert captured_system is not None
        assert "DATAFIELD CATALOG" in captured_system
        assert "`borrow_cost`" in captured_system

        assert batch.datafield_catalog_version is not None
        assert "borrow_cost" in (batch.allowed_fields or [])

        # Candidate 0 is valid, Candidate 1 is invalid due to unknown field
        assert len(batch.candidates) == 2
        assert batch.candidates[0].validation.is_valid is True
        assert batch.candidates[1].validation.is_valid is False
        assert "Unknown or ineligible data field: hallucinated_metric" in (
            batch.candidates[1].validation.error or ""
        )

    asyncio.run(_run())


def test_invalid_expressions_are_rejected_before_simulation_dispatch(tmp_path: Any) -> None:
    """Verifies candidates failing field validation are marked INVALID and not dispatched."""
    async def _run() -> None:
        db_path = tmp_path / "test_p1_05_pre_sim.db"
        db = Database(db_path)
        await db.create_all()

        async def mock_generate(system: str, **kwargs: Any) -> Any:
            resp = MagicMock()
            resp.text = (
                '{"candidates": ['
                '{"hypothesis": "Invalid field candidate", "expression": "ts_rank(hallucinated_score, 10)"}'
                ']}'
            )
            resp.total_tokens = 150
            return resp

        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(side_effect=mock_generate)

        discovery = CatalogDiscovery()
        pipeline = AlphaGenerationPipeline(llm=mock_llm, discovery=discovery)
        orchestrator = MultiLLMOrchestrator(db=db, pipeline=pipeline)

        async with db.session() as session:
            sess = ResearchSession(
                name="Pre-Sim Test Session",
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
                    model_ref="google:gemini-2.5-flash",
                    label="Test Researcher",
                    strategy_focus="momentum",
                    target_dataset_id="short_interest_us",
                    dataset_allocation_policy=DatasetAllocationPolicy.EXPLICIT.value,
                )
            ],
        )

        # Run generation round
        valid_ids = await orchestrator.run_generation_round(
            session_id=sess_id,
            total_target_candidates=1,
            auto_repair=False,
        )

        # Candidate was invalid -> valid_ids must be empty (preventing simulation dispatch)
        assert len(valid_ids) == 0

        async with db.session() as session:
            cand = (await session.execute(select(ResearchCandidate))).scalars().first()
            assert cand is not None
            assert cand.is_valid is False
            assert cand.status == CandidateStatus.INVALID.value
            assert "Unknown or ineligible data field: hallucinated_score" in (cand.validation_error or "")

        await db.dispose()

    asyncio.run(_run())


# =============================================================================
# 6. Expression Repair Loop & Feedback
# =============================================================================


def test_repair_loop_receives_useful_field_validation_hints() -> None:
    """Verifies that the repair prompt contains specific field error and alternative fields."""
    async def _run() -> None:
        captured_user_msgs: list[str] = []

        async def mock_generate(system: str, user: str, **kwargs: Any) -> Any:
            captured_user_msgs.append(user)
            resp = MagicMock()
            # Repaired expression fixes unknown field to borrow_cost
            resp.text = '{"fixed_expression": "ts_rank(borrow_cost, 20)"}'
            resp.total_tokens = 100
            return resp

        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(side_effect=mock_generate)

        discovery = CatalogDiscovery()
        pipeline = AlphaGenerationPipeline(llm=mock_llm, discovery=discovery)

        registry = DatasetRegistry()
        ds = registry.get_dataset("short_interest_us")
        cat = registry.get_datafield_catalog(dataset=ds, region="USA", delay=1)

        result = await pipeline._repair_expression(
            model_ref="google:gemini-2.5-flash",
            hypothesis="Borrow cost spike",
            faulty_expression="ts_rank(bogus_cost, 20)",
            validation_error="Unknown or ineligible data field: bogus_cost",
            max_attempts=1,
            allowed_field_names=cat.get_allowed_field_names(),
            datafield_catalog=cat,
        )

        assert len(captured_user_msgs) == 1
        user_msg = captured_user_msgs[0]
        assert "Unknown or ineligible data field: bogus_cost" in user_msg
        assert "Available eligible fields:" in user_msg
        assert "borrow_cost" in user_msg
        assert result.is_valid is True
        assert result.expression == "ts_rank(borrow_cost, 20)"

    asyncio.run(_run())


def test_repaired_expressions_are_fully_revalidated() -> None:
    """Verifies that repaired expressions are checked against the same catalog scope."""
    async def _run() -> None:
        # LLM attempts to fix bogus_cost with another unknown field fake_metric
        async def mock_generate(system: str, user: str, **kwargs: Any) -> Any:
            resp = MagicMock()
            resp.text = '{"fixed_expression": "ts_rank(fake_metric, 20)"}'
            resp.total_tokens = 100
            return resp

        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(side_effect=mock_generate)

        discovery = CatalogDiscovery()
        pipeline = AlphaGenerationPipeline(llm=mock_llm, discovery=discovery)

        registry = DatasetRegistry()
        ds = registry.get_dataset("short_interest_us")
        cat = registry.get_datafield_catalog(dataset=ds, region="USA", delay=1)

        result = await pipeline._repair_expression(
            model_ref="google:gemini-2.5-flash",
            hypothesis="Borrow cost spike",
            faulty_expression="ts_rank(bogus_cost, 20)",
            validation_error="Unknown or ineligible data field: bogus_cost",
            max_attempts=1,
            allowed_field_names=cat.get_allowed_field_names(),
            datafield_catalog=cat,
        )

        # Repaired expression used fake_metric, which is still invalid
        assert result.is_valid is False
        assert "Unknown or ineligible data field: fake_metric" in (result.error or "")

    asyncio.run(_run())


# =============================================================================
# 7. Persistence & Lineage Tracking
# =============================================================================


def test_generation_attempt_and_research_candidate_retain_catalog_version_lineage(tmp_path: Any) -> None:
    """Verifies datafield_catalog_version is stored on GenerationAttempt and ResearchCandidate."""
    async def _run() -> None:
        db_path = tmp_path / "test_p1_05_lineage.db"
        db = Database(db_path)
        await db.create_all()

        async def mock_generate(system: str, **kwargs: Any) -> Any:
            resp = MagicMock()
            resp.text = (
                '{"candidates": ['
                '{"hypothesis": "Borrow cost anomaly", "expression": "ts_rank(borrow_cost, 20)"}'
                ']}'
            )
            resp.total_tokens = 200
            return resp

        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(side_effect=mock_generate)

        discovery = CatalogDiscovery()
        pipeline = AlphaGenerationPipeline(llm=mock_llm, discovery=discovery)
        orchestrator = MultiLLMOrchestrator(db=db, pipeline=pipeline)

        async with db.session() as session:
            sess = ResearchSession(
                name="Lineage Session",
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
                    model_ref="google:gemini-2.5-flash",
                    label="Lineage Researcher",
                    strategy_focus="short_interest",
                    target_dataset_id="short_interest_us",
                    dataset_allocation_policy=DatasetAllocationPolicy.EXPLICIT.value,
                )
            ],
        )

        valid_ids = await orchestrator.run_generation_round(
            session_id=sess_id,
            total_target_candidates=1,
            auto_repair=False,
        )
        assert len(valid_ids) == 1

        async with db.session() as session:
            attempt = (await session.execute(select(GenerationAttempt))).scalars().first()
            assert attempt is not None
            assert attempt.datafield_catalog_version is not None
            assert attempt.datafield_catalog_version.startswith("df_v1_")
            assert attempt.allowed_fields is not None
            assert "borrow_cost" in attempt.allowed_fields

            cand = (await session.execute(select(ResearchCandidate))).scalars().first()
            assert cand is not None
            assert cand.datafield_catalog_version == attempt.datafield_catalog_version

        await db.dispose()

    asyncio.run(_run())


def test_candidate_field_family_lineage_is_persisted_accurately_where_available(tmp_path: Any) -> None:
    """Verifies that field_families metadata is recorded on ResearchCandidate."""
    async def _run() -> None:
        db_path = tmp_path / "test_p1_05_field_fam.db"
        db = Database(db_path)
        await db.create_all()

        async def mock_generate(system: str, **kwargs: Any) -> Any:
            resp = MagicMock()
            # Uses borrow_cost (Short Interest) and close (Price Volume)
            resp.text = (
                '{"candidates": ['
                '{"hypothesis": "Composite Signal", "expression": "rank(borrow_cost) * rank(close)"}'
                ']}'
            )
            resp.total_tokens = 200
            return resp

        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(side_effect=mock_generate)

        discovery = CatalogDiscovery()
        pipeline = AlphaGenerationPipeline(llm=mock_llm, discovery=discovery)
        orchestrator = MultiLLMOrchestrator(db=db, pipeline=pipeline)

        async with db.session() as session:
            sess = ResearchSession(
                name="Family Lineage Session",
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
                    model_ref="google:gemini-2.5-flash",
                    label="Family Researcher",
                    strategy_focus="short_interest",
                    target_dataset_id="short_interest_us",
                    dataset_allocation_policy=DatasetAllocationPolicy.EXPLICIT.value,
                )
            ],
        )

        await orchestrator.run_generation_round(
            session_id=sess_id,
            total_target_candidates=1,
            auto_repair=False,
        )

        async with db.session() as session:
            cand = (await session.execute(select(ResearchCandidate))).scalars().first()
            assert cand is not None
            assert cand.field_families is not None
            assert cand.field_families.get("close") == "Price Volume"
            assert cand.field_families.get("borrow_cost") == "Short Interest"

        await db.dispose()

    asyncio.run(_run())


# =============================================================================
# 8. Fallback Behavior & Fail-Closed Robustness
# =============================================================================


def test_builtin_fallback_behavior_deterministic_when_external_catalog_data_unavailable() -> None:
    """Verifies that built-in dataset/datafield catalog operates deterministically without external db."""
    discovery = CatalogDiscovery(catalog=None)
    df_cat1 = discovery.get_datafield_catalog(
        dataset=discovery.registry.get_dataset("short_interest_us"),
        region="USA",
        delay=1,
    )
    df_cat2 = discovery.get_datafield_catalog(
        dataset=discovery.registry.get_dataset("short_interest_us"),
        region="USA",
        delay=1,
    )

    assert df_cat1.catalog_version == df_cat2.catalog_version
    assert "borrow_cost" in df_cat1.get_allowed_field_names()


def test_missing_or_incomplete_catalog_metadata_fails_closed() -> None:
    """Verifies validator fails closed when catalog metadata is empty rather than accepting names."""
    # Catalog with no dataset attached
    empty_cat = DatafieldCatalog(dataset=None, region="USA", delay=1)
    allowed = empty_cat.get_allowed_field_names()

    # Only core market and grouping fields are allowed
    assert "fake_field" not in allowed

    res = validate_expression("ts_rank(fake_field, 10)", datafield_catalog=empty_cat)
    assert res.is_valid is False
    assert "Unknown or ineligible data field: fake_field" in (res.error or "")


# =============================================================================
# 9. SQLite Migration Idempotency & Legacy Record Compatibility
# =============================================================================


def test_sqlite_migrations_remain_idempotent_and_legacy_compatible(tmp_path: Any) -> None:
    """Verifies schema migrations add columns idempotently and legacy records have honest NULL defaults."""
    async def _run() -> None:
        db_path = tmp_path / "test_p1_05_migration.db"
        db = Database(db_path)

        # 1. Initial schema creation
        await db.create_all()

        # 2. Insert legacy records lacking P1-05 fields
        async with db.session() as session:
            sess = ResearchSession(name="Legacy Migration Session")
            session.add(sess)
            await session.flush()

            # Insert legacy candidate without P1-05 fields
            await session.execute(
                text(
                    """
                    INSERT INTO research_candidate (session_id, hypothesis, expression, is_valid, status, created_at)
                    VALUES (:sess_id, 'Legacy Alpha', 'close', 1, 'VALID', datetime('now'))
                    """
                ),
                {"sess_id": sess.id},
            )
            # Insert legacy generation attempt without P1-05 fields
            await session.execute(
                text(
                    """
                    INSERT INTO generation_attempt (session_id, model_ref, created_at)
                    VALUES (:sess_id, 'legacy-model', datetime('now'))
                    """
                ),
                {"sess_id": sess.id},
            )

        # 3. Run migration twice
        async with db._engine.begin() as conn:
            await migrate(conn)
        async with db._engine.begin() as conn:
            await migrate(conn)

        # 4. Verify legacy records have honest NULL values, not fabricated versions
        async with db.session() as session:
            cand_row = (await session.execute(select(ResearchCandidate))).scalars().first()
            assert cand_row is not None
            assert cand_row.datafield_catalog_version is None
            assert cand_row.field_families is None

            att_row = (await session.execute(select(GenerationAttempt))).scalars().first()
            assert att_row is not None
            assert att_row.datafield_catalog_version is None
            assert att_row.allowed_fields is None

        await db.dispose()

    asyncio.run(_run())
