"""P1-04: Dataset Selection, Registry, Eligibility, and Allocation Regression Tests.

Verifies:
1. Deterministic catalog versioning and metadata stability.
2. Authoritative eligibility validation and fail-closed rejections (unknown or incompatible datasets).
3. Pre-LLM failure: invalid target datasets fail before any model or simulation invocation.
4. Prompt context injection honoring explicit target_dataset_id.
5. Deterministic, persistent controlled exploration allocation across restarts and concurrency.
6. Documented category-aligned allocation and safe fallbacks for unmapped categories.
7. Persistence of catalog_version, target_dataset_id, policy, and rationale on GenerationAttempt and ResearchCandidate.
8. Clear distinction between target_dataset_id and expression datasets_used.
9. Idempotent SQLite migrations and legacy compatibility.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select, text

from alpha_harness.db.models import (
    DatasetAllocationState,
    GenerationAttempt,
    PromptStrategy,
    PromptVersion,
    ResearchCandidate,
    ResearchResearcher,
    ResearchSession,
)
from alpha_harness.db.sqlite import Database, migrate
from alpha_harness.discovery.catalog import (
    BUILTIN_DATASETS_BY_REGION,
    CatalogDiscovery,
    DatasetAllocationPolicy,
    DatasetAllocationTracker,
    DatasetEligibilityError,
    DatasetNotFoundError,
    DatasetRegistry,
    DatasetSelectionContext,
    DiscoveredDataset,
    DiscoveredField,
)
from alpha_harness.generator.pipeline import (
    AlphaGenerationPipeline,
    CandidateValidationResult,
    GeneratedCandidate,
    GenerationBatch,
)
from alpha_harness.orchestrator.researchers import (
    MultiLLMOrchestrator,
    ResearcherConfig,
)
from alpha_harness.prompts.library import StrategyCategory


# =============================================================================
# 1. Deterministic Catalog Versioning & Metadata Stability
# =============================================================================


def test_deterministic_catalog_versioning() -> None:
    """Verifies that catalog version is a deterministic fingerprint of registered datasets."""
    reg1 = DatasetRegistry()
    reg2 = DatasetRegistry()

    assert reg1.catalog_version.startswith("cat_v1_")
    assert reg1.catalog_version == reg2.catalog_version

    # Mutating metadata produces a distinct catalog version
    mutated_datasets = {
        "USA": [
            DiscoveredDataset(
                dataset_id="custom_ds",
                name="Custom Dataset",
                category="Alternative",
                description="Custom mutated description",
                fields=(DiscoveredField("custom_f", "MATRIX", 0.9, "Field", "custom_ds", "Alternative"),),
                region="USA",
            )
        ]
    }
    reg_custom = DatasetRegistry(datasets_by_region=mutated_datasets)
    assert reg_custom.catalog_version != reg1.catalog_version
    assert reg_custom.catalog_version.startswith("cat_v1_")


def test_registry_metadata_and_version_stability() -> None:
    """Verifies registry exposes built-in datasets with accurate metadata across all regions."""
    registry = DatasetRegistry()

    for region in ("USA", "EUR", "ASI", "CHN", "GLB"):
        eligible = registry.get_eligible_datasets(region=region)
        assert len(eligible) > 0
        for ds in eligible:
            assert ds.dataset_id
            assert ds.name
            assert ds.category
            assert len(ds.fields) > 0
            if region != "GLB":
                assert ds.region.upper() in (region.upper(), "GLB")


# =============================================================================
# 2. Eligibility Validation & Fail-Closed Rejections
# =============================================================================


def test_successful_eligibility_validation() -> None:
    """Verifies valid combinations of dataset ID, region, and delay pass eligibility."""
    registry = DatasetRegistry()

    # Fundamental US in USA
    ds_us = registry.validate_dataset_eligibility("fundamental_us", region="USA", delay=1)
    assert ds_us.dataset_id == "fundamental_us"
    assert ds_us.region == "USA"

    # Global sentiment is eligible in USA
    ds_glb = registry.validate_dataset_eligibility("sentiment_global", region="USA", delay=1)
    assert ds_glb.dataset_id == "sentiment_global"
    assert ds_glb.region == "GLB"

    # European fundamental in EUR
    ds_eu = registry.validate_dataset_eligibility("fundamental_eu", region="EUR", delay=1)
    assert ds_eu.dataset_id == "fundamental_eu"
    assert ds_eu.region == "EUR"


def test_rejection_of_unknown_dataset_ids() -> None:
    """Verifies that unknown or empty dataset IDs fail closed with DatasetNotFoundError."""
    registry = DatasetRegistry()

    with pytest.raises(DatasetNotFoundError, match="not found in authoritative catalog"):
        registry.validate_dataset_eligibility("non_existent_dataset_123", region="USA")

    with pytest.raises(DatasetNotFoundError, match="cannot be empty"):
        registry.validate_dataset_eligibility("", region="USA")


def test_rejection_of_demonstrably_incompatible_dataset_requests() -> None:
    """Verifies that region-incompatible datasets fail closed with DatasetEligibilityError."""
    registry = DatasetRegistry()

    # fundamental_eu requested for USA region
    with pytest.raises(DatasetEligibilityError, match="incompatible with target region 'USA'"):
        registry.validate_dataset_eligibility("fundamental_eu", region="USA")

    # fundamental_us requested for EUR region
    with pytest.raises(DatasetEligibilityError, match="incompatible with target region 'EUR'"):
        registry.validate_dataset_eligibility("fundamental_us", region="EUR")

    # fundamental_china requested for ASI region
    with pytest.raises(DatasetEligibilityError, match="incompatible with target region 'ASI'"):
        registry.validate_dataset_eligibility("fundamental_china", region="ASI")


# =============================================================================
# 3. Pre-LLM Failure & Pipeline Target Flow
# =============================================================================


def test_invalid_requests_fail_before_any_llm_call() -> None:
    """Verifies that an invalid target dataset fails closed before any LLM API invocation."""
    async def _run() -> None:
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock()

        discovery = CatalogDiscovery()
        pipeline = AlphaGenerationPipeline(llm=mock_llm, discovery=discovery)

        # 1. Unknown dataset ID fails before LLM call
        with pytest.raises(DatasetNotFoundError):
            await pipeline.generate_batch(
                model_ref="google:gemini-2.5-flash",
                target_dataset_id="invalid_dataset_xyz",
                region="USA",
            )
        mock_llm.generate.assert_not_called()

        # 2. Region-incompatible dataset fails before LLM call
        with pytest.raises(DatasetEligibilityError):
            await pipeline.generate_batch(
                model_ref="google:gemini-2.5-flash",
                target_dataset_id="fundamental_eu",
                region="USA",
            )
        mock_llm.generate.assert_not_called()

    asyncio.run(_run())


def test_explicit_target_dataset_id_reaches_prompt_generation() -> None:
    """Verifies that explicit target_dataset_id is injected into prompt and returned in batch."""
    async def _run() -> None:
        captured_system_prompt: str | None = None

        async def mock_generate(system: str, **kwargs: Any) -> Any:
            nonlocal captured_system_prompt
            captured_system_prompt = system
            resp = MagicMock()
            resp.text = '{"candidates": [{"hypothesis": "Borrow cost anomaly", "expression": "ts_rank(borrow_cost, 20)"}]}'
            resp.total_tokens = 250
            return resp

        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(side_effect=mock_generate)

        discovery = CatalogDiscovery()
        pipeline = AlphaGenerationPipeline(llm=mock_llm, discovery=discovery)

        batch = await pipeline.generate_batch(
            model_ref="google:gemini-2.5-flash",
            target_dataset_id="short_interest_us",
            region="USA",
        )

        # Prompt received the targeted dataset and its fields
        assert captured_system_prompt is not None
        assert "DATASET: `short_interest_us`" in captured_system_prompt
        assert "`borrow_cost`" in captured_system_prompt

        # Returned batch and candidates retain lineage
        assert batch.target_dataset_id == "short_interest_us"
        assert batch.dataset_selection_policy == DatasetAllocationPolicy.EXPLICIT.value
        assert batch.catalog_version == discovery.registry.catalog_version
        assert len(batch.candidates) == 1
        assert batch.candidates[0].target_dataset == "short_interest_us"

    asyncio.run(_run())


# =============================================================================
# 4. Controlled Exploration: Balanced Coverage, Concurrency & Reproducibility
# =============================================================================


def test_controlled_exploration_balanced_coverage_across_cycles() -> None:
    """Verifies that eligible datasets receive balanced, non-starving coverage over scheduling cycles."""
    registry = DatasetRegistry()
    eligible = registry.get_eligible_datasets(region="USA")
    num_eligible = len(eligible)
    assert num_eligible >= 4

    # Case A: 2 researchers over num_eligible generations (2 * num_eligible allocations total)
    num_researchers = 2
    generations = num_eligible
    allocations_a: list[str] = []
    for gen in range(generations):
        for r_idx in range(num_researchers):
            ctx = registry.allocate_dataset(
                policy=DatasetAllocationPolicy.CONTROLLED_EXPLORATION,
                region="USA",
                session_id=100,
                generation=gen,
                researcher_index=r_idx,
                num_researchers=num_researchers,
            )
            allocations_a.append(ctx.selected_dataset_id)

    total_alloc_a = num_researchers * generations
    assert len(allocations_a) == total_alloc_a
    # Over 2 complete scheduling cycles, each dataset receives exactly 2 allocations
    counts_a = {ds.dataset_id: allocations_a.count(ds.dataset_id) for ds in eligible}
    for ds_id, count in counts_a.items():
        assert count == 2, f"Dataset '{ds_id}' received {count} allocations, expected balanced count of 2"

    # Case B: num_eligible researchers over 3 generations (3 * num_eligible allocations total)
    num_researchers_b = num_eligible
    generations_b = 3
    allocations_b: list[str] = []
    for gen in range(generations_b):
        for r_idx in range(num_researchers_b):
            ctx = registry.allocate_dataset(
                policy=DatasetAllocationPolicy.CONTROLLED_EXPLORATION,
                region="USA",
                session_id=200,
                generation=gen,
                researcher_index=r_idx,
                num_researchers=num_researchers_b,
            )
            allocations_b.append(ctx.selected_dataset_id)

    total_alloc_b = num_researchers_b * generations_b
    assert len(allocations_b) == total_alloc_b
    # Over 3 complete scheduling cycles, each dataset receives exactly 3 allocations
    counts_b = {ds.dataset_id: allocations_b.count(ds.dataset_id) for ds in eligible}
    for ds_id, count in counts_b.items():
        assert count == 3, f"Dataset '{ds_id}' received {count} allocations, expected balanced count of 3"


def test_controlled_exploration_distinct_allocations_within_round() -> None:
    """Verifies distinct dataset allocation within a round when researchers <= eligible datasets,
    and balanced reuse when researchers > eligible datasets.
    """
    registry = DatasetRegistry()
    eligible = registry.get_eligible_datasets(region="USA")
    num_eligible = len(eligible)
    assert num_eligible >= 4

    # 1. When researchers <= eligible (M = num_eligible): strictly distinct assignments
    allocations_exact = [
        registry.allocate_dataset(
            policy=DatasetAllocationPolicy.CONTROLLED_EXPLORATION,
            region="USA",
            session_id=1,
            generation=0,
            researcher_index=i,
            num_researchers=num_eligible,
        ).selected_dataset_id
        for i in range(num_eligible)
    ]
    assert len(set(allocations_exact)) == num_eligible, "Expected all distinct datasets when M == N"

    # 2. When researchers < eligible (M = 3 < num_eligible): strictly distinct assignments
    allocations_subset = [
        registry.allocate_dataset(
            policy=DatasetAllocationPolicy.CONTROLLED_EXPLORATION,
            region="USA",
            session_id=1,
            generation=0,
            researcher_index=i,
            num_researchers=3,
        ).selected_dataset_id
        for i in range(3)
    ]
    assert len(set(allocations_subset)) == 3, "Expected all distinct datasets when M < N"

    # 3. When researchers > eligible (M = num_eligible + 2): covers all datasets with balanced reuse
    m_excess = num_eligible + 2
    allocations_excess = [
        registry.allocate_dataset(
            policy=DatasetAllocationPolicy.CONTROLLED_EXPLORATION,
            region="USA",
            session_id=1,
            generation=0,
            researcher_index=i,
            num_researchers=m_excess,
        ).selected_dataset_id
        for i in range(m_excess)
    ]
    assert len(set(allocations_excess)) == num_eligible
    # The first two datasets wrap around in balanced order
    assert allocations_excess[num_eligible] == allocations_excess[0]
    assert allocations_excess[num_eligible + 1] == allocations_excess[1]


def test_controlled_exploration_reproducibility_after_restart(tmp_path: Any) -> None:
    """Verifies reproducibility across process restarts for both registry and persisted tracker."""
    # 1. Pure registry reproducibility
    reg1 = DatasetRegistry()
    reg2 = DatasetRegistry()

    ctx1 = reg1.allocate_dataset(
        policy=DatasetAllocationPolicy.CONTROLLED_EXPLORATION,
        region="USA",
        session_id=10,
        generation=2,
        researcher_index=1,
        num_researchers=3,
    )
    ctx2 = reg2.allocate_dataset(
        policy=DatasetAllocationPolicy.CONTROLLED_EXPLORATION,
        region="USA",
        session_id=10,
        generation=2,
        researcher_index=1,
        num_researchers=3,
    )

    assert ctx1.selected_dataset_id == ctx2.selected_dataset_id
    assert ctx1.catalog_version == ctx2.catalog_version
    assert ctx1.selection_reason == ctx2.selection_reason
    assert ctx1.allocation_policy == DatasetAllocationPolicy.CONTROLLED_EXPLORATION.value

    # 2. Persisted tracker reproducibility across simulated process restarts
    async def _run() -> None:
        db_path = tmp_path / "test_tracker_restart.db"
        db = Database(db_path)
        await db.create_all()

        registry = DatasetRegistry()
        eligible = registry.get_eligible_datasets(region="USA")

        # Session 1: allocate 2 datasets with first tracker instance
        tracker1 = DatasetAllocationTracker(db=db)
        ds1, slot1 = await tracker1.get_next_allocation(session_id=77, scope_key="USA", eligible_datasets=eligible)
        ds2, slot2 = await tracker1.get_next_allocation(session_id=77, scope_key="USA", eligible_datasets=eligible)

        assert ds1.dataset_id == eligible[0].dataset_id
        assert ds2.dataset_id == eligible[1].dataset_id

        # Simulate restart: dispose old connections and instantiate new tracker
        await db.dispose()
        db_restarted = Database(db_path)
        tracker2 = DatasetAllocationTracker(db=db_restarted)

        # Allocate next 2 datasets with fresh tracker instance
        ds3, slot3 = await tracker2.get_next_allocation(session_id=77, scope_key="USA", eligible_datasets=eligible)
        ds4, slot4 = await tracker2.get_next_allocation(session_id=77, scope_key="USA", eligible_datasets=eligible)

        assert ds3.dataset_id == eligible[2].dataset_id
        assert ds4.dataset_id == eligible[3].dataset_id
        await db_restarted.dispose()

    asyncio.run(_run())


def test_concurrent_allocation_consistency(tmp_path: Any) -> None:
    """Verifies concurrent allocations produce consistent selection records and avoid race conditions."""
    async def _run() -> None:
        # 1. Pure registry concurrency
        registry = DatasetRegistry()

        async def _allocate(i: int) -> DatasetSelectionContext:
            return registry.allocate_dataset(
                policy=DatasetAllocationPolicy.CONTROLLED_EXPLORATION,
                region="USA",
                session_id=42,
                generation=1,
                researcher_id=i % 4,
            )

        tasks = [_allocate(i) for i in range(20)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 20
        # Consistency check: tasks with identical coordinates yield identical selection
        assert results[0].selected_dataset_id == results[4].selected_dataset_id
        assert results[0].selected_dataset_id == results[8].selected_dataset_id

        # 2. Persisted tracker concurrency with transactional SQLite coordination
        db_path = tmp_path / "test_tracker_concurrency.db"
        db = Database(db_path)
        await db.create_all()

        eligible = registry.get_eligible_datasets(region="USA")
        num_eligible = len(eligible)
        tracker = DatasetAllocationTracker(db=db)

        # 2 complete scheduling cycles across concurrent tasks
        total_tasks = num_eligible * 2

        async def _concurrent_task() -> tuple[DiscoveredDataset, int]:
            return await tracker.get_next_allocation(
                session_id=99,
                scope_key="USA",
                eligible_datasets=eligible,
            )

        concurrent_allocations = await asyncio.gather(*[_concurrent_task() for _ in range(total_tasks)])
        assert len(concurrent_allocations) == total_tasks

        # Verify sequential slot assignment without race conditions or dropped counts
        slots = [slot for _, slot in concurrent_allocations]
        for s in range(num_eligible):
            assert slots.count(s) == 2, f"Slot {s} was allocated {slots.count(s)} times, expected 2"

        await db.dispose()

    asyncio.run(_run())



def test_controlled_exploration_changing_eligible_set() -> None:
    """Verifies that changes in the eligible dataset set never select an ineligible dataset."""
    registry = DatasetRegistry()

    # Full eligible set for USA
    all_eligible = registry.get_eligible_datasets(region="USA")
    assert len(all_eligible) >= 4

    # Restrict eligible set to only 2 datasets
    subset_eligible = all_eligible[:2]
    ineligible_ds_ids = {d.dataset_id for d in all_eligible[2:]}

    # Custom registry with only the subset
    custom_datasets = {"USA": list(subset_eligible)}
    reg_restricted = DatasetRegistry(datasets_by_region=custom_datasets)

    # Allocations across several generations must never select any excluded dataset
    for gen in range(5):
        for r_idx in range(3):
            ctx = reg_restricted.allocate_dataset(
                policy=DatasetAllocationPolicy.CONTROLLED_EXPLORATION,
                region="USA",
                session_id=1,
                generation=gen,
                researcher_index=r_idx,
            )
            assert ctx.selected_dataset_id not in ineligible_ds_ids
            assert ctx.selected_dataset_id in {d.dataset_id for d in subset_eligible}


# =============================================================================
# 5. Category Alignment & Safe Fallback
# =============================================================================


def test_category_alignment_uses_documented_mappings_only() -> None:
    """Verifies documented category mappings select the expected dataset category."""
    registry = DatasetRegistry()

    # Value -> Fundamental
    ctx_val = registry.allocate_dataset(
        policy=DatasetAllocationPolicy.CATEGORY_ALIGNED,
        strategy_category="value",
        region="USA",
    )
    assert ctx_val.dataset.category == "Fundamental"
    assert ctx_val.allocation_policy == DatasetAllocationPolicy.CATEGORY_ALIGNED.value

    # Quality -> Fundamental
    ctx_qual = registry.allocate_dataset(
        policy=DatasetAllocationPolicy.CATEGORY_ALIGNED,
        strategy_category="quality",
        region="USA",
    )
    assert ctx_qual.dataset.category == "Fundamental"

    # Sentiment -> Sentiment
    ctx_sent = registry.allocate_dataset(
        policy=DatasetAllocationPolicy.CATEGORY_ALIGNED,
        strategy_category="sentiment",
        region="USA",
    )
    assert ctx_sent.dataset.category == "Sentiment"

    # Analyst -> Analyst
    ctx_analyst = registry.allocate_dataset(
        policy=DatasetAllocationPolicy.CATEGORY_ALIGNED,
        strategy_category="analyst",
        region="USA",
    )
    assert ctx_analyst.dataset.category == "Analyst"

    # Short Interest -> Short Interest
    ctx_si = registry.allocate_dataset(
        policy=DatasetAllocationPolicy.CATEGORY_ALIGNED,
        strategy_category="short_interest",
        region="USA",
    )
    assert ctx_si.dataset.category == "Short Interest"


def test_safe_fallback_when_category_unmapped() -> None:
    """Verifies unmapped categories document the limitation and fall back to controlled exploration."""
    registry = DatasetRegistry()

    # General category has no single domain -> falls back to controlled exploration
    ctx_gen = registry.allocate_dataset(
        policy=DatasetAllocationPolicy.CATEGORY_ALIGNED,
        strategy_category="general",
        region="USA",
    )
    assert ctx_gen.allocation_policy == DatasetAllocationPolicy.CONTROLLED_EXPLORATION.value
    assert "general category baseline" in ctx_gen.selection_reason

    # Arbitrary unmapped category -> falls back to controlled exploration with documented reason
    ctx_unmapped = registry.allocate_dataset(
        policy=DatasetAllocationPolicy.CATEGORY_ALIGNED,
        strategy_category="exotic_arbitrage",
        region="USA",
    )
    assert ctx_unmapped.allocation_policy == DatasetAllocationPolicy.CONTROLLED_EXPLORATION.value
    assert "Fallback from unmapped category 'exotic_arbitrage'" in ctx_unmapped.selection_reason


# =============================================================================
# 6. Database Lineage & Orchestrator Integration
# =============================================================================


def test_candidate_and_attempt_persist_dataset_lineage(tmp_path: Any) -> None:
    """Verifies that orchestrator generation round persists dataset lineage to database."""
    async def _run() -> None:
        db_path = tmp_path / "test_p1_04_lineage.db"
        db = Database(db_path)
        await db.create_all()

        # Create dummy candidate batch
        val = CandidateValidationResult(
            is_valid=True,
            expression="ts_rank(borrow_cost, 20)",
            operator_count=1,
            field_count=1,
            fields_used=["borrow_cost"],
        )
        cand = GeneratedCandidate(
            hypothesis="High borrow cost signal",
            expression="ts_rank(borrow_cost, 20)",
            category="short_interest",
            target_dataset="short_interest_us",
            primary_fields=["borrow_cost"],
            validation=val,
        )
        batch = GenerationBatch(
            summary="Test generation round",
            candidates=[cand],
            model_ref="google:gemini-2.5-flash",
            total_tokens=150,
            duration_seconds=1.0,
            target_dataset_id="short_interest_us",
            catalog_version="cat_v1_test123",
            dataset_selection_policy=DatasetAllocationPolicy.EXPLICIT.value,
            dataset_selection_reason="Explicit researcher target",
        )

        pipeline = AlphaGenerationPipeline(llm=MagicMock())
        pipeline.generate_batch = AsyncMock(return_value=batch)

        orch = MultiLLMOrchestrator(db=db, pipeline=pipeline)

        # Initialize session and researchers with explicit target dataset
        async with db.session() as session:
            sess = ResearchSession(name="P1-04 Test Session", region="USA", universe="TOP3000")
            session.add(sess)
            await session.flush()
            sess_id = sess.id

        roster = [
            ResearcherConfig(
                model_ref="google:gemini-2.5-flash",
                label="Short Interest Specialist",
                strategy_focus=StrategyCategory.GENERAL,
                target_dataset_id="short_interest_us",
            )
        ]
        await orch.initialize_session_researchers(sess_id, custom_roster=roster)

        # Run generation round
        cand_ids = await orch.run_generation_round(sess_id, total_target_candidates=1)
        assert len(cand_ids) >= 1

        # Verify database persistence
        async with db.session() as session:
            # 1. Verify GenerationAttempt has dataset selection lineage
            att_stmt = select(GenerationAttempt).where(GenerationAttempt.session_id == sess_id)
            attempt = (await session.execute(att_stmt)).scalars().first()
            assert attempt is not None
            assert attempt.target_dataset_id == "short_interest_us"
            assert attempt.catalog_version is not None
            assert attempt.dataset_selection_policy == DatasetAllocationPolicy.EXPLICIT.value
            assert "Explicit" in str(attempt.dataset_selection_reason)

            # 2. Verify ResearchCandidate retains dataset lineage and distinguishes datasets_used
            c_stmt = select(ResearchCandidate).where(ResearchCandidate.id.in_(cand_ids))
            candidates = list((await session.execute(c_stmt)).scalars().all())
            assert len(candidates) >= 1
            for c in candidates:
                assert c.target_dataset_id == "short_interest_us"
                assert c.catalog_version is not None
                assert c.dataset_selection_policy == DatasetAllocationPolicy.EXPLICIT.value
                assert c.dataset_selection_reason is not None
                # Target dataset is distinguished from fields/datasets used
                assert c.fields_used == ["borrow_cost"]

        await db.dispose()

    asyncio.run(_run())


def test_candidate_lineage_distinguishes_target_from_datasets_used() -> None:
    """Verifies that target_dataset_id is not confused with expression datasets_used."""
    val = CandidateValidationResult(
        is_valid=True,
        expression="rank(close / ts_mean(close, 20))",  # Uses core market fields
        operator_count=2,
        field_count=1,
        fields_used=["close"],
    )
    cand = GeneratedCandidate(
        hypothesis="Core price reversion under fundamental target",
        expression="rank(close / ts_mean(close, 20))",
        category="value",
        target_dataset="fundamental_us",  # Target was fundamental_us
        primary_fields=["close"],
        validation=val,
    )

    # Target is fundamental_us, but expression only referenced core market field
    assert cand.target_dataset == "fundamental_us"
    assert "close" in cand.validation.fields_used
    assert "fundamental_us" not in cand.validation.fields_used


# =============================================================================
# 7. SQLite Migration Idempotency & Legacy Compatibility
# =============================================================================


def test_sqlite_migration_idempotent_and_legacy_compatible(tmp_path: Any) -> None:
    """Verifies migrations add columns and backfill legacy records idempotently without regressions."""
    async def _run() -> None:
        db_path = tmp_path / "test_p1_04_migration.db"
        db = Database(db_path)

        # First migration
        await db.create_all()

        # Insert a simulated legacy record missing dataset selection
        async with db.session() as session:
            sess = ResearchSession(name="Legacy Migration Session")
            session.add(sess)
            await session.flush()

            # Insert legacy candidate with explicit raw SQL without target_dataset_id
            await session.execute(
                text(
                    """
                    INSERT INTO research_candidate (session_id, hypothesis, expression, is_valid, status, created_at)
                    VALUES (:sess_id, 'Legacy alpha', 'close', 1, 'VALID', datetime('now'))
                    """
                ),
                {"sess_id": sess.id},
            )
            # Insert legacy generation attempt without dataset fields
            await session.execute(
                text(
                    """
                    INSERT INTO generation_attempt (session_id, model_ref, created_at)
                    VALUES (:sess_id, 'legacy-model', datetime('now'))
                    """
                ),
                {"sess_id": sess.id},
            )

        # Re-run migration (idempotency check)
        async with db._engine.begin() as conn:
            await migrate(conn)

        # Verify legacy records have documented non-fabricated fallback policy
        async with db.session() as session:
            cand_row = (await session.execute(select(ResearchCandidate))).scalars().first()
            assert cand_row is not None
            assert cand_row.dataset_selection_policy == "LEGACY_UNSPECIFIED"
            assert cand_row.target_dataset_id is None

            att_row = (await session.execute(select(GenerationAttempt))).scalars().first()
            assert att_row is not None
            assert att_row.dataset_selection_policy == "LEGACY_UNSPECIFIED"

        # Re-run migration third time
        async with db._engine.begin() as conn:
            await migrate(conn)

        await db.dispose()

    asyncio.run(_run())
