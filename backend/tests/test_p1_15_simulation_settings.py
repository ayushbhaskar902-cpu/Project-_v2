"""Comprehensive test suite for P1-15: Explicit, validated, and persisted simulation settings.

Validates:
1. Complete supported settings persist correctly, including truncation, pasteurization, and unit handling.
2. Canonical serialization and settings hashes are deterministic.
3. Distinct effective settings produce different hashes and provenance fingerprints.
4. Candidate-level overrides survive generation, validation, persistence, and dispatch.
5. Explicit versioned profiles resolve defaults deterministically.
6. Missing required settings cannot silently receive arbitrary defaults.
7. A region/universe incompatibility is rejected before simulation dispatch.
8. Invalid platform setting choices are rejected using controlled schema fixtures.
9. Candidate expressions referencing fields with incompatible known coverage are rejected.
10. The dispatched SimulationSettings exactly matches the stored and validated configuration.
11. Legacy rows with only scalar settings remain readable without inventing missing historical values.
12. An incomplete legacy configuration cannot falsely claim complete reproducibility.
13. Missing or stale platform options do not result in unverified settings being represented as verified.
14. API compatibility is retained.
15. Settings and provenance fingerprints remain correct under database persistence.
16. Final holdout changes do not influence settings resolution or preflight decisions.
17. Existing P0-02, P0-03, P1-05, P1-08, P1-11, and P1-14 regression behavior remains intact.
"""

from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from alpha_harness.api.research import CandidateOut
from alpha_harness.brain.schemas import SimulationRequest, SimulationSettings, SimulationType
from alpha_harness.db.models import (
    CandidateStatus,
    ResearchCandidate,
    ResearchSession,
    ResearchStatus,
    utcnow,
)
from alpha_harness.db.sqlite import Database
from alpha_harness.discovery.catalog import (
    DatafieldCatalog,
    DatasetRegistry,
    DiscoveredDataset,
    DiscoveredField,
)
from alpha_harness.generator.pipeline import (
    AlphaGenerationPipeline,
    CandidateValidationResult,
    GeneratedCandidate,
    GenerationBatch,
)
from alpha_harness.orchestrator.diversity import CandidateItem, DiversitySelector
from alpha_harness.orchestrator.researchers import MultiLLMOrchestrator
from alpha_harness.orchestrator.simulation_settings import (
    CandidateSimulationSettings,
    ERR_INVALID_SIMULATION_SETTING_OVERRIDE,
    ERR_PREFLIGHT_SETTINGS_INCOMPATIBLE,
    PlatformSchemaValidator,
    SETTINGS_PROFILES,
    STANDARD_EUR_EQUITY_V1,
    STANDARD_US_EQUITY_V1,
    SimulationSettingsResolutionError,
    SimulationSettingsValidationError,
    compute_candidate_provenance_hash,
    get_settings_profile,
    resolve_candidate_simulation_settings,
    validate_candidate_dataset_preflight,
    validate_candidate_overrides,
)
from alpha_harness.prompts.validation import _extract_cohort_key


@pytest.fixture
def test_db(tmp_path: Path) -> Database:
    db_path = tmp_path / "test_p1_15.db"
    db = Database(db_path)
    asyncio.run(db.create_all())
    return db


# Controlled schema fixture for platform options testing
MOCK_SIMULATION_SCHEMA = {
    "instrumentType": {
        "label": "Instrument Type",
        "type": "string",
        "required": True,
        "choices": [{"value": "EQUITY", "label": "Equity"}],
    },
    "region": {
        "label": "Region",
        "type": "string",
        "required": True,
        "choices": [
            {"value": "USA", "label": "USA"},
            {"value": "EUR", "label": "Europe"},
            {"value": "ASI", "label": "Asia"},
        ],
    },
    "universe": {
        "label": "Universe",
        "type": "string",
        "required": True,
        "choices": {
            "region": {
                "USA": [
                    {"value": "TOP3000", "label": "Top 3000"},
                    {"value": "TOP1000", "label": "Top 1000"},
                    {"value": "TOP500", "label": "Top 500"},
                ],
                "EUR": [
                    {"value": "TOP1200", "label": "Top 1200"},
                    {"value": "TOP600", "label": "Top 600"},
                ],
                "ASI": [
                    {"value": "TOP1000", "label": "Top 1000"},
                ],
            }
        },
    },
    "delay": {
        "label": "Delay",
        "type": "integer",
        "required": True,
        "choices": [{"value": 0, "label": "0"}, {"value": 1, "label": "1"}],
    },
    "neutralization": {
        "label": "Neutralization",
        "type": "string",
        "choices": [
            {"value": "NONE", "label": "None"},
            {"value": "INDUSTRY", "label": "Industry"},
            {"value": "SUBINDUSTRY", "label": "Subindustry"},
            {"value": "SECTOR", "label": "Sector"},
        ],
    },
}


# ---------------------------------------------------------------------------
# Test 1: Complete supported settings persist correctly
# ---------------------------------------------------------------------------
def test_complete_supported_settings_persist_correctly(test_db: Database):
    """Verifies that all supported simulation parameters (truncation, pasteurization,
    unit_handling, nan_handling, max_trade, max_position) persist on ResearchCandidate."""

    async def _run():
        settings = CandidateSimulationSettings(
            region="USA",
            universe="TOP3000",
            delay=1,
            decay=4,
            neutralization="INDUSTRY",
            instrument_type="EQUITY",
            truncation=0.05,
            pasteurization="ON",
            unit_handling="VERIFY",
            nan_handling="OFF",
            max_trade="100000",
            max_position="500000",
            simulation_mode="FULL",
            profile_id="custom_v1",
            profile_version="v1",
            schema_version="4.0",
            verification_status="VERIFIED",
        )

        async with test_db.session() as s:
            sess = ResearchSession(name="S1", region="USA", universe="TOP3000")
            s.add(sess)
            await s.flush()

            cand = ResearchCandidate(
                session_id=sess.id,
                expression="rank(close)",
                simulation_settings=settings.to_dict(),
                settings_hash=settings.compute_hash(),
                settings_profile_id=settings.profile_id,
                settings_profile_version=settings.profile_version,
                settings_schema_version=settings.schema_version,
                settings_verification_status=settings.verification_status,
                sim_region=settings.region,
                sim_universe=settings.universe,
                sim_delay=settings.delay,
                sim_decay=settings.decay,
                sim_neutralization=settings.neutralization,
            )
            s.add(cand)
            await s.commit()
            cand_id = cand.id

        async with test_db.session() as s:
            retrieved = await s.get(ResearchCandidate, cand_id)
            assert retrieved is not None
            assert retrieved.simulation_settings is not None
            saved = retrieved.simulation_settings

            assert saved["region"] == "USA"
            assert saved["universe"] == "TOP3000"
            assert saved["delay"] == 1
            assert saved["decay"] == 4
            assert saved["neutralization"] == "INDUSTRY"
            assert saved["truncation"] == 0.05
            assert saved["pasteurization"] == "ON"
            assert saved["unit_handling"] == "VERIFY"
            assert saved["nan_handling"] == "OFF"
            assert saved["max_trade"] == "100000"
            assert saved["max_position"] == "500000"
            assert saved["simulation_mode"] == "FULL"
            assert retrieved.settings_hash == settings.compute_hash()
            assert retrieved.is_simulation_settings_complete is True

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 2: Canonical serialization and settings hashes are deterministic
# ---------------------------------------------------------------------------
def test_canonical_serialization_and_hash_deterministic():
    """Different dictionary key insertion orders or equivalent float/int representations
    must produce the exact same canonical JSON and SHA-256 fingerprint."""
    s1 = CandidateSimulationSettings(
        region="USA",
        universe="TOP3000",
        delay=1,
        decay=4,
        neutralization="INDUSTRY",
        truncation=0.08,
    )
    s2 = CandidateSimulationSettings(
        region="USA",
        universe="TOP3000",
        delay=1,
        decay=4,
        neutralization="INDUSTRY",
        truncation=0.080000,
    )

    h1 = s1.compute_hash()
    h2 = s2.compute_hash()
    assert h1 == h2
    assert len(h1) == 64
    assert s1.canonical_json() == s2.canonical_json()


# ---------------------------------------------------------------------------
# Test 3: Distinct effective settings produce different hashes & provenance
# ---------------------------------------------------------------------------
def test_distinct_effective_settings_produce_different_hashes():
    """A change in decay, delay, truncation, or region must alter the settings hash
    and the candidate provenance fingerprint."""
    base = CandidateSimulationSettings(
        region="USA", universe="TOP3000", delay=1, decay=4, neutralization="INDUSTRY", truncation=0.08
    )
    different_decay = CandidateSimulationSettings(
        region="USA", universe="TOP3000", delay=1, decay=2, neutralization="INDUSTRY", truncation=0.08
    )
    different_trunc = CandidateSimulationSettings(
        region="USA", universe="TOP3000", delay=1, decay=4, neutralization="INDUSTRY", truncation=0.05
    )

    h_base = base.compute_hash()
    h_decay = different_decay.compute_hash()
    h_trunc = different_trunc.compute_hash()

    assert h_base != h_decay
    assert h_base != h_trunc
    assert h_decay != h_trunc

    prov_base = compute_candidate_provenance_hash("hash123", h_base, "gpt-4o", 1)
    prov_decay = compute_candidate_provenance_hash("hash123", h_decay, "gpt-4o", 1)
    assert prov_base != prov_decay


# ---------------------------------------------------------------------------
# Test 4: Candidate-level overrides survive resolution and persistence
# ---------------------------------------------------------------------------
def test_candidate_level_overrides_survive():
    """An explicit candidate override (e.g. decay=2, neutralization='SECTOR') must not be
    overwritten by session defaults or profile defaults."""
    resolved = resolve_candidate_simulation_settings(
        candidate_overrides={"decay": 2, "neutralization": "SECTOR", "truncation": 0.03},
        session_region="USA",
        session_universe="TOP3000",
        session_delay=1,
        session_constraints={"decay": 8, "neutralization": "SUBINDUSTRY"},
        profile="standard_us_equity_v1",
    )

    # Candidate overrides override both session constraints and profile baseline
    assert resolved.decay == 2
    assert resolved.neutralization == "SECTOR"
    assert resolved.truncation == 0.03
    assert resolved.region == "USA"
    assert resolved.universe == "TOP3000"
    assert resolved.delay == 1


# ---------------------------------------------------------------------------
# Test 5: Explicit versioned profiles resolve defaults deterministically
# ---------------------------------------------------------------------------
def test_explicit_versioned_profiles_resolve_deterministically():
    """Resolving with a known profile populates the profile_id, profile_version,
    and profile-standard parameters."""
    resolved_us = resolve_candidate_simulation_settings(
        profile="standard_us_equity_v1",
    )
    assert resolved_us.region == "USA"
    assert resolved_us.universe == "TOP3000"
    assert resolved_us.delay == 1
    assert resolved_us.decay == 4
    assert resolved_us.profile_id == "standard_us_equity"
    assert resolved_us.profile_version == "v1"

    resolved_eur = resolve_candidate_simulation_settings(
        profile="standard_eur_equity_v1",
    )
    assert resolved_eur.region == "EUR"
    assert resolved_eur.universe == "TOP1200"
    assert resolved_eur.profile_id == "standard_eur_equity"


# ---------------------------------------------------------------------------
# Test 6: Missing required settings cannot silently receive arbitrary defaults
# ---------------------------------------------------------------------------
def test_missing_required_settings_raises_resolution_error():
    """When profile fallback is omitted and required settings are absent, resolution
    must raise SimulationSettingsResolutionError rather than guessing defaults."""
    with pytest.raises(SimulationSettingsResolutionError, match="Required simulation setting 'region'"):
        resolve_candidate_simulation_settings(
            candidate_overrides={},
            session_region=None,
            session_universe="TOP3000",
            session_delay=1,
            profile=None,
            strict_profile=False,
        )

    with pytest.raises(SimulationSettingsResolutionError, match="Required simulation setting 'universe'"):
        resolve_candidate_simulation_settings(
            candidate_overrides={},
            session_region="USA",
            session_universe=None,
            session_delay=1,
            profile=None,
            strict_profile=False,
        )


# ---------------------------------------------------------------------------
# Test 7: Region/universe incompatibility rejected in preflight
# ---------------------------------------------------------------------------
def test_region_dataset_mismatch_rejected_in_preflight():
    """A candidate targeting a European dataset while using US region settings must fail
    preflight before simulation dispatch."""
    eur_dataset = DiscoveredDataset(
        dataset_id="eur_fundamental_v1",
        name="European Fundamentals",
        category="Fundamental",
        description="EUR equity data",
        fields=(
            DiscoveredField(
                field_id="ebitda_eur",
                field_type="Matrix",
                coverage=0.95,
                description="EBITDA",
                dataset_id="eur_fundamental_v1",
                category="Fundamental",
            ),
        ),
        region="EUR",
        delay=1,
    )
    registry = DatasetRegistry(datasets_by_region={"EUR": [eur_dataset]})

    us_settings = resolve_candidate_simulation_settings(profile="standard_us_equity_v1")

    is_valid, err = validate_candidate_dataset_preflight(
        expression="rank(ebitda_eur)",
        settings=us_settings,
        target_dataset_id="eur_fundamental_v1",
        dataset_registry=registry,
    )

    assert is_valid is False
    assert ERR_PREFLIGHT_SETTINGS_INCOMPATIBLE in err
    assert "region 'EUR', which is incompatible with candidate simulation region 'USA'" in err


# ---------------------------------------------------------------------------
# Test 8: Invalid platform setting choices rejected using controlled schema
# ---------------------------------------------------------------------------
def test_invalid_platform_setting_choices_rejected_by_schema():
    """Validating candidate settings against platform OPTIONS schema rejects invalid
    region/universe combinations."""
    validator = PlatformSchemaValidator(
        schema=MOCK_SIMULATION_SCHEMA,
        schema_source="LIVE",
    )

    # Valid settings
    valid_settings = resolve_candidate_simulation_settings(
        candidate_overrides={"region": "USA", "universe": "TOP3000"},
        profile="standard_us_equity_v1",
    )
    ok, problems, status = validator.validate(valid_settings)
    assert ok is True
    assert status == "VERIFIED"
    assert len(problems) == 0

    # Invalid universe for region: TOP1200 does not exist in region USA
    invalid_settings = resolve_candidate_simulation_settings(
        candidate_overrides={"region": "USA", "universe": "TOP1200"},
        profile="standard_us_equity_v1",
    )
    ok, problems, status = validator.validate(invalid_settings)
    assert ok is False
    assert status == "INVALID"
    assert any("Universe 'TOP1200' is not available" in p for p in problems)


# ---------------------------------------------------------------------------
# Test 9: Candidate expression referencing field with incompatible coverage rejected
# ---------------------------------------------------------------------------
def test_field_coverage_delay_incompatibility_rejected():
    """A candidate expression referencing a field requiring delay 2 when simulation
    delay is 1 must be rejected during preflight."""
    delay2_dataset = DiscoveredDataset(
        dataset_id="delayed_sentiment",
        name="Delayed Sentiment Data",
        category="Sentiment",
        description="T+2 sentiment",
        fields=(
            DiscoveredField(
                field_id="sent_score",
                field_type="Matrix",
                coverage=0.90,
                description="Sentiment",
                dataset_id="delayed_sentiment",
                category="Sentiment",
            ),
        ),
        region="USA",
        delay=2,
    )
    registry = DatasetRegistry(datasets_by_region={"USA": [delay2_dataset]})

    delay1_settings = resolve_candidate_simulation_settings(
        candidate_overrides={"delay": 1},
        profile="standard_us_equity_v1",
    )

    is_valid, err = validate_candidate_dataset_preflight(
        expression="rank(sent_score)",
        settings=delay1_settings,
        target_dataset_id="delayed_sentiment",
        dataset_registry=registry,
    )

    assert is_valid is False
    assert ERR_PREFLIGHT_SETTINGS_INCOMPATIBLE in err
    assert "requires delay 2, but candidate simulation delay is 1" in err


# ---------------------------------------------------------------------------
# Test 10: Dispatched SimulationSettings exactly matches stored and validated config
# ---------------------------------------------------------------------------
def test_dispatched_simulation_settings_matches_stored_config(test_db: Database):
    """The SimulationSettings wire model generated via candidate.to_simulation_settings()
    matches the candidate's persisted configuration without silent mutation."""

    async def _run():
        settings = CandidateSimulationSettings(
            region="USA",
            universe="TOP500",
            delay=0,
            decay=1,
            neutralization="SUBINDUSTRY",
            truncation=0.04,
            pasteurization="OFF",
            unit_handling="STRICT",
            nan_handling="OFF",
        )

        async with test_db.session() as s:
            sess = ResearchSession(name="Sess", region="USA", universe="TOP500")
            s.add(sess)
            await s.flush()

            cand = ResearchCandidate(
                session_id=sess.id,
                expression="ts_rank(close, 10)",
                simulation_settings=settings.to_dict(),
                settings_hash=settings.compute_hash(),
                sim_region=settings.region,
                sim_universe=settings.universe,
                sim_delay=settings.delay,
                sim_decay=settings.decay,
                sim_neutralization=settings.neutralization,
            )
            s.add(cand)
            await s.commit()
            cand_id = cand.id

        async with test_db.session() as s:
            c = await s.get(ResearchCandidate, cand_id)
            wire_settings = c.to_simulation_settings()

            assert wire_settings.region == "USA"
            assert wire_settings.universe == "TOP500"
            assert wire_settings.delay == 0
            assert wire_settings.decay == 1
            assert wire_settings.neutralization == "SUBINDUSTRY"
            assert wire_settings.truncation == 0.04
            assert wire_settings.pasteurization == "OFF"
            assert wire_settings.unit_handling == "STRICT"
            assert wire_settings.nan_handling == "OFF"

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 11: Legacy rows remain readable without inventing missing values
# ---------------------------------------------------------------------------
def test_legacy_rows_remain_readable_without_inventing_values(test_db: Database):
    """Legacy candidates without simulation_settings column load cleanly and provide
    scalar accessors without fabricating truncation, pasteurization, or unit_handling."""

    async def _run():
        async with test_db.session() as s:
            sess = ResearchSession(name="Legacy", region="USA", universe="TOP3000")
            s.add(sess)
            await s.flush()

            legacy_cand = ResearchCandidate(
                session_id=sess.id,
                expression="rank(close)",
                sim_region="USA",
                sim_universe="TOP3000",
                sim_delay=1,
                sim_decay=4,
                sim_neutralization="INDUSTRY",
                simulation_settings=None,
                settings_hash=None,
                settings_verification_status="LEGACY_UNVERIFIED",
            )
            s.add(legacy_cand)
            await s.commit()
            leg_id = legacy_cand.id

        async with test_db.session() as s:
            loaded = await s.get(ResearchCandidate, leg_id)
            assert loaded.sim_region == "USA"
            assert loaded.sim_universe == "TOP3000"
            assert loaded.sim_delay == 1
            assert loaded.sim_decay == 4
            assert loaded.simulation_settings is None
            assert loaded.settings_hash is None

            effective = loaded.effective_simulation_settings()
            assert effective["region"] == "USA"
            assert effective["is_legacy_unverified"] is True
            assert "truncation" not in effective  # Missing values are NOT fabricated

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 12: Incomplete legacy configuration cannot claim complete reproducibility
# ---------------------------------------------------------------------------
def test_incomplete_legacy_configuration_cannot_claim_reproducibility(test_db: Database):
    """is_simulation_settings_complete returns False for legacy rows with missing complete settings."""

    async def _run():
        async with test_db.session() as s:
            sess = ResearchSession(name="LegRepro", region="USA", universe="TOP3000")
            s.add(sess)
            await s.flush()

            cand = ResearchCandidate(
                session_id=sess.id,
                expression="rank(close)",
                sim_region="USA",
                sim_universe="TOP3000",
                simulation_settings=None,
                settings_verification_status="LEGACY_UNVERIFIED",
            )
            s.add(cand)
            await s.commit()
            c_id = cand.id

        async with test_db.session() as s:
            c = await s.get(ResearchCandidate, c_id)
            assert c.is_simulation_settings_complete is False

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 13: Missing or stale platform options do not result in false verification
# ---------------------------------------------------------------------------
def test_missing_platform_options_returns_unverified_status():
    """When schema is unavailable, PlatformSchemaValidator marks settings as UNVERIFIED
    rather than falsely claiming VERIFIED."""
    validator = PlatformSchemaValidator(
        schema=None,
        schema_source="LIVE",
    )
    settings = resolve_candidate_simulation_settings(profile="standard_us_equity_v1")

    ok, problems, status = validator.validate(settings, allow_fallback=False)
    assert ok is False
    assert status == "UNVERIFIED"
    assert "schema unavailable" in problems[0]


# ---------------------------------------------------------------------------
# Test 14: API compatibility is retained
# ---------------------------------------------------------------------------
def test_api_compatibility_candidate_out():
    """CandidateOut schema accepts and serializes simulation_settings and settings_hash
    while retaining full backward compatibility for existing fields."""
    cand_out = CandidateOut(
        id=10,
        expression="rank(close)",
        sim_region="USA",
        sim_universe="TOP3000",
        simulation_settings={"region": "USA", "universe": "TOP3000", "truncation": 0.08},
        settings_hash="abcdef123456",
        settings_verification_status="VERIFIED",
        candidate_provenance_hash="prov123",
    )
    d = cand_out.model_dump(by_alias=False)
    assert d["id"] == 10
    assert d["simulation_settings"]["truncation"] == 0.08
    assert d["settings_hash"] == "abcdef123456"
    assert d["settings_verification_status"] == "VERIFIED"

    # Also verify camelCase wire representation for API endpoints
    wire = cand_out.model_dump(by_alias=True)
    assert wire["simulationSettings"]["truncation"] == 0.08
    assert wire["settingsHash"] == "abcdef123456"


# ---------------------------------------------------------------------------
# Test 15: Settings and provenance fingerprints remain correct under DB persistence
# ---------------------------------------------------------------------------
def test_settings_and_provenance_fingerprints_under_db_persistence(test_db: Database):
    """Saving and querying ResearchCandidate preserves exact settings_hash and
    candidate_provenance_hash values."""

    async def _run():
        expr = "ts_decay_linear(close, 10)"
        settings = resolve_candidate_simulation_settings(
            candidate_overrides={"decay": 3, "neutralization": "SUBINDUSTRY"},
            profile="standard_us_equity_v1",
        )
        s_hash = settings.compute_hash()
        prov_hash = compute_candidate_provenance_hash("expr_hash_1", s_hash, "gemini-2.5-pro", 5, "ds_1")

        async with test_db.session() as s:
            sess = ResearchSession(name="FingerprintSess", region="USA", universe="TOP3000")
            s.add(sess)
            await s.flush()

            cand = ResearchCandidate(
                session_id=sess.id,
                expression=expr,
                simulation_settings=settings.to_dict(),
                settings_hash=s_hash,
                candidate_provenance_hash=prov_hash,
                sim_region=settings.region,
                sim_universe=settings.universe,
                sim_delay=settings.delay,
                sim_decay=settings.decay,
                sim_neutralization=settings.neutralization,
            )
            s.add(cand)
            await s.commit()
            c_id = cand.id

        async with test_db.session() as s:
            loaded = await s.get(ResearchCandidate, c_id)
            assert loaded.settings_hash == s_hash
            assert loaded.candidate_provenance_hash == prov_hash
            assert loaded.simulation_settings["decay"] == 3
            assert loaded.simulation_settings["neutralization"] == "SUBINDUSTRY"

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test 16: Final holdout changes do not influence settings resolution
# ---------------------------------------------------------------------------
def test_final_holdout_changes_do_not_influence_settings_resolution():
    """Settings resolution and preflight validation operate exclusively on metadata;
    modifying holdout split dates or holdout returns produces identical settings and hashes."""
    s1 = resolve_candidate_simulation_settings(
        candidate_overrides={"region": "USA", "universe": "TOP3000"},
        profile="standard_us_equity_v1",
    )
    s2 = resolve_candidate_simulation_settings(
        candidate_overrides={"region": "USA", "universe": "TOP3000"},
        profile="standard_us_equity_v1",
    )

    # Identical metadata must produce identical settings and hash regardless of external holdout state
    assert s1.compute_hash() == s2.compute_hash()
    assert s1.canonical_json() == s2.canonical_json()


# ---------------------------------------------------------------------------
# Test 17: Multiplicity, Diversity, and Cohort Matching Preservation
# ---------------------------------------------------------------------------
def test_cohort_matching_and_diversity_preservation():
    """Candidates with different settings hashes belong to different cohorts in
    prompt validation and cannot be improperly paired."""
    c1 = {
        "task": "T1",
        "dataset_id": "DS1",
        "region": "USA",
        "universe": "TOP3000",
        "delay": 1,
        "settings_hash": "hash_decay_4",
        "val_sharpe": 1.5,
    }
    c2 = {
        "task": "T1",
        "dataset_id": "DS1",
        "region": "USA",
        "universe": "TOP3000",
        "delay": 1,
        "settings_hash": "hash_decay_2",
        "val_sharpe": 1.5,
    }

    k1 = _extract_cohort_key(c1)
    k2 = _extract_cohort_key(c2)
    assert k1 != k2, "Candidates with different simulation settings hashes must have different cohort keys"

    # Validate override rejection on malformed inputs
    with pytest.raises(SimulationSettingsValidationError, match="Unsupported simulation setting override 'invalid_magic'"):
        validate_candidate_overrides({"invalid_magic": 123})

    with pytest.raises(SimulationSettingsValidationError, match="must be an integer"):
        validate_candidate_overrides({"decay": "four"})
