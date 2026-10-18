"""Explicit, validated, and persisted candidate simulation settings.

Implements P1-15:
- CandidateSimulationSettings: Canonical configuration model representing all simulation parameters.
- Deterministic canonical serialization and SHA-256 settings hashing.
- Explicit three-tier resolution policy:
  1. Valid candidate-level override.
  2. Valid, persisted research-session configuration.
  3. Explicitly selected and versioned settings profile.
- Preflight dataset coverage compatibility validation (region, universe, delay).
- Platform schema validation against OPTIONS /simulations (live, cached, or controlled test fixtures).
- Candidate provenance fingerprint integration under P0-02 rules.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any

from ..brain.schemas import SimulationSettings
from ..brain.settings_schema import validate_settings as brain_validate_settings
from ..labs.fastexpr import data_fields, parse

# Allowed candidate override keys
ALLOWED_OVERRIDE_FIELDS: set[str] = {
    "region",
    "universe",
    "delay",
    "decay",
    "neutralization",
    "instrument_type",
    "truncation",
    "pasteurization",
    "unit_handling",
    "nan_handling",
    "language",
    "visualization",
    "test_period",
    "max_trade",
    "max_position",
    "simulation_mode",
    "selection_handling",
    "selection_limit",
    "component_activation",
}

# Diagnostic error constants
ERR_INVALID_SIMULATION_SETTING_OVERRIDE = "ERR_INVALID_SIMULATION_SETTING_OVERRIDE"
ERR_PREFLIGHT_SETTINGS_INCOMPATIBLE = "ERR_PREFLIGHT_SETTINGS_INCOMPATIBLE"


class SimulationSettingsResolutionError(ValueError):
    """Raised when required simulation settings cannot be resolved from any valid tier."""


class SimulationSettingsValidationError(ValueError):
    """Raised when simulation settings fail type, override, or schema validation."""


@dataclass(frozen=True, slots=True)
class SimulationProfile:
    """An explicit, versioned simulation profile providing identifiably sourced baseline settings."""

    profile_id: str
    profile_version: str
    name: str
    description: str
    settings: dict[str, Any]


# Authoritative versioned simulation profiles
STANDARD_US_EQUITY_V1 = SimulationProfile(
    profile_id="standard_us_equity",
    profile_version="v1",
    name="Standard US Equity (Daily, Delay 1)",
    description="Default WorldQuant BRAIN US equity profile with delay 1, decay 4, and industry neutralization.",
    settings={
        "region": "USA",
        "universe": "TOP3000",
        "delay": 1,
        "decay": 4,
        "neutralization": "INDUSTRY",
        "truncation": 0.08,
        "pasteurization": "ON",
        "unit_handling": "VERIFY",
        "nan_handling": "OFF",
        "instrument_type": "EQUITY",
        "language": "FASTEXPR",
        "visualization": False,
    },
)

INTRADAY_US_EQUITY_V1 = SimulationProfile(
    profile_id="intraday_us_equity",
    profile_version="v1",
    name="Intraday US Equity (Delay 0)",
    description="US equity profile configured for delay 0 intraday alphas with subindustry neutralization.",
    settings={
        "region": "USA",
        "universe": "TOP3000",
        "delay": 0,
        "decay": 0,
        "neutralization": "SUBINDUSTRY",
        "truncation": 0.05,
        "pasteurization": "ON",
        "unit_handling": "VERIFY",
        "nan_handling": "OFF",
        "instrument_type": "EQUITY",
        "language": "FASTEXPR",
        "visualization": False,
    },
)

STANDARD_EUR_EQUITY_V1 = SimulationProfile(
    profile_id="standard_eur_equity",
    profile_version="v1",
    name="Standard European Equity (Daily, Delay 1)",
    description="European equity profile with TOP1200 universe, delay 1, decay 4, and industry neutralization.",
    settings={
        "region": "EUR",
        "universe": "TOP1200",
        "delay": 1,
        "decay": 4,
        "neutralization": "INDUSTRY",
        "truncation": 0.08,
        "pasteurization": "ON",
        "unit_handling": "VERIFY",
        "nan_handling": "OFF",
        "instrument_type": "EQUITY",
        "language": "FASTEXPR",
        "visualization": False,
    },
)

STANDARD_ASIA_EQUITY_V1 = SimulationProfile(
    profile_id="standard_asia_equity",
    profile_version="v1",
    name="Standard Asian Equity (Daily, Delay 1)",
    description="Asian equity profile with TOP1000 universe, delay 1, decay 4, and industry neutralization.",
    settings={
        "region": "ASI",
        "universe": "TOP1000",
        "delay": 1,
        "decay": 4,
        "neutralization": "INDUSTRY",
        "truncation": 0.08,
        "pasteurization": "ON",
        "unit_handling": "VERIFY",
        "nan_handling": "OFF",
        "instrument_type": "EQUITY",
        "language": "FASTEXPR",
        "visualization": False,
    },
)

SETTINGS_PROFILES: dict[str, SimulationProfile] = {
    "standard_us_equity_v1": STANDARD_US_EQUITY_V1,
    "standard_us_equity": STANDARD_US_EQUITY_V1,
    "intraday_us_equity_v1": INTRADAY_US_EQUITY_V1,
    "intraday_us_equity": INTRADAY_US_EQUITY_V1,
    "standard_eur_equity_v1": STANDARD_EUR_EQUITY_V1,
    "standard_eur_equity": STANDARD_EUR_EQUITY_V1,
    "standard_asia_equity_v1": STANDARD_ASIA_EQUITY_V1,
    "standard_asia_equity": STANDARD_ASIA_EQUITY_V1,
}


def get_settings_profile(profile_id: str | None) -> SimulationProfile | None:
    """Looks up a registered simulation profile by id or alias."""
    if not profile_id:
        return None
    return SETTINGS_PROFILES.get(profile_id.strip().lower())


@dataclass(slots=True)
class CandidateSimulationSettings:
    """Canonical, auditable simulation settings configuration for a research candidate."""

    region: str
    universe: str
    delay: int
    decay: int = 0
    neutralization: str = "NONE"
    instrument_type: str = "EQUITY"
    truncation: float = 0.08
    pasteurization: str = "ON"
    unit_handling: str = "VERIFY"
    nan_handling: str = "OFF"
    language: str = "FASTEXPR"
    visualization: bool = False
    test_period: str | None = None
    max_trade: str | None = None
    max_position: str | None = None
    simulation_mode: str | None = None
    selection_handling: str | None = None
    selection_limit: int | None = None
    component_activation: str | None = None
    profile_id: str | None = None
    profile_version: str | None = None
    schema_version: str | None = None
    verification_status: str = "UNVERIFIED"
    raw_overrides: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Returns deterministic Python dictionary containing all defined settings."""
        d: dict[str, Any] = {
            "component_activation": self.component_activation,
            "decay": int(self.decay),
            "delay": int(self.delay),
            "instrument_type": str(self.instrument_type),
            "language": str(self.language),
            "max_position": self.max_position,
            "max_trade": self.max_trade,
            "nan_handling": str(self.nan_handling),
            "neutralization": str(self.neutralization),
            "pasteurization": str(self.pasteurization),
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "region": str(self.region),
            "schema_version": self.schema_version,
            "selection_handling": self.selection_handling,
            "selection_limit": self.selection_limit,
            "simulation_mode": self.simulation_mode,
            "test_period": self.test_period,
            "truncation": round(float(self.truncation), 6),
            "unit_handling": str(self.unit_handling),
            "universe": str(self.universe),
            "verification_status": str(self.verification_status),
            "visualization": bool(self.visualization),
        }
        # Filter None values for canonical representation of genuinely optional fields
        return {k: v for k, v in d.items() if v is not None}

    def to_wire_dict(self) -> dict[str, Any]:
        """Converts settings to camelCase wire dictionary expected by BRAIN API and settings_schema."""
        wire: dict[str, Any] = {
            "instrumentType": self.instrument_type,
            "region": self.region,
            "universe": self.universe,
            "delay": self.delay,
            "decay": self.decay,
            "neutralization": self.neutralization,
            "truncation": self.truncation,
            "pasteurization": self.pasteurization,
            "unitHandling": self.unit_handling,
            "nanHandling": self.nan_handling,
            "language": self.language,
            "visualization": self.visualization,
        }
        if self.test_period is not None:
            wire["testPeriod"] = self.test_period
        if self.max_trade is not None:
            wire["maxTrade"] = self.max_trade
        if self.max_position is not None:
            wire["maxPosition"] = self.max_position
        if self.simulation_mode is not None:
            wire["simulationMode"] = self.simulation_mode
        if self.selection_handling is not None:
            wire["selectionHandling"] = self.selection_handling
        if self.selection_limit is not None:
            wire["selectionLimit"] = self.selection_limit
        if self.component_activation is not None:
            wire["componentActivation"] = self.component_activation
        return wire

    def canonical_json(self) -> str:
        """Returns deterministic JSON string sorted by keys with minimal whitespace separators."""
        data = self.to_dict()
        return json.dumps(data, sort_keys=True, separators=(",", ":"))

    def compute_hash(self) -> str:
        """Computes deterministic SHA-256 settings fingerprint."""
        raw = self.canonical_json().encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def to_brain_settings(self) -> SimulationSettings:
        """Converts into a SimulationSettings instance for live BRAIN dispatch."""
        return SimulationSettings(
            instrument_type=self.instrument_type,
            region=self.region,
            universe=self.universe,
            delay=self.delay,
            decay=self.decay,
            neutralization=self.neutralization,
            truncation=self.truncation,
            pasteurization=self.pasteurization,
            unit_handling=self.unit_handling,
            nan_handling=self.nan_handling,
            language=self.language,
            visualization=self.visualization,
            test_period=self.test_period,
            max_trade=self.max_trade,
            max_position=self.max_position,
            simulation_mode=self.simulation_mode,
            selection_handling=self.selection_handling,
            selection_limit=self.selection_limit,
            component_activation=self.component_activation,
        )


def validate_candidate_overrides(overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Validates candidate-level override keys and types."""
    if not overrides:
        return {}

    cleaned: dict[str, Any] = {}
    for k, v in overrides.items():
        k_norm = k.strip().lower()
        if k_norm not in ALLOWED_OVERRIDE_FIELDS:
            raise SimulationSettingsValidationError(
                f"{ERR_INVALID_SIMULATION_SETTING_OVERRIDE}: Unsupported simulation setting override '{k}'"
            )

        # Type checks
        if k_norm in ("delay", "decay", "selection_limit"):
            try:
                cleaned[k_norm] = int(v)
            except (ValueError, TypeError) as err:
                raise SimulationSettingsValidationError(
                    f"{ERR_INVALID_SIMULATION_SETTING_OVERRIDE}: Field '{k}' must be an integer, got {v!r}"
                ) from err
        elif k_norm == "truncation":
            try:
                cleaned[k_norm] = float(v)
            except (ValueError, TypeError) as err:
                raise SimulationSettingsValidationError(
                    f"{ERR_INVALID_SIMULATION_SETTING_OVERRIDE}: Field '{k}' must be a float, got {v!r}"
                ) from err
        elif k_norm == "visualization":
            cleaned[k_norm] = bool(v)
        else:
            cleaned[k_norm] = str(v).strip()

    return cleaned


def resolve_candidate_simulation_settings(
    candidate_overrides: dict[str, Any] | None = None,
    session_region: str | None = None,
    session_universe: str | None = None,
    session_delay: int | None = None,
    session_constraints: dict[str, Any] | None = None,
    profile: SimulationProfile | str | None = "standard_us_equity_v1",
    schema_version: str | None = None,
    strict_profile: bool = True,
) -> CandidateSimulationSettings:
    """Resolves simulation settings using an explicit 3-tier precedence policy:

    Tier 1: Valid candidate-level override.
    Tier 2: Valid, persisted research-session configuration.
    Tier 3: An explicitly selected and versioned settings profile.

    Defaults are permitted only when derived from an explicit, identifiable profile.
    Missing required fields that cannot be resolved from any tier raise SimulationSettingsResolutionError.
    """
    valid_overrides = validate_candidate_overrides(candidate_overrides)

    # Resolve baseline profile
    prof_obj: SimulationProfile | None = None
    if isinstance(profile, SimulationProfile):
        prof_obj = profile
    elif isinstance(profile, str):
        prof_obj = get_settings_profile(profile)
        if prof_obj is None and strict_profile:
            raise SimulationSettingsResolutionError(
                f"Unknown settings profile identifier: '{profile}'"
            )

    prof_settings = copy.deepcopy(prof_obj.settings) if prof_obj else {}

    # Extract session constraints
    sess_constraints = copy.deepcopy(session_constraints or {})
    if session_region:
        sess_constraints["region"] = session_region
    if session_universe:
        sess_constraints["universe"] = session_universe
    if session_delay is not None:
        sess_constraints["delay"] = session_delay

    # Tiered resolution helper
    def _resolve_val(key: str, default: Any = None) -> Any:
        # Tier 1: Candidate Override
        if key in valid_overrides:
            return valid_overrides[key]
        # Tier 2: Session Configuration
        if key in sess_constraints and sess_constraints[key] is not None:
            return sess_constraints[key]
        # Tier 3: Profile Baseline
        if key in prof_settings and prof_settings[key] is not None:
            return prof_settings[key]
        return default

    # Required fields must be resolved
    region = _resolve_val("region")
    universe = _resolve_val("universe")
    delay = _resolve_val("delay")

    if not region:
        raise SimulationSettingsResolutionError(
            "Required simulation setting 'region' could not be resolved from candidate overrides, session, or profile."
        )
    if not universe:
        raise SimulationSettingsResolutionError(
            "Required simulation setting 'universe' could not be resolved from candidate overrides, session, or profile."
        )
    if delay is None:
        raise SimulationSettingsResolutionError(
            "Required simulation setting 'delay' could not be resolved from candidate overrides, session, or profile."
        )

    decay = int(_resolve_val("decay", 0))
    neutralization = str(_resolve_val("neutralization", "NONE"))
    instrument_type = str(_resolve_val("instrument_type", "EQUITY"))
    truncation = float(_resolve_val("truncation", 0.08))
    pasteurization = str(_resolve_val("pasteurization", "ON"))
    unit_handling = str(_resolve_val("unit_handling", "VERIFY"))
    nan_handling = str(_resolve_val("nan_handling", "OFF"))
    language = str(_resolve_val("language", "FASTEXPR"))
    visualization = bool(_resolve_val("visualization", False))
    test_period = _resolve_val("test_period")
    max_trade = _resolve_val("max_trade")
    max_position = _resolve_val("max_position")
    simulation_mode = _resolve_val("simulation_mode")
    selection_handling = _resolve_val("selection_handling")
    selection_limit = _resolve_val("selection_limit")
    if selection_limit is not None:
        selection_limit = int(selection_limit)
    component_activation = _resolve_val("component_activation")

    return CandidateSimulationSettings(
        region=str(region).upper().strip(),
        universe=str(universe).upper().strip(),
        delay=int(delay),
        decay=decay,
        neutralization=str(neutralization).upper().strip(),
        instrument_type=str(instrument_type).upper().strip(),
        truncation=truncation,
        pasteurization=str(pasteurization).upper().strip(),
        unit_handling=str(unit_handling).upper().strip(),
        nan_handling=str(nan_handling).upper().strip(),
        language=language,
        visualization=visualization,
        test_period=test_period,
        max_trade=max_trade,
        max_position=max_position,
        simulation_mode=simulation_mode,
        selection_handling=selection_handling,
        selection_limit=selection_limit,
        component_activation=component_activation,
        profile_id=prof_obj.profile_id if prof_obj else None,
        profile_version=prof_obj.profile_version if prof_obj else None,
        schema_version=schema_version,
        verification_status="UNVERIFIED",
        raw_overrides=valid_overrides,
    )


def compute_candidate_provenance_hash(
    expression_hash: str,
    settings_hash: str,
    model_ref: str | None = None,
    prompt_version_id: int | None = None,
    target_dataset_id: str | None = None,
    catalog_version: str | None = None,
) -> str:
    """Computes candidate provenance fingerprint incorporating expression and effective simulation settings.

    Under P0-02 provenance integrity, any change to effective simulation settings alters the
    settings_hash and consequently produces a distinct candidate provenance fingerprint.
    """
    payload = {
        "catalog_version": catalog_version,
        "expression_hash": expression_hash,
        "model_ref": model_ref,
        "prompt_version_id": prompt_version_id,
        "settings_hash": settings_hash,
        "target_dataset_id": target_dataset_id,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class PlatformSchemaValidator:
    """Validates candidate simulation settings against dynamic platform OPTIONS /simulations schemas."""

    def __init__(
        self,
        schema: dict[str, Any] | None = None,
        schema_source: str = "LIVE",
        schema_version: str = "4.0",
    ) -> None:
        self.schema = schema
        self.schema_source = schema_source.upper().strip()
        self.schema_version = schema_version

    def validate(
        self,
        settings: CandidateSimulationSettings,
        *,
        allow_fallback: bool = False,
    ) -> tuple[bool, list[str], str]:
        """Validates settings against platform options.

        Returns (is_valid, problems, verification_status).
        verification_status can be:
        - "VERIFIED" (live authoritative platform schema)
        - "CACHED_VERIFIED" (cached platform schema)
        - "TEST_FIXTURE_VERIFIED" (explicit test fixture)
        - "UNVERIFIED" (fallback or missing schema)
        - "INVALID" (failed schema checks)
        """
        if self.schema is None:
            if allow_fallback or self.schema_source == "OFFLINE_TEST_FIXTURE":
                return True, [], "TEST_FIXTURE_VERIFIED"
            return False, ["Platform settings schema unavailable and unverified."], "UNVERIFIED"

        wire_settings = settings.to_wire_dict()
        problems = brain_validate_settings(self.schema, wire_settings, require_all=False)

        if problems:
            return False, problems, "INVALID"

        status_map = {
            "LIVE": "VERIFIED",
            "CACHED": "CACHED_VERIFIED",
            "OFFLINE_TEST_FIXTURE": "TEST_FIXTURE_VERIFIED",
            "FALLBACK": "UNVERIFIED",
        }
        v_status = status_map.get(self.schema_source, "UNVERIFIED")
        return True, [], v_status


def validate_candidate_dataset_preflight(
    expression: str,
    settings: CandidateSimulationSettings,
    target_dataset_id: str | None = None,
    dataset_registry: Any = None,
    datafield_catalog: Any = None,
) -> tuple[bool, str | None]:
    """Validates dataset eligibility and field coverage before consuming a simulation queue slot.

    Rejects:
    1. Dataset region mismatch (e.g. European dataset with US region setting).
    2. Dataset delay requirement exceeding simulation delay.
    3. Referenced fields outside dataset coverage or incompatible with target scope.
    """
    # 1. Check target dataset scope compatibility if registry is present
    if target_dataset_id and dataset_registry and hasattr(dataset_registry, "get_dataset"):
        ds = dataset_registry.get_dataset(target_dataset_id)
        if ds is not None:
            ds_reg = getattr(ds, "region", "USA").upper().strip()
            if ds_reg != "GLB" and settings.region != "GLB" and ds_reg != settings.region:
                return (
                    False,
                    f"{ERR_PREFLIGHT_SETTINGS_INCOMPATIBLE}: Target dataset '{target_dataset_id}' belongs to region '{ds_reg}', which is incompatible with candidate simulation region '{settings.region}'.",
                )

            ds_delay = getattr(ds, "delay", None)
            if ds_delay is not None and ds_delay > settings.delay:
                return (
                    False,
                    f"{ERR_PREFLIGHT_SETTINGS_INCOMPATIBLE}: Target dataset '{target_dataset_id}' requires delay {ds_delay}, but candidate simulation delay is {settings.delay}.",
                )

    # 2. Extract referenced fields from expression
    referenced_fields: list[str] = []
    if expression and expression.strip():
        try:
            tree = parse(expression.strip())
            referenced_fields = sorted(data_fields(tree))
        except Exception:
            # Syntax validation is handled by earlier stage
            pass

    # 3. Check field validity against DatafieldCatalog if available
    if referenced_fields and datafield_catalog and hasattr(datafield_catalog, "validate_field"):
        for fld in referenced_fields:
            is_valid, reason = datafield_catalog.validate_field(fld)
            if not is_valid:
                return (
                    False,
                    f"{ERR_PREFLIGHT_SETTINGS_INCOMPATIBLE}: Referenced field '{fld}' failed preflight validation: {reason}",
                )

    return True, None
