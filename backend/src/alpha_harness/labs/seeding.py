"""Stable, auditable seeds for persisted research experiments.

Seed derivation v1 hashes a canonical JSON document containing a unique persisted
experiment identity, the full relevant configuration, and an optional user seed. A new
run therefore gets a distinct identity, while replaying that identity and configuration
reproduces the same seed across processes and Python hash randomization settings.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

SEED_DERIVATION_VERSION = "sha256-canonical-json-v1"
MAX_SEED = (1 << 63) - 1


def canonical_configuration(configuration: Any) -> str:
    """Serialize JSON-compatible run configuration in a stable canonical form."""
    return json.dumps(
        configuration,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def derive_seed(
    experiment_id: str, configuration: Any, *, explicit_seed: int | None = None
) -> int:
    """Return a stable SQLite-compatible seed; identities must be unique per new run."""
    if not experiment_id or not experiment_id.strip():
        raise ValueError("A persisted experiment identity is required for seed derivation.")
    if explicit_seed is not None and explicit_seed < 0:
        raise ValueError("An explicit seed must be a non-negative integer.")
    material = canonical_configuration(
        {
            "version": SEED_DERIVATION_VERSION,
            "experiment_id": experiment_id,
            "configuration": configuration,
            "explicit_seed": explicit_seed,
        }
    )
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & MAX_SEED


def reproducibility_metadata(
    experiment_id: str,
    configuration: Any,
    effective_seed: int,
    *,
    explicit_seed: int | None = None,
) -> dict[str, Any]:
    """Metadata persisted beside Study.seed; config is kept as canonical JSON text."""
    return {
        "experiment_id": experiment_id,
        "derivation_version": SEED_DERIVATION_VERSION,
        "effective_seed": effective_seed,
        "explicit_seed": explicit_seed,
        "canonical_configuration": canonical_configuration(configuration),
    }


def reproducibility_known(seed: int | None, metadata: Any) -> bool:
    """Legacy rows without matching persisted derivation evidence are explicitly unknown."""
    return (
        seed is not None
        and isinstance(metadata, dict)
        and metadata.get("effective_seed") == seed
        and metadata.get("derivation_version") == SEED_DERIVATION_VERSION
        and isinstance(metadata.get("canonical_configuration"), str)
        and bool(metadata["canonical_configuration"])
    )
