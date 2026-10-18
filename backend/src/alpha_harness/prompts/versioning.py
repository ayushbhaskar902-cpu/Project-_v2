"""Immutable prompt versioning, canonical normalization, and integrity verification.

Implements P1-03 requirements:
- Append-only prompt snapshots with SHA-256 content hashes
- Option A canonical normalization (empty list [], None guidance, verbatim template)
- Fail-closed prompt integrity verification
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..db.models import PromptVersion


class PromptIntegrityError(RuntimeError):
    """Raised when a PromptVersion's stored content does not match its computed content_hash."""


class PromptResolutionError(RuntimeError):
    """Raised when no valid immutable PromptVersion can be resolved."""


def normalize_prompt_components(
    template_text: str,
    guidance: str | dict[str, Any] | None = None,
    examples: list[Any] | None = None,
    anti_patterns: list[Any] | None = None,
) -> tuple[str, str | None, list[Any], list[Any]]:
    """Normalizes prompt components into canonical representation before hashing and persistence (Option A).

    Rules:
    - template_text: non-empty string preserved verbatim.
    - guidance: None if absent, empty, or whitespace-only; non-empty string or canonical JSON string if provided.
    - examples: normalized to list, empty list [] if None or empty.
    - anti_patterns: normalized to list, empty list [] if None or empty.
    """
    if guidance is None:
        norm_guidance: str | None = None
    elif isinstance(guidance, str):
        stripped = guidance.strip()
        norm_guidance = stripped if stripped else None
    elif isinstance(guidance, dict):
        norm_guidance = json.dumps(guidance, sort_keys=True) if guidance else None
    else:
        norm_guidance = str(guidance).strip() or None

    norm_examples: list[Any] = list(examples) if examples is not None else []
    norm_anti_patterns: list[Any] = list(anti_patterns) if anti_patterns is not None else []

    return template_text, norm_guidance, norm_examples, norm_anti_patterns


def compute_prompt_content_hash(
    template_text: str,
    guidance: str | dict[str, Any] | None = None,
    examples: list[Any] | None = None,
    anti_patterns: list[Any] | None = None,
) -> str:
    """Computes deterministic SHA-256 content-only hash over canonical prompt components."""
    t_text, n_guidance, n_examples, n_anti_patterns = normalize_prompt_components(
        template_text=template_text,
        guidance=guidance,
        examples=examples,
        anti_patterns=anti_patterns,
    )
    payload = {
        "anti_patterns": n_anti_patterns,
        "examples": n_examples,
        "guidance": n_guidance,
        "template_text": t_text,
    }
    canonical_bytes = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


def verify_prompt_integrity(pv: PromptVersion) -> None:
    """Verifies that a stored PromptVersion matches its recorded content hash.

    Fails closed: raises PromptIntegrityError if any component has been altered.
    """
    computed = compute_prompt_content_hash(
        template_text=pv.template_text,
        guidance=pv.guidance,
        examples=pv.examples,
        anti_patterns=pv.anti_patterns,
    )
    if computed != pv.content_hash:
        raise PromptIntegrityError(
            f"PromptVersion id={pv.id} content hash mismatch: stored={pv.content_hash}, computed={computed}"
        )
