"""Diversity-aware candidate generation, selection, deduplication, and portfolio curation (P1-14).

Enforces deterministic upstream deduplication, parameter-jitter near-duplicate pruning,
reproducible family quota allocation with exploration reserves, and strict separation
of pre-simulation structural diversity from post-simulation statistical quality.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any

import structlog

from ..labs.fastexpr import (
    CandidateFingerprint,
    compute_expression_hash,
    extract_candidate_fingerprint,
    is_near_duplicate,
)

log = structlog.get_logger(__name__)


@dataclass(slots=True)
class DiversityConfig:
    """Configuration for deterministic pre-simulation diversity and quota allocation."""

    exact_dedup: bool = True
    prune_near_duplicates: bool = True
    family_quotas: dict[str, float] | None = None  # e.g. {"MOMENTUM": 0.25, "MEAN_REVERSION": 0.25, ...}
    exploration_reserve_ratio: float = 0.20        # 20% reserved for underrepresented/exploratory candidates
    field_family_diversity: bool = True
    operator_family_diversity: bool = True
    seed: int = 42


@dataclass(slots=True)
class CandidateItem:
    """Standard representation of a generated candidate for diversity analysis."""

    expression: str
    hypothesis: str = ""
    category: str = "GENERAL"
    target_dataset_id: str | None = None
    fields_used: list[str] = field(default_factory=list)
    field_families: dict[str, str] = field(default_factory=dict)
    operators_used: list[str] = field(default_factory=list)
    operator_families: dict[str, str] = field(default_factory=dict)
    is_valid: bool = True
    candidate_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CandidateAuditRecord:
    """Durable audit record documenting candidate retention, pruning, and allocation decisions."""

    expression: str
    expression_hash: str
    ast_skeleton: str
    category: str
    fields_used: list[str]
    operators_used: list[str]
    is_valid: bool
    decision: str  # DISPATCHED, EXACT_DUPLICATE_PRUNED, NEAR_DUPLICATE_PRUNED, INVALID_EXPRESSION, RESERVED_OVER_CAPACITY
    reason: str
    candidate_id: int | None = None


@dataclass(slots=True)
class DiversityFunnelMetrics:
    """Comprehensive observability metrics across the diversity and deduplication funnel."""

    generated_count: int = 0
    valid_count: int = 0
    invalid_count: int = 0
    exact_duplicate_count: int = 0
    near_duplicate_count: int = 0
    retained_count: int = 0
    dispatched_count: int = 0
    category_counts: dict[str, int] = field(default_factory=dict)
    field_family_counts: dict[str, int] = field(default_factory=dict)
    operator_family_counts: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class DiversitySelectionResult:
    """Result of diversity filtering and simulation queue selection."""

    dispatched_candidates: list[CandidateItem]
    pruned_candidates: list[CandidateItem]
    funnel_metrics: DiversityFunnelMetrics
    audit_records: list[CandidateAuditRecord]


class DiversitySelector:
    """Executes deterministic upstream deduplication, near-duplicate pruning, and diversity allocation."""

    def __init__(self, config: DiversityConfig | None = None) -> None:
        self.config = config or DiversityConfig()

    def select_candidates(
        self,
        candidates: list[CandidateItem],
        capacity: int,
        *,
        session_seen_hashes: set[str] | None = None,
        custom_config: DiversityConfig | None = None,
    ) -> DiversitySelectionResult:
        """Applies multi-stage deterministic deduplication, near-duplicate pruning, and diversity allocation.

        Lifecycle Stages:
        1. Validity Gate: Rejects syntactically or structurally invalid candidates.
        2. Exact Hash Deduplication: Eliminates normalized SHA-256 duplicate expressions.
        3. AST Near-Duplicate Pruning: Prunes parameter-jitter expressions sharing identical AST skeleton,
           fields, operators, and hypothesis category.
        4. Diversity-Aware Quota Allocation: Distributes simulation capacity across hypothesis families,
           datafield families, and exploration reserve without candidate fabrication or empty fallback.
        """
        cfg = custom_config or self.config
        seen_hashes = set(session_seen_hashes or set())
        audit_records: list[CandidateAuditRecord] = []

        metrics = DiversityFunnelMetrics(generated_count=len(candidates))

        # 1. Separate valid from invalid
        valid_candidates: list[tuple[CandidateItem, CandidateFingerprint]] = []
        for c in candidates:
            if not c.is_valid or not c.expression or not c.expression.strip():
                metrics.invalid_count += 1
                fp = extract_candidate_fingerprint(c.expression or "")
                audit_records.append(
                    CandidateAuditRecord(
                        candidate_id=c.candidate_id,
                        expression=c.expression,
                        expression_hash=fp.expr_hash,
                        ast_skeleton=fp.skeleton,
                        category=c.category,
                        fields_used=c.fields_used,
                        operators_used=c.operators_used,
                        is_valid=False,
                        decision="INVALID_EXPRESSION",
                        reason=c.metadata.get("validation_error") or "Expression failed AST validation",
                    )
                )
                continue

            metrics.valid_count += 1
            fp = extract_candidate_fingerprint(c.expression)
            valid_candidates.append((c, fp))

        # 2. Exact Hash Deduplication
        unique_hash_candidates: list[tuple[CandidateItem, CandidateFingerprint]] = []
        for c, fp in valid_candidates:
            if cfg.exact_dedup and fp.expr_hash in seen_hashes:
                metrics.exact_duplicate_count += 1
                audit_records.append(
                    CandidateAuditRecord(
                        candidate_id=c.candidate_id,
                        expression=c.expression,
                        expression_hash=fp.expr_hash,
                        ast_skeleton=fp.skeleton,
                        category=c.category,
                        fields_used=c.fields_used,
                        operators_used=c.operators_used,
                        is_valid=True,
                        decision="EXACT_DUPLICATE_PRUNED",
                        reason="Identical normalized expression hash already generated or active in session",
                    )
                )
            else:
                seen_hashes.add(fp.expr_hash)
                unique_hash_candidates.append((c, fp))

        # 3. Near-Duplicate Pruning (AST Skeleton + identical fields + identical operators)
        # Structural similarity collapses parameter-jitter sweeps (e.g. ts_decay_linear(close, 10) vs 12)
        # while strictly preserving expressions using different data fields or different hypothesis categories.
        pruned_candidates: list[CandidateItem] = []
        retained_candidates: list[tuple[CandidateItem, CandidateFingerprint]] = []

        near_dup_clusters: dict[tuple, CandidateItem] = {}

        for c, fp in unique_hash_candidates:
            if not cfg.prune_near_duplicates:
                retained_candidates.append((c, fp))
                continue

            # Cluster key distinguishes fields, operators, and hypothesis category
            cluster_key = (
                fp.skeleton,
                fp.fields,
                fp.operators,
                c.category.strip().lower(),
            )

            if cluster_key in near_dup_clusters:
                metrics.near_duplicate_count += 1
                lead = near_dup_clusters[cluster_key]
                pruned_candidates.append(c)
                audit_records.append(
                    CandidateAuditRecord(
                        candidate_id=c.candidate_id,
                        expression=c.expression,
                        expression_hash=fp.expr_hash,
                        ast_skeleton=fp.skeleton,
                        category=c.category,
                        fields_used=c.fields_used,
                        operators_used=c.operators_used,
                        is_valid=True,
                        decision="NEAR_DUPLICATE_PRUNED",
                        reason=(
                            f"Near-duplicate parameter jitter of representative candidate '{lead.expression}' "
                            f"with matching skeleton '{fp.skeleton}', fields {fp.fields}, and category '{c.category}'"
                        ),
                    )
                )
            else:
                near_dup_clusters[cluster_key] = c
                retained_candidates.append((c, fp))

        metrics.retained_count = len(retained_candidates)

        # 4. Diversity-Aware Quota Allocation & Exploration Reserve
        # If retained count <= capacity, all are dispatched
        if len(retained_candidates) <= capacity:
            dispatched = [c for c, _ in retained_candidates]
            for c, fp in retained_candidates:
                audit_records.append(
                    CandidateAuditRecord(
                        candidate_id=c.candidate_id,
                        expression=c.expression,
                        expression_hash=fp.expr_hash,
                        ast_skeleton=fp.skeleton,
                        category=c.category,
                        fields_used=c.fields_used,
                        operators_used=c.operators_used,
                        is_valid=True,
                        decision="DISPATCHED",
                        reason="Retained within simulation capacity",
                    )
                )
        else:
            # Deterministic reproducible candidate scoring for selection
            # Sort with deterministic salt based on seed and expression hash
            rng = random.Random(cfg.seed)

            # Group candidates by category
            by_category: dict[str, list[tuple[CandidateItem, CandidateFingerprint]]] = {}
            for c, fp in retained_candidates:
                cat_key = c.category.strip().upper()
                by_category.setdefault(cat_key, []).append((c, fp))

            # Shuffle candidates within each category deterministically using pseudo-random salt
            for cat, items in by_category.items():
                items.sort(
                    key=lambda item: hashlib.sha256(
                        f"{cfg.seed}:{cat}:{item[1].expr_hash}".encode()
                    ).hexdigest()
                )

            # Calculate quotas
            reserve_capacity = max(1, int(round(capacity * cfg.exploration_reserve_ratio)))
            quota_capacity = capacity - reserve_capacity

            # Target quota allocation across categories
            dispatched_items: list[tuple[CandidateItem, CandidateFingerprint]] = []
            leftover_items: list[tuple[CandidateItem, CandidateFingerprint]] = []

            # Default to equal category quotas if none provided
            categories = sorted(by_category.keys())
            n_cats = len(categories)

            family_quotas = cfg.family_quotas or {}
            target_per_cat: dict[str, int] = {}
            for cat in categories:
                if cat in family_quotas:
                    target_per_cat[cat] = int(round(quota_capacity * family_quotas[cat]))
                else:
                    target_per_cat[cat] = max(1, quota_capacity // max(1, n_cats))

            # Step 4a: Take quota targets from each family
            for cat, items in by_category.items():
                target = target_per_cat.get(cat, 1)
                selected = items[:target]
                remaining = items[target:]
                dispatched_items.extend(selected)
                leftover_items.extend(remaining)

            # Step 4b: Allocate exploration reserve (prioritizing diverse field and operator families)
            # Sort leftover items deterministically
            leftover_items.sort(
                key=lambda item: (
                    len(item[0].field_families or {}),
                    len(item[0].operators_used or []),
                    hashlib.sha256(f"reserve:{cfg.seed}:{item[1].expr_hash}".encode()).hexdigest(),
                ),
                reverse=True,
            )

            slots_remaining = capacity - len(dispatched_items)
            if slots_remaining > 0:
                reserve_selected = leftover_items[:slots_remaining]
                dispatched_items.extend(reserve_selected)
                over_capacity = leftover_items[slots_remaining:]
            else:
                over_capacity = leftover_items

            dispatched = [c for c, _ in dispatched_items]

            for c, fp in dispatched_items:
                audit_records.append(
                    CandidateAuditRecord(
                        candidate_id=c.candidate_id,
                        expression=c.expression,
                        expression_hash=fp.expr_hash,
                        ast_skeleton=fp.skeleton,
                        category=c.category,
                        fields_used=c.fields_used,
                        operators_used=c.operators_used,
                        is_valid=True,
                        decision="DISPATCHED",
                        reason="Selected through diversity quota and exploration allocation",
                    )
                )

            for c, fp in over_capacity:
                audit_records.append(
                    CandidateAuditRecord(
                        candidate_id=c.candidate_id,
                        expression=c.expression,
                        expression_hash=fp.expr_hash,
                        ast_skeleton=fp.skeleton,
                        category=c.category,
                        fields_used=c.fields_used,
                        operators_used=c.operators_used,
                        is_valid=True,
                        decision="RESERVED_OVER_CAPACITY",
                        reason="Valid candidate retained for subsequent simulation rounds",
                    )
                )

        metrics.dispatched_count = len(dispatched)

        # Tabulate category, field family, and operator family distributions
        for c in dispatched:
            cat = c.category.strip().upper()
            metrics.category_counts[cat] = metrics.category_counts.get(cat, 0) + 1

            if c.field_families:
                for fld, fam in c.field_families.items():
                    metrics.field_family_counts[fam] = metrics.field_family_counts.get(fam, 0) + 1
            else:
                metrics.field_family_counts["UNCLASSIFIED"] = (
                    metrics.field_family_counts.get("UNCLASSIFIED", 0) + 1
                )

            if c.operators_used:
                for op in c.operators_used:
                    metrics.operator_family_counts[op] = metrics.operator_family_counts.get(op, 0) + 1

        log.info(
            "diversity.selection_complete",
            generated=metrics.generated_count,
            valid=metrics.valid_count,
            exact_duplicates=metrics.exact_duplicate_count,
            near_duplicates=metrics.near_duplicate_count,
            retained=metrics.retained_count,
            dispatched=metrics.dispatched_count,
            category_counts=metrics.category_counts,
        )

        return DiversitySelectionResult(
            dispatched_candidates=dispatched,
            pruned_candidates=pruned_candidates,
            funnel_metrics=metrics,
            audit_records=audit_records,
        )
