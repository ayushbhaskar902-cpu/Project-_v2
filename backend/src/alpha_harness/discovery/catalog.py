"""Catalog discovery service for prompt context generation.

Queries DuckDB for high-quality data fields, datasets, and categories matching the research
scope, and formats them into token-efficient context blocks for LLM researchers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from ..db.duck import Catalog

log = structlog.get_logger(__name__)

# Essential market fields present across equity markets
CORE_MARKET_FIELDS = (
    ("close", "Daily closing price (MATRIX)", 1.0),
    ("open", "Daily opening price (MATRIX)", 1.0),
    ("high", "Daily highest price (MATRIX)", 1.0),
    ("low", "Daily lowest price (MATRIX)", 1.0),
    ("volume", "Daily trading volume in shares (MATRIX)", 1.0),
    ("vwap", "Volume-weighted average price (MATRIX)", 1.0),
    ("adv20", "20-day average daily volume in currency (MATRIX)", 1.0),
    ("returns", "Daily close-to-close return (MATRIX)", 1.0),
    ("cap", "Market capitalization (MATRIX)", 1.0),
)

GROUPING_FIELDS = (
    ("sector", "Broad sector classification (GROUP)"),
    ("industry", "Specific industry classification (GROUP)"),
    ("subindustry", "Sub-industry classification (GROUP)"),
    ("country", "Country of risk / exchange (GROUP)"),
    ("market", "Overall market group (GROUP)"),
)

STANDARD_OPERATORS = [
    ("ts_rank(x, d)", "Time-series percentile rank over d trading days"),
    ("ts_zscore(x, d)", "Time-series z-score (mean / std) over d trading days"),
    ("ts_delta(x, d)", "Difference x - ts_delay(x, d)"),
    ("ts_delay(x, d)", "Value of x lagged by d trading days"),
    ("ts_mean(x, d)", "Moving average of x over d trading days"),
    ("ts_std_dev(x, d)", "Moving standard deviation of x over d trading days"),
    ("ts_corr(x, y, d)", "Rolling correlation between x and y over d trading days"),
    ("rank(x)", "Cross-sectional rank across all stocks in universe [0, 1]"),
    ("group_rank(x, group)", "Cross-sectional rank within sector or industry"),
    ("group_neutralize(x, group)", "Subtract industry or sector mean from x"),
    ("trade_when(cond, alpha, -1)", "Hold or enter when cond is true, exit/neutral otherwise"),
    ("quantile(x, driver=y, quantiles=5)", "Rank into quantile buckets by driver"),
    ("scale(x, target=1)", "Scale weights so sum(abs(weights)) == target"),
    ("ts_backfill(x, d)", "Fill missing values with most recent value up to d days (UNCOUNTED)"),
    ("vec_avg(x)", "Collapse vector field to scalar mean across components"),
    ("vec_sum(x)", "Collapse vector field to scalar sum across components"),
]


@dataclass(frozen=True, slots=True)
class DiscoveredField:
    field_id: str
    field_type: str
    coverage: float | None
    description: str
    dataset_id: str
    category: str


@dataclass(frozen=True, slots=True)
class DiscoveredDataset:
    dataset_id: str
    name: str
    category: str
    description: str
    fields: tuple[DiscoveredField, ...]


class CatalogDiscovery:
    """Discovers and formats datasets and fields for LLM research prompts."""

    def __init__(self, catalog: Catalog | None = None) -> None:
        self._catalog = catalog

    async def get_datasets_for_scope(
        self,
        region: str = "USA",
        delay: int = 1,
        universe: str = "TOP3000",
        instrument_type: str = "EQUITY",
        limit: int = 5,
        min_coverage: float = 0.6,
        category: str | None = None,
    ) -> list[DiscoveredDataset]:
        """Fetches top datasets with their best fields for a target market scope."""
        if self._catalog is None:
            return []

        try:
            # First find candidate datasets
            cat_filter = "AND (category_name = ? OR category_id = ?)" if category else ""
            cat_params = [category, category] if category else []

            dataset_query = f"""
                SELECT dataset_id, name, description, category_name, subcategory_name
                FROM data_set
                WHERE region = ? AND delay = ? {cat_filter}
                ORDER BY name ASC
                LIMIT ?
            """
            dataset_rows = await self._catalog.query(
                dataset_query, [region, delay, *cat_params, limit]
            )

            results: list[DiscoveredDataset] = []
            for d_row in dataset_rows:
                d_id = str(d_row["dataset_id"])
                name = str(d_row.get("name") or d_id)
                cat = " › ".join(
                    str(c)
                    for c in (d_row.get("category_name"), d_row.get("subcategory_name"))
                    if c
                )
                desc = str(d_row.get("description") or "")[:300]

                # Get top fields for this dataset
                field_query = """
                    SELECT field_id, field_type, coverage, description
                    FROM data_field
                    WHERE instrument_type = ? AND region = ? AND delay = ? AND universe = ?
                      AND dataset_id = ? AND coverage >= ?
                    ORDER BY coverage DESC, alpha_count DESC
                    LIMIT 20
                """
                field_rows = await self._catalog.query(
                    field_query, [instrument_type, region, delay, universe, d_id, min_coverage]
                )

                fields = tuple(
                    DiscoveredField(
                        field_id=str(f["field_id"]),
                        field_type=str(f["field_type"]),
                        coverage=float(f["coverage"]) if f["coverage"] is not None else None,
                        description=str(f.get("description") or "")[:120],
                        dataset_id=d_id,
                        category=cat,
                    )
                    for f in field_rows
                )

                if fields:
                    results.append(
                        DiscoveredDataset(
                            dataset_id=d_id,
                            name=name,
                            category=cat,
                            description=desc,
                            fields=fields,
                        )
                    )

            return results
        except Exception as exc:
            log.warning("discovery.catalog_query_failed", error=str(exc))
            return []

    def format_catalog_context(self, datasets: list[DiscoveredDataset]) -> str:
        """Formats discovered datasets and fields into a token-efficient prompt string."""
        blocks: list[str] = []

        # 1. Always list basic market fields
        blocks.append("1. CORE MARKET FIELDS (Available for all stocks):")
        for f_name, f_desc, cov in CORE_MARKET_FIELDS:
            blocks.append(f"   - `{f_name}`: {f_desc} (Coverage: {int(cov*100)}%)")

        blocks.append("\n2. GROUPING FIELDS (Use in group_rank, group_neutralize; DO NOT count as data fields):")
        for g_name, g_desc in GROUPING_FIELDS:
            blocks.append(f"   - `{g_name}`: {g_desc}")

        # 2. Add specific dataset fields
        if datasets:
            blocks.append("\n3. SPECIALIZED DATASETS & FIELDS:")
            for ds in datasets:
                blocks.append(f"\n   DATASET: `{ds.dataset_id}` ({ds.name})")
                if ds.category:
                    blocks.append(f"   Category: {ds.category}")
                if ds.description:
                    blocks.append(f"   Description: {ds.description}")
                blocks.append("   Fields:")
                for f in ds.fields:
                    cov_pct = f"{int(f.coverage*100)}%" if f.coverage is not None else "N/A"
                    blocks.append(
                        f"     * `{f.field_id}` [{f.field_type}]: {f.description} (Coverage: {cov_pct})"
                    )
        else:
            blocks.append("\n3. SPECIALIZED DATASETS: No custom dataset loaded; use CORE MARKET fields above.")

        return "\n".join(blocks)

    def format_operator_summary(self, extra_operators: list[dict[str, Any]] | None = None) -> str:
        """Formats the list of valid operators and arities."""
        lines = []
        for sig, desc in STANDARD_OPERATORS:
            lines.append(f"- `{sig}`: {desc}")

        if extra_operators:
            for op in extra_operators[:20]:
                name = op.get("name")
                desc = op.get("description", "")
                if name and not any(sig.startswith(str(name)) for sig, _ in STANDARD_OPERATORS):
                    lines.append(f"- `{name}`: {desc}")

        return "\n".join(lines)
