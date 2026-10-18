"""Catalog discovery service for prompt context generation.

Queries DuckDB for high-quality data fields, datasets, and categories matching the research
scope, and formats them into token-efficient context blocks for LLM researchers.
"""

from __future__ import annotations

import asyncio
import enum
import hashlib
import json
from dataclasses import dataclass, field
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

CORE_MARKET_FIELD_NAMES: frozenset[str] = frozenset(f[0] for f in CORE_MARKET_FIELDS)
STANDARD_GROUPING_FIELD_NAMES: frozenset[str] = frozenset(
    {"market", "sector", "industry", "subindustry", "country", "exchange", "currency", "split", "adjfactor"}
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


# =============================================================================
# P1-06: Authoritative Operator Catalog with Versioned Fingerprinting
# =============================================================================


#: Comprehensive built-in operator definitions for WorldQuant BRAIN.
#: Each entry mirrors the schema returned by GET /operators.
#: Scope: REGULAR = usable in regular alphas; COMBO = combo-only operators.
BUILTIN_OPERATOR_DEFS: list[dict[str, Any]] = [
    # ---- Time-Series Operators ----
    {"name": "ts_rank", "definition": "ts_rank(x, d)", "category": "Time Series",
     "scope": ["REGULAR", "COMBO"], "description": "Time-series percentile rank over d trading days"},
    {"name": "ts_zscore", "definition": "ts_zscore(x, d)", "category": "Time Series",
     "scope": ["REGULAR", "COMBO"], "description": "Time-series z-score (mean / std) over d trading days"},
    {"name": "ts_delta", "definition": "ts_delta(x, d)", "category": "Time Series",
     "scope": ["REGULAR", "COMBO"], "description": "Difference x - ts_delay(x, d)"},
    {"name": "ts_delay", "definition": "ts_delay(x, d)", "category": "Time Series",
     "scope": ["REGULAR", "COMBO"], "description": "Value of x lagged by d trading days"},
    {"name": "ts_mean", "definition": "ts_mean(x, d)", "category": "Time Series",
     "scope": ["REGULAR", "COMBO"], "description": "Moving average of x over d trading days"},
    {"name": "ts_std_dev", "definition": "ts_std_dev(x, d)", "category": "Time Series",
     "scope": ["REGULAR", "COMBO"], "description": "Moving standard deviation of x over d trading days"},
    {"name": "ts_corr", "definition": "ts_corr(x, y, d)", "category": "Time Series",
     "scope": ["REGULAR", "COMBO"], "description": "Rolling correlation between x and y over d trading days"},
    {"name": "ts_covariance", "definition": "ts_covariance(x, y, d)", "category": "Time Series",
     "scope": ["REGULAR", "COMBO"], "description": "Rolling covariance between x and y over d trading days"},
    {"name": "ts_sum", "definition": "ts_sum(x, d)", "category": "Time Series",
     "scope": ["REGULAR", "COMBO"], "description": "Rolling sum of x over d trading days"},
    {"name": "ts_product", "definition": "ts_product(x, d)", "category": "Time Series",
     "scope": ["REGULAR", "COMBO"], "description": "Rolling product of x over d trading days"},
    {"name": "ts_decay_linear", "definition": "ts_decay_linear(x, d)", "category": "Time Series",
     "scope": ["REGULAR", "COMBO"], "description": "Linearly weighted moving average of x over d trading days"},
    {"name": "ts_argmax", "definition": "ts_argmax(x, d)", "category": "Time Series",
     "scope": ["REGULAR", "COMBO"], "description": "Number of days since the max of x over d trading days"},
    {"name": "ts_argmin", "definition": "ts_argmin(x, d)", "category": "Time Series",
     "scope": ["REGULAR", "COMBO"], "description": "Number of days since the min of x over d trading days"},
    {"name": "ts_backfill", "definition": "ts_backfill(x, d)", "category": "Time Series",
     "scope": ["REGULAR", "COMBO"], "description": "Fill missing values with most recent value up to d days (UNCOUNTED)"},
    {"name": "ts_max", "definition": "ts_max(x, d)", "category": "Time Series",
     "scope": ["REGULAR", "COMBO"], "description": "Rolling maximum of x over d trading days"},
    {"name": "ts_min", "definition": "ts_min(x, d)", "category": "Time Series",
     "scope": ["REGULAR", "COMBO"], "description": "Rolling minimum of x over d trading days"},
    {"name": "ts_returns", "definition": "ts_returns(x, d)", "category": "Time Series",
     "scope": ["REGULAR", "COMBO"], "description": "Returns of x over d trading days: (x - ts_delay(x, d)) / ts_delay(x, d)"},
    # ---- Cross-Sectional / Ranking Operators ----
    {"name": "rank", "definition": "rank(x)", "category": "Cross-Sectional",
     "scope": ["REGULAR", "COMBO"], "description": "Cross-sectional rank across all stocks in universe [0, 1]"},
    {"name": "quantile", "definition": "quantile(x, driver=y, quantiles=5)", "category": "Cross-Sectional",
     "scope": ["REGULAR", "COMBO"], "description": "Rank into quantile buckets by driver"},
    {"name": "scale", "definition": "scale(x, target=1)", "category": "Cross-Sectional",
     "scope": ["REGULAR", "COMBO"], "description": "Scale weights so sum(abs(weights)) == target"},
    # ---- Group Operators ----
    {"name": "group_rank", "definition": "group_rank(x, group)", "category": "Group",
     "scope": ["REGULAR", "COMBO"], "description": "Cross-sectional rank within sector or industry"},
    {"name": "group_neutralize", "definition": "group_neutralize(x, group)", "category": "Group",
     "scope": ["REGULAR", "COMBO"], "description": "Subtract industry or sector mean from x"},
    {"name": "group_mean", "definition": "group_mean(x, group)", "category": "Group",
     "scope": ["REGULAR", "COMBO"], "description": "Mean of x within group"},
    {"name": "group_zscore", "definition": "group_zscore(x, group)", "category": "Group",
     "scope": ["REGULAR", "COMBO"], "description": "Z-score of x within group"},
    # ---- Math Operators ----
    {"name": "abs", "definition": "abs(x)", "category": "Math",
     "scope": ["REGULAR", "COMBO"], "description": "Absolute value of x"},
    {"name": "log", "definition": "log(x)", "category": "Math",
     "scope": ["REGULAR", "COMBO"], "description": "Natural logarithm of x"},
    {"name": "sign", "definition": "sign(x)", "category": "Math",
     "scope": ["REGULAR", "COMBO"], "description": "Sign of x (-1, 0, 1)"},
    {"name": "sqrt", "definition": "sqrt(x)", "category": "Math",
     "scope": ["REGULAR", "COMBO"], "description": "Square root of x"},
    {"name": "fraction", "definition": "fraction(x)", "category": "Math",
     "scope": ["REGULAR", "COMBO"], "description": "Fractional part of x"},
    {"name": "max", "definition": "max(x, y)", "category": "Math",
     "scope": ["REGULAR", "COMBO"], "description": "Element-wise maximum of x and y"},
    {"name": "min", "definition": "min(x, y)", "category": "Math",
     "scope": ["REGULAR", "COMBO"], "description": "Element-wise minimum of x and y"},
    {"name": "power", "definition": "power(x, y)", "category": "Math",
     "scope": ["REGULAR", "COMBO"], "description": "x raised to the power of y"},
    # ---- Logic / Conditional Operators ----
    {"name": "trade_when", "definition": "trade_when(cond, alpha, exit)", "category": "Logic",
     "scope": ["REGULAR", "COMBO"], "description": "Hold or enter when cond is true, exit/neutral otherwise"},
    # ---- Vector Operators ----
    {"name": "vec_avg", "definition": "vec_avg(x)", "category": "Vector",
     "scope": ["REGULAR", "COMBO"], "description": "Collapse vector field to scalar mean across components"},
    {"name": "vec_sum", "definition": "vec_sum(x)", "category": "Vector",
     "scope": ["REGULAR", "COMBO"], "description": "Collapse vector field to scalar sum across components"},
]


class OperatorCatalogError(RuntimeError):
    """Base exception for operator catalog errors."""


class UnknownOperatorError(OperatorCatalogError):
    """Raised when an unknown or unsupported operator is referenced."""


class OperatorArityError(OperatorCatalogError):
    """Raised when an operator call has an invalid number of arguments."""


class OperatorScopeError(OperatorCatalogError):
    """Raised when an operator is not available in the target alpha scope."""


@dataclass(frozen=True, slots=True)
class OperatorEntry:
    """Authoritative metadata for a single BRAIN operator."""

    name: str
    category: str
    definition: str
    description: str
    required_inputs: int
    max_inputs: int | None
    params: tuple[str, ...]
    scope: tuple[str, ...]
    level: str | None = None


class OperatorCatalog:
    """Authoritative, versioned registry of BRAIN operators with deterministic fingerprinting.

    Provides:
    - Deterministic SHA-256 catalog versioning (op_v1_<sha256[:16]>).
    - Scope-filtered operator tables for REGULAR and COMBO alphas.
    - Built-in fallback catalog from BUILTIN_OPERATOR_DEFS when BRAIN API is offline.
    - Sync support from GET /operators and MetadataCache.
    - Arity validation via the fastexpr OperatorInfo interface.
    """

    def __init__(
        self,
        operator_defs: list[dict[str, Any]] | None = None,
        scope: str = "REGULAR",
        source: str | None = None,
    ) -> None:
        self._scope = scope.upper().strip()
        self._source = source or ("BUILTIN_FALLBACK" if operator_defs is None else "METADATA_CACHE")
        raw_defs = operator_defs if operator_defs is not None else BUILTIN_OPERATOR_DEFS
        self._all_entries: dict[str, OperatorEntry] = {}
        self._scoped_entries: dict[str, OperatorEntry] = {}

        for op in raw_defs:
            entry = self._parse_operator_def(op)
            if entry is not None:
                self._all_entries[entry.name] = entry
                # Apply scope filter
                if self._scope in entry.scope or not entry.scope:
                    self._scoped_entries[entry.name] = entry

        self._catalog_version = self._compute_fingerprint()

    @staticmethod
    def _parse_operator_def(op: dict[str, Any]) -> OperatorEntry | None:
        """Parses a single operator definition dict into an OperatorEntry."""
        name = op.get("name")
        definition = op.get("definition", "")
        if not isinstance(name, str) or not name:
            return None

        # Parse params from definition using the fastexpr approach
        import re
        sig_match = re.match(r"([A-Za-z_]\w*)\s*\(", str(definition))
        if sig_match is None or sig_match.group(1) != name:
            # No valid signature; still register with zero-arity
            return OperatorEntry(
                name=name,
                category=str(op.get("category") or ""),
                definition=str(definition),
                description=str(op.get("description") or ""),
                required_inputs=0,
                max_inputs=None,
                params=(),
                scope=tuple(op.get("scope") or []),
                level=op.get("level"),
            )

        # Extract parameter list
        paren_start = sig_match.end()
        depth, current, params_list = 0, [], []
        quoted = False
        for ch in str(definition)[paren_start:]:
            if ch in "\"'""":
                quoted = not quoted
            elif not quoted and ch == "(":
                depth += 1
            elif not quoted and ch == ")":
                if depth == 0:
                    tail = "".join(current).strip()
                    if tail or params_list:
                        params_list.append(tail)
                    break
                depth -= 1
            elif not quoted and depth == 0 and ch == ",":
                params_list.append("".join(current).strip())
                current = []
                continue
            current.append(ch)

        variadic = any(".." in p for p in params_list)
        required = sum(1 for p in params_list if "=" not in p and ".." not in p)
        maximum = None if variadic else len(params_list)

        return OperatorEntry(
            name=name,
            category=str(op.get("category") or ""),
            definition=str(definition),
            description=str(op.get("description") or ""),
            required_inputs=required,
            max_inputs=maximum,
            params=tuple(params_list),
            scope=tuple(op.get("scope") or []),
            level=op.get("level"),
        )

    def _compute_fingerprint(self) -> str:
        """Computes deterministic SHA-256 fingerprint over scoped operator definitions."""
        records = []
        for name in sorted(self._scoped_entries.keys()):
            entry = self._scoped_entries[name]
            records.append({
                "category": entry.category,
                "definition": entry.definition,
                "max_inputs": entry.max_inputs,
                "name": entry.name,
                "params": list(entry.params),
                "required_inputs": entry.required_inputs,
                "scope": list(entry.scope),
            })
        payload = {
            "operators": records,
            "scope": self._scope,
        }
        canonical_str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        h = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()
        return f"op_v1_{h[:16]}"

    @property
    def catalog_version(self) -> str:
        """Deterministic version fingerprint of the active operator catalog."""
        return self._catalog_version

    @property
    def version(self) -> str:
        """Alias for catalog_version."""
        return self._catalog_version

    @property
    def source(self) -> str:
        """The metadata provenance source ('BRAIN_LIVE', 'METADATA_CACHE', or 'BUILTIN_FALLBACK')."""
        return self._source

    @property
    def is_fallback(self) -> bool:
        """Whether this catalog is using unverified built-in fallback definitions."""
        return self._source == "BUILTIN_FALLBACK"

    @property
    def is_verified(self) -> bool:
        """Whether this catalog was verified against live BRAIN or MetadataCache."""
        return self._source in ("BRAIN_LIVE", "METADATA_CACHE")

    @property
    def scope(self) -> str:
        """The active scope filter (e.g. REGULAR, COMBO)."""
        return self._scope

    def get_operator(self, name: str) -> OperatorEntry | None:
        """Returns the scoped operator entry for the given name, or None."""
        return self._scoped_entries.get(name)

    def get_all_operator(self, name: str) -> OperatorEntry | None:
        """Returns the operator entry regardless of scope, or None."""
        return self._all_entries.get(name)

    def get_operator_names(self) -> set[str]:
        """Returns the set of operator names available in the active scope."""
        return set(self._scoped_entries.keys())

    def to_operator_table(self) -> dict[str, Any]:
        """Converts the scoped catalog to the fastexpr OperatorInfo table format.

        Returns a dict[str, OperatorInfo] compatible with fastexpr.validate().
        """
        from ..labs.fastexpr import OperatorInfo

        table: dict[str, OperatorInfo] = {}
        for name, entry in self._scoped_entries.items():
            table[name] = OperatorInfo(
                name=entry.name,
                category=entry.category,
                required=entry.required_inputs,
                maximum=entry.max_inputs,
                params=entry.params,
            )
        return table

    def validate_operator(self, name: str, arg_count: int) -> tuple[bool, str | None]:
        """Validates an operator reference against this catalog.

        Returns (is_valid, error_message).
        """
        scoped = self._scoped_entries.get(name)
        if scoped is not None:
            if arg_count < scoped.required_inputs:
                return False, (
                    f"Invalid operator arity: {name} requires at least "
                    f"{scoped.required_inputs} inputs (got {arg_count})"
                )
            if scoped.max_inputs is not None and arg_count > scoped.max_inputs:
                return False, (
                    f"Invalid operator arity: {name} accepts at most "
                    f"{scoped.max_inputs} inputs (got {arg_count})"
                )
            return True, None

        # Check if it exists in a different scope
        all_entry = self._all_entries.get(name)
        if all_entry is not None:
            return False, (
                f"Operator '{name}' is only available in scope "
                f"{', '.join(all_entry.scope)}, not in {self._scope}"
            )

        return False, f"Unknown or unsupported operator: {name}"

    def validate_call_arguments(
        self,
        name: str,
        args: list[Any],
        kwargs: list[tuple[str, Any]] | None = None,
    ) -> tuple[bool, str | None]:
        """Validates operator call argument contracts (lookback window d > 0, grouping argument)."""
        arg_count = len(args)
        is_valid, err = self.validate_operator(name, arg_count)
        if not is_valid:
            return False, err

        scoped = self._scoped_entries.get(name)
        if not scoped:
            return False, f"Unknown or unsupported operator: {name}"

        # 1. Time-Series window contract (d must be positive integer)
        if name.startswith("ts_") and scoped.params:
            if "d" in scoped.params and arg_count >= scoped.required_inputs:
                d_idx = scoped.params.index("d")
                if d_idx < len(args):
                    d_node = args[d_idx]
                    val: float | None = None
                    if hasattr(d_node, "kind"):
                        if d_node.kind == "num":
                            try:
                                val = float(d_node.value)
                            except ValueError:
                                pass
                        elif d_node.kind == "unary" and d_node.value == "-" and d_node.args and d_node.args[0].kind == "num":
                            try:
                                val = -float(d_node.args[0].value)
                            except ValueError:
                                pass
                    elif isinstance(d_node, (int, float)):
                        val = float(d_node)

                    if val is not None:
                        if val <= 0:
                            return False, f"Time-series operator '{name}' lookback window 'd' must be greater than 0 (got {val:g})"
                        if not val.is_integer():
                            return False, f"Time-series operator '{name}' lookback window 'd' must be an integer number of days (got {val:g})"

        # 2. Group operator grouping argument contract
        if name.startswith("group_") and "group" in scoped.params:
            group_idx = scoped.params.index("group")
            if group_idx < len(args):
                g_node = args[group_idx]
                is_num = False
                if hasattr(g_node, "kind"):
                    if g_node.kind == "num" or (g_node.kind == "unary" and g_node.value == "-" and g_node.args and g_node.args[0].kind == "num"):
                        is_num = True
                elif isinstance(g_node, (int, float)):
                    is_num = True

                if is_num:
                    return False, f"Group operator '{name}' requires a grouping field (e.g. sector, industry) as argument 2, not a number"

        return True, None

    def get_operators_used(self, operator_names_list: list[str]) -> list[str]:
        """Filters and returns only the operators from the list that exist in this catalog."""
        return [n for n in operator_names_list if n in self._scoped_entries]

    def format_prompt_summary(self) -> str:
        """Formats the operator catalog for prompt injection."""
        blocks: list[str] = [
            f"OPERATOR CATALOG (Version: {self.catalog_version}, Scope: {self._scope}):",
        ]
        # Group by category
        by_category: dict[str, list[OperatorEntry]] = {}
        for entry in self._scoped_entries.values():
            cat = entry.category or "Other"
            by_category.setdefault(cat, []).append(entry)

        for cat in sorted(by_category.keys()):
            blocks.append(f"\n  {cat}:")
            for entry in sorted(by_category[cat], key=lambda e: e.name):
                blocks.append(f"    - `{entry.definition}`: {entry.description}")

        blocks.append(
            "\nSTRICT OPERATOR CONSTRAINTS: You may ONLY use the operators listed above. "
            "Any expression using unknown, unlisted, or out-of-scope operators will fail "
            "pre-simulation validation."
        )
        return "\n".join(blocks)

    @classmethod
    def from_brain_operators(
        cls,
        operators: list[Any],
        scope: str = "REGULAR",
    ) -> OperatorCatalog:
        """Constructs an OperatorCatalog from BRAIN API operator list (GET /operators).

        Args:
            operators: List of Operator Pydantic models or raw dicts from BrainEndpoints.list_operators().
            scope: Target alpha scope filter.
        """
        defs: list[dict[str, Any]] = []
        for op in operators:
            if hasattr(op, "model_dump"):
                d = op.model_dump()
            elif isinstance(op, dict):
                d = op
            else:
                continue
            defs.append({
                "name": d.get("name"),
                "definition": d.get("definition"),
                "category": d.get("category"),
                "scope": d.get("scope", []),
                "description": d.get("description"),
                "level": d.get("level"),
            })
        return cls(operator_defs=defs, scope=scope, source="BRAIN_LIVE")

    @classmethod
    def from_metadata_cache(
        cls,
        cached_operators: list[dict[str, Any]],
        scope: str = "REGULAR",
    ) -> OperatorCatalog:
        """Constructs an OperatorCatalog from MetadataCache stored operator data."""
        return cls(operator_defs=cached_operators, scope=scope, source="METADATA_CACHE")


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
    region: str = "USA"
    delay: int = 1
    universe: str = "ALL"
    instrument_type: str = "EQUITY"
# Built-in specialized datasets by region and domain for WorldQuant BRAIN
BUILTIN_DATASETS_BY_REGION: dict[str, list[DiscoveredDataset]] = {
    "USA": [
        DiscoveredDataset(
            dataset_id="fundamental_us",
            name="Compustat / FactSet Corporate Financials (USA)",
            category="Fundamental",
            description="Comprehensive income statement, balance sheet, and cash flow items for US equities.",
            fields=(
                DiscoveredField("assets", "MATRIX", 0.95, "Total assets in currency", "fundamental_us", "Fundamental"),
                DiscoveredField("liabilities", "MATRIX", 0.95, "Total liabilities in currency", "fundamental_us", "Fundamental"),
                DiscoveredField("equity", "MATRIX", 0.95, "Common stockholders equity (book value)", "fundamental_us", "Fundamental"),
                DiscoveredField("debt", "MATRIX", 0.92, "Total short-term and long-term interest-bearing debt", "fundamental_us", "Fundamental"),
                DiscoveredField("revenue", "MATRIX", 0.95, "Total revenue/sales over reporting period", "fundamental_us", "Fundamental"),
                DiscoveredField("net_income", "MATRIX", 0.95, "Net income after tax and expenses", "fundamental_us", "Fundamental"),
                DiscoveredField("ebit", "MATRIX", 0.93, "Earnings before interest and taxes", "fundamental_us", "Fundamental"),
                DiscoveredField("ebitda", "MATRIX", 0.93, "Earnings before interest, taxes, depreciation, and amortization", "fundamental_us", "Fundamental"),
                DiscoveredField("operating_cash_flow", "MATRIX", 0.92, "Cash generated from operating activities", "fundamental_us", "Fundamental"),
                DiscoveredField("free_cash_flow", "MATRIX", 0.90, "Operating cash flow minus capital expenditures", "fundamental_us", "Fundamental"),
                DiscoveredField("capex", "MATRIX", 0.91, "Capital expenditures on plant, property, equipment", "fundamental_us", "Fundamental"),
                DiscoveredField("cash", "MATRIX", 0.95, "Cash and cash equivalents on balance sheet", "fundamental_us", "Fundamental"),
                DiscoveredField("gross_profit", "MATRIX", 0.94, "Revenue minus cost of goods sold (COGS)", "fundamental_us", "Fundamental"),
                DiscoveredField("book_value", "MATRIX", 0.95, "Total book value of common equity", "fundamental_us", "Fundamental"),
                DiscoveredField("roa", "MATRIX", 0.92, "Return on assets (net_income / assets)", "fundamental_us", "Fundamental"),
                DiscoveredField("roe", "MATRIX", 0.92, "Return on equity (net_income / equity)", "fundamental_us", "Fundamental"),
                DiscoveredField("gross_margin", "MATRIX", 0.94, "Gross profit margin (gross_profit / revenue)", "fundamental_us", "Fundamental"),
                DiscoveredField("operating_margin", "MATRIX", 0.93, "Operating margin (ebit / revenue)", "fundamental_us", "Fundamental"),
                DiscoveredField("asset_turnover", "MATRIX", 0.92, "Asset turnover efficiency (revenue / assets)", "fundamental_us", "Fundamental"),
                DiscoveredField("pe", "MATRIX", 0.91, "Price-to-Earnings ratio (cap / net_income)", "fundamental_us", "Fundamental"),
                DiscoveredField("pb", "MATRIX", 0.93, "Price-to-Book ratio (cap / book_value)", "fundamental_us", "Fundamental"),
                DiscoveredField("ps", "MATRIX", 0.94, "Price-to-Sales ratio (cap / revenue)", "fundamental_us", "Fundamental"),
            ),
        ),
        DiscoveredDataset(
            dataset_id="analyst_estimates_us",
            name="I/B/E/S Consensus Analyst Estimates (USA)",
            category="Analyst",
            description="Consensus forward earnings estimates, revenue forecasts, price targets, and revisions.",
            fields=(
                DiscoveredField("est_eps", "MATRIX", 0.88, "Consensus next fiscal year earnings-per-share estimate", "analyst_estimates_us", "Analyst"),
                DiscoveredField("est_sales", "MATRIX", 0.86, "Consensus next fiscal year sales/revenue forecast", "analyst_estimates_us", "Analyst"),
                DiscoveredField("est_target_price", "MATRIX", 0.89, "Mean 12-month analyst consensus target price", "analyst_estimates_us", "Analyst"),
                DiscoveredField("est_recommendation", "MATRIX", 0.87, "Mean analyst recommendation score (1=Strong Buy, 5=Sell)", "analyst_estimates_us", "Analyst"),
                DiscoveredField("est_eps_mean", "MATRIX", 0.88, "Mean consensus EPS estimate across covering analysts", "analyst_estimates_us", "Analyst"),
                DiscoveredField("est_eps_std", "MATRIX", 0.85, "Standard deviation (dispersion) of EPS estimates", "analyst_estimates_us", "Analyst"),
                DiscoveredField("est_revision_up", "MATRIX", 0.82, "Count of upward estimate revisions in last 30 days", "analyst_estimates_us", "Analyst"),
                DiscoveredField("est_revision_down", "MATRIX", 0.82, "Count of downward estimate revisions in last 30 days", "analyst_estimates_us", "Analyst"),
                DiscoveredField("est_fwd_pe", "MATRIX", 0.87, "Forward Price-to-Earnings based on consensus EPS", "analyst_estimates_us", "Analyst"),
            ),
        ),
        DiscoveredDataset(
            dataset_id="sentiment_us",
            name="RavenPack News & Social Media Sentiment (USA)",
            category="Sentiment",
            description="Real-time media sentiment polarity, news buzz, and narrative attention scores.",
            fields=(
                DiscoveredField("news_sentiment", "MATRIX", 0.80, "Aggregate news sentiment polarity score [-1.0, 1.0]", "sentiment_us", "Sentiment"),
                DiscoveredField("sentiment_buzz", "MATRIX", 0.78, "Abnormal news volume buzz relative to 30-day baseline", "sentiment_us", "Sentiment"),
                DiscoveredField("social_sentiment_score", "MATRIX", 0.75, "Social media sentiment indicator score", "sentiment_us", "Sentiment"),
                DiscoveredField("news_volume", "MATRIX", 0.82, "Daily count of financial news articles mentioning firm", "sentiment_us", "Sentiment"),
                DiscoveredField("media_attention", "MATRIX", 0.79, "Relative media attention score across financial wires", "sentiment_us", "Sentiment"),
            ),
        ),
        DiscoveredDataset(
            dataset_id="short_interest_us",
            name="S3 Partners Short Interest & Stock Loan (USA)",
            category="Short Interest",
            description="Daily short interest positions, shares on loan, utilization rates, and borrow costs.",
            fields=(
                DiscoveredField("short_interest", "MATRIX", 0.85, "Total shares currently held short", "short_interest_us", "Short Interest"),
                DiscoveredField("short_interest_ratio", "MATRIX", 0.85, "Short interest as percentage of free float", "short_interest_us", "Short Interest"),
                DiscoveredField("days_to_cover", "MATRIX", 0.84, "Days to cover short positions (short_interest / adv20)", "short_interest_us", "Short Interest"),
                DiscoveredField("shares_on_loan", "MATRIX", 0.83, "Total shares borrowed in securities lending market", "short_interest_us", "Short Interest"),
                DiscoveredField("utilization", "MATRIX", 0.82, "Lending utilization rate (% of lendable supply on loan)", "short_interest_us", "Short Interest"),
                DiscoveredField("borrow_cost", "MATRIX", 0.80, "Annualized fee to borrow stock for shorting (bps)", "short_interest_us", "Short Interest"),
            ),
        ),
        DiscoveredDataset(
            dataset_id="risk_factors_us",
            name="Barra / Equity Style Risk Factors (USA)",
            category="Model Data",
            description="Systematic cross-sectional risk factors and idiosyncratic volatility measures.",
            fields=(
                DiscoveredField("beta", "MATRIX", 0.94, "60-day rolling market beta relative to universe", "risk_factors_us", "Model Data"),
                DiscoveredField("residual_volatility", "MATRIX", 0.92, "Idiosyncratic volatility after stripping market returns", "risk_factors_us", "Model Data"),
                DiscoveredField("momentum_12m", "MATRIX", 0.95, "12-month minus 1-month cross-sectional price momentum", "risk_factors_us", "Model Data"),
                DiscoveredField("size", "MATRIX", 0.95, "Log market capitalization cross-sectional factor", "risk_factors_us", "Model Data"),
                DiscoveredField("value_factor", "MATRIX", 0.93, "Composite book-to-price and earnings yield factor", "risk_factors_us", "Model Data"),
            ),
        ),
    ],
    "EUR": [
        DiscoveredDataset(
            dataset_id="fundamental_eu",
            name="European Financial Fundamentals (EUR)",
            category="Fundamental",
            description="Key accounting, profitability, and balance sheet indicators for European equities.",
            fields=(
                DiscoveredField("revenue", "MATRIX", 0.92, "Net sales and operating revenue in EUR", "fundamental_eu", "Fundamental"),
                DiscoveredField("ebitda", "MATRIX", 0.90, "Operating earnings before D&A in EUR", "fundamental_eu", "Fundamental"),
                DiscoveredField("ebit", "MATRIX", 0.90, "Operating profit (EBIT) in EUR", "fundamental_eu", "Fundamental"),
                DiscoveredField("net_income", "MATRIX", 0.92, "Net income attributable to common shareholders", "fundamental_eu", "Fundamental"),
                DiscoveredField("total_debt", "MATRIX", 0.89, "Total short and long-term financial debt", "fundamental_eu", "Fundamental"),
                DiscoveredField("equity", "MATRIX", 0.92, "Shareholders equity book value", "fundamental_eu", "Fundamental"),
                DiscoveredField("free_cash_flow", "MATRIX", 0.88, "Free cash flow to firm", "fundamental_eu", "Fundamental"),
                DiscoveredField("dividend_yield", "MATRIX", 0.91, "Trailing 12-month dividend yield", "fundamental_eu", "Fundamental"),
                DiscoveredField("operating_margin", "MATRIX", 0.90, "EBIT divided by revenue", "fundamental_eu", "Fundamental"),
                DiscoveredField("roe", "MATRIX", 0.89, "Return on equity", "fundamental_eu", "Fundamental"),
                DiscoveredField("book_value", "MATRIX", 0.92, "Total book value", "fundamental_eu", "Fundamental"),
            ),
        ),
        DiscoveredDataset(
            dataset_id="analyst_eu",
            name="I/B/E/S European Consensus (EUR)",
            category="Analyst",
            description="Analyst forecasts, earnings estimates, and target prices for European companies.",
            fields=(
                DiscoveredField("est_eps", "MATRIX", 0.85, "Forward EPS consensus estimate in local currency", "analyst_eu", "Analyst"),
                DiscoveredField("est_sales", "MATRIX", 0.83, "Forward revenue consensus estimate", "analyst_eu", "Analyst"),
                DiscoveredField("est_target_price", "MATRIX", 0.86, "Consensus 12-month target price", "analyst_eu", "Analyst"),
                DiscoveredField("est_recommendation", "MATRIX", 0.84, "Mean broker recommendation score (1=Buy, 5=Sell)", "analyst_eu", "Analyst"),
                DiscoveredField("est_revision_up", "MATRIX", 0.79, "Count of upward revisions in last 30 days", "analyst_eu", "Analyst"),
                DiscoveredField("est_revision_down", "MATRIX", 0.79, "Count of downward revisions in last 30 days", "analyst_eu", "Analyst"),
            ),
        ),
        DiscoveredDataset(
            dataset_id="sentiment_eu",
            name="European Media & News Sentiment (EUR)",
            category="Sentiment",
            description="Multilingual financial news and wire sentiment scores across Europe.",
            fields=(
                DiscoveredField("news_sentiment", "MATRIX", 0.78, "Overall European financial news sentiment score", "sentiment_eu", "Sentiment"),
                DiscoveredField("news_volume", "MATRIX", 0.80, "Daily publication count covering security", "sentiment_eu", "Sentiment"),
                DiscoveredField("media_attention", "MATRIX", 0.77, "Abnormal news attention index", "sentiment_eu", "Sentiment"),
            ),
        ),
    ],
    "ASI": [
        DiscoveredDataset(
            dataset_id="fundamental_asia",
            name="Asian Financial Fundamentals (ASI)",
            category="Fundamental",
            description="Core accounting and valuation metrics across Japan, Hong Kong, Korea, Taiwan, Singapore.",
            fields=(
                DiscoveredField("revenue", "MATRIX", 0.90, "Total operating revenue in reporting currency", "fundamental_asia", "Fundamental"),
                DiscoveredField("net_income", "MATRIX", 0.90, "Net income attributable to parent", "fundamental_asia", "Fundamental"),
                DiscoveredField("operating_cash_flow", "MATRIX", 0.87, "Cash flow from operations", "fundamental_asia", "Fundamental"),
                DiscoveredField("equity", "MATRIX", 0.90, "Total common equity (book value)", "fundamental_asia", "Fundamental"),
                DiscoveredField("assets", "MATRIX", 0.90, "Total balance sheet assets", "fundamental_asia", "Fundamental"),
                DiscoveredField("dividend_yield", "MATRIX", 0.89, "Cash dividend yield percentage", "fundamental_asia", "Fundamental"),
                DiscoveredField("roa", "MATRIX", 0.88, "Return on assets (net_income / assets)", "fundamental_asia", "Fundamental"),
                DiscoveredField("roe", "MATRIX", 0.88, "Return on equity (net_income / equity)", "fundamental_asia", "Fundamental"),
                DiscoveredField("pe", "MATRIX", 0.89, "Trailing Price-to-Earnings ratio", "fundamental_asia", "Fundamental"),
                DiscoveredField("pb", "MATRIX", 0.90, "Price-to-Book ratio", "fundamental_asia", "Fundamental"),
            ),
        ),
        DiscoveredDataset(
            dataset_id="flow_asia",
            name="Asian Retail Flow & Ownership (ASI)",
            category="Flow & Sentiment",
            description="Retail participation, turnover velocity, and foreign institutional holding changes.",
            fields=(
                DiscoveredField("turnover_ratio", "MATRIX", 0.94, "Daily volume divided by floating shares outstanding", "flow_asia", "Flow & Sentiment"),
                DiscoveredField("foreign_holding_pct", "MATRIX", 0.82, "Percentage of shares held by foreign qualified institutions", "flow_asia", "Flow & Sentiment"),
                DiscoveredField("retail_flow", "MATRIX", 0.78, "Estimated retail investor net buy/sell order imbalance", "flow_asia", "Flow & Sentiment"),
                DiscoveredField("news_sentiment", "MATRIX", 0.75, "Asian market local and regional news sentiment", "flow_asia", "Flow & Sentiment"),
            ),
        ),
    ],
    "CHN": [
        DiscoveredDataset(
            dataset_id="fundamental_china",
            name="China A-Share Financial Indicators (CHN)",
            category="Fundamental",
            description="Balance sheet, margin, and earnings metrics for mainland Shanghai & Shenzhen listings.",
            fields=(
                DiscoveredField("revenue", "MATRIX", 0.93, "Quarterly operating revenue in RMB", "fundamental_china", "Fundamental"),
                DiscoveredField("net_income", "MATRIX", 0.93, "Quarterly net profit in RMB", "fundamental_china", "Fundamental"),
                DiscoveredField("operating_cash_flow", "MATRIX", 0.89, "Cash flow from operations in RMB", "fundamental_china", "Fundamental"),
                DiscoveredField("total_debt", "MATRIX", 0.90, "Total interest-bearing liabilities", "fundamental_china", "Fundamental"),
                DiscoveredField("equity", "MATRIX", 0.93, "Net book assets / stockholders equity", "fundamental_china", "Fundamental"),
                DiscoveredField("roe", "MATRIX", 0.91, "Return on equity", "fundamental_china", "Fundamental"),
                DiscoveredField("roa", "MATRIX", 0.91, "Return on assets", "fundamental_china", "Fundamental"),
                DiscoveredField("pb", "MATRIX", 0.92, "Price-to-Book ratio", "fundamental_china", "Fundamental"),
                DiscoveredField("pe", "MATRIX", 0.92, "Price-to-Earnings ratio", "fundamental_china", "Fundamental"),
            ),
        ),
        DiscoveredDataset(
            dataset_id="microstructure_china",
            name="China Order Flow & Limit Bounds (CHN)",
            category="Price Volume",
            description="A-share daily turnover, margin financing, and price limit proximity indicators.",
            fields=(
                DiscoveredField("turnover_ratio", "MATRIX", 0.96, "Daily turnover rate percentage", "microstructure_china", "Price Volume"),
                DiscoveredField("limit_up_down_flag", "MATRIX", 0.94, "Daily limit-up or limit-down indicator flag", "microstructure_china", "Price Volume"),
                DiscoveredField("retail_order_imbalance", "MATRIX", 0.82, "Small-lot order buy/sell imbalance proxy", "microstructure_china", "Price Volume"),
                DiscoveredField("margin_trading_balance", "MATRIX", 0.85, "Total outstanding margin financing balance in RMB", "microstructure_china", "Price Volume"),
            ),
        ),
    ],
    "GLB": [
        DiscoveredDataset(
            dataset_id="fundamental_global",
            name="Global Multi-Country Financial Fundamentals (GLB)",
            category="Fundamental",
            description="Cross-border standardized financials for global large and mid-cap equity universes.",
            fields=(
                DiscoveredField("revenue", "MATRIX", 0.91, "Standardized total revenue in USD", "fundamental_global", "Fundamental"),
                DiscoveredField("net_income", "MATRIX", 0.91, "Standardized net income in USD", "fundamental_global", "Fundamental"),
                DiscoveredField("ebitda", "MATRIX", 0.89, "Standardized EBITDA in USD", "fundamental_global", "Fundamental"),
                DiscoveredField("equity", "MATRIX", 0.91, "Standardized total common equity in USD", "fundamental_global", "Fundamental"),
                DiscoveredField("assets", "MATRIX", 0.91, "Standardized total assets in USD", "fundamental_global", "Fundamental"),
                DiscoveredField("operating_margin", "MATRIX", 0.89, "Operating profit margin", "fundamental_global", "Fundamental"),
                DiscoveredField("pe", "MATRIX", 0.90, "Standardized P/E ratio", "fundamental_global", "Fundamental"),
                DiscoveredField("pb", "MATRIX", 0.90, "Standardized P/B ratio", "fundamental_global", "Fundamental"),
            ),
        ),
        DiscoveredDataset(
            dataset_id="sentiment_global",
            name="Global News & Geopolitical Sentiment (GLB)",
            category="Sentiment",
            description="Global wire news sentiment and cross-market narrative indicators.",
            fields=(
                DiscoveredField("news_sentiment", "MATRIX", 0.80, "Global media sentiment polarity [-1.0, 1.0]", "sentiment_global", "Sentiment"),
                DiscoveredField("sentiment_buzz", "MATRIX", 0.77, "Global media coverage volume buzz", "sentiment_global", "Sentiment"),
            ),
        ),
    ],
}



# =============================================================================
# P1-04: Authoritative Dataset Registry, Versioning, and Allocation Policies
# =============================================================================


class DatasetRegistryError(RuntimeError):
    """Base exception for dataset registry errors."""


class DatasetNotFoundError(DatasetRegistryError):
    """Raised when a requested dataset ID does not exist in the authoritative catalog."""


class DatasetEligibilityError(DatasetRegistryError):
    """Raised when a requested dataset is incompatible with the target experiment configuration."""


class DatasetAllocationPolicy(str, enum.Enum):
    """Allocation policy governing how autonomous researchers select datasets."""

    EXPLICIT = "EXPLICIT"
    CONTROLLED_EXPLORATION = "CONTROLLED_EXPLORATION"
    CATEGORY_ALIGNED = "CATEGORY_ALIGNED"


@dataclass(frozen=True, slots=True)
class DatasetSelectionContext:
    """Auditable context recording deliberate dataset allocation and catalog lineage."""

    selected_dataset_id: str
    catalog_version: str
    allocation_policy: str
    selection_reason: str
    dataset: DiscoveredDataset
    candidate_datafields: tuple[DiscoveredField, ...]
    session_id: int | None = None
    researcher_id: int | None = None
    generation: int | None = None
    datafield_catalog_version: str | None = None
    allowed_field_names: tuple[str, ...] | None = None


class DatafieldCatalog:
    """Authoritative, scoped registry of datafields with deterministic versioning.

    Binds datafields to a specific research scope (region, universe, delay, instrument_type)
    and selected dataset. Guarantees:
    - Deterministic SHA-256 catalog fingerprinting (df_v1_<hash>).
    - Exposes exact legal field names (core market fields, grouping fields, and dataset-specific fields).
    - Rejects hallucinated, out-of-dataset, region-incompatible, or delay-incompatible fields.
    - Provides field family classification (Fundamental, Price Volume, Analyst, etc.).
    """

    def __init__(
        self,
        dataset: DiscoveredDataset | None = None,
        region: str = "USA",
        universe: str = "TOP3000",
        delay: int = 1,
        instrument_type: str = "EQUITY",
        all_datasets: dict[str, DiscoveredDataset] | None = None,
    ) -> None:
        self.dataset = dataset
        self.region = region.upper().strip()
        self.universe = universe.upper().strip()
        self.delay = delay
        self.instrument_type = instrument_type.upper().strip()
        self._all_datasets = all_datasets or {}

        # Resolve eligible dataset fields
        self._fields: dict[str, DiscoveredField] = {}
        if self.dataset is not None:
            # Check dataset eligibility for target scope
            ds_reg = self.dataset.region.upper().strip()
            if ds_reg == self.region or ds_reg == "GLB" or self.region == "GLB":
                if self.dataset.delay is None or self.dataset.delay <= self.delay:
                    for f in self.dataset.fields:
                        self._fields[f.field_id] = f

        self._allowed_names: set[str] = (
            set(CORE_MARKET_FIELD_NAMES) | set(STANDARD_GROUPING_FIELD_NAMES) | set(self._fields.keys())
        )
        self._catalog_version = self._compute_fingerprint()

    def _compute_fingerprint(self) -> str:
        """Computes deterministic SHA-256 fingerprint over active datafield definitions and scope."""
        field_records = []
        for fid in sorted(self._fields.keys()):
            f = self._fields[fid]
            field_records.append({
                "category": f.category,
                "coverage": f.coverage,
                "dataset_id": f.dataset_id,
                "description": f.description,
                "field_id": f.field_id,
                "field_type": f.field_type,
            })
        core_records = [
            {"field_id": name, "field_type": "MATRIX", "coverage": cov, "category": "Price Volume"}
            for name, _, cov in sorted(CORE_MARKET_FIELDS, key=lambda x: x[0])
        ]
        grouping_records = [
            {"field_id": name, "field_type": "GROUP", "category": "Grouping"}
            for name in sorted(STANDARD_GROUPING_FIELD_NAMES)
        ]
        payload = {
            "dataset_id": self.dataset.dataset_id if self.dataset else "NONE",
            "delay": self.delay,
            "fields": field_records,
            "core_fields": core_records,
            "grouping_fields": grouping_records,
            "instrument_type": self.instrument_type,
            "region": self.region,
            "universe": self.universe,
        }
        canonical_str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        h = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()
        return f"df_v1_{h[:16]}"

    @property
    def catalog_version(self) -> str:
        return self._catalog_version

    def get_allowed_field_names(self) -> set[str]:
        return set(self._allowed_names)

    def get_field(self, field_id: str) -> DiscoveredField | None:
        return self._fields.get(field_id)

    def get_field_family(self, field_id: str) -> str | None:
        if field_id in CORE_MARKET_FIELD_NAMES:
            return "Price Volume"
        if field_id in STANDARD_GROUPING_FIELD_NAMES:
            return "Grouping"
        f = self._fields.get(field_id)
        if f:
            return f.category
        # Check if field exists in any registered dataset
        for ds in self._all_datasets.values():
            for df in ds.fields:
                if df.field_id == field_id:
                    return df.category
        return None

    def validate_field(self, field_id: str) -> tuple[bool, str | None]:
        """Validates a single referenced field name against this authoritative catalog scope.

        Returns (is_valid, error_reason).
        """
        if field_id in self._allowed_names:
            return True, None

        # Check why it's not allowed
        for ds_id, ds in self._all_datasets.items():
            for f in ds.fields:
                if f.field_id == field_id:
                    # In another dataset
                    if self.dataset and ds.dataset_id != self.dataset.dataset_id:
                        return False, f"Field '{field_id}' belongs to dataset '{ds.dataset_id}', not selected dataset '{self.dataset.dataset_id}'"
                    ds_reg = ds.region.upper().strip()
                    if ds_reg != self.region and ds_reg != "GLB" and self.region != "GLB":
                        return False, f"Field '{field_id}' belongs to region '{ds_reg}', which is incompatible with target region '{self.region}'"
                    if ds.delay is not None and ds.delay > self.delay:
                        return False, f"Field '{field_id}' requires delay {ds.delay}, exceeding target delay {self.delay}"
        return False, f"Unknown or ineligible data field: {field_id}"

    def format_prompt_context(self) -> str:
        """Formats the datafield catalog context for prompt injection."""
        blocks: list[str] = [
            f"DATAFIELD CATALOG (Version: {self.catalog_version}):",
            "\n1. CORE MARKET FIELDS (Universal for all equities):",
        ]
        for f_name, f_desc, cov in CORE_MARKET_FIELDS:
            blocks.append(f"   - `{f_name}`: {f_desc} (Coverage: {int(cov*100)}%)")

        blocks.append("\n2. GROUPING FIELDS (Use in group_rank, group_neutralize; DO NOT count as data fields):")
        for g_name, g_desc in GROUPING_FIELDS:
            blocks.append(f"   - `{g_name}`: {g_desc}")

        if self.dataset:
            blocks.append(f"\n3. SELECTED DATASET FIELDS:")
            blocks.append(f"   DATASET: `{self.dataset.dataset_id}` ({self.dataset.name})")
            if self.dataset.category:
                blocks.append(f"   Category: {self.dataset.category}")
            for f in self.dataset.fields:
                cov_pct = f"{int(f.coverage*100)}%" if f.coverage is not None else "N/A"
                blocks.append(f"   - `{f.field_id}` [{f.field_type}]: {f.description} (Coverage: {cov_pct})")
        else:
            blocks.append("\n3. SPECIALIZED DATASETS: None selected. Use CORE MARKET fields above.")

        blocks.append(
            "\nSTRICT DATAFIELD CONSTRAINTS: You may ONLY reference the CORE MARKET FIELDS, "
            "GROUPING FIELDS, and SELECTED DATASET FIELDS listed above. Any expression using "
            "unknown, unlisted, or out-of-dataset field names will fail pre-simulation validation."
        )
        return "\n".join(blocks)


#: Documented mappings from quantitative strategy categories to eligible dataset categories/keywords
CATEGORY_TO_DATASET_PATTERNS: dict[str, tuple[str, ...]] = {
    "value": ("Fundamental",),
    "quality": ("Fundamental",),
    "momentum": ("Model Data", "Price Volume"),
    "volatility": ("Model Data", "Price Volume"),
    "sentiment": ("Sentiment", "Flow & Sentiment"),
    "analyst": ("Analyst",),
    "short_interest": ("Short Interest",),
}


class DatasetRegistry:
    """Authoritative registry of datasets with deterministic versioning and eligibility verification.

    Provides:
    - Deterministic catalog versioning computed from registered metadata and fields.
    - Explicit target validation failing closed on unknown or incompatible datasets.
    - Controlled exploration allocation across eligible datasets without starvation.
    - Category-aligned allocation with documented fallback logic.
    """

    def __init__(
        self,
        datasets_by_region: dict[str, list[DiscoveredDataset]] | None = None,
        catalog: Catalog | None = None,
    ) -> None:
        self._catalog = catalog
        self._datasets_by_region: dict[str, list[DiscoveredDataset]] = (
            datasets_by_region if datasets_by_region is not None else BUILTIN_DATASETS_BY_REGION
        )
        self._all_datasets: dict[str, DiscoveredDataset] = {}
        for reg, ds_list in self._datasets_by_region.items():
            for ds in ds_list:
                # Ensure region attribute matches the dictionary key if defaulted
                if ds.region == "USA" and reg != "USA":
                    ds = DiscoveredDataset(
                        dataset_id=ds.dataset_id,
                        name=ds.name,
                        category=ds.category,
                        description=ds.description,
                        fields=ds.fields,
                        region=reg,
                        delay=ds.delay,
                        universe=ds.universe,
                        instrument_type=ds.instrument_type,
                    )
                self._all_datasets[ds.dataset_id] = ds

        self._catalog_version = self._compute_catalog_version()

    def _compute_catalog_version(self) -> str:
        """Computes deterministic SHA-256 fingerprint over registered datasets and fields."""
        items = []
        for ds_id in sorted(self._all_datasets.keys()):
            ds = self._all_datasets[ds_id]
            field_items = [
                {
                    "coverage": f.coverage,
                    "description": f.description,
                    "field_id": f.field_id,
                    "field_type": f.field_type,
                }
                for f in sorted(ds.fields, key=lambda x: x.field_id)
            ]
            items.append({
                "category": ds.category,
                "dataset_id": ds.dataset_id,
                "delay": ds.delay,
                "description": ds.description,
                "fields": field_items,
                "instrument_type": ds.instrument_type,
                "name": ds.name,
                "region": ds.region,
            })
        canonical_str = json.dumps(items, sort_keys=True, separators=(",", ":"))
        h = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()
        return f"cat_v1_{h[:16]}"

    @property
    def catalog_version(self) -> str:
        """Deterministic version identifier representing the active catalog contents."""
        return self._catalog_version

    def get_dataset(self, dataset_id: str) -> DiscoveredDataset | None:
        return self._all_datasets.get(dataset_id)

    def validate_dataset_eligibility(
        self,
        dataset_id: str,
        region: str = "USA",
        universe: str = "TOP3000",
        delay: int = 1,
        instrument_type: str = "EQUITY",
    ) -> DiscoveredDataset:
        """Validates that a dataset exists and is eligible for target region/settings.

        Fails closed with DatasetNotFoundError or DatasetEligibilityError.
        """
        if not dataset_id:
            raise DatasetNotFoundError("Dataset ID cannot be empty")

        ds = self._all_datasets.get(dataset_id)
        if not ds:
            raise DatasetNotFoundError(
                f"Dataset '{dataset_id}' not found in authoritative catalog (catalog_version: {self.catalog_version})"
            )

        reg_norm = region.upper().strip()
        ds_reg = ds.region.upper().strip()

        # Regional compatibility check
        if ds_reg != reg_norm and ds_reg != "GLB" and reg_norm != "GLB":
            raise DatasetEligibilityError(
                f"Dataset '{dataset_id}' belongs to region '{ds_reg}', which is incompatible with target region '{reg_norm}'"
            )

        # Delay compatibility check
        if ds.delay is not None and delay is not None and ds.delay > delay:
            raise DatasetEligibilityError(
                f"Dataset '{dataset_id}' requires minimum delay {ds.delay}, but requested delay is {delay}"
            )

        return ds

    def get_eligible_datasets(
        self,
        region: str = "USA",
        universe: str = "TOP3000",
        delay: int = 1,
        instrument_type: str = "EQUITY",
        category: str | None = None,
    ) -> list[DiscoveredDataset]:
        """Returns all registered datasets eligible for the target research scope."""
        reg_norm = region.upper().strip()
        eligible: list[DiscoveredDataset] = []

        for ds_id, ds in self._all_datasets.items():
            ds_reg = ds.region.upper().strip()
            if ds_reg == reg_norm or ds_reg == "GLB" or reg_norm == "GLB":
                if ds.delay is not None and delay is not None and ds.delay > delay:
                    continue
                eligible.append(ds)

        if category and category.lower() != "general":
            cat_lower = category.lower().strip()
            filtered = [
                d
                for d in eligible
                if cat_lower in d.category.lower()
                or cat_lower in d.name.lower()
                or cat_lower in d.dataset_id.lower()
            ]
            if filtered:
                return filtered

        return sorted(eligible, key=lambda d: d.dataset_id)

    def allocate_dataset(
        self,
        policy: str | DatasetAllocationPolicy = DatasetAllocationPolicy.CATEGORY_ALIGNED,
        target_dataset_id: str | None = None,
        strategy_category: str | None = None,
        region: str = "USA",
        universe: str = "TOP3000",
        delay: int = 1,
        instrument_type: str = "EQUITY",
        session_id: int | None = None,
        researcher_id: int | None = None,
        generation: int = 0,
        researcher_index: int | None = None,
        num_researchers: int | None = None,
    ) -> DatasetSelectionContext:
        """Allocates a dataset according to explicit target, category alignment, or controlled exploration."""
        pol_str = policy.value if isinstance(policy, DatasetAllocationPolicy) else str(policy).upper()

    def get_datafield_catalog(
        self,
        dataset: DiscoveredDataset | str | None = None,
        region: str = "USA",
        universe: str = "TOP3000",
        delay: int = 1,
        instrument_type: str = "EQUITY",
    ) -> DatafieldCatalog:
        """Returns the authoritative DatafieldCatalog for a dataset and scope."""
        ds: DiscoveredDataset | None = None
        if isinstance(dataset, str):
            ds = self.get_dataset(dataset)
        elif isinstance(dataset, DiscoveredDataset):
            ds = dataset
        return DatafieldCatalog(
            dataset=ds,
            region=region,
            universe=universe,
            delay=delay,
            instrument_type=instrument_type,
            all_datasets=self._all_datasets,
        )

    def allocate_dataset(
        self,
        policy: str | DatasetAllocationPolicy = DatasetAllocationPolicy.CATEGORY_ALIGNED,
        target_dataset_id: str | None = None,
        strategy_category: str | None = None,
        region: str = "USA",
        universe: str = "TOP3000",
        delay: int = 1,
        instrument_type: str = "EQUITY",
        session_id: int | None = None,
        researcher_id: int | None = None,
        generation: int = 0,
        researcher_index: int | None = None,
        num_researchers: int | None = None,
    ) -> DatasetSelectionContext:
        """Allocates a dataset according to explicit target, category alignment, or controlled exploration."""
        pol_str = policy.value if isinstance(policy, DatasetAllocationPolicy) else str(policy).upper()

        # 1. EXPLICIT Policy
        if pol_str == DatasetAllocationPolicy.EXPLICIT.value or (
            target_dataset_id
            and pol_str
            not in (
                DatasetAllocationPolicy.CONTROLLED_EXPLORATION.value,
                DatasetAllocationPolicy.CATEGORY_ALIGNED.value,
            )
        ):
            if not target_dataset_id:
                raise DatasetEligibilityError(
                    "Explicit allocation policy requires a non-empty target_dataset_id"
                )
            ds = self.validate_dataset_eligibility(
                dataset_id=target_dataset_id,
                region=region,
                universe=universe,
                delay=delay,
                instrument_type=instrument_type,
            )
            df_cat = self.get_datafield_catalog(
                dataset=ds, region=region, universe=universe, delay=delay, instrument_type=instrument_type
            )
            return DatasetSelectionContext(
                selected_dataset_id=ds.dataset_id,
                catalog_version=self.catalog_version,
                allocation_policy=DatasetAllocationPolicy.EXPLICIT.value,
                selection_reason=f"Explicitly requested target dataset '{ds.dataset_id}'",
                dataset=ds,
                candidate_datafields=ds.fields,
                session_id=session_id,
                researcher_id=researcher_id,
                generation=generation,
                datafield_catalog_version=df_cat.catalog_version,
                allowed_field_names=tuple(sorted(df_cat.get_allowed_field_names())),
            )

        eligible = self.get_eligible_datasets(
            region=region,
            universe=universe,
            delay=delay,
            instrument_type=instrument_type,
        )
        if not eligible:
            raise DatasetEligibilityError(
                f"No eligible datasets found for region '{region}', delay {delay}, universe '{universe}'"
            )

        # 2. CATEGORY_ALIGNED Policy
        if pol_str == DatasetAllocationPolicy.CATEGORY_ALIGNED.value:
            cat_norm = (strategy_category or "").lower().strip()
            patterns = CATEGORY_TO_DATASET_PATTERNS.get(cat_norm)
            if patterns:
                matched = [
                    d
                    for d in eligible
                    if any(
                        pat.lower() in d.category.lower() or pat.lower() in d.name.lower()
                        for pat in patterns
                    )
                ]
                if matched:
                    # Pick highest field count, stable sort
                    selected = sorted(matched, key=lambda d: len(d.fields), reverse=True)[0]
                    df_cat = self.get_datafield_catalog(
                        dataset=selected, region=region, universe=universe, delay=delay, instrument_type=instrument_type
                    )
                    return DatasetSelectionContext(
                        selected_dataset_id=selected.dataset_id,
                        catalog_version=self.catalog_version,
                        allocation_policy=DatasetAllocationPolicy.CATEGORY_ALIGNED.value,
                        selection_reason=(
                            f"Category-aligned allocation for '{strategy_category}' "
                            f"matching '{selected.category}' dataset '{selected.dataset_id}'"
                        ),
                        dataset=selected,
                        candidate_datafields=selected.fields,
                        session_id=session_id,
                        researcher_id=researcher_id,
                        generation=generation,
                        datafield_catalog_version=df_cat.catalog_version,
                        allowed_field_names=tuple(sorted(df_cat.get_allowed_field_names())),
                    )
            # Documented fallback when category metadata is unmapped
            fallback_prefix = (
                f"Fallback from unmapped category '{strategy_category}' to controlled exploration: "
                if cat_norm and cat_norm != "general"
                else "Controlled exploration (general category baseline): "
            )
            selected, idx = self._balanced_rotation(
                eligible, session_id, generation, researcher_id, researcher_index, num_researchers
            )
            df_cat = self.get_datafield_catalog(
                dataset=selected, region=region, universe=universe, delay=delay, instrument_type=instrument_type
            )
            return DatasetSelectionContext(
                selected_dataset_id=selected.dataset_id,
                catalog_version=self.catalog_version,
                allocation_policy=DatasetAllocationPolicy.CONTROLLED_EXPLORATION.value,
                selection_reason=f"{fallback_prefix}allocated slot {idx} of {len(eligible)} ({selected.dataset_id})",
                dataset=selected,
                candidate_datafields=selected.fields,
                session_id=session_id,
                researcher_id=researcher_id,
                generation=generation,
                datafield_catalog_version=df_cat.catalog_version,
                allowed_field_names=tuple(sorted(df_cat.get_allowed_field_names())),
            )

        # 3. CONTROLLED_EXPLORATION Policy
        selected, idx = self._balanced_rotation(
            eligible, session_id, generation, researcher_id, researcher_index, num_researchers
        )
        df_cat = self.get_datafield_catalog(
            dataset=selected, region=region, universe=universe, delay=delay, instrument_type=instrument_type
        )
        return DatasetSelectionContext(
            selected_dataset_id=selected.dataset_id,
            catalog_version=self.catalog_version,
            allocation_policy=DatasetAllocationPolicy.CONTROLLED_EXPLORATION.value,
            selection_reason=(
                f"Controlled exploration allocation (balanced slot {idx} of {len(eligible)}): "
                f"{selected.dataset_id}"
            ),
            dataset=selected,
            candidate_datafields=selected.fields,
            session_id=session_id,
            researcher_id=researcher_id,
            generation=generation,
            datafield_catalog_version=df_cat.catalog_version,
            allowed_field_names=tuple(sorted(df_cat.get_allowed_field_names())),
        )

    def _balanced_rotation(
        self,
        eligible: list[DiscoveredDataset],
        session_id: int | None,
        generation: int | None,
        researcher_id: int | None,
        researcher_index: int | None,
        num_researchers: int | None = None,
    ) -> tuple[DiscoveredDataset, int]:
        """Calculates stable, balanced allocation ensuring distinct coverage within a round
        and balanced exploration across generations without dataset starvation.

        Guarantees:
        1. Distinct allocations: When researcher_index < len(eligible) within a generation,
           distinct researchers receive strictly distinct datasets.
        2. Balanced reuse: When num_researchers > len(eligible), allocations wrap around evenly,
           guaranteeing each dataset is allocated either floor(M/N) or ceil(M/N) times.
        3. Scheduling cycle coverage: Over consecutive allocations across generations, each continuously
           eligible dataset receives allocations over a complete scheduling cycle.
           Note: Absolute starvation-free guarantees hold strictly under continuous eligibility;
           if a dataset becomes dynamically ineligible, it is immediately excluded from allocation.
        4. Reproducibility across restarts: The schedule is a deterministic closed-form function of
           (session_id, generation, index, eligible), requiring no volatile in-memory state.
        5. Concurrency safety: Pure functional evaluation produces identical, consistent selection records
           across concurrent tasks.
        """
        num_eligible = len(eligible)
        if num_eligible == 0:
            raise DatasetEligibilityError("Cannot rotate among 0 eligible datasets")

        s_id = session_id if session_id is not None else 0
        gen = generation if generation is not None else 0

        # Determine researcher's intra-round index
        if researcher_index is not None:
            r_idx = researcher_index
        elif researcher_id is not None:
            r_idx = researcher_id
        else:
            r_idx = 0

        # Stride between successive generations
        stride = num_researchers if (num_researchers is not None and num_researchers > 0) else 1

        session_seed = (
            int(hashlib.sha256(f"session:{s_id}".encode("utf-8")).hexdigest(), 16)
            if s_id != 0
            else 0
        )

        sequence_index = gen * stride + r_idx
        slot = (session_seed + sequence_index) % num_eligible
        return eligible[slot], slot

    def _deterministic_rotation(
        self,
        eligible: list[DiscoveredDataset],
        session_id: int | None,
        generation: int | None,
        researcher_id: int | None,
        researcher_index: int | None,
        num_researchers: int | None = None,
    ) -> tuple[DiscoveredDataset, int]:
        """Backward-compatible alias pointing to balanced rotation."""
        return self._balanced_rotation(
            eligible=eligible,
            session_id=session_id,
            generation=generation,
            researcher_id=researcher_id,
            researcher_index=researcher_index,
            num_researchers=num_researchers,
        )

    def get_datafield_catalog(
        self,
        dataset: DiscoveredDataset | None = None,
        region: str = "USA",
        universe: str = "TOP3000",
        delay: int = 1,
        instrument_type: str = "EQUITY",
    ) -> DatafieldCatalog:
        """Constructs an authoritative DatafieldCatalog for the target scope."""
        return DatafieldCatalog(
            dataset=dataset,
            region=region,
            universe=universe,
            delay=delay,
            instrument_type=instrument_type,
            all_datasets=self._all_datasets,
        )



class DatasetAllocationTracker:
    """Persisted, transactional allocation tracker coordinating balanced dataset selection.

    Guarantees:
    - Atomically advances allocation counter in SQLite to eliminate race conditions among concurrent researchers.
    - Persists allocation state across process restarts.
    - Selects exclusively from currently eligible datasets without selecting ineligible entries.
    """

    def __init__(self, db: Any | None = None) -> None:
        self.db = db
        self._lock = asyncio.Lock()

    async def get_next_allocation(
        self,
        session_id: int | None,
        scope_key: str,
        eligible_datasets: list[DiscoveredDataset],
        session: Any | None = None,
    ) -> tuple[DiscoveredDataset, int]:
        """Atomically allocates the next dataset in sequence using persisted SQLite state."""
        if not eligible_datasets:
            raise DatasetEligibilityError("No eligible datasets available for allocation")

        s_id = session_id if session_id is not None else 0

        async def _do_allocate(s: Any) -> tuple[DiscoveredDataset, int]:
            from ..db.models import DatasetAllocationState
            from sqlalchemy import select

            stmt = (
                select(DatasetAllocationState)
                .where(
                    DatasetAllocationState.session_id == s_id,
                    DatasetAllocationState.scope_key == scope_key,
                )
                .with_for_update()
            )
            res = await s.execute(stmt)
            record = res.scalar_one_or_none()

            if record is None:
                record = DatasetAllocationState(
                    session_id=s_id,
                    scope_key=scope_key,
                    allocation_count=0,
                )
                s.add(record)
                await s.flush()

            count = record.allocation_count
            record.allocation_count = count + 1

            # Map sequence strictly to currently eligible datasets
            slot = count % len(eligible_datasets)
            selected = eligible_datasets[slot]
            record.last_dataset_id = selected.dataset_id
            await s.flush()
            return selected, slot

        async with self._lock:
            if session is not None:
                return await _do_allocate(session)
            elif self.db is not None:
                async with self.db.session() as s:
                    result = await _do_allocate(s)
                    await s.commit()
                    return result
            else:
                raise RuntimeError("DatasetAllocationTracker requires a Database or AsyncSession")




class CatalogDiscovery:
    """Discovers and formats datasets and fields for LLM research prompts."""

    def __init__(
        self,
        catalog: Catalog | None = None,
        metadata: Any | None = None,
    ) -> None:
        self._catalog = catalog
        self._metadata = metadata
        self.registry = DatasetRegistry(catalog=catalog)

    async def get_datasets_for_scope(
        self,
        region: str = "USA",
        delay: int = 1,
        universe: str = "TOP3000",
        instrument_type: str = "EQUITY",
        limit: int = 5,
        min_coverage: float = 0.5,
        category: str | None = None,
    ) -> list[DiscoveredDataset]:
        """Fetches top datasets with their best fields for a target market scope."""
        # 1. First, check if DuckDB has synced datasets
        if self._catalog is not None:
            try:
                cat_filter = ""
                cat_params: list[Any] = []
                if category and category.lower() != "general":
                    cat_filter = """
                        AND (
                            LOWER(category_name) LIKE ?
                            OR LOWER(subcategory_name) LIKE ?
                            OR LOWER(name) LIKE ?
                            OR LOWER(description) LIKE ?
                        )
                    """
                    kw = f"%{category.lower()}%"
                    cat_params = [kw, kw, kw, kw]

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

                if dataset_rows:
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

                    if results:
                        return results
            except Exception as exc:
                log.warning("discovery.catalog_query_failed", error=str(exc))

        # 2. Rich fallback: Use authoritative DatasetRegistry for this scope
        eligible = self.registry.get_eligible_datasets(
            region=region,
            universe=universe,
            delay=delay,
            category=category,
        )
        return eligible[:limit]

    def get_datafield_catalog(
        self,
        dataset: DiscoveredDataset | str | None = None,
        region: str = "USA",
        universe: str = "TOP3000",
        delay: int = 1,
        instrument_type: str = "EQUITY",
    ) -> DatafieldCatalog:
        """Returns the authoritative DatafieldCatalog for a dataset and research scope."""
        return self.registry.get_datafield_catalog(
            dataset=dataset,
            region=region,
            universe=universe,
            delay=delay,
            instrument_type=instrument_type,
        )

    def format_catalog_context(
        self,
        datasets: list[DiscoveredDataset],
        datafield_catalog: DatafieldCatalog | None = None,
    ) -> str:
        """Formats discovered datasets and fields into a token-efficient prompt string."""
        if datafield_catalog is not None:
            return datafield_catalog.format_prompt_context()

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
            blocks.append("\n3. SPECIALIZED DATASETS & FIELDS (Freely combine with CORE MARKET fields):")
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
            blocks.append("\n3. SPECIALIZED DATASETS: Use CORE MARKET fields above.")

        blocks.append(
            "\nSTRICT DATAFIELD CONSTRAINTS: You may ONLY reference the CORE MARKET FIELDS, "
            "GROUPING FIELDS, and SPECIALIZED DATASETS fields listed above. Any expression using "
            "unknown, unlisted, or out-of-dataset field names will fail pre-simulation validation."
        )
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

    async def resolve_operator_catalog(
        self,
        scope: str = "REGULAR",
        *,
        force_refresh: bool = False,
    ) -> OperatorCatalog:
        """Resolves authoritative OperatorCatalog from PlatformMetadata if available, else built-in fallback."""
        if self._metadata is not None and hasattr(self._metadata, "get_operator_catalog"):
            return await self._metadata.get_operator_catalog(scope=scope, force_refresh=force_refresh)
        return OperatorCatalog(scope=scope, source="BUILTIN_FALLBACK")

    def get_operator_catalog(self, scope: str = "REGULAR") -> OperatorCatalog:
        """Returns the authoritative OperatorCatalog (synchronous, built-in fallback)."""
        return OperatorCatalog(scope=scope, source="BUILTIN_FALLBACK")

    def get_datafield_catalog(
        self,
        dataset: DiscoveredDataset | None = None,
        region: str = "USA",
        universe: str = "TOP3000",
        delay: int = 1,
        instrument_type: str = "EQUITY",
    ) -> DatafieldCatalog:
        """Constructs an authoritative DatafieldCatalog for the target scope."""
        return self.registry.get_datafield_catalog(
            dataset=dataset,
            region=region,
            universe=universe,
            delay=delay,
            instrument_type=instrument_type,
        )


