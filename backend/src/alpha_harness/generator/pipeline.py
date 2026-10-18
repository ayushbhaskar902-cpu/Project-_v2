"""Fast Expression generation and AST validation pipeline with automated syntax retry.

Orchestrates LLMs to generate quantitative alpha candidates, validates each candidate
locally using the Fast Expression AST parser and operator checker, and executes automated
repair/retry loops for expressions violating BRAIN's 8-operator or 3-field constraints.
"""

from __future__ import annotations

import enum
import json
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from ..labs.fastexpr import (
    GROUPING,
    MAX_FIELDS,
    MAX_OPERATORS,
    Node,
    OperatorInfo,
    ParseError,
    data_fields,
    node_at,
    operator_count,
    operator_names,
    operator_table,
    parse,
    render,
    validate,
    walk,
)
from ..discovery.catalog import (
    CORE_MARKET_FIELD_NAMES,
    CatalogDiscovery,
    DatafieldCatalog,
    DatasetAllocationPolicy,
    DatasetEligibilityError,
    DatasetNotFoundError,
    DatasetRegistry,
    DatasetSelectionContext,
    DiscoveredDataset,
    OperatorCatalog,
)
from ..prompts.library import PromptLibrary, StrategyCategory

if TYPE_CHECKING:
    from ..llm.service import LLMService

log = structlog.get_logger(__name__)

# P1-07: Authoritative Alpha Language Policy
SUPPORTED_ALPHA_LANGUAGES: frozenset[str] = frozenset({"FASTEXPR"})
DEFAULT_ALPHA_LANGUAGE: str = "FASTEXPR"


# P1-08: Validation Stage & Machine-Readable Error Codes
class ValidationStage(str, enum.Enum):
    """Explicit, sequential stages of the candidate validation pipeline."""

    LANGUAGE_SCOPE = "LANGUAGE_SCOPE"
    SYNTAX_PARSE = "SYNTAX_PARSE"
    BRAIN_CONSTRAINTS = "BRAIN_CONSTRAINTS"
    OPERATOR_SIGNATURE = "OPERATOR_SIGNATURE"
    FIELD_MEMBERSHIP = "FIELD_MEMBERSHIP"
    SEMANTIC_TYPE_UNIT = "SEMANTIC_TYPE_UNIT"
    PREFLIGHT_SIM = "PREFLIGHT_SIM"


class ValidationErrorCode(str, enum.Enum):
    """Machine-readable error codes for candidate validation failures."""

    ERR_UNSUPPORTED_LANGUAGE = "ERR_UNSUPPORTED_LANGUAGE"
    ERR_EMPTY_EXPRESSION = "ERR_EMPTY_EXPRESSION"
    ERR_SYNTAX_PARSE = "ERR_SYNTAX_PARSE"
    ERR_MAX_OPERATORS_EXCEEDED = "ERR_MAX_OPERATORS_EXCEEDED"
    ERR_MAX_FIELDS_EXCEEDED = "ERR_MAX_FIELDS_EXCEEDED"
    ERR_UNKNOWN_OPERATOR = "ERR_UNKNOWN_OPERATOR"
    ERR_OPERATOR_ARITY = "ERR_OPERATOR_ARITY"
    ERR_OPERATOR_SCOPE = "ERR_OPERATOR_SCOPE"
    ERR_OPERATOR_ARG_CONTRACT = "ERR_OPERATOR_ARG_CONTRACT"
    ERR_UNKNOWN_FIELD = "ERR_UNKNOWN_FIELD"
    ERR_FIELD_INELIGIBLE = "ERR_FIELD_INELIGIBLE"
    ERR_TYPE_INCOMPATIBLE = "ERR_TYPE_INCOMPATIBLE"
    ERR_UNIT_INCOMPATIBLE = "ERR_UNIT_INCOMPATIBLE"
    ERR_INVALID_LOOKBACK = "ERR_INVALID_LOOKBACK"
    ERR_PREFLIGHT_DATASET_MISMATCH = "ERR_PREFLIGHT_DATASET_MISMATCH"
    ERR_PREFLIGHT_CONFIGURATION_INCOMPLETE = "ERR_PREFLIGHT_CONFIGURATION_INCOMPLETE"


class UnsupportedAlphaLanguageError(ValueError):
    """Raised when an alpha generation or validation request specifies an unsupported language."""

    error_code: str = ValidationErrorCode.ERR_UNSUPPORTED_LANGUAGE.value
    failed_stage: ValidationStage = ValidationStage.LANGUAGE_SCOPE


def validate_alpha_language(language: str | None) -> str:
    """Validates that the requested alpha language is supported.

    WorldQuant BRAIN Fast Expression ('FASTEXPR') is the sole supported alpha generation
    and validation language. Python alpha generation is explicitly out of scope.

    Returns the normalized uppercase language ('FASTEXPR') if valid.
    Raises UnsupportedAlphaLanguageError if an unsupported or unknown language is requested.
    """
    if language is None:
        return DEFAULT_ALPHA_LANGUAGE
    normalized = str(language).strip().upper()
    if normalized in SUPPORTED_ALPHA_LANGUAGES:
        return normalized
    raise UnsupportedAlphaLanguageError(
        f"Unsupported alpha language: '{language}'. "
        f"WorldQuant BRAIN Fast Expression ('{DEFAULT_ALPHA_LANGUAGE}') is the sole supported "
        "alpha generation target. Python alpha generation is explicitly out of scope."
    )


# JSON schema enforced for Google Gemini models and requested from all providers
CANDIDATE_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "batch_summary": {"type": "string"},
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "hypothesis": {"type": "string"},
                    "expression": {"type": "string"},
                    "category": {"type": "string"},
                    "target_dataset": {"type": "string"},
                    "primary_fields": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["hypothesis", "expression"],
            },
        },
    },
    "required": ["candidates"],
}


CORE_BRAIN_OPERATOR_DEFS = [
    "ts_rank(x, d)",
    "ts_zscore(x, d)",
    "ts_delta(x, d)",
    "ts_delay(x, d)",
    "ts_mean(x, d)",
    "ts_std_dev(x, d)",
    "ts_corr(x, y, d)",
    "ts_covariance(x, y, d)",
    "ts_sum(x, d)",
    "ts_product(x, d)",
    "ts_decay_linear(x, d)",
    "ts_argmax(x, d)",
    "ts_argmin(x, d)",
    "ts_backfill(x, d)",
    "rank(x)",
    "group_rank(x, group)",
    "group_neutralize(x, group)",
    "group_mean(x, group)",
    "group_zscore(x, group)",
    "abs(x)",
    "log(x)",
    "sign(x)",
    "sqrt(x)",
    "fraction(x)",
    "max(x, y)",
    "min(x, y)",
    "power(x, y)",
    "scale(x, target=1)",
    "trade_when(cond, alpha, exit)",
    "vec_avg(x)",
    "vec_sum(x)",
]

BUILTIN_OPERATOR_TABLE: dict[str, OperatorInfo] = operator_table(
    [
        {"name": d.split("(")[0], "definition": d, "category": "CORE"}
        for d in CORE_BRAIN_OPERATOR_DEFS
    ]
)


def sanitize_expression(raw_expr: str) -> str:
    """Pre-cleans and normalizes LLM-generated Fast Expressions.

    Strips markdown code fences, wrapping quotes, common variable assignment prefixes
    (e.g. 'alpha =', 'return '), Python logical keywords ('and', 'or', 'not'),
    and trailing semicolons.
    """
    if not raw_expr:
        return ""

    expr = raw_expr.strip()

    # 1. Strip markdown fences if present
    fence_match = re.search(r"```(?:\w+)?\s*([\s\S]*?)\s*```", expr)
    if fence_match:
        expr = fence_match.group(1).strip()

    # 2. Strip surrounding quotes (double, single, backticks)
    if (expr.startswith(('"', "'", "`")) and expr.endswith(('"', "'", "`"))) and len(expr) >= 2:
        expr = expr[1:-1].strip()

    # 3. Strip common assignment prefixes like "alpha =", "signal =", "result =", "return "
    expr = re.sub(
        r"^(?:alpha|signal|expression|result|factor|ret)\s*=\s*", "", expr, flags=re.IGNORECASE
    )
    expr = re.sub(r"^return\s+", "", expr, flags=re.IGNORECASE)

    # 4. Strip trailing semicolons
    expr = expr.rstrip(";").strip()

    # 5. Replace Python logical keywords with Fast Expression symbols
    expr = re.sub(r"\bnot\s+", "!", expr)
    expr = re.sub(r"\band\b", "&&", expr)
    expr = re.sub(r"\bor\b", "||", expr)

    # 6. Normalize multiline expressions into a single line
    expr = re.sub(r"\s+", " ", expr).strip()

    return expr


@dataclass(slots=True)
class CandidateValidationResult:
    """Outcome of offline multi-stage validation against WorldQuant BRAIN grammar and simulation context."""

    is_valid: bool
    expression: str
    error: str | None = None
    is_syntactically_valid: bool = False
    failed_stage: ValidationStage | None = None
    error_code: str | None = None
    stage_results: dict[str, str] = field(default_factory=dict)
    stage_passed: dict[str, bool] = field(default_factory=dict)
    operator_count: int = 0
    field_count: int = 0
    fields_used: list[str] = field(default_factory=list)
    operators_used: list[str] = field(default_factory=list)
    retry_count: int = 0
    repair_history: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GeneratedCandidate:
    """One candidate alpha factor generated by an LLM researcher."""

    hypothesis: str
    expression: str
    category: str
    target_dataset: str | None
    primary_fields: list[str]
    validation: CandidateValidationResult
    raw_response: dict[str, Any] = field(default_factory=dict)
    simulation_settings_overrides: dict[str, Any] | None = None


@dataclass(slots=True)
class GenerationBatch:
    """A complete round of candidate alphas from one researcher."""

    summary: str
    candidates: list[GeneratedCandidate]
    model_ref: str
    total_tokens: int
    duration_seconds: float
    system_prompt: str = ""
    user_prompt: str = ""
    response_schema: dict[str, Any] = field(default_factory=dict)
    target_dataset_id: str | None = None
    catalog_version: str | None = None
    dataset_selection_policy: str | None = None
    dataset_selection_reason: str | None = None
    datafield_catalog_version: str | None = None
    allowed_fields: list[str] | None = None
    #: P1-06: Operator catalog versioning and lineage
    operator_catalog_version: str | None = None
    #: P1-07: Alpha generation language (FASTEXPR only)
    language: str = DEFAULT_ALPHA_LANGUAGE

    @property
    def valid_count(self) -> int:
        return sum(1 for c in self.candidates if c.validation.is_valid)

    @property
    def valid_rate(self) -> float:
        return (self.valid_count / len(self.candidates)) if self.candidates else 0.0


def _extract_num_value(node: Node) -> float | None:
    """Extracts a numeric literal value from an AST node, handling unary negation."""
    if node.kind == "num":
        try:
            return float(node.value)
        except (ValueError, TypeError):
            return None
    elif node.kind == "unary" and node.value == "-" and node.args and node.args[0].kind == "num":
        try:
            return -float(node.args[0].value)
        except (ValueError, TypeError):
            return None
    return None


def _check_lookback_contracts(
    tree: Node,
    operator_catalog: OperatorCatalog | None,
) -> tuple[bool, str | None]:
    """Validates that time-series lookback windows are positive integers."""
    for _, node in walk(tree):
        if node.kind == "call" and node.value.startswith("ts_"):
            d_idx: int | None = None
            if operator_catalog is not None:
                op_entry = operator_catalog.get_operator(node.value) or operator_catalog.get_all_operator(node.value)
                if op_entry and "d" in op_entry.params:
                    d_idx = op_entry.params.index("d")
            if d_idx is None:
                d_idx = len(node.args) - 1

            if 0 <= d_idx < len(node.args):
                val = _extract_num_value(node.args[d_idx])
                if val is not None:
                    if val <= 0:
                        return (
                            False,
                            f"Time-series operator '{node.value}' lookback window 'd' must be greater than 0 (got {val:g})",
                        )
                    if not val.is_integer():
                        return (
                            False,
                            f"Time-series operator '{node.value}' lookback window 'd' must be an integer number of days (got {val:g})",
                        )
    return True, None


def _check_grouping_contracts(tree: Node) -> tuple[bool, str | None, str | None]:
    """Validates grouping argument slots and prevents grouping fields in numeric operations."""
    for _, node in walk(tree):
        if node.kind == "call" and node.value.startswith("group_"):
            if len(node.args) >= 2:
                g_arg = node.args[1]
                val = _extract_num_value(g_arg)
                if val is not None:
                    return (
                        False,
                        f"Group operator '{node.value}' requires a grouping field (e.g. sector, industry) as argument 2, not a number ({val:g})",
                        ValidationErrorCode.ERR_OPERATOR_ARG_CONTRACT.value,
                    )

    NUMERIC_FUNCS = frozenset({
        "rank", "log", "abs", "sign", "sqrt", "scale", "fraction",
        "ts_rank", "ts_mean", "ts_delta", "ts_delay", "ts_zscore", "ts_std_dev",
        "ts_decay_linear", "ts_sum", "ts_product", "ts_max", "ts_min",
    })
    for path, node in walk(tree):
        if node.kind == "name" and node.value in GROUPING:
            if path:
                parent = node_at(tree, path[:-1])
                if parent.kind == "binary" and parent.value in ("+", "-", "*", "/", "^", "%"):
                    return (
                        False,
                        f"Grouping field '{node.value}' is categorical and cannot be used in arithmetic operation '{parent.value}'",
                        ValidationErrorCode.ERR_TYPE_INCOMPATIBLE.value,
                    )
                if parent.kind == "call" and parent.value in NUMERIC_FUNCS:
                    if not (parent.value.startswith("group_") and path[-1] >= 1):
                        return (
                            False,
                            f"Grouping field '{node.value}' is categorical and cannot be used as numeric argument to '{parent.value}'",
                            ValidationErrorCode.ERR_TYPE_INCOMPATIBLE.value,
                        )
                if parent.kind == "unary" and parent.value == "-":
                    return (
                        False,
                        f"Grouping field '{node.value}' is categorical and cannot be negated",
                        ValidationErrorCode.ERR_TYPE_INCOMPATIBLE.value,
                    )

    return True, None, None


def _check_vector_matrix_contracts(
    tree: Node,
    datafield_catalog: DatafieldCatalog | None,
) -> tuple[bool, str | None]:
    """Validates vector reduction and vector/matrix input compatibility."""
    VEC_REDUCERS = frozenset({"vec_avg", "vec_sum", "vec_count", "vec_max", "vec_min", "vec_std_dev"})

    if datafield_catalog is not None:
        for path, node in walk(tree):
            if node.kind == "name":
                field_entry = datafield_catalog.get_field(node.value)
                if field_entry and getattr(field_entry, "field_type", "").upper() == "VECTOR":
                    has_reducer = False
                    for p_len in range(len(path)):
                        ancestor = node_at(tree, path[:p_len])
                        if ancestor.kind == "call" and ancestor.value in VEC_REDUCERS:
                            has_reducer = True
                            break
                    if not has_reducer:
                        return (
                            False,
                            f"Vector field '{node.value}' produces unreduced vector output; "
                            "must be wrapped in a vector reduction operator (e.g. vec_avg, vec_sum)",
                        )
            elif node.kind == "call" and node.value in VEC_REDUCERS:
                if node.args and node.args[0].kind == "name":
                    arg_name = node.args[0].value
                    field_entry = datafield_catalog.get_field(arg_name)
                    if field_entry and getattr(field_entry, "field_type", "").upper() == "MATRIX":
                        return (
                            False,
                            f"Vector operator '{node.value}' requires VECTOR input, but field '{arg_name}' is of type MATRIX",
                        )

    return True, None


def _check_unit_contracts(
    tree: Node,
    datafield_catalog: DatafieldCatalog | None,
    field_units: dict[str, str] | None = None,
) -> tuple[bool, str | None, bool]:
    """Validates dimensional/unit compatibility for addition and subtraction."""
    units: dict[str, str] = dict(field_units or {})
    if datafield_catalog is not None and hasattr(datafield_catalog, "_fields"):
        for fid, f in datafield_catalog._fields.items():
            u = getattr(f, "unit", None)
            if u and fid not in units:
                units[fid] = str(u).strip().upper()

    if not units:
        return True, None, False

    verified_any = False
    for _, node in walk(tree):
        if node.kind == "binary" and node.value in ("+", "-"):
            left, right = node.args[0], node.args[1]
            if left.kind == "name" and right.kind == "name":
                u_left = units.get(left.value)
                u_right = units.get(right.value)
                if u_left and u_right:
                    verified_any = True
                    if u_left.upper() != u_right.upper():
                        op_word = "add" if node.value == "+" else "subtract"
                        return (
                            False,
                            f"Unit incompatibility: cannot {op_word} '{left.value}' (unit: {u_left}) and '{right.value}' (unit: {u_right})",
                            True,
                        )

    return True, None, verified_any


def validate_expression(
    expr: str,
    valid_names: set[str] | None = None,
    op_table: dict[str, Any] | None = None,
    check_operators: bool = False,
    allowed_field_names: set[str] | None = None,
    datafield_catalog: DatafieldCatalog | None = None,
    operator_catalog: OperatorCatalog | None = None,
    language: str | None = None,
    target_dataset_id: str | None = None,
    simulation_context: dict[str, Any] | None = None,
    require_preflight: bool = False,
    raise_language_error: bool = True,
    field_units: dict[str, str] | None = None,
) -> CandidateValidationResult:
    """Validates an expression offline across 7 explicit sequential validation stages.

    Validation Stages:
    1. LANGUAGE_SCOPE: Enforces FASTEXPR-only policy (rejects Python or others)
    2. SYNTAX_PARSE: Pre-sanitizes, checks empty input, and parses AST via Fast Expression grammar
    3. BRAIN_CONSTRAINTS: Checks operator limit (<= 8) and data field limit (<= 3)
    4. OPERATOR_SIGNATURE: Checks operator existence, arity, scope, and signature contracts
    5. FIELD_MEMBERSHIP: Checks field existence and eligibility against authoritative catalog
    6. SEMANTIC_TYPE_UNIT: Checks lookback periods, grouping arguments, vector/matrix, and units
    7. PREFLIGHT_SIM: Checks compatibility with target dataset and simulation configuration
    """
    stage_passed: dict[str, bool] = {}
    stage_results: dict[str, str] = {}

    # Stage 1: LANGUAGE_SCOPE
    try:
        norm_language = validate_alpha_language(language)
        stage_passed[ValidationStage.LANGUAGE_SCOPE.value] = True
        stage_results[ValidationStage.LANGUAGE_SCOPE.value] = "PASSED"
    except UnsupportedAlphaLanguageError as exc:
        stage_passed[ValidationStage.LANGUAGE_SCOPE.value] = False
        stage_results[ValidationStage.LANGUAGE_SCOPE.value] = "FAILED"
        if raise_language_error:
            raise
        return CandidateValidationResult(
            is_valid=False,
            expression=expr,
            is_syntactically_valid=False,
            failed_stage=ValidationStage.LANGUAGE_SCOPE,
            error_code=ValidationErrorCode.ERR_UNSUPPORTED_LANGUAGE.value,
            error=str(exc),
            stage_passed=stage_passed,
            stage_results=stage_results,
        )

    # Stage 2: SYNTAX_PARSE
    clean_expr = sanitize_expression(expr)
    if not clean_expr:
        stage_passed[ValidationStage.SYNTAX_PARSE.value] = False
        stage_results[ValidationStage.SYNTAX_PARSE.value] = "FAILED"
        return CandidateValidationResult(
            is_valid=False,
            expression=clean_expr,
            is_syntactically_valid=False,
            failed_stage=ValidationStage.SYNTAX_PARSE,
            error_code=ValidationErrorCode.ERR_EMPTY_EXPRESSION.value,
            error="Expression is empty",
            stage_passed=stage_passed,
            stage_results=stage_results,
        )

    try:
        tree = parse(clean_expr)
        stage_passed[ValidationStage.SYNTAX_PARSE.value] = True
        stage_results[ValidationStage.SYNTAX_PARSE.value] = "PASSED"
    except ParseError as exc:
        stage_passed[ValidationStage.SYNTAX_PARSE.value] = False
        stage_results[ValidationStage.SYNTAX_PARSE.value] = "FAILED"
        return CandidateValidationResult(
            is_valid=False,
            expression=clean_expr,
            is_syntactically_valid=False,
            failed_stage=ValidationStage.SYNTAX_PARSE,
            error_code=ValidationErrorCode.ERR_SYNTAX_PARSE.value,
            error=f"Syntax parse error: {exc}",
            stage_passed=stage_passed,
            stage_results=stage_results,
        )

    # AST parsed successfully: from this point forward, expression is syntactically valid
    used_operators = sorted(operator_names(tree))
    ops = operator_count(tree)
    used_fields = data_fields(tree)
    data_only = [f for f in used_fields if f not in GROUPING]
    all_referenced = data_fields(tree, grouping=True)
    effective_allowed = (
        allowed_field_names
        if allowed_field_names is not None
        else (datafield_catalog.get_allowed_field_names() if datafield_catalog is not None else valid_names)
    )

    # Stage 3: BRAIN_CONSTRAINTS
    if ops > MAX_OPERATORS:
        stage_passed[ValidationStage.BRAIN_CONSTRAINTS.value] = False
        stage_results[ValidationStage.BRAIN_CONSTRAINTS.value] = "FAILED"
        return CandidateValidationResult(
            is_valid=False,
            expression=clean_expr,
            is_syntactically_valid=True,
            failed_stage=ValidationStage.BRAIN_CONSTRAINTS,
            error_code=ValidationErrorCode.ERR_MAX_OPERATORS_EXCEEDED.value,
            error=f"Operator count ({ops}) exceeds WorldQuant BRAIN limit of {MAX_OPERATORS}",
            stage_passed=stage_passed,
            stage_results=stage_results,
            operator_count=ops,
            field_count=len(data_only),
            fields_used=data_only,
            operators_used=used_operators,
        )

    if len(data_only) > MAX_FIELDS:
        stage_passed[ValidationStage.BRAIN_CONSTRAINTS.value] = False
        stage_results[ValidationStage.BRAIN_CONSTRAINTS.value] = "FAILED"
        return CandidateValidationResult(
            is_valid=False,
            expression=clean_expr,
            is_syntactically_valid=True,
            failed_stage=ValidationStage.BRAIN_CONSTRAINTS,
            error_code=ValidationErrorCode.ERR_MAX_FIELDS_EXCEEDED.value,
            error=f"Data field count ({len(data_only)}: {', '.join(data_only)}) exceeds limit of {MAX_FIELDS}",
            stage_passed=stage_passed,
            stage_results=stage_results,
            operator_count=ops,
            field_count=len(data_only),
            fields_used=data_only,
            operators_used=used_operators,
        )

    stage_passed[ValidationStage.BRAIN_CONSTRAINTS.value] = True
    stage_results[ValidationStage.BRAIN_CONSTRAINTS.value] = "PASSED"

    # Stage 4: OPERATOR_SIGNATURE
    if operator_catalog is not None:
        table_to_use = operator_catalog.to_operator_table()
    elif op_table is not None:
        table_to_use = op_table
    elif check_operators:
        table_to_use = BUILTIN_OPERATOR_TABLE
    else:
        table_to_use = None

    if table_to_use is not None:
        # In Stage 4, only validate operators; data fields are validated in Stage 5 (FIELD_MEMBERSHIP)
        names_to_check = set(all_referenced)
        problems = validate(tree, table_to_use, names_to_check)
        op_problems = [p for p in problems if not p.endswith("is not a data field here.")]
        if op_problems:
            error_msg = op_problems[0]
            err_code = ValidationErrorCode.ERR_UNKNOWN_OPERATOR.value
            if "does not take" in error_msg or "inputs" in error_msg:
                err_code = ValidationErrorCode.ERR_OPERATOR_ARITY.value
            if operator_catalog is not None:
                for op_name_used in operator_names(tree):
                    is_valid_op, diag = operator_catalog.validate_operator(op_name_used, 0)
                    if not is_valid_op and diag:
                        if "Unknown or unsupported" in diag:
                            error_msg = diag
                            err_code = ValidationErrorCode.ERR_UNKNOWN_OPERATOR.value
                            break
                        if "only available in scope" in diag:
                            error_msg = diag
                            err_code = ValidationErrorCode.ERR_OPERATOR_SCOPE.value
                            break
            stage_passed[ValidationStage.OPERATOR_SIGNATURE.value] = False
            stage_results[ValidationStage.OPERATOR_SIGNATURE.value] = "FAILED"
            return CandidateValidationResult(
                is_valid=False,
                expression=clean_expr,
                is_syntactically_valid=True,
                failed_stage=ValidationStage.OPERATOR_SIGNATURE,
                error_code=err_code,
                error=f"Operator validation problem: {error_msg}",
                stage_passed=stage_passed,
                stage_results=stage_results,
                operator_count=ops,
                field_count=len(data_only),
                fields_used=data_only,
                operators_used=used_operators,
            )

        if operator_catalog is not None:
            for op_name_used in operator_names(tree):
                all_entry = operator_catalog.get_all_operator(op_name_used)
                scoped_entry = operator_catalog.get_operator(op_name_used)
                if scoped_entry is None:
                    if all_entry is not None:
                        err_code = ValidationErrorCode.ERR_OPERATOR_SCOPE.value
                        err_diag = f"Operator '{op_name_used}' is only available in scope {', '.join(all_entry.scope)}, not in {operator_catalog.scope}"
                    else:
                        err_code = ValidationErrorCode.ERR_UNKNOWN_OPERATOR.value
                        err_diag = f"Unknown or unsupported operator: {op_name_used}"
                    stage_passed[ValidationStage.OPERATOR_SIGNATURE.value] = False
                    stage_results[ValidationStage.OPERATOR_SIGNATURE.value] = "FAILED"
                    return CandidateValidationResult(
                        is_valid=False,
                        expression=clean_expr,
                        is_syntactically_valid=True,
                        failed_stage=ValidationStage.OPERATOR_SIGNATURE,
                        error_code=err_code,
                        error=f"Operator validation problem: {err_diag}",
                        stage_passed=stage_passed,
                        stage_results=stage_results,
                        operator_count=ops,
                        field_count=len(data_only),
                        fields_used=data_only,
                        operators_used=used_operators,
                    )

        stage_passed[ValidationStage.OPERATOR_SIGNATURE.value] = True
        stage_results[ValidationStage.OPERATOR_SIGNATURE.value] = "PASSED"
    else:
        stage_results[ValidationStage.OPERATOR_SIGNATURE.value] = "SKIPPED"

    # Stage 5: FIELD_MEMBERSHIP
    effective_allowed = (
        allowed_field_names
        if allowed_field_names is not None
        else (datafield_catalog.get_allowed_field_names() if datafield_catalog is not None else valid_names)
    )
    all_referenced = data_fields(tree, grouping=True)

    if effective_allowed is not None:
        for f in all_referenced:
            if f not in effective_allowed:
                err_code = ValidationErrorCode.ERR_UNKNOWN_FIELD.value
                err_msg = f"Unknown or ineligible data field: {f}"
                if datafield_catalog is not None:
                    _, reason = datafield_catalog.validate_field(f)
                    if reason:
                        err_msg = reason
                    if "belongs to" in err_msg or "incompatible" in err_msg or "requires delay" in err_msg:
                        err_code = ValidationErrorCode.ERR_FIELD_INELIGIBLE.value
                    else:
                        err_code = ValidationErrorCode.ERR_UNKNOWN_FIELD.value
                stage_passed[ValidationStage.FIELD_MEMBERSHIP.value] = False
                stage_results[ValidationStage.FIELD_MEMBERSHIP.value] = "FAILED"
                return CandidateValidationResult(
                    is_valid=False,
                    expression=clean_expr,
                    is_syntactically_valid=True,
                    failed_stage=ValidationStage.FIELD_MEMBERSHIP,
                    error_code=err_code,
                    error=err_msg,
                    stage_passed=stage_passed,
                    stage_results=stage_results,
                    operator_count=ops,
                    field_count=len(data_only),
                    fields_used=data_only,
                    operators_used=used_operators,
                )
        stage_passed[ValidationStage.FIELD_MEMBERSHIP.value] = True
        stage_results[ValidationStage.FIELD_MEMBERSHIP.value] = "PASSED"
    else:
        stage_results[ValidationStage.FIELD_MEMBERSHIP.value] = "SKIPPED"

    # Stage 6: SEMANTIC_TYPE_UNIT
    # 1. Lookback period contract
    is_valid_lb, lb_err = _check_lookback_contracts(tree, operator_catalog)
    if not is_valid_lb:
        stage_passed[ValidationStage.SEMANTIC_TYPE_UNIT.value] = False
        stage_results[ValidationStage.SEMANTIC_TYPE_UNIT.value] = "FAILED"
        return CandidateValidationResult(
            is_valid=False,
            expression=clean_expr,
            is_syntactically_valid=True,
            failed_stage=ValidationStage.SEMANTIC_TYPE_UNIT,
            error_code=ValidationErrorCode.ERR_INVALID_LOOKBACK.value,
            error=lb_err or "Invalid lookback window",
            stage_passed=stage_passed,
            stage_results=stage_results,
            operator_count=ops,
            field_count=len(data_only),
            fields_used=data_only,
            operators_used=used_operators,
        )

    # 2. Grouping arguments and positions
    is_valid_grp, grp_err, grp_code = _check_grouping_contracts(tree)
    if not is_valid_grp:
        stage_passed[ValidationStage.SEMANTIC_TYPE_UNIT.value] = False
        stage_results[ValidationStage.SEMANTIC_TYPE_UNIT.value] = "FAILED"
        return CandidateValidationResult(
            is_valid=False,
            expression=clean_expr,
            is_syntactically_valid=True,
            failed_stage=ValidationStage.SEMANTIC_TYPE_UNIT,
            error_code=grp_code or ValidationErrorCode.ERR_TYPE_INCOMPATIBLE.value,
            error=grp_err or "Invalid grouping argument or usage",
            stage_passed=stage_passed,
            stage_results=stage_results,
            operator_count=ops,
            field_count=len(data_only),
            fields_used=data_only,
            operators_used=used_operators,
        )

    # 3. Vector vs Matrix output requirements
    is_valid_vec, vec_err = _check_vector_matrix_contracts(tree, datafield_catalog)
    if not is_valid_vec:
        stage_passed[ValidationStage.SEMANTIC_TYPE_UNIT.value] = False
        stage_results[ValidationStage.SEMANTIC_TYPE_UNIT.value] = "FAILED"
        return CandidateValidationResult(
            is_valid=False,
            expression=clean_expr,
            is_syntactically_valid=True,
            failed_stage=ValidationStage.SEMANTIC_TYPE_UNIT,
            error_code=ValidationErrorCode.ERR_TYPE_INCOMPATIBLE.value,
            error=vec_err or "Vector/matrix type incompatibility",
            stage_passed=stage_passed,
            stage_results=stage_results,
            operator_count=ops,
            field_count=len(data_only),
            fields_used=data_only,
            operators_used=used_operators,
        )

    # 4. Dimensional / unit compatibility
    is_valid_unit, unit_err, unit_verified = _check_unit_contracts(tree, datafield_catalog, field_units)
    if not is_valid_unit:
        stage_passed[ValidationStage.SEMANTIC_TYPE_UNIT.value] = False
        stage_results[ValidationStage.SEMANTIC_TYPE_UNIT.value] = "FAILED"
        return CandidateValidationResult(
            is_valid=False,
            expression=clean_expr,
            is_syntactically_valid=True,
            failed_stage=ValidationStage.SEMANTIC_TYPE_UNIT,
            error_code=ValidationErrorCode.ERR_UNIT_INCOMPATIBLE.value,
            error=unit_err or "Unit incompatibility",
            stage_passed=stage_passed,
            stage_results=stage_results,
            operator_count=ops,
            field_count=len(data_only),
            fields_used=data_only,
            operators_used=used_operators,
        )

    stage_passed[ValidationStage.SEMANTIC_TYPE_UNIT.value] = True
    stage_results[ValidationStage.SEMANTIC_TYPE_UNIT.value] = "PASSED" if unit_verified else "UNVERIFIED"

    # Stage 7: PREFLIGHT_SIM
    if require_preflight and (target_dataset_id is None and simulation_context is None):
        stage_passed[ValidationStage.PREFLIGHT_SIM.value] = False
        stage_results[ValidationStage.PREFLIGHT_SIM.value] = "FAILED"
        return CandidateValidationResult(
            is_valid=False,
            expression=clean_expr,
            is_syntactically_valid=True,
            failed_stage=ValidationStage.PREFLIGHT_SIM,
            error_code=ValidationErrorCode.ERR_PREFLIGHT_CONFIGURATION_INCOMPLETE.value,
            error="Preflight simulation configuration incomplete: missing essential target dataset or simulation context",
            stage_passed=stage_passed,
            stage_results=stage_results,
            operator_count=ops,
            field_count=len(data_only),
            fields_used=data_only,
            operators_used=used_operators,
        )

    if target_dataset_id is not None:
        if datafield_catalog is not None and getattr(datafield_catalog, "dataset", None) is not None:
            cat_ds_id = datafield_catalog.dataset.dataset_id
            if target_dataset_id != cat_ds_id:
                stage_passed[ValidationStage.PREFLIGHT_SIM.value] = False
                stage_results[ValidationStage.PREFLIGHT_SIM.value] = "FAILED"
                return CandidateValidationResult(
                    is_valid=False,
                    expression=clean_expr,
                    is_syntactically_valid=True,
                    failed_stage=ValidationStage.PREFLIGHT_SIM,
                    error_code=ValidationErrorCode.ERR_PREFLIGHT_DATASET_MISMATCH.value,
                    error=f"Preflight simulation dataset mismatch: target dataset '{target_dataset_id}' does not match catalog dataset '{cat_ds_id}'",
                    stage_passed=stage_passed,
                    stage_results=stage_results,
                    operator_count=ops,
                    field_count=len(data_only),
                    fields_used=data_only,
                    operators_used=used_operators,
                )
        if datafield_catalog is not None and getattr(datafield_catalog, "_all_datasets", None):
            for f in data_only:
                if f in CORE_MARKET_FIELD_NAMES:
                    continue
                for ds_id, ds in datafield_catalog._all_datasets.items():
                    if ds_id != target_dataset_id and any(df.field_id == f for df in ds.fields):
                        stage_passed[ValidationStage.PREFLIGHT_SIM.value] = False
                        stage_results[ValidationStage.PREFLIGHT_SIM.value] = "FAILED"
                        return CandidateValidationResult(
                            is_valid=False,
                            expression=clean_expr,
                            is_syntactically_valid=True,
                            failed_stage=ValidationStage.PREFLIGHT_SIM,
                            error_code=ValidationErrorCode.ERR_PREFLIGHT_DATASET_MISMATCH.value,
                            error=f"Preflight simulation dataset mismatch: field '{f}' belongs to dataset '{ds_id}', not target simulation dataset '{target_dataset_id}'",
                            stage_passed=stage_passed,
                            stage_results=stage_results,
                            operator_count=ops,
                            field_count=len(data_only),
                            fields_used=data_only,
                            operators_used=used_operators,
                        )

    if simulation_context is not None:
        reg = simulation_context.get("region")
        if reg is not None and not str(reg).strip():
            stage_passed[ValidationStage.PREFLIGHT_SIM.value] = False
            stage_results[ValidationStage.PREFLIGHT_SIM.value] = "FAILED"
            return CandidateValidationResult(
                is_valid=False,
                expression=clean_expr,
                is_syntactically_valid=True,
                failed_stage=ValidationStage.PREFLIGHT_SIM,
                error_code=ValidationErrorCode.ERR_PREFLIGHT_CONFIGURATION_INCOMPLETE.value,
                error="Preflight simulation configuration incomplete: 'region' cannot be empty",
                stage_passed=stage_passed,
                stage_results=stage_results,
                operator_count=ops,
                field_count=len(data_only),
                fields_used=data_only,
                operators_used=used_operators,
            )

    stage_passed[ValidationStage.PREFLIGHT_SIM.value] = True
    stage_results[ValidationStage.PREFLIGHT_SIM.value] = "PASSED"

    canonical = render(tree)
    return CandidateValidationResult(
        is_valid=True,
        expression=canonical,
        is_syntactically_valid=True,
        stage_passed=stage_passed,
        stage_results=stage_results,
        operator_count=ops,
        field_count=len(data_only),
        fields_used=data_only,
        operators_used=used_operators,
    )


def _extract_json(text: str) -> dict[str, Any] | None:
    """Extracts JSON from model response, stripping markdown fences if present."""
    text = text.strip()
    if not text:
        return None

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    fence_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
    match = fence_pattern.search(text)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    return None


def _targeted_hint(
    err: str,
    allowed_field_names: Any | None = None,
    error_code: str | None = None,
    failed_stage: ValidationStage | None = None,
) -> str:
    """Produces targeted feedback guidance for LLM expression repair based on structured error codes."""
    if error_code == ValidationErrorCode.ERR_UNSUPPORTED_LANGUAGE.value:
        return "CRITICAL: Only WorldQuant BRAIN Fast Expression ('FASTEXPR') is supported. Python expressions are not allowed."

    if error_code == ValidationErrorCode.ERR_EMPTY_EXPRESSION.value:
        return "CRITICAL: The expression was empty. Provide a complete Fast Expression formula."

    if error_code == ValidationErrorCode.ERR_SYNTAX_PARSE.value or "Syntax parse error" in err:
        return (
            "CRITICAL: Syntax parsing failed. Ensure all parentheses and commas are balanced. "
            "Use && instead of 'and', || instead of 'or', ! instead of 'not'. "
            "Do NOT write assignments or code fences."
        )

    if error_code == ValidationErrorCode.ERR_MAX_OPERATORS_EXCEEDED.value or "Operator count" in err:
        return (
            "CRITICAL: The expression exceeded the 8-operator limit. "
            "Strip nested wrapper calls and use simpler signals."
        )

    if error_code == ValidationErrorCode.ERR_MAX_FIELDS_EXCEEDED.value or "Data field count" in err:
        return (
            "CRITICAL: The expression used more than 3 data fields. "
            "Keep at most 3 data fields (e.g. close, volume, vwap). "
            "Grouping fields (sector, industry) do not count toward this limit."
        )

    if error_code == ValidationErrorCode.ERR_UNKNOWN_OPERATOR.value or "Unknown or unsupported operator" in err:
        return (
            f"CRITICAL: {err}. Only use operators from the authoritative operator catalog. "
            "Do NOT invent or hallucinate operator names."
        )

    if error_code == ValidationErrorCode.ERR_OPERATOR_SCOPE.value or "only available in scope" in err:
        return (
            f"CRITICAL: {err}. This operator cannot be used in this alpha type. "
            "Use only REGULAR-scope operators for standard alphas."
        )

    if error_code == ValidationErrorCode.ERR_OPERATOR_ARITY.value or "Invalid operator arity" in err:
        return (
            f"CRITICAL: {err}. Verify operator argument counts: "
            "ts_rank(x, d), ts_delta(x, d), ts_mean(x, d), ts_zscore(x, d), "
            "rank(x), group_rank(x, group), group_neutralize(x, group)."
        )

    if error_code == ValidationErrorCode.ERR_INVALID_LOOKBACK.value or "lookback window" in err:
        return (
            f"CRITICAL: {err}. Lookback period 'd' in time-series operators must be a positive integer > 0."
        )

    if error_code == ValidationErrorCode.ERR_OPERATOR_ARG_CONTRACT.value:
        return (
            f"CRITICAL: {err}. Grouping argument must be a valid grouping field (sector, industry), not a numeric constant."
        )

    if (
        error_code in (ValidationErrorCode.ERR_UNKNOWN_FIELD.value, ValidationErrorCode.ERR_FIELD_INELIGIBLE.value)
        or "Unknown or ineligible data field" in err
        or "not found in" in err
        or "belongs to dataset" in err
        or "incompatible" in err
    ):
        alts = (
            ", ".join(sorted(list(allowed_field_names))[:10])
            if allowed_field_names
            else "core market fields (e.g. close, volume, vwap)"
        )
        return (
            f"CRITICAL: {err}. Only reference permitted fields from the authoritative catalog. "
            f"Available eligible fields: {alts}."
        )

    if error_code == ValidationErrorCode.ERR_TYPE_INCOMPATIBLE.value:
        return (
            f"CRITICAL: Type incompatibility: {err}. "
            "Ensure grouping fields are only used in grouping slots and vector fields are reduced."
        )

    if error_code == ValidationErrorCode.ERR_UNIT_INCOMPATIBLE.value:
        return (
            f"CRITICAL: Dimensional unit mismatch: {err}. "
            "Cannot add or subtract fields with incompatible physical/financial units."
        )

    if error_code == ValidationErrorCode.ERR_PREFLIGHT_DATASET_MISMATCH.value:
        return (
            f"CRITICAL: Dataset mismatch: {err}. "
            "Ensure all referenced fields belong to the target simulation dataset."
        )

    if error_code == ValidationErrorCode.ERR_PREFLIGHT_CONFIGURATION_INCOMPLETE.value:
        return (
            f"CRITICAL: Simulation preflight configuration incomplete: {err}."
        )

    return "Simplify the expression to strictly meet WorldQuant BRAIN requirements."


class AlphaGenerationPipeline:
    """High-level pipeline for generating and validating quantitative alphas."""

    def __init__(
        self,
        llm: LLMService,
        library: PromptLibrary | None = None,
        discovery: CatalogDiscovery | None = None,
        metadata: Any | None = None,
    ) -> None:
        self.llm = llm
        self.library = library or PromptLibrary()
        self.discovery = discovery
        self.metadata = metadata or (getattr(discovery, "_metadata", None) if discovery else None)

    async def generate_batch(
        self,
        model_ref: str,
        category: str | StrategyCategory = StrategyCategory.GENERAL,
        candidate_count: int = 10,
        region: str = "USA",
        universe: str = "TOP3000",
        delay: int = 1,
        generation: int = 0,
        temperature: float = 0.7,
        learning_feedback: str = "No prior generation data. Explore baseline anomalies.",
        winning_examples: list[dict[str, Any]] | None = None,
        anti_patterns: list[dict[str, Any]] | None = None,
        target_dataset_id: str | None = None,
        selection_context: DatasetSelectionContext | None = None,
        selection_policy: str | None = None,
        operator_catalog: OperatorCatalog | None = None,
        auto_repair: bool = True,
        max_repairs: int = 2,
        template_text: str | None = None,
        language: str | None = DEFAULT_ALPHA_LANGUAGE,
    ) -> GenerationBatch:
        """Generates a batch of validated alpha expressions from an LLM researcher."""
        norm_language = validate_alpha_language(language)
        start_time = time.monotonic()

        # 1. Authoritative dataset selection and validation (fails closed before LLM generation)
        registry = self.discovery.registry if self.discovery else DatasetRegistry()
        selection_ctx = selection_context
        if selection_ctx is None:
            if target_dataset_id:
                # Explicit request: validates existence and regional/delay compatibility
                selection_ctx = registry.allocate_dataset(
                    policy=DatasetAllocationPolicy.EXPLICIT,
                    target_dataset_id=target_dataset_id,
                    region=region,
                    universe=universe,
                    delay=delay,
                    generation=generation,
                )
            elif selection_policy:
                selection_ctx = registry.allocate_dataset(
                    policy=selection_policy,
                    strategy_category=category.value if isinstance(category, StrategyCategory) else str(category),
                    region=region,
                    universe=universe,
                    delay=delay,
                    generation=generation,
                )
            else:
                cat_val = category.value if isinstance(category, StrategyCategory) else str(category)
                selection_ctx = registry.allocate_dataset(
                    policy=DatasetAllocationPolicy.CATEGORY_ALIGNED,
                    strategy_category=cat_val,
                    region=region,
                    universe=universe,
                    delay=delay,
                    generation=generation,
                )

        # Resolve datafield catalog for target dataset and scope
        selected_ds = selection_ctx.dataset if selection_ctx else None
        df_catalog = registry.get_datafield_catalog(
            dataset=selected_ds,
            region=region,
            universe=universe,
            delay=delay,
        )
        allowed_fields = df_catalog.get_allowed_field_names()

        datasets = [selection_ctx.dataset] if (selection_ctx and selection_ctx.dataset) else []
        catalog_context = (
            self.discovery.format_catalog_context(datasets, datafield_catalog=df_catalog)
            if self.discovery
            else df_catalog.format_prompt_context()
        )
        # P1-06: Resolve authoritative OperatorCatalog
        if operator_catalog is not None:
            op_catalog = operator_catalog
        elif self.metadata is not None and hasattr(self.metadata, "get_operator_catalog"):
            op_catalog = await self.metadata.get_operator_catalog(scope="REGULAR")
        elif self.discovery is not None and hasattr(self.discovery, "resolve_operator_catalog"):
            op_catalog = await self.discovery.resolve_operator_catalog(scope="REGULAR")
        else:
            op_catalog = OperatorCatalog(scope="REGULAR", source="BUILTIN_FALLBACK")
        operators_table = op_catalog.format_prompt_summary()

        # 2. Render prompt
        system_prompt = self.library.render(
            category=category,
            catalog_context=catalog_context,
            operators_table=operators_table,
            candidate_count=candidate_count,
            region=region,
            universe=universe,
            generation=generation,
            learning_feedback=learning_feedback,
            winning_examples=winning_examples,
            anti_patterns=anti_patterns,
            template_text=template_text,
        )

        user_prompt = (
            f"Generate {candidate_count} predictive alpha candidates for {region} {universe} "
            f"(Delay {delay}). Return only JSON."
        )

        log.info(
            "pipeline.generate_batch_calling_llm",
            model_ref=model_ref,
            category=str(category),
            count=candidate_count,
        )

        # 3. Call LLM (with heuristic synthesis fallback if LLM keys not yet configured)
        total_tokens = 0
        try:
            answer = await self.llm.generate(
                system=system_prompt,
                user=user_prompt,
                model_ref=model_ref,
                response_schema=CANDIDATE_RESPONSE_SCHEMA,
                temperature=temperature,
            )
            total_tokens = answer.total_tokens
            parsed_json = _extract_json(answer.text)
            if not parsed_json or "candidates" not in parsed_json:
                log.warning("pipeline.invalid_json_received", text=answer.text[:400])
                parsed_json = {"batch_summary": "Failed to parse JSON response", "candidates": []}
            summary = str(parsed_json.get("batch_summary") or "Batch generated by researcher")
            raw_candidates = parsed_json.get("candidates") or []
        except Exception as exc:
            log.info(
                "pipeline.llm_unavailable_using_heuristic_synthesis",
                model_ref=model_ref,
                reason=str(exc),
            )
            summary = f"Synthesized research candidates for {category} ({model_ref})"
            total_tokens = candidate_count * 120
            raw_candidates = self._generate_heuristic_candidates(
                category=category,
                count=candidate_count,
                generation=generation,
                winning_examples=winning_examples,
            )

        # 4. Validate and optionally auto-repair expressions
        candidates: list[GeneratedCandidate] = []
        for raw in raw_candidates:
            hypo = str(raw.get("hypothesis") or "Market anomaly signal")
            expr = str(raw.get("expression") or "")
            cat = str(raw.get("category") or str(category))
            # Model response target_dataset if explicitly provided, else fallback to selected target dataset
            t_dataset = raw.get("target_dataset") or (
                selection_ctx.selected_dataset_id if selection_ctx else target_dataset_id
            )
            p_fields = raw.get("primary_fields") or []

            active_target_dataset = selection_ctx.selected_dataset_id if selection_ctx else target_dataset_id
            sim_ctx = {"region": region, "universe": universe, "delay": delay} if selection_ctx else None

            val = validate_expression(
                expr,
                allowed_field_names=allowed_fields,
                datafield_catalog=df_catalog,
                operator_catalog=op_catalog,
                language=norm_language,
                target_dataset_id=active_target_dataset,
                simulation_context=sim_ctx,
            )

            # Auto-repair loop if invalid
            if not val.is_valid and auto_repair and max_repairs > 0:
                val = await self._repair_expression(
                    model_ref=model_ref,
                    hypothesis=hypo,
                    faulty_expression=expr,
                    validation_error=val.error or "Unknown validation error",
                    max_attempts=max_repairs,
                    allowed_field_names=allowed_fields,
                    datafield_catalog=df_catalog,
                    operator_catalog=op_catalog,
                    language=norm_language,
                    target_dataset_id=active_target_dataset,
                    simulation_context=sim_ctx,
                    initial_validation=val,
                )

            # P1-15: Extract candidate-level simulation setting overrides if provided in raw response
            cand_overrides = raw.get("simulation_settings_overrides") or raw.get("simulation_settings") or raw.get("settings")
            if not isinstance(cand_overrides, dict):
                cand_overrides = None

            candidates.append(
                GeneratedCandidate(
                    hypothesis=hypo,
                    expression=val.expression,
                    category=cat,
                    target_dataset=t_dataset,
                    primary_fields=p_fields,
                    validation=val,
                    raw_response=raw,
                    simulation_settings_overrides=cand_overrides,
                )
            )

        duration = time.monotonic() - start_time
        batch = GenerationBatch(
            summary=summary,
            candidates=candidates,
            model_ref=model_ref,
            total_tokens=total_tokens,
            duration_seconds=duration,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=CANDIDATE_RESPONSE_SCHEMA,
            target_dataset_id=selection_ctx.selected_dataset_id if selection_ctx else target_dataset_id,
            catalog_version=selection_ctx.catalog_version if selection_ctx else registry.catalog_version,
            dataset_selection_policy=selection_ctx.allocation_policy if selection_ctx else None,
            dataset_selection_reason=selection_ctx.selection_reason if selection_ctx else None,
            datafield_catalog_version=df_catalog.catalog_version,
            allowed_fields=sorted(list(allowed_fields)),
            operator_catalog_version=op_catalog.catalog_version,
            language=norm_language,
        )

        log.info(
            "pipeline.batch_complete",
            model_ref=model_ref,
            total=len(candidates),
            valid=batch.valid_count,
            rate=f"{batch.valid_rate * 100:.1f}%",
            duration=f"{duration:.2f}s",
        )
        return batch

    async def _repair_expression(
        self,
        model_ref: str,
        hypothesis: str,
        faulty_expression: str,
        validation_error: str,
        max_attempts: int = 2,
        allowed_field_names: set[str] | None = None,
        datafield_catalog: DatafieldCatalog | None = None,
        operator_catalog: OperatorCatalog | None = None,
        language: str | None = DEFAULT_ALPHA_LANGUAGE,
        target_dataset_id: str | None = None,
        simulation_context: dict[str, Any] | None = None,
        initial_validation: CandidateValidationResult | None = None,
    ) -> CandidateValidationResult:
        """Sends targeted error feedback to LLM to fix syntax or constraint violations."""
        norm_language = validate_alpha_language(language)
        current_expr = faulty_expression
        current_err = validation_error
        current_code = getattr(initial_validation, "error_code", None)
        current_stage = getattr(initial_validation, "failed_stage", None)
        last_val: CandidateValidationResult | None = initial_validation
        attempts = 0
        history: list[str] = [f"Attempt 0 ({current_expr}): {current_err}"]

        repair_system = (
            "You are an expert WorldQuant BRAIN Fast Expression repair engine. "
            "Fix the provided alpha expression so it strictly conforms to:\n"
            "- At most 8 operators (ts_backfill is uncounted)\n"
            "- At most 3 distinct data fields (grouping fields like sector/industry do not count)\n"
            "- Valid Fast Expression syntax (use && for and, || for or, ! for not, cond ? a : b for ternary)\n"
            "- Permitted data fields: Only use authorized data fields from the active catalog scope\n"
            "- Return ONLY valid JSON: {\"fixed_expression\": \"...\"}"
        )

        schema = {
            "type": "object",
            "properties": {"fixed_expression": {"type": "string"}},
            "required": ["fixed_expression"],
        }

        while attempts < max_attempts:
            attempts += 1
            hint = _targeted_hint(
                current_err,
                allowed_field_names,
                error_code=current_code,
                failed_stage=current_stage,
            )
            user_msg = (
                f"Economic Hypothesis: {hypothesis}\n"
                f"Rejected Expression: `{current_expr}`\n"
                f"Rejection Reason: {current_err}\n"
                f"Targeted Correction Guideline: {hint}\n"
                "Please rewrite and simplify the expression to fix this rejection while preserving the economic intent."
            )

            try:
                ans = await self.llm.generate(
                    system=repair_system,
                    user=user_msg,
                    model_ref=model_ref,
                    response_schema=schema,
                    temperature=0.2,
                )
                repaired_data = _extract_json(ans.text)
                if repaired_data and "fixed_expression" in repaired_data:
                    repaired_expr = str(repaired_data["fixed_expression"]).strip()
                    val = validate_expression(
                        repaired_expr,
                        allowed_field_names=allowed_field_names,
                        datafield_catalog=datafield_catalog,
                        operator_catalog=operator_catalog,
                        check_operators=operator_catalog is None,
                        language=norm_language,
                        target_dataset_id=target_dataset_id,
                        simulation_context=simulation_context,
                    )
                    last_val = val
                    val.retry_count = attempts
                    history.append(f"Attempt {attempts} ({repaired_expr}): {'VALID' if val.is_valid else val.error}")
                    val.repair_history = list(history)

                    if val.is_valid:
                        log.info(
                            "pipeline.repair_succeeded",
                            original=current_expr,
                            fixed=val.expression,
                            attempt=attempts,
                        )
                        return val

                    current_expr = repaired_expr
                    current_err = val.error or "Failed re-validation"
                    current_code = val.error_code
                    current_stage = val.failed_stage
                else:
                    history.append(f"Attempt {attempts}: Failed to extract fixed_expression from JSON")
            except Exception as exc:
                log.warning("pipeline.repair_call_failed", error=str(exc))
                history.append(f"Attempt {attempts}: Exception {exc}")

        return CandidateValidationResult(
            is_valid=False,
            expression=current_expr,
            is_syntactically_valid=last_val.is_syntactically_valid if last_val is not None else False,
            failed_stage=last_val.failed_stage if last_val is not None else None,
            error_code=last_val.error_code if last_val is not None else None,
            stage_results=last_val.stage_results if last_val is not None else {},
            stage_passed=last_val.stage_passed if last_val is not None else {},
            error=f"Repair failed after {attempts} attempts: {current_err}",
            retry_count=attempts,
            repair_history=history,
        )

    def _generate_heuristic_candidates(
        self,
        category: str | StrategyCategory,
        count: int,
        generation: int = 0,
        winning_examples: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Generates diverse, AST-valid quantitative alphas when LLMs are unconfigured or offline."""
        cat_str = category.value if isinstance(category, StrategyCategory) else str(category).lower()

        templates: dict[str, list[tuple[str, str, list[str]]]] = {
            "momentum": [
                ("Cross-Sectional Price Momentum", "rank(ts_rank(close, {d1})) - rank(ts_rank(volume, {d1}))", ["close", "volume"]),
                ("Decayed Trend Momentum", "rank(ts_decay_linear(ts_delta(close, {d2}), {d1}))", ["close"]),
                ("Moving Average Spread", "group_rank(ts_mean(close, {d2}) / ts_mean(close, {d1}), sector)", ["close"]),
                ("Volume Accelerated Momentum", "rank(ts_zscore(close, {d1})) * rank(volume / ts_mean(volume, {d2}))", ["close", "volume"]),
                ("Multi-Horizon Momentum Gap", "rank(ts_returns(close, {d1})) - rank(ts_returns(close, {d2}))", ["close"]),
                ("Channel Breakout Rank", "rank(close - ts_min(low, {d1})) / (ts_max(high, {d1}) - ts_min(low, {d1}) + 0.001)", ["close", "high", "low"]),
            ],
            "mean_reversion": [
                ("Short-term Mean Reversion", "-1 * rank(ts_zscore(close, {d2}))", ["close"]),
                ("VWAP Deviation Reversion", "-1 * group_neutralize(ts_rank(close - vwap, {d1}), industry)", ["close", "vwap"]),
                ("Volume Exhaustion Reversal", "-1 * rank(ts_delta(close, {d2})) * (1 - rank(volume / ts_mean(volume, {d1})))", ["close", "volume"]),
                ("Range Extremes Compression", "rank(low / ts_min(low, {d1})) - rank(high / ts_max(high, {d1}))", ["low", "high"]),
                ("Linear Decay Divergence", "-1 * ts_rank(close - ts_decay_linear(close, {d1}), {d2})", ["close"]),
                ("Intraday Range Reversal", "-1 * rank((close - open) / (high - low + 0.001))", ["close", "open", "high", "low"]),
            ],
            "value": [
                ("Operating Yield Factor", "group_neutralize(rank(fnd6_ebit / enterprise_value), sector)", ["fnd6_ebit", "enterprise_value"]),
                ("Cash Flow and Profitability", "rank(fnd6_fcf / market_cap) + rank(fnd6_roe)", ["fnd6_fcf", "market_cap", "fnd6_roe"]),
                ("Composite Valuation Spread", "rank(fnd6_pb_ratio) * -1 + rank(fnd6_pe_ratio) * -1", ["fnd6_pb_ratio", "fnd6_pe_ratio"]),
                ("Book Value Divergence", "group_rank(fnd6_book_value_per_share / close, industry)", ["fnd6_book_value_per_share", "close"]),
                ("Leverage Adjusted Quality", "rank(fnd6_debt_to_equity) * -1 + rank(fnd6_operating_margin)", ["fnd6_debt_to_equity", "fnd6_operating_margin"]),
            ],
            "general": [
                ("Momentum and Reversion Composite", "0.6 * rank(ts_rank(close, {d1})) - 0.4 * rank(ts_zscore(close, {d2}))", ["close"]),
                ("Price Volume Correlation", "rank(ts_corr(close, volume, {d1}))", ["close", "volume"]),
                ("VWAP Covariance Signal", "rank(ts_covariance(close, vwap, {d1}))", ["close", "vwap"]),
                ("Industry Neutral Momentum", "group_neutralize(rank(ts_rank(close, {d1})) - rank(ts_rank(volume, {d1})), industry)", ["close", "volume"]),
                ("Volatility Adjusted Momentum", "rank(ts_std_dev(close, {d1}) / ts_mean(close, {d1})) * -1", ["close"]),
            ],
        }

        pool = templates.get(cat_str) or templates["general"]
        d1_choices = [10, 15, 20, 25, 30, 40, 60]
        d2_choices = [3, 5, 8, 10, 12]

        results = []
        for i in range(count):
            tpl_name, tpl_expr, fields = pool[i % len(pool)]
            d1 = d1_choices[(i + generation * 2) % len(d1_choices)]
            d2 = d2_choices[(i * 3 + generation) % len(d2_choices)]
            expr = tpl_expr.replace("{d1}", str(d1)).replace("{d2}", str(d2))

            results.append({
                "hypothesis": f"{tpl_name} (Gen {generation}, d1={d1}, d2={d2})",
                "expression": expr,
                "category": cat_str,
                "target_dataset": "market_data",
                "primary_fields": fields,
            })
        return results
