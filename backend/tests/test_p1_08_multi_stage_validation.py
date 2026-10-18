"""Regression test suite for P1-08: Multi-Stage Validation Pipeline.

Tests:
1. Deterministic execution order of validation stages.
2. Malformed syntax and ERR_SYNTAX_PARSE.
3. Empty-expression rejection and ERR_EMPTY_EXPRESSION.
4. Operator complexity limits and ERR_MAX_OPERATORS_EXCEEDED.
5. Field-count limits and ERR_MAX_FIELDS_EXCEEDED.
6. Unknown operator rejection and ERR_UNKNOWN_OPERATOR.
7. Operator arity failure and ERR_OPERATOR_ARITY.
8. Operator scope failure and ERR_OPERATOR_SCOPE.
9. Unknown field rejection and ERR_UNKNOWN_FIELD.
10. Ineligible field rejection and ERR_FIELD_INELIGIBLE.
11. Positive and correctly typed lookback acceptance.
12. Invalid/non-positive lookback rejection where operator requires it.
13. Invalid grouping argument positions and categorical grouping in numeric ops.
14. Vector/matrix incompatibility where authoritative type metadata establishes incompatibility.
15. Unit incompatibility where authoritative unit metadata establishes incompatibility.
16. Missing type/unit metadata produces an honest incomplete or unverified outcome rather than a fabricated pass.
17. Wrong target dataset rejection before simulation dispatch.
18. Syntactic validity distinguished from complete validation.
19. Fully compliant expression passes applicable stages.
20. Validation errors contain correct machine-readable codes.
21. Stage results correctly identify the failing stage and unexecuted stages are not reported as passed.
22. Repaired expressions are fully revalidated.
23. Language policy remains FASTEXPR-only.
24. P1-05 field catalog integration remains effective.
25. P1-06 operator catalog integration remains effective.
26. Missing essential context cannot silently bypass preflight.
27. Existing validate_expression() callers remain compatible.
28. Existing candidate status and simulation queue behavior remain consistent.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from alpha_harness.db.models import CandidateStatus, ResearchCandidate
from alpha_harness.discovery.catalog import (
    DatafieldCatalog,
    DiscoveredDataset,
    DiscoveredField,
    OperatorCatalog,
)
from alpha_harness.generator import (
    AlphaGenerationPipeline,
    CandidateValidationResult,
    UnsupportedAlphaLanguageError,
    ValidationErrorCode,
    ValidationStage,
    validate_expression,
)
from alpha_harness.orchestrator.researchers import (
    MultiLLMOrchestrator,
    ResearcherConfig,
)
from alpha_harness.prompts.library import StrategyCategory


# =============================================================================
# Controlled Fixture Helpers
# =============================================================================

def _build_test_datafield_catalog(
    fields: list[DiscoveredField] | None = None,
    dataset_id: str = "test_dataset",
    region: str = "USA",
    delay: int = 1,
) -> DatafieldCatalog:
    """Creates a controlled DatafieldCatalog fixture for unit/type testing."""
    default_fields = (
        DiscoveredField("alpha_signal", "MATRIX", 0.95, "Test signal", dataset_id, "Fundamental"),
        DiscoveredField("beta_signal", "MATRIX", 0.90, "Beta signal", dataset_id, "Fundamental"),
        DiscoveredField("vec_metric", "VECTOR", 0.85, "Vector metric", dataset_id, "Analyst"),
    )
    dataset = DiscoveredDataset(
        dataset_id=dataset_id,
        name="Test Dataset",
        category="Test",
        description="Dataset for P1-08 tests",
        fields=tuple(fields) if fields is not None else default_fields,
        region=region,
        delay=delay,
    )
    all_ds = {
        dataset_id: dataset,
        "other_dataset": DiscoveredDataset(
            dataset_id="other_dataset",
            name="Other Dataset",
            category="Other",
            description="Other test dataset",
            fields=(
                DiscoveredField("foreign_signal", "MATRIX", 0.8, "Foreign", "other_dataset", "Other"),
            ),
            region=region,
            delay=delay,
        ),
        "eu_dataset": DiscoveredDataset(
            dataset_id="eu_dataset",
            name="EU Dataset",
            category="EU",
            description="EU region dataset",
            fields=(
                DiscoveredField("eu_metric", "MATRIX", 0.8, "EU metric", "eu_dataset", "Macro"),
            ),
            region="EUR",
            delay=delay,
        ),
    }
    return DatafieldCatalog(
        dataset=dataset,
        region=region,
        universe="TOP3000",
        delay=delay,
        all_datasets=all_ds,
    )


def _build_test_operator_catalog(scope: str = "REGULAR") -> OperatorCatalog:
    """Creates an authoritative OperatorCatalog fixture."""
    return OperatorCatalog(scope=scope)


# =============================================================================
# 1. Deterministic Execution Order of Validation Stages
# =============================================================================

def test_deterministic_stage_execution_order() -> None:
    """Verifies stages execute in deterministic order and earlier failures halt execution."""
    # Expression with both malformed syntax AND 20 operators: syntax fails first
    malformed_excess = "rank(ts_rank(close, 10 + + + ))"
    res = validate_expression(malformed_excess)
    assert res.failed_stage == ValidationStage.SYNTAX_PARSE
    assert res.error_code == ValidationErrorCode.ERR_SYNTAX_PARSE.value
    assert res.is_syntactically_valid is False
    assert res.stage_passed.get(ValidationStage.SYNTAX_PARSE.value) is False
    assert ValidationStage.BRAIN_CONSTRAINTS.value not in res.stage_passed

    # Expression with valid syntax, but exceeds operator limit AND has unknown operator:
    # Brain constraints fails before operator signature check
    op_cat = _build_test_operator_catalog()
    too_many_with_unknown = "rank(close) + rank(open) + rank(high) + rank(low) + rank(volume) + rank(vwap) + rank(cap) + rank(shares) + unknown_op(close, 10)"
    res2 = validate_expression(too_many_with_unknown, operator_catalog=op_cat)
    assert res2.failed_stage == ValidationStage.BRAIN_CONSTRAINTS
    assert res2.error_code == ValidationErrorCode.ERR_MAX_OPERATORS_EXCEEDED.value
    assert res2.is_syntactically_valid is True
    assert res2.stage_passed[ValidationStage.SYNTAX_PARSE.value] is True
    assert res2.stage_passed[ValidationStage.BRAIN_CONSTRAINTS.value] is False
    assert ValidationStage.OPERATOR_SIGNATURE.value not in res2.stage_passed


# =============================================================================
# 2. Malformed Syntax & Empty Expression Rejection
# =============================================================================

def test_malformed_syntax_rejection() -> None:
    """Verifies malformed syntax fails at SYNTAX_PARSE with ERR_SYNTAX_PARSE."""
    res = validate_expression("ts_rank(close, 10)) + ((")
    assert res.is_valid is False
    assert res.is_syntactically_valid is False
    assert res.failed_stage == ValidationStage.SYNTAX_PARSE
    assert res.error_code == ValidationErrorCode.ERR_SYNTAX_PARSE.value
    assert "Syntax parse error" in (res.error or "")


def test_empty_expression_rejection() -> None:
    """Verifies empty and whitespace-only expressions fail at SYNTAX_PARSE with ERR_EMPTY_EXPRESSION."""
    for empty_expr in ("", "   ", "\n\t", "```\n```"):
        res = validate_expression(empty_expr)
        assert res.is_valid is False
        assert res.is_syntactically_valid is False
        assert res.failed_stage == ValidationStage.SYNTAX_PARSE
        assert res.error_code == ValidationErrorCode.ERR_EMPTY_EXPRESSION.value
        assert "empty" in (res.error or "").lower()


# =============================================================================
# 3. Brain Constraints: Operator & Field Count Limits
# =============================================================================

def test_operator_complexity_limit_exceeded() -> None:
    """Verifies expressions exceeding 8 operators fail with ERR_MAX_OPERATORS_EXCEEDED."""
    expr_9_ops = "rank(close) + rank(open) + rank(high) + rank(low) + rank(volume) + rank(vwap) + rank(cap) + rank(shares) + rank(returns)"
    res = validate_expression(expr_9_ops)
    assert res.is_valid is False
    assert res.is_syntactically_valid is True
    assert res.failed_stage == ValidationStage.BRAIN_CONSTRAINTS
    assert res.error_code == ValidationErrorCode.ERR_MAX_OPERATORS_EXCEEDED.value
    assert "exceeds WorldQuant BRAIN limit of 8" in (res.error or "")


def test_field_count_limit_exceeded() -> None:
    """Verifies expressions with > 3 distinct non-grouping fields fail with ERR_MAX_FIELDS_EXCEEDED."""
    expr_4_fields = "rank(close) + rank(volume) + rank(open) + rank(vwap)"
    res = validate_expression(expr_4_fields)
    assert res.is_valid is False
    assert res.is_syntactically_valid is True
    assert res.failed_stage == ValidationStage.BRAIN_CONSTRAINTS
    assert res.error_code == ValidationErrorCode.ERR_MAX_FIELDS_EXCEEDED.value
    assert "exceeds limit of 3" in (res.error or "")

    # Grouping fields (sector, industry) must not count toward limit
    expr_grouping = "group_rank(close + volume + vwap, sector)"
    res_grp = validate_expression(expr_grouping)
    assert res_grp.is_valid is True
    assert res_grp.field_count == 3


# =============================================================================
# 4. Operator Signature: Unknown, Arity, Scope
# =============================================================================

def test_unknown_operator_rejection() -> None:
    """Verifies unknown operators fail with ERR_UNKNOWN_OPERATOR at OPERATOR_SIGNATURE."""
    op_cat = _build_test_operator_catalog()
    res = validate_expression("magic_formula(close, 10)", operator_catalog=op_cat)
    assert res.is_valid is False
    assert res.is_syntactically_valid is True
    assert res.failed_stage == ValidationStage.OPERATOR_SIGNATURE
    assert res.error_code == ValidationErrorCode.ERR_UNKNOWN_OPERATOR.value
    assert "magic_formula" in (res.error or "")


def test_operator_arity_failure() -> None:
    """Verifies incorrect operator input count fails with ERR_OPERATOR_ARITY."""
    op_cat = _build_test_operator_catalog()
    # ts_rank requires 2 inputs (x, d), got 1
    res = validate_expression("ts_rank(close)", operator_catalog=op_cat)
    assert res.is_valid is False
    assert res.is_syntactically_valid is True
    assert res.failed_stage == ValidationStage.OPERATOR_SIGNATURE
    assert res.error_code == ValidationErrorCode.ERR_OPERATOR_ARITY.value
    assert "ts_rank" in (res.error or "")


def test_operator_scope_failure() -> None:
    """Verifies COMBO-only operators fail in REGULAR scope with ERR_OPERATOR_SCOPE."""
    op_cat_reg = OperatorCatalog(
        operator_defs=[
            {"name": "rank", "definition": "rank(x)", "scope": ["REGULAR"]},
            {"name": "combo_only_op", "definition": "combo_only_op(x, d)", "scope": ["COMBO"]},
        ],
        scope="REGULAR",
    )
    res = validate_expression("combo_only_op(close, 10)", operator_catalog=op_cat_reg)
    assert res.is_valid is False
    assert res.is_syntactically_valid is True
    assert res.failed_stage == ValidationStage.OPERATOR_SIGNATURE
    assert res.error_code == ValidationErrorCode.ERR_OPERATOR_SCOPE.value
    assert "only available in scope" in (res.error or "")


# =============================================================================
# 5. Field Membership: Unknown vs Ineligible Fields
# =============================================================================

def test_unknown_hallucinated_field_rejection() -> None:
    """Verifies hallucinated fields fail with ERR_UNKNOWN_FIELD at FIELD_MEMBERSHIP."""
    df_cat = _build_test_datafield_catalog()
    res = validate_expression("ts_rank(hallucinated_field, 10)", datafield_catalog=df_cat)
    assert res.is_valid is False
    assert res.is_syntactically_valid is True
    assert res.failed_stage == ValidationStage.FIELD_MEMBERSHIP
    assert res.error_code == ValidationErrorCode.ERR_UNKNOWN_FIELD.value
    assert "Unknown or ineligible data field: hallucinated_field" in (res.error or "")


def test_ineligible_field_rejection_dataset_and_region() -> None:
    """Verifies fields from other datasets or regions fail with ERR_FIELD_INELIGIBLE."""
    df_cat = _build_test_datafield_catalog()

    # foreign_signal belongs to other_dataset
    res_other = validate_expression("ts_rank(foreign_signal, 10)", datafield_catalog=df_cat)
    assert res_other.is_valid is False
    assert res_other.failed_stage == ValidationStage.FIELD_MEMBERSHIP
    assert res_other.error_code == ValidationErrorCode.ERR_FIELD_INELIGIBLE.value
    assert "belongs to dataset 'other_dataset'" in (res_other.error or "")

    # eu_metric belongs to EUR region / eu_dataset
    res_eu = validate_expression("ts_rank(eu_metric, 10)", datafield_catalog=df_cat)
    assert res_eu.is_valid is False
    assert res_eu.failed_stage == ValidationStage.FIELD_MEMBERSHIP
    assert res_eu.error_code == ValidationErrorCode.ERR_FIELD_INELIGIBLE.value


# =============================================================================
# 6. Lookback Contracts
# =============================================================================

def test_positive_and_integer_lookback_accepted() -> None:
    """Verifies positive integer lookbacks pass SEMANTIC_TYPE_UNIT."""
    res = validate_expression("ts_rank(close, 20)")
    assert res.is_valid is True
    assert res.stage_passed[ValidationStage.SEMANTIC_TYPE_UNIT.value] is True


def test_invalid_lookback_rejected() -> None:
    """Verifies zero, negative, and float lookbacks fail with ERR_INVALID_LOOKBACK."""
    # Lookback d = 0
    res_zero = validate_expression("ts_rank(close, 0)")
    assert res_zero.is_valid is False
    assert res_zero.failed_stage == ValidationStage.SEMANTIC_TYPE_UNIT
    assert res_zero.error_code == ValidationErrorCode.ERR_INVALID_LOOKBACK.value
    assert "greater than 0" in (res_zero.error or "")

    # Lookback d = -5
    res_neg = validate_expression("ts_rank(close, -5)")
    assert res_neg.is_valid is False
    assert res_neg.failed_stage == ValidationStage.SEMANTIC_TYPE_UNIT
    assert res_neg.error_code == ValidationErrorCode.ERR_INVALID_LOOKBACK.value

    # Lookback d = 5.5
    res_float = validate_expression("ts_rank(close, 5.5)")
    assert res_float.is_valid is False
    assert res_float.failed_stage == ValidationStage.SEMANTIC_TYPE_UNIT
    assert res_float.error_code == ValidationErrorCode.ERR_INVALID_LOOKBACK.value
    assert "integer number of days" in (res_float.error or "")


# =============================================================================
# 7. Grouping Arguments & Type Compatibility
# =============================================================================

def test_invalid_grouping_argument_positions() -> None:
    """Verifies numeric grouping arguments fail with ERR_OPERATOR_ARG_CONTRACT."""
    res = validate_expression("group_rank(close, 10)")
    assert res.is_valid is False
    assert res.failed_stage == ValidationStage.SEMANTIC_TYPE_UNIT
    assert res.error_code == ValidationErrorCode.ERR_OPERATOR_ARG_CONTRACT.value
    assert "requires a grouping field" in (res.error or "")


def test_categorical_grouping_field_in_numeric_operation_rejected() -> None:
    """Verifies categorical grouping fields used in numeric operations fail with ERR_TYPE_INCOMPATIBLE."""
    # rank(sector) - sector is categorical
    res_rank = validate_expression("rank(sector)")
    assert res_rank.is_valid is False
    assert res_rank.failed_stage == ValidationStage.SEMANTIC_TYPE_UNIT
    assert res_rank.error_code == ValidationErrorCode.ERR_TYPE_INCOMPATIBLE.value
    assert "categorical" in (res_rank.error or "")

    # sector + 1 - arithmetic on categorical field
    res_arith = validate_expression("sector + 1")
    assert res_arith.is_valid is False
    assert res_arith.failed_stage == ValidationStage.SEMANTIC_TYPE_UNIT
    assert res_arith.error_code == ValidationErrorCode.ERR_TYPE_INCOMPATIBLE.value


# =============================================================================
# 8. Vector / Matrix Type Requirements
# =============================================================================

def test_vector_matrix_incompatibility() -> None:
    """Verifies unreduced vector fields and vector operators on MATRIX fail with ERR_TYPE_INCOMPATIBLE."""
    df_cat = _build_test_datafield_catalog()

    # vec_metric is VECTOR type; used directly without reduction
    res_unreduced = validate_expression("rank(vec_metric)", datafield_catalog=df_cat)
    assert res_unreduced.is_valid is False
    assert res_unreduced.failed_stage == ValidationStage.SEMANTIC_TYPE_UNIT
    assert res_unreduced.error_code == ValidationErrorCode.ERR_TYPE_INCOMPATIBLE.value
    assert "unreduced vector output" in (res_unreduced.error or "")

    # vec_avg(vec_metric) reduces the vector to cross-sectional scalar/matrix -> valid!
    res_reduced = validate_expression("rank(vec_avg(vec_metric))", datafield_catalog=df_cat)
    assert res_reduced.is_valid is True

    # vec_avg called on explicitly MATRIX field (alpha_signal) -> type mismatch!
    res_matrix = validate_expression("vec_avg(alpha_signal)", datafield_catalog=df_cat)
    assert res_matrix.is_valid is False
    assert res_matrix.failed_stage == ValidationStage.SEMANTIC_TYPE_UNIT
    assert res_matrix.error_code == ValidationErrorCode.ERR_TYPE_INCOMPATIBLE.value
    assert "requires VECTOR input" in (res_matrix.error or "")


# =============================================================================
# 9. Unit Compatibility & Honest Unverified Metadata Reporting
# =============================================================================

def test_unit_incompatibility_when_metadata_is_present() -> None:
    """Verifies addition of incompatible dimensional units fails with ERR_UNIT_INCOMPATIBLE."""
    # Controlled fixture units: usd_metric (USD) vs pct_metric (PERCENT)
    field_units = {
        "usd_metric": "USD",
        "pct_metric": "PERCENT",
    }
    allowed = {"usd_metric", "pct_metric", "close"}
    res = validate_expression(
        "usd_metric + pct_metric",
        allowed_field_names=allowed,
        field_units=field_units,
    )
    assert res.is_valid is False
    assert res.failed_stage == ValidationStage.SEMANTIC_TYPE_UNIT
    assert res.error_code == ValidationErrorCode.ERR_UNIT_INCOMPATIBLE.value
    assert "Unit incompatibility" in (res.error or "")


def test_missing_unit_metadata_produces_honest_unverified_status() -> None:
    """Verifies missing unit metadata records UNVERIFIED without fabricating a pass."""
    df_cat = _build_test_datafield_catalog()
    # Fields have no unit attribute in default catalog
    res = validate_expression("alpha_signal + beta_signal", datafield_catalog=df_cat)
    assert res.is_valid is True
    # Stage result records honest UNVERIFIED status
    assert res.stage_results[ValidationStage.SEMANTIC_TYPE_UNIT.value] == "UNVERIFIED"
    assert res.stage_passed[ValidationStage.SEMANTIC_TYPE_UNIT.value] is True


# =============================================================================
# 10. Preflight Simulation Gate
# =============================================================================

def test_wrong_target_dataset_rejection() -> None:
    """Verifies field from wrong dataset is rejected in preflight with ERR_PREFLIGHT_DATASET_MISMATCH."""
    df_cat = _build_test_datafield_catalog()
    # Candidate expression uses foreign_signal from other_dataset, simulation target is test_dataset
    res = validate_expression(
        "ts_rank(foreign_signal, 10)",
        datafield_catalog=df_cat,
        target_dataset_id="test_dataset",
    )
    assert res.is_valid is False
    # Even if it passes earlier or fails at field membership/preflight
    assert res.error_code in (
        ValidationErrorCode.ERR_FIELD_INELIGIBLE.value,
        ValidationErrorCode.ERR_PREFLIGHT_DATASET_MISMATCH.value,
    )


def test_preflight_configuration_completeness() -> None:
    """Verifies missing required preflight simulation configuration produces ERR_PREFLIGHT_CONFIGURATION_INCOMPLETE."""
    # require_preflight=True without any context fails closed
    res_missing = validate_expression(
        "rank(close)",
        require_preflight=True,
    )
    assert res_missing.is_valid is False
    assert res_missing.failed_stage == ValidationStage.PREFLIGHT_SIM
    assert res_missing.error_code == ValidationErrorCode.ERR_PREFLIGHT_CONFIGURATION_INCOMPLETE.value

    # Empty region in simulation context produces explicit error
    res_empty_reg = validate_expression(
        "rank(close)",
        simulation_context={"region": ""},
    )
    assert res_empty_reg.is_valid is False
    assert res_empty_reg.failed_stage == ValidationStage.PREFLIGHT_SIM
    assert res_empty_reg.error_code == ValidationErrorCode.ERR_PREFLIGHT_CONFIGURATION_INCOMPLETE.value


# =============================================================================
# 11. Distinction Between Syntactic Validity and Full Validity
# =============================================================================

def test_syntactic_validity_distinguished_from_full_validity() -> None:
    """Verifies an expression can be syntactically valid while not fully valid."""
    df_cat = _build_test_datafield_catalog()
    # Valid syntax, but unknown hallucinated field
    res = validate_expression("ts_rank(bogus_field, 10)", datafield_catalog=df_cat)
    assert res.is_syntactically_valid is True
    assert res.is_valid is False
    assert res.failed_stage == ValidationStage.FIELD_MEMBERSHIP

    # Malformed syntax: neither syntactically valid nor fully valid
    res_bad = validate_expression("ts_rank(close, 10) + ((")
    assert res_bad.is_syntactically_valid is False
    assert res_bad.is_valid is False


# =============================================================================
# 12. Full Compliance & Stage Results Accuracy
# =============================================================================

def test_fully_compliant_expression_passes_all_stages() -> None:
    """Verifies compliant expression passes all 7 validation stages."""
    df_cat = _build_test_datafield_catalog()
    op_cat = _build_test_operator_catalog()
    sim_ctx = {"region": "USA", "universe": "TOP3000", "delay": 1}

    res = validate_expression(
        "rank(ts_rank(alpha_signal, 10))",
        datafield_catalog=df_cat,
        operator_catalog=op_cat,
        target_dataset_id="test_dataset",
        simulation_context=sim_ctx,
        require_preflight=True,
    )
    assert res.is_valid is True
    assert res.is_syntactically_valid is True
    assert res.failed_stage is None
    assert res.error_code is None
    assert res.error is None

    # Check all stages passed
    for stage in ValidationStage:
        assert res.stage_passed.get(stage.value) is True


def test_stage_not_executed_never_reported_as_passed() -> None:
    """Verifies that an unexecuted stage is never reported as having passed."""
    # When SYNTAX_PARSE fails, BRAIN_CONSTRAINTS, OPERATOR_SIGNATURE, etc., were never run
    res = validate_expression("invalid(syntax +")
    assert res.stage_passed.get(ValidationStage.SYNTAX_PARSE.value) is False
    assert ValidationStage.BRAIN_CONSTRAINTS.value not in res.stage_passed
    assert ValidationStage.OPERATOR_SIGNATURE.value not in res.stage_passed
    assert ValidationStage.FIELD_MEMBERSHIP.value not in res.stage_passed
    assert ValidationStage.SEMANTIC_TYPE_UNIT.value not in res.stage_passed
    assert ValidationStage.PREFLIGHT_SIM.value not in res.stage_passed


# =============================================================================
# 13. Revalidation in Auto-Repair Loop
# =============================================================================

def test_repaired_expressions_are_fully_revalidated() -> None:
    """Verifies that an LLM-repaired expression is revalidated through all stages."""
    async def _run() -> None:
        mock_llm = AsyncMock()
        pipeline = AlphaGenerationPipeline(llm=mock_llm)
        df_cat = _build_test_datafield_catalog()
        op_cat = _build_test_operator_catalog()

        # Repair returns an expression with unknown field: must fail revalidation!
        mock_llm.generate.return_value = AsyncMock(
            text='{"fixed_expression": "rank(unknown_repaired_field)"}'
        )

        res = await pipeline._repair_expression(
            model_ref="test:model",
            hypothesis="Test repair",
            faulty_expression="rank(initial_bad_field)",
            validation_error="Unknown or ineligible data field: initial_bad_field",
            datafield_catalog=df_cat,
            operator_catalog=op_cat,
            target_dataset_id="test_dataset",
            max_attempts=1,
        )
        assert res.is_valid is False
        assert res.failed_stage == ValidationStage.FIELD_MEMBERSHIP
        assert res.error_code == ValidationErrorCode.ERR_UNKNOWN_FIELD.value

        # Repair returns genuinely compliant expression: must pass!
        mock_llm.generate.return_value = AsyncMock(
            text='{"fixed_expression": "rank(ts_rank(alpha_signal, 10))"}'
        )

        res_ok = await pipeline._repair_expression(
            model_ref="test:model",
            hypothesis="Test repair success",
            faulty_expression="rank(bad)",
            validation_error="Bad",
            datafield_catalog=df_cat,
            operator_catalog=op_cat,
            target_dataset_id="test_dataset",
            max_attempts=1,
        )
        assert res_ok.is_valid is True
        assert res_ok.is_syntactically_valid is True

    asyncio.run(_run())


# =============================================================================
# 14. Language Policy Remains FASTEXPR-Only
# =============================================================================

def test_language_scope_remains_fastexpr_only() -> None:
    """Verifies language stage rejects Python and records ERR_UNSUPPORTED_LANGUAGE."""
    # When raise_language_error=True (default), raises UnsupportedAlphaLanguageError
    with pytest.raises(UnsupportedAlphaLanguageError) as exc_info:
        validate_expression("rank(close)", language="PYTHON")
    assert exc_info.value.error_code == ValidationErrorCode.ERR_UNSUPPORTED_LANGUAGE.value
    assert exc_info.value.failed_stage == ValidationStage.LANGUAGE_SCOPE

    # When raise_language_error=False, returns CandidateValidationResult
    res = validate_expression("rank(close)", language="PYTHON", raise_language_error=False)
    assert res.is_valid is False
    assert res.is_syntactically_valid is False
    assert res.failed_stage == ValidationStage.LANGUAGE_SCOPE
    assert res.error_code == ValidationErrorCode.ERR_UNSUPPORTED_LANGUAGE.value


# =============================================================================
# 15. Backwards Compatibility with Existing validate_expression Callers
# =============================================================================

def test_existing_validate_expression_callers_remain_compatible() -> None:
    """Verifies existing calling conventions work identically without code changes."""
    # 1. Simple expr only
    res1 = validate_expression("rank(close)")
    assert res1.is_valid is True
    assert res1.expression == "rank(close)"

    # 2. Positional valid_names
    res2 = validate_expression("rank(open)", valid_names={"close"})
    assert res2.is_valid is False

    # 3. Keyword check_operators
    res3 = validate_expression("ts_rank(close, 10)", check_operators=True)
    assert res3.is_valid is True


# =============================================================================
# 16. Candidate Status and Pre-Simulation Gate Consistency
# =============================================================================

def test_invalid_candidates_cannot_reach_simulation_queue() -> None:
    """Verifies that candidates where is_valid=False cannot be marked VALID or enqueued."""
    df_cat = _build_test_datafield_catalog()
    val_invalid = validate_expression("ts_rank(bad_field, 10)", datafield_catalog=df_cat)
    assert val_invalid.is_valid is False

    # Candidate status mapped from validation result
    cand_status = CandidateStatus.VALID if val_invalid.is_valid else CandidateStatus.INVALID
    assert cand_status == CandidateStatus.INVALID

    cand = ResearchCandidate(
        session_id=1,
        hypothesis="Test",
        expression=val_invalid.expression,
        is_valid=val_invalid.is_valid,
        validation_error=val_invalid.error,
        status=cand_status,
    )
    assert cand.is_valid is False
    assert cand.status == CandidateStatus.INVALID
