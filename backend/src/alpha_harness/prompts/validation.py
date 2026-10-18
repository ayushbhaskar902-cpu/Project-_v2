"""P1-13: Prompt evolution validation, controlled shadow benchmarking, and deterministic rollback.

Key statistical guarantees:
1. Every newly evolved prompt version starts as EXPERIMENTAL (is_active = False).
2. Controlled matched-cohort benchmarking against the incumbent baseline.
3. Predeclared non-inferiority gate: LCB95(Delta) > -delta where Delta = SR_exp - SR_inc.
4. Independent absolute validation Sharpe gate: SR_exp > 0 (not confused with non-inferiority).
5. AST validity gate (>= 85%) and token budget compliance.
6. Multiplicity correction (Bonferroni) when multiple experimental prompt variants are evaluated.
7. Atomic promotion and deterministic rollback preserving full version lineage and audit logs.
8. Strict P0-03 holdout quarantine: zero holdout data in benchmarking or promotion decisions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import func, select, update

from ..db.models import (
    PromptLifecycleEvent,
    PromptStrategy,
    PromptVersion,
    ResearchCandidate,
    ResearchEvaluation,
    utcnow,
)
from ..prompts.library import normalize_category
from ..prompts.versioning import (
    compute_prompt_content_hash,
    normalize_prompt_components,
)

if TYPE_CHECKING:
    from ..db.sqlite import Database

log = structlog.get_logger(__name__)


class PromptLifecycleState(StrEnum):
    EXPERIMENTAL = "EXPERIMENTAL"
    SHADOW_TESTING = "SHADOW_TESTING"
    PROMOTED = "PROMOTED"
    REJECTED = "REJECTED"
    ROLLED_BACK = "ROLLED_BACK"


@dataclass(slots=True)
class PredeclaredPromotionCriteria:
    """Predeclared promotion gates specified before examining experiment results."""

    primary_metric: str = "val_sharpe"
    observation_window_years: float = 4.0
    non_inferiority_margin: float = 0.15  # delta > 0, requirement: LCB95(Delta) > -delta
    min_sample_size: int = 5              # Minimum evaluated candidates per cohort
    min_ast_valid_rate: float = 0.85      # Audited 85% threshold
    max_token_ratio: float = 1.25         # Avg tokens per candidate <= 1.25x incumbent baseline
    require_positive_sharpe: bool = True  # Distinct absolute gate: SR_exp > 0
    confidence_level: float = 0.95
    required_provenance: str | None = None
    allow_missing_metrics: bool = False
    require_matched_cohorts: bool = True


@dataclass(slots=True)
class PromptValidationResult:
    """Auditable statistical evidence and decision for a prompt promotion evaluation."""

    decision: str  # "PROMOTED", "REJECTED", "INSUFFICIENT_EVIDENCE", "RETAIN_EXPERIMENTAL", "INCOMPATIBLE_COHORTS"
    is_promoted: bool
    category: str
    experimental_strategy_id: int
    incumbent_strategy_id: int | None
    delta_sharpe: float | None = None
    delta_se: float | None = None
    lcb95_delta: float | None = None
    non_inferiority_margin: float = 0.15
    non_inferiority_passed: bool = False
    absolute_sharpe_passed: bool = False
    ast_valid_rate_passed: bool = False
    token_budget_passed: bool = False
    sample_size_passed: bool = False
    provenance_passed: bool = True
    unmatched_evaluations_count: int = 0
    missing_metrics_count: int = 0
    reasons: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


def _normal_quantile(p: float) -> float:
    """Rational approximation for inverse standard normal CDF (Acklam / Abramowitz & Stegun)."""
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p must be in (0, 1)")
    # For common confidence levels:
    if abs(p - 0.975) < 1e-4:
        return 1.95996
    if abs(p - 0.95) < 1e-4:
        return 1.64485
    if abs(p - 0.995) < 1e-4:
        return 2.57583
    if abs(p - 0.99) < 1e-4:
        return 2.32635

    # Abramowitz and Stegun 26.2.23 approximation
    if p > 0.5:
        q = 1.0 - p
        sign = 1.0
    else:
        q = p
        sign = -1.0
    t = math.sqrt(-2.0 * math.log(q))
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308
    return sign * (t - ((c2 * t + c1) * t + c0) / (((d3 * t + d2) * t + d1) * t + 1.0))


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for regularized incomplete beta function (Lentz's method)."""
    max_iter = 200
    eps = 3.0e-14
    fp_min = 1.0e-30
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fp_min:
        d = fp_min
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fp_min:
            d = fp_min
        c = 1.0 + aa / c
        if abs(c) < fp_min:
            c = fp_min
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fp_min:
            d = fp_min
        c = 1.0 + aa / c
        if abs(c) < fp_min:
            c = fp_min
        d = 1.0 / d
        delh = d * c
        h *= delh
        if abs(delh - 1.0) <= eps:
            break
    return h


def _ibeta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    bt = math.exp(-lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _student_t_cdf(t_val: float, df: int) -> float:
    """CDF of Student's t distribution with df degrees of freedom."""
    if t_val == 0.0:
        return 0.5
    x = df / (df + t_val * t_val)
    prob_tail = 0.5 * _ibeta(df / 2.0, 0.5, x)
    return 1.0 - prob_tail if t_val > 0 else prob_tail


def _student_t_quantile_fallback(p: float, df: int) -> float:
    """High-precision numerical inversion of Student's t CDF via binary search and Halley/Newton steps."""
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -_student_t_quantile_fallback(1.0 - p, df)

    # Initial bounds based on Cornish-Fisher or Cauchy/Normal
    z = _normal_quantile(p)
    low = max(0.0, z * 0.5)
    high = max(100.0, z * math.sqrt(df))
    # Expand upper bound if needed
    while _student_t_cdf(high, df) < p and high < 1e6:
        high *= 2.0

    # Binary search for robust convergence to machine precision
    for _ in range(80):
        mid = (low + high) / 2.0
        if _student_t_cdf(mid, df) < p:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def student_t_quantile(p: float, df: int) -> float:
    """Exact Student's t inverse CDF with high-precision incomplete beta inversion fallback.

    Used when sample variance is estimated from small matched-cohort samples.
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p must be in (0, 1)")
    if df < 1:
        raise ValueError("Degrees of freedom must be >= 1")
    try:
        import scipy.stats as st

        return float(st.t.ppf(p, df))
    except Exception:
        return _student_t_quantile_fallback(p, df)


def compute_matched_non_inferiority(
    experimental_sharpes: list[float],
    incumbent_sharpes: list[float],
    margin: float = 0.15,
    num_variants_tested: int = 1,
) -> tuple[float, float, float, bool]:
    """Computes matched-cohort Sharpe difference Delta and Student's t lower confidence bound LCB95(Delta).

    Delta_i = SR_experimental,i - SR_incumbent,i
    Delta_bar = mean(Delta_i)
    SE(Delta_bar) = s_delta / sqrt(N)
    LCB95(Delta) = Delta_bar - t_crit * SE(Delta_bar)
    Non-inferiority: LCB95(Delta) > -margin

    Statistical Design:
    - Matched Design: Pairs observations strictly on shared cohorts (N = len(pairs)).
    - Degrees of Freedom: df = N - 1 for N matched pairs.
    - Student's t Critical Value: When variance is estimated from a small sample of paired differences,
      uses Student's t critical value t_crit rather than a normal critical value.
    - Multiple Comparisons: Bonferroni-adjusted significance level alpha_adj = 0.05 / M
      where M is the candidate comparison family count, preserving two-sided confidence interval
      lower endpoint convention: p_tail = 1.0 - (alpha_adj / 2.0).
    """
    n_pairs = min(len(experimental_sharpes), len(incumbent_sharpes))

    if n_pairs < 2:
        return 0.0, float("inf"), float("-inf"), False

    diffs = [
        float(e) - float(i)
        for e, i in zip(experimental_sharpes[:n_pairs], incumbent_sharpes[:n_pairs], strict=False)
    ]
    delta = round(sum(diffs) / n_pairs, 4)
    var_d = sum((d - delta) ** 2 for d in diffs) / (n_pairs - 1)
    se_delta = round(math.sqrt(max(1e-9, var_d) / n_pairs), 4)

    # Multiplicity correction across M variants in family
    m = max(1, num_variants_tested)
    alpha = 0.05 / m
    p_tail = 1.0 - (alpha / 2.0)
    df = n_pairs - 1
    t_crit = student_t_quantile(p_tail, df=df)

    lcb = round(delta - t_crit * se_delta, 4)
    non_inferiority_passed = lcb > -margin

    return delta, se_delta, lcb, non_inferiority_passed




def _extract_cohort_key(c: dict[str, Any]) -> tuple:
    """Extracts comparability metadata tuple for matched-cohort benchmarking."""
    task = c.get("task") or c.get("specification") or c.get("task_id")
    dataset = c.get("dataset_id") or c.get("dataset")
    region = c.get("region")
    universe = c.get("universe")
    delay = c.get("delay")
    sim_settings = str(c.get("settings_hash") or c.get("simulation_settings") or "")
    val_window = (c.get("val_start_date"), c.get("val_end_date"))
    provenance = c.get("provenance")
    return (task, dataset, region, universe, delay, sim_settings, val_window, provenance)


class PromptValidationService:
    """Manages prompt validation lifecycle, controlled tournaments, promotion gates, and rollback."""

    def __init__(self, db: Database) -> None:
        self.db = db

    async def create_experimental_version(
        self,
        *,
        session_id: int,
        category: str,
        template_text: str,
        guidance: dict[str, Any] | str | None = None,
        examples: list[Any] | None = None,
        anti_patterns: list[Any] | None = None,
        authoring_source: str = "EVOLUTION",
        parent_strategy_id: int | None = None,
    ) -> tuple[PromptStrategy, PromptVersion]:
        """Creates an experimental prompt version.

        Guarantees:
        - Starts as EXPERIMENTAL (is_active = False).
        - Cannot silently replace the active promoted version.
        - Immutable snapshot persisted with cryptographic content_hash.
        - Traceable lifecycle event recorded.
        """
        cat_norm = normalize_category(category)

        t_text, n_guidance, n_examples, n_anti_patterns = normalize_prompt_components(
            template_text=template_text,
            guidance=guidance,
            examples=examples,
            anti_patterns=anti_patterns,
        )
        c_hash = compute_prompt_content_hash(
            template_text=t_text,
            guidance=n_guidance,
            examples=n_examples,
            anti_patterns=n_anti_patterns,
        )

        async with self.db.session() as session:
            # Query highest version for this category/session
            v_stmt = (
                select(func.max(PromptStrategy.version))
                .where(
                    PromptStrategy.session_id == session_id,
                    func.lower(func.trim(PromptStrategy.category)) == cat_norm,
                )
            )
            v_res = await session.execute(v_stmt)
            curr_max = v_res.scalar() or 0
            new_version = curr_max + 1

            new_strategy = PromptStrategy(
                session_id=session_id,
                name=f"Experimental {cat_norm.title()} v{new_version}",
                version=new_version,
                category=cat_norm,
                parent_id=parent_strategy_id,
                template_text=t_text,
                guidance=guidance if isinstance(guidance, dict) else {},
                examples=n_examples,
                anti_patterns=n_anti_patterns,
                is_active=False,  # EXPERIMENTAL PROMPTS MUST NOT BE ACTIVE IN PRODUCTION
                validation_status=PromptLifecycleState.EXPERIMENTAL.value,
            )
            session.add(new_strategy)
            await session.flush()

            new_pv = PromptVersion(
                origin_session_id=session_id,
                origin_strategy_id=new_strategy.id,
                category=cat_norm,
                version=new_version,
                authoring_source=authoring_source,
                template_text=t_text,
                guidance=n_guidance,
                examples=n_examples,
                anti_patterns=n_anti_patterns,
                content_hash=c_hash,
            )
            session.add(new_pv)
            await session.flush()

            new_strategy.current_prompt_version_id = new_pv.id

            event = PromptLifecycleEvent(
                session_id=session_id,
                category=cat_norm,
                strategy_id=new_strategy.id,
                prompt_version_id=new_pv.id,
                event_type="CREATED_EXPERIMENTAL",
                from_status=None,
                to_status=PromptLifecycleState.EXPERIMENTAL.value,
                reason="Evolved prompt candidate generated; pending controlled benchmark validation.",
            )
            session.add(event)
            await session.commit()

        log.info(
            "prompt.experimental_created",
            session_id=session_id,
            category=cat_norm,
            strategy_id=new_strategy.id,
            version=new_version,
        )
        return new_strategy, new_pv

    async def evaluate_tournament_and_promote(
        self,
        *,
        session_id: int,
        category: str,
        experimental_strategy_id: int,
        experimental_candidates: list[dict[str, Any]],
        incumbent_candidates: list[dict[str, Any]],
        criteria: PredeclaredPromotionCriteria | None = None,
        num_variants_tested: int = 1,
    ) -> PromptValidationResult:
        """Evaluates an experimental prompt against the incumbent on matched validation cohorts.

        Strict P0-03 Holdout Isolation:
        - Evaluates exclusively on validation partition data (val_sharpe, AST syntax valid rate).
        - Rejects any presence of holdout metrics.
        - Strict matched-cohort verification and conservative handling of missing/invalid provenance.
        """
        crit = criteria or PredeclaredPromotionCriteria()
        cat_norm = normalize_category(category)

        # 1. Strict P0-03 Holdout Isolation Check
        for c in experimental_candidates + incumbent_candidates:
            if any("holdout" in k.lower() for k in c):
                raise ValueError("Holdout data cannot influence prompt promotion decisions (P0-03)")

        # 2. Conservative handling of missing metrics and provenance
        exp_valid_candidates: list[dict[str, Any]] = []
        inc_valid_candidates: list[dict[str, Any]] = []
        missing_metrics_count = 0
        provenance_failures = 0

        for c in experimental_candidates:
            val_metric = c.get("val_sharpe")
            prov = c.get("provenance")
            if val_metric is None:
                missing_metrics_count += 1
                if not crit.allow_missing_metrics:
                    continue
            if prov in ("UNVERIFIED", "CORRUPTED", "PROVENANCE_FAILED"):
                provenance_failures += 1
                continue
            if crit.required_provenance and prov != crit.required_provenance:
                provenance_failures += 1
                continue
            exp_valid_candidates.append(c)

        for c in incumbent_candidates:
            val_metric = c.get("val_sharpe")
            prov = c.get("provenance")
            if val_metric is None:
                missing_metrics_count += 1
                if not crit.allow_missing_metrics:
                    continue
            if prov in ("UNVERIFIED", "CORRUPTED", "PROVENANCE_FAILED"):
                provenance_failures += 1
                continue
            if crit.required_provenance and prov != crit.required_provenance:
                provenance_failures += 1
                continue
            inc_valid_candidates.append(c)

        # 3. Matched-Cohort Stratification and Pairing
        unmatched_evaluations_count = 0
        unmatched_reasons: list[str] = []

        has_cohort_tags = any(
            any(c.get(k) is not None for k in ("task", "specification", "dataset_id", "dataset", "region", "universe"))
            for c in (exp_valid_candidates + inc_valid_candidates)
        )

        paired_exp_candidates: list[dict[str, Any]] = []
        paired_inc_candidates: list[dict[str, Any]] = []

        if crit.require_matched_cohorts and has_cohort_tags:
            inc_cohort_keys = {_extract_cohort_key(c) for c in inc_valid_candidates}
            exp_cohort_keys = {_extract_cohort_key(c) for c in exp_valid_candidates}
            shared_cohorts = inc_cohort_keys.intersection(exp_cohort_keys)

            if not shared_cohorts:
                return PromptValidationResult(
                    decision="INCOMPATIBLE_COHORTS",
                    is_promoted=False,
                    category=cat_norm,
                    experimental_strategy_id=experimental_strategy_id,
                    incumbent_strategy_id=None,
                    sample_size_passed=False,
                    provenance_passed=(provenance_failures == 0),
                    unmatched_evaluations_count=len(experimental_candidates) + len(incumbent_candidates),
                    missing_metrics_count=missing_metrics_count,
                    reasons=["INCOMPATIBLE_COHORTS: Experimental candidates evaluated on incompatible tasks/cohorts with zero incumbent overlap."],
                    evidence={
                        "experimental_cohorts": [str(k) for k in exp_cohort_keys],
                        "incumbent_cohorts": [str(k) for k in inc_cohort_keys],
                    },
                )

            # Pair strictly within each shared cohort
            for ch in sorted(shared_cohorts, key=str):
                ch_exp = [c for c in exp_valid_candidates if _extract_cohort_key(c) == ch]
                ch_inc = [c for c in inc_valid_candidates if _extract_cohort_key(c) == ch]
                pair_count = min(len(ch_exp), len(ch_inc))
                paired_exp_candidates.extend(ch_exp[:pair_count])
                paired_inc_candidates.extend(ch_inc[:pair_count])

                cohort_unmatched = abs(len(ch_exp) - len(ch_inc))
                if cohort_unmatched > 0:
                    unmatched_evaluations_count += cohort_unmatched
                    unmatched_reasons.append(
                        f"Cohort {ch}: {len(ch_exp)} experimental vs {len(ch_inc)} incumbent candidates ({cohort_unmatched} unmatched)."
                    )

            # Record non-shared cohorts as excluded
            non_shared_exp = sum(1 for c in exp_valid_candidates if _extract_cohort_key(c) not in shared_cohorts)
            non_shared_inc = sum(1 for c in inc_valid_candidates if _extract_cohort_key(c) not in shared_cohorts)
            if non_shared_exp > 0:
                unmatched_evaluations_count += non_shared_exp
                unmatched_reasons.append(f"{non_shared_exp} experimental candidates in cohorts with no incumbent counterpart.")
            if non_shared_inc > 0:
                unmatched_evaluations_count += non_shared_inc
                unmatched_reasons.append(f"{non_shared_inc} incumbent candidates in cohorts with no experimental counterpart.")
        else:
            # Parallel list pairing (e.g. synthetic test suites where candidates are aligned arrays)
            pair_count = min(len(exp_valid_candidates), len(inc_valid_candidates))
            paired_exp_candidates = exp_valid_candidates[:pair_count]
            paired_inc_candidates = inc_valid_candidates[:pair_count]
            len_diff = abs(len(exp_valid_candidates) - len(inc_valid_candidates))
            if len_diff > 0:
                unmatched_evaluations_count += len_diff
                unmatched_reasons.append(f"{len_diff} trailing candidates unmatched between arms.")

        # 4. Extract validation metrics from paired cohorts
        exp_sharpes = [float(c["val_sharpe"]) for c in paired_exp_candidates if c.get("val_sharpe") is not None]
        inc_sharpes = [float(c["val_sharpe"]) for c in paired_inc_candidates if c.get("val_sharpe") is not None]
        n_matched_pairs = min(len(exp_sharpes), len(inc_sharpes))

        n_exp = len(experimental_candidates)
        ast_valid_exp = sum(1 for c in experimental_candidates if c.get("is_valid", False) or c.get("ast_valid", False))
        ast_valid_rate = (ast_valid_exp / n_exp) if n_exp > 0 else 0.0

        tokens_exp = [float(c.get("tokens", 0) or 0) for c in experimental_candidates]
        tokens_inc = [float(c.get("tokens", 0) or 0) for c in incumbent_candidates]
        avg_tokens_exp = sum(tokens_exp) / len(tokens_exp) if tokens_exp else 0.0
        avg_tokens_inc = sum(tokens_inc) / len(tokens_inc) if tokens_inc else 0.0

        token_ratio = (avg_tokens_exp / avg_tokens_inc) if avg_tokens_inc > 0 else 1.0

        # Query incumbent strategy
        async with self.db.session() as session:
            inc_stmt = (
                select(PromptStrategy)
                .where(
                    PromptStrategy.session_id == session_id,
                    func.lower(func.trim(PromptStrategy.category)) == cat_norm,
                    PromptStrategy.is_active.is_(True),
                )
            )
            inc_res = await session.execute(inc_stmt)
            inc_strat = inc_res.scalars().first()
            incumbent_id = inc_strat.id if inc_strat else None

        # 5. Check sample size requirement on matched pairs
        reasons: list[str] = []
        sample_size_passed = n_matched_pairs >= crit.min_sample_size
        if not sample_size_passed:
            reasons.append(
                f"INSUFFICIENT_EVIDENCE: Valid matched pairs ({n_matched_pairs}) below required minimum ({crit.min_sample_size})."
            )
            if unmatched_evaluations_count > 0:
                reasons.append(f"Excluded {unmatched_evaluations_count} unmatched evaluations across cohorts.")
            return PromptValidationResult(
                decision="INSUFFICIENT_EVIDENCE",
                is_promoted=False,
                category=cat_norm,
                experimental_strategy_id=experimental_strategy_id,
                incumbent_strategy_id=incumbent_id,
                sample_size_passed=False,
                provenance_passed=(provenance_failures == 0),
                unmatched_evaluations_count=unmatched_evaluations_count,
                missing_metrics_count=missing_metrics_count,
                reasons=reasons,
                evidence={
                    "n_matched_pairs": n_matched_pairs,
                    "min_required": crit.min_sample_size,
                    "unmatched_evaluations": unmatched_evaluations_count,
                    "unmatched_reasons": unmatched_reasons,
                    "missing_metrics": missing_metrics_count,
                },
            )

        # 6. Statistical non-inferiority calculation: LCB95(Delta) > -delta
        delta, se_delta, lcb_delta, non_inf_passed = compute_matched_non_inferiority(
            experimental_sharpes=exp_sharpes[:n_matched_pairs],
            incumbent_sharpes=inc_sharpes[:n_matched_pairs],
            margin=crit.non_inferiority_margin,
            num_variants_tested=num_variants_tested,
        )


        # 7. Independent absolute validation Sharpe gate
        mean_exp_sharpe = sum(exp_sharpes) / len(exp_sharpes)
        if crit.require_positive_sharpe:
            abs_sharpe_passed = mean_exp_sharpe > 0.0
        else:
            abs_sharpe_passed = True

        # 8. AST validity gate (>= 85%)
        ast_passed = ast_valid_rate >= crit.min_ast_valid_rate

        # 9. Token budget gate (<= 1.25x incumbent)
        token_passed = token_ratio <= crit.max_token_ratio

        # 10. Provenance gate
        prov_passed = provenance_failures == 0

        if not non_inf_passed:
            reasons.append(
                f"NON_INFERIORITY_FAILED: LCB95(Delta)={lcb_delta:.4f} <= -margin({-crit.non_inferiority_margin:.4f}). "
                f"Mean delta={delta:.4f}, SE={se_delta:.4f}."
            )
        if not abs_sharpe_passed:
            reasons.append(
                f"ABSOLUTE_SHARPE_FAILED: Mean validation Sharpe ({mean_exp_sharpe:.4f}) is non-positive."
            )
        if not ast_passed:
            reasons.append(
                f"AST_VALIDITY_FAILED: Valid rate ({ast_valid_rate:.1%}) below required {crit.min_ast_valid_rate:.1%}."
            )
        if not token_passed:
            reasons.append(
                f"TOKEN_BUDGET_EXCEEDED: Token ratio ({token_ratio:.2f}x) exceeds allowed {crit.max_token_ratio:.2f}x."
            )
        if not prov_passed:
            reasons.append(
                f"PROVENANCE_FAILED: Observed {provenance_failures} unverified or invalid provenance evaluations."
            )

        all_gates_passed = non_inf_passed and abs_sharpe_passed and ast_passed and token_passed and sample_size_passed and prov_passed

        evidence_dict = {
            "mean_experimental_sharpe": round(mean_exp_sharpe, 4),
            "mean_incumbent_sharpe": round(sum(inc_sharpes) / len(inc_sharpes), 4),
            "delta": delta,
            "se_delta": se_delta,
            "lcb95_delta": lcb_delta,
            "non_inferiority_margin": crit.non_inferiority_margin,
            "df": n_matched_pairs - 1,
            "n_matched_pairs": n_matched_pairs,
            "ast_valid_rate": round(ast_valid_rate, 4),
            "token_ratio": round(token_ratio, 3),
            "num_variants_tested": num_variants_tested,
            "unmatched_evaluations_count": unmatched_evaluations_count,
            "unmatched_reasons": unmatched_reasons,
            "missing_metrics_count": missing_metrics_count,
        }


        if all_gates_passed:
            await self.promote_prompt_version(
                session_id=session_id,
                category=cat_norm,
                strategy_id=experimental_strategy_id,
                evidence=evidence_dict,
            )
            return PromptValidationResult(
                decision="PROMOTED",
                is_promoted=True,
                category=cat_norm,
                experimental_strategy_id=experimental_strategy_id,
                incumbent_strategy_id=incumbent_id,
                delta_sharpe=delta,
                delta_se=se_delta,
                lcb95_delta=lcb_delta,
                non_inferiority_margin=crit.non_inferiority_margin,
                non_inferiority_passed=True,
                absolute_sharpe_passed=True,
                ast_valid_rate_passed=True,
                token_budget_passed=True,
                sample_size_passed=True,
                provenance_passed=True,
                unmatched_evaluations_count=unmatched_evaluations_count,
                missing_metrics_count=missing_metrics_count,
                reasons=["All predeclared non-inferiority, absolute Sharpe, validity, and cost gates passed."],
                evidence=evidence_dict,
            )
        else:
            await self.reject_prompt_version(
                session_id=session_id,
                category=cat_norm,
                strategy_id=experimental_strategy_id,
                reasons=reasons,
                evidence=evidence_dict,
            )
            return PromptValidationResult(
                decision="REJECTED",
                is_promoted=False,
                category=cat_norm,
                experimental_strategy_id=experimental_strategy_id,
                incumbent_strategy_id=incumbent_id,
                delta_sharpe=delta,
                delta_se=se_delta,
                lcb95_delta=lcb_delta,
                non_inferiority_margin=crit.non_inferiority_margin,
                non_inferiority_passed=non_inf_passed,
                absolute_sharpe_passed=abs_sharpe_passed,
                ast_valid_rate_passed=ast_passed,
                token_budget_passed=token_passed,
                sample_size_passed=sample_size_passed,
                provenance_passed=prov_passed,
                unmatched_evaluations_count=unmatched_evaluations_count,
                missing_metrics_count=missing_metrics_count,
                reasons=reasons,
                evidence=evidence_dict,
            )

    async def promote_prompt_version(
        self,
        *,
        session_id: int,
        category: str,
        strategy_id: int,
        evidence: dict[str, Any] | None = None,
    ) -> bool:
        """Promotes an experimental strategy to active production status atomically.

        Maintains an explicit link to the previously promoted version for safe rollback.
        """
        cat_norm = normalize_category(category)
        async with self.db.session() as session:
            # 1. Fetch experimental strategy
            strat = await session.get(PromptStrategy, strategy_id)
            if not strat or strat.session_id != session_id or normalize_category(strat.category) != cat_norm:
                raise ValueError(f"Strategy {strategy_id} not found for session {session_id} and category {cat_norm}")

            # 2. Query currently active incumbent strategy
            inc_stmt = (
                select(PromptStrategy)
                .where(
                    PromptStrategy.session_id == session_id,
                    func.lower(func.trim(PromptStrategy.category)) == cat_norm,
                    PromptStrategy.is_active.is_(True),
                )
            )
            inc_res = await session.execute(inc_stmt)
            curr_active = inc_res.scalars().first()

            prev_strat_id = curr_active.id if curr_active else None
            prev_pv_id = curr_active.current_prompt_version_id if curr_active else None

            # 3. Deactivate prior active strategies
            await session.execute(
                update(PromptStrategy)
                .where(
                    PromptStrategy.session_id == session_id,
                    func.lower(func.trim(PromptStrategy.category)) == cat_norm,
                    PromptStrategy.is_active.is_(True),
                )
                .values(is_active=False, updated_at=utcnow())
            )

            # 4. Activate new strategy
            strat.is_active = True
            strat.validation_status = PromptLifecycleState.PROMOTED.value
            strat.previous_promoted_strategy_id = prev_strat_id
            strat.previous_promoted_version_id = prev_pv_id
            strat.promoted_at = utcnow()
            strat.promotion_criteria = evidence or {}
            strat.updated_at = utcnow()

            # 5. Log lifecycle event
            event = PromptLifecycleEvent(
                session_id=session_id,
                category=cat_norm,
                strategy_id=strat.id,
                prompt_version_id=strat.current_prompt_version_id,
                target_strategy_id=prev_strat_id,
                target_version_id=prev_pv_id,
                event_type="PROMOTED",
                from_status=PromptLifecycleState.EXPERIMENTAL.value,
                to_status=PromptLifecycleState.PROMOTED.value,
                delta_sharpe=evidence.get("delta") if evidence else None,
                delta_se=evidence.get("se_delta") if evidence else None,
                lcb95_delta=evidence.get("lcb95_delta") if evidence else None,
                non_inferiority_margin=evidence.get("non_inferiority_margin") if evidence else None,
                non_inferiority_passed=True,
                ast_valid_rate=evidence.get("ast_valid_rate") if evidence else None,
                reason="Predeclared non-inferiority validation gates satisfied.",
                details=evidence,
            )
            session.add(event)
            await session.commit()

        log.info(
            "prompt.promoted",
            session_id=session_id,
            category=cat_norm,
            strategy_id=strategy_id,
            previous_strategy_id=prev_strat_id,
        )
        return True

    async def reject_prompt_version(
        self,
        *,
        session_id: int,
        category: str,
        strategy_id: int,
        reasons: list[str],
        evidence: dict[str, Any] | None = None,
    ) -> bool:
        """Marks an experimental strategy as REJECTED without mutating the active incumbent."""
        cat_norm = normalize_category(category)
        async with self.db.session() as session:
            strat = await session.get(PromptStrategy, strategy_id)
            if not strat:
                return False

            strat.is_active = False
            strat.validation_status = PromptLifecycleState.REJECTED.value
            strat.rejection_reason = " | ".join(reasons)
            strat.updated_at = utcnow()

            event = PromptLifecycleEvent(
                session_id=session_id,
                category=cat_norm,
                strategy_id=strat.id,
                prompt_version_id=strat.current_prompt_version_id,
                event_type="REJECTED",
                from_status=PromptLifecycleState.EXPERIMENTAL.value,
                to_status=PromptLifecycleState.REJECTED.value,
                reason=" | ".join(reasons),
                details=evidence,
            )
            session.add(event)
            await session.commit()

        log.info(
            "prompt.rejected",
            session_id=session_id,
            category=cat_norm,
            strategy_id=strategy_id,
            reasons=reasons,
        )
        return True

    async def rollback_prompt_version(
        self,
        *,
        session_id: int,
        category: str,
        target_version_id: int | None = None,
    ) -> dict[str, Any]:
        """Rolls back the active prompt version to the intended stable prior version.

        Guarantees:
        - Fails safely if no valid prior promoted version exists (does not invent one).
        - Demotes currently active strategy to ROLLED_BACK.
        - Reactivates target prior strategy to PROMOTED and is_active = True.
        - Preserves all historical evaluation and version records.
        - Records traceable lifecycle event.
        """
        cat_norm = normalize_category(category)

        async with self.db.session() as session:
            # 1. Fetch currently active strategy
            curr_stmt = (
                select(PromptStrategy)
                .where(
                    PromptStrategy.session_id == session_id,
                    func.lower(func.trim(PromptStrategy.category)) == cat_norm,
                    PromptStrategy.is_active.is_(True),
                )
            )
            curr_res = await session.execute(curr_stmt)
            curr_strat = curr_res.scalars().first()

            if not curr_strat:
                raise ValueError(
                    f"No active prompt strategy found for session {session_id} and category {cat_norm} to roll back."
                )

            # 2. Identify target rollback strategy
            target_strat: PromptStrategy | None = None
            if target_version_id is not None:
                # Look up specific prior strategy or prompt version
                tgt_stmt = (
                    select(PromptStrategy)
                    .where(
                        PromptStrategy.session_id == session_id,
                        func.lower(func.trim(PromptStrategy.category)) == cat_norm,
                        PromptStrategy.version == target_version_id,
                    )
                )
                tgt_res = await session.execute(tgt_stmt)
                target_strat = tgt_res.scalars().first()
            elif curr_strat.previous_promoted_strategy_id:
                target_strat = await session.get(PromptStrategy, curr_strat.previous_promoted_strategy_id)

            # Fallback to the latest prior promoted or seed strategy if explicit pointer not set
            if not target_strat:
                fallback_stmt = (
                    select(PromptStrategy)
                    .where(
                        PromptStrategy.session_id == session_id,
                        func.lower(func.trim(PromptStrategy.category)) == cat_norm,
                        PromptStrategy.id != curr_strat.id,
                        PromptStrategy.validation_status.in_(
                            [PromptLifecycleState.PROMOTED.value, "ACTIVE", "SEED"]
                        ),
                    )
                    .order_by(PromptStrategy.version.desc())
                    .limit(1)
                )
                fb_res = await session.execute(fallback_stmt)
                target_strat = fb_res.scalars().first()

            if not target_strat:
                raise ValueError(
                    f"Safe rollback failed: No valid prior promoted prompt version exists for category '{cat_norm}'. "
                    f"Refusing to fabricate an arbitrary target version."
                )

            # 3. Atomically demote current and reactivate target
            curr_strat.is_active = False
            curr_strat.validation_status = PromptLifecycleState.ROLLED_BACK.value
            curr_strat.rolled_back_at = utcnow()
            curr_strat.updated_at = utcnow()

            target_strat.is_active = True
            target_strat.validation_status = PromptLifecycleState.PROMOTED.value
            target_strat.updated_at = utcnow()

            # 4. Log audit event
            event = PromptLifecycleEvent(
                session_id=session_id,
                category=cat_norm,
                strategy_id=curr_strat.id,
                prompt_version_id=curr_strat.current_prompt_version_id,
                target_strategy_id=target_strat.id,
                target_version_id=target_strat.current_prompt_version_id,
                event_type="ROLLED_BACK",
                from_status=PromptLifecycleState.PROMOTED.value,
                to_status=PromptLifecycleState.ROLLED_BACK.value,
                reason=f"Rollback to prior stable version {target_strat.version} (strategy {target_strat.id}) executed.",
            )
            session.add(event)
            await session.commit()

        log.info(
            "prompt.rolled_back",
            session_id=session_id,
            category=cat_norm,
            demoted_strategy_id=curr_strat.id,
            restored_strategy_id=target_strat.id,
            restored_version=target_strat.version,
        )
        return {
            "success": True,
            "session_id": session_id,
            "category": cat_norm,
            "demoted_strategy_id": curr_strat.id,
            "demoted_version": curr_strat.version,
            "restored_strategy_id": target_strat.id,
            "restored_version": target_strat.version,
        }

    async def get_prompt_lifecycle_history(
        self,
        session_id: int,
        category: str | None = None,
    ) -> list[PromptLifecycleEvent]:
        """Returns the audit trail of prompt lifecycle events."""
        async with self.db.session() as session:
            stmt = select(PromptLifecycleEvent).where(PromptLifecycleEvent.session_id == session_id)
            if category is not None:
                stmt = stmt.where(PromptLifecycleEvent.category == normalize_category(category))
            stmt = stmt.order_by(PromptLifecycleEvent.created_at.desc(), PromptLifecycleEvent.id.desc())
            res = await session.execute(stmt)
            return list(res.scalars().all())
