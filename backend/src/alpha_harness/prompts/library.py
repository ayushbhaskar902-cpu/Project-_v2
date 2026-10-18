"""Prompt strategy library for autonomous quantitative alpha generation.

Provides structured, versioned prompt templates for diverse quantitative research styles
(momentum, mean reversion, value, quality, volatility, multi-factor), with slots for
dynamic context injection:
- Real data fields from DuckDB catalog
- Allowed BRAIN Fast Expression operators and arities
- Few-shot winning examples from previous generations
- Anti-patterns and common failure modes to avoid
- Self-learning feedback rules extracted by the learning loop
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class StrategyCategory(StrEnum):
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VALUE = "value"
    QUALITY = "quality"
    VOLATILITY = "volatility"
    GENERAL = "general"


def normalize_category(cat: str | StrategyCategory | None) -> str:
    """Canonicalizes category names while preserving valid custom category identities.

    Rules:
    1. None, empty string, or whitespace-only -> 'general' (the documented default fallback).
    2. Matches recognized StrategyCategory enum (by value or name, case-insensitive) -> canonical enum value.
    3. Typographical variants of canonical categories (e.g. 'mean-reversion', 'mean reversion') -> canonical enum value.
    4. Non-empty custom string (e.g. 'reversal', 'sentiment') -> lowercased, trimmed string identity (preserved as-is).
    """
    if cat is None:
        return StrategyCategory.GENERAL.value
    if isinstance(cat, StrategyCategory):
        return cat.value

    s = str(cat).strip()
    if not s:
        return StrategyCategory.GENERAL.value

    s_lower = s.lower()
    for member in StrategyCategory:
        if s_lower == member.value or s.upper() == member.name:
            return member.value

    # Canonical typographical aliases for existing StrategyCategory members
    canonical_aliases: dict[str, str] = {
        "mean-reversion": StrategyCategory.MEAN_REVERSION.value,
        "mean reversion": StrategyCategory.MEAN_REVERSION.value,
        "meanreversion": StrategyCategory.MEAN_REVERSION.value,
    }
    if s_lower in canonical_aliases:
        return canonical_aliases[s_lower]

    return s_lower



@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """A versioned template definition for LLM researchers."""

    name: str
    category: StrategyCategory
    version: int
    system_role: str
    strategy_guidance: str
    operator_guidance: str
    template_text: str
    default_examples: list[dict[str, Any]] = field(default_factory=list)
    anti_patterns: list[dict[str, Any]] = field(default_factory=list)


# --- Core System Role & Rule Base ---

BASE_SYSTEM_ROLE = """\
You are an expert quantitative researcher and portfolio manager developing predictive alpha \
factors for WorldQuant BRAIN. You write mathematical expressions in BRAIN's Fast Expression syntax.

Your goal is to discover genuine, economically sound market anomalies with high risk-adjusted \
returns (Sharpe ratio > 1.5), strong information coefficient (Fitness > 1.0), and low turnover (< 50%).
"""

FAST_EXPR_RULES = """\
CRITICAL WORLDQUANT BRAIN FAST EXPRESSION RULES:
1. OPERATOR BUDGET: Strictly at most 8 operators per expression.
   - Count every function call: rank(), ts_rank(), ts_delta(), ts_mean(), ts_zscore(), etc.
   - Count every arithmetic/comparison operator: +, -, *, /, ^, <, <=, >, >=, ==, !=
   - Count ternary conditionals: cond ? a : b counts as 2 operators (? and :)
   - Count unary negation: -x counts as 1 operator
   - ts_backfill() and group_backfill() are UNCOUNTED (zero operator penalty)
   - Expressions with > 8 operators are AUTOMATICALLY REJECTED.

2. DATA FIELD BUDGET: Strictly at most 3 distinct data fields per expression.
   - Grouping fields (market, sector, industry, subindustry, country, exchange) do NOT count as data fields.
   - Matrix/Vector price & volume fields (close, open, high, low, volume, vwap, cap) DO count as data fields.
   - You MUST include at least one field from the provided dataset.

3. VALID OPERATOR SIGNATURES & ARITIES:
   - ts_rank(x, d): Cross-time percentile rank over lookback d days (e.g. ts_rank(close, 20))
   - ts_zscore(x, d): (x - ts_mean(x, d)) / ts_std_dev(x, d)
   - ts_delta(x, d): x - ts_delay(x, d)
   - ts_mean(x, d), ts_std_dev(x, d), ts_sum(x, d)
   - ts_corr(x, y, d): Correlation between x and y over d days
   - rank(x): Cross-sectional rank normalized [0, 1]
   - group_rank(x, group): Rank x within sector or industry, e.g. group_rank(x, industry)
   - group_neutralize(x, group): Subtract group mean
   - trade_when(condition, alpha, -1): Exit or hold position when condition fails
   - vec_avg(x), vec_sum(x): Collapse vector field to scalar matrix

4. LOOKBACK DAYS (trading days):
   - 5 = 1 week, 10 = 2 weeks, 20 = 1 month, 60 = 1 quarter, 120 = 6 months, 252 = 1 year.
"""

OUTPUT_FORMAT_INSTRUCTIONS = """\
RESPONSE FORMAT REQUIREMENTS:
You MUST respond with valid JSON ONLY. No conversational prelude or markdown codeblock wrapping outside the JSON.
The JSON must adhere to this exact schema:
{{
  "batch_summary": "Brief 1-2 sentence overview of your quantitative hypotheses for this round",
  "candidates": [
    {{
      "hypothesis": "Clear economic rationale explaining why this market signal generates risk premium",
      "expression": "rank(-ts_delta(close, 5))",
      "category": "momentum",
      "target_dataset": "dataset_id_here",
      "primary_fields": ["close"]
    }}
  ]
}}
"""

# --- Strategy-Specific Prompt Templates ---

MOMENTUM_GUIDANCE = """\
MOMENTUM & TREND RESEARCH OBJECTIVE:
Capture continuation of intermediate-term returns and cross-sectional trend persistence while \
mitigating short-term reversal noise.
Key Ideas:
- Time-series momentum: ts_delta or ts_zscore of price, volume-weighted metrics, or fundamental trends.
- Cross-sectional momentum: rank or group_rank across industry/sector to avoid sector bets.
- Volume-confirmed trends: scale momentum by relative volume (volume / ts_mean(volume, 20)).
- Acceleration: second derivative of price or growth metrics (ts_delta(ts_delta(x, 10), 10)).
- Medium-term lookbacks (20 to 120 days) often outperform ultra-short (1-3 days) lookbacks.
"""

MEAN_REVERSION_GUIDANCE = """\
MEAN REVERSION & OSCILLATOR RESEARCH OBJECTIVE:
Identify short-term overreaction and liquidity shocks where asset prices deviate excessively \
from their moving averages or peer averages and are likely to revert.
Key Ideas:
- Standardized deviations: -ts_zscore(close, 5) or -ts_zscore(vwap, 10).
- Extreme percentile bounds: rank(ts_rank(close, 10) < 0.05 ? 1 : (ts_rank(close, 10) > 0.95 ? -1 : 0)).
- Distance from VWAP: -(close - vwap) / ts_std_dev(close, 20).
- Industry-relative reversion: group_rank(-(close / ts_mean(close, 10) - 1), industry).
- Always use negative sign or inverted ranking to position against the short-term spike.
"""

VALUE_GUIDANCE = """\
VALUE & FUNDAMENTAL RATIO RESEARCH OBJECTIVE:
Exploit valuation discrepancies between price and underlying fundamentals (earnings, cash flow, \
sales, asset book value).
Key Ideas:
- Earnings / Cash Flow yield: fundamental_metric / cap or fundamental_metric / close.
- Cross-sectional percentile ranking: rank(fundamental_metric / ts_backfill(cap, 60)).
- Industry neutralization: group_neutralize(rank(ebitda / cap), industry).
- Trend in valuation: ts_delta(fundamental_metric / close, 60).
- Remember fundamental data has lower update frequency; always use ts_backfill(field, 60 or 252).
"""

QUALITY_GUIDANCE = """\
QUALITY & PROFITABILITY RESEARCH OBJECTIVE:
Identify companies with superior return on capital, high earnings quality, conservative accounting, \
and low financial distress.
Key Ideas:
- Margin expansion: ts_delta(operating_income / sales, 120).
- Capital efficiency: net_income / ts_backfill(total_assets, 252).
- Accruals anomaly: -(net_income - operating_cash_flow) / total_assets.
- Stability of earnings: -ts_std_dev(ebit, 120) / ts_mean(ebit, 120).
"""

VOLATILITY_GUIDANCE = """\
VOLATILITY & LIQUIDITY RESEARCH OBJECTIVE:
Harvest low-volatility anomaly and liquidity premia.
Key Ideas:
- Low-beta / low-volatility: -ts_std_dev(returns, 60) or -ts_std_dev(close / ts_delay(close, 1) - 1, 40).
- Volatility surprise: -(ts_std_dev(close, 10) / ts_std_dev(close, 60) - 1).
- Liquidity shocks: rank(-volume / adv20) or rank(abs(returns) / (volume * close)).
"""

GENERAL_GUIDANCE = """\
MULTI-FACTOR QUANTITATIVE EXPLORATION:
Combine structural market features (price, volume, fundamentals, analyst forecasts, sentiment) \
into coherent multi-signal expressions.
Key Ideas:
- Conditioning: trade_when(volatility < historical_volatility, trend_signal, reversion_signal).
- Signal interaction: rank(momentum_signal) * rank(quality_signal).
- Ratio comparison: field_a / ts_mean(field_b, 20).
"""


def _build_template(
    name: str,
    category: StrategyCategory,
    version: int,
    guidance: str,
    examples: list[dict[str, Any]],
    anti_patterns: list[dict[str, Any]],
) -> PromptTemplate:
    template_text = f"""\
{{base_system_role}}

{{strategy_guidance}}

{FAST_EXPR_RULES}

AVAILABLE DATASETS & FIELDS:
{{catalog_context}}

ALLOWED OPERATOR TABLE:
{{operators_table}}

LEARNED HEURISTICS & STRATEGY UPDATE (Generation {{generation}}):
{{learning_feedback}}

TOP-PERFORMING WINNING EXAMPLES (FEW-SHOT):
{{winning_examples}}

COMMON ANTI-PATTERNS & FAILURE MODES TO AVOID:
{{anti_patterns}}

TASK INSTRUCTION:
Generate {{candidate_count}} distinct, diverse, and robust alpha candidates for region {{region}} \
and universe {{universe}}.
Every candidate must strictly satisfy the 8-operator and 3-field limits.
Ensure high variety across operators and field combinations.

{OUTPUT_FORMAT_INSTRUCTIONS}
"""
    return PromptTemplate(
        name=name,
        category=category,
        version=version,
        system_role=BASE_SYSTEM_ROLE,
        strategy_guidance=guidance,
        operator_guidance=FAST_EXPR_RULES,
        template_text=template_text,
        default_examples=examples,
        anti_patterns=anti_patterns,
    )


# --- Seed Catalog of Prompt Templates ---

SEED_TEMPLATES: dict[StrategyCategory, PromptTemplate] = {
    StrategyCategory.MOMENTUM: _build_template(
        name="Momentum & Trend Follower v1",
        category=StrategyCategory.MOMENTUM,
        version=1,
        guidance=MOMENTUM_GUIDANCE,
        examples=[
            {
                "hypothesis": "12-month cross-sectional price momentum with 1-month reversal skip",
                "expression": "group_rank(ts_delta(close, 252) - ts_delta(close, 20), industry)",
                "sharpe": 1.74,
            },
            {
                "hypothesis": "Volume-weighted price acceleration over 20 days",
                "expression": "rank(ts_zscore(close * volume, 20))",
                "sharpe": 1.62,
            },
        ],
        anti_patterns=[
            {
                "flaw": "Exceeded 8 operators budget",
                "expression": "rank(ts_rank(close, 5)) + rank(ts_rank(open, 5)) + rank(ts_rank(high, 5))",
            },
            {
                "flaw": "Used too many fields (> 3)",
                "expression": "close + open + high + low",
            },
        ],
    ),
    StrategyCategory.MEAN_REVERSION: _build_template(
        name="Short-Term Mean Reversion v1",
        category=StrategyCategory.MEAN_REVERSION,
        version=1,
        guidance=MEAN_REVERSION_GUIDANCE,
        examples=[
            {
                "hypothesis": "5-day standardized price reversal normalized within sector",
                "expression": "group_neutralize(-ts_zscore(close, 5), sector)",
                "sharpe": 1.82,
            },
            {
                "hypothesis": "Extreme price deviation from VWAP over 10 days",
                "expression": "rank(-(close - vwap) / ts_std_dev(close, 10))",
                "sharpe": 1.68,
            },
        ],
        anti_patterns=[
            {
                "flaw": "Positive correlation with momentum during crash periods",
                "expression": "ts_delta(close, 3)",
            }
        ],
    ),
    StrategyCategory.VALUE: _build_template(
        name="Fundamental Value Explorer v1",
        category=StrategyCategory.VALUE,
        version=1,
        guidance=VALUE_GUIDANCE,
        examples=[
            {
                "hypothesis": "Operating earnings yield backfilled over 60 days with industry ranking",
                "expression": "group_rank(ts_backfill(operating_income, 60) / cap, industry)",
                "sharpe": 1.58,
            }
        ],
        anti_patterns=[
            {
                "flaw": "Forgetting ts_backfill on low-coverage quarterly fundamental data",
                "expression": "rank(operating_income / cap)",
            }
        ],
    ),
    StrategyCategory.QUALITY: _build_template(
        name="Quality & Profitability v1",
        category=StrategyCategory.QUALITY,
        version=1,
        guidance=QUALITY_GUIDANCE,
        examples=[
            {
                "hypothesis": "Return on assets momentum normalized across subindustry",
                "expression": "group_rank(ts_delta(ts_backfill(net_income, 60) / ts_backfill(cap, 60), 120), subindustry)",
                "sharpe": 1.65,
            }
        ],
        anti_patterns=[],
    ),
    StrategyCategory.VOLATILITY: _build_template(
        name="Volatility & Liquidity v1",
        category=StrategyCategory.VOLATILITY,
        version=1,
        guidance=VOLATILITY_GUIDANCE,
        examples=[
            {
                "hypothesis": "Low volatility anomaly over 60 days",
                "expression": "rank(-ts_std_dev(close / ts_delay(close, 1) - 1, 60))",
                "sharpe": 1.55,
            }
        ],
        anti_patterns=[],
    ),
    StrategyCategory.GENERAL: _build_template(
        name="General Quant Exploration v1",
        category=StrategyCategory.GENERAL,
        version=1,
        guidance=GENERAL_GUIDANCE,
        examples=[
            {
                "hypothesis": "Volume surprise scaled price momentum",
                "expression": "rank(ts_delta(close, 20)) * rank(volume / adv20)",
                "sharpe": 1.69,
            }
        ],
        anti_patterns=[],
    ),
}


class PromptLibrary:
    """Manages strategy prompt templates and renders full prompts for researchers."""

    def __init__(self) -> None:
        self._templates: dict[str, PromptTemplate] = {}
        for category, tmpl in SEED_TEMPLATES.items():
            self._templates[category.value] = tmpl

    def get_template(self, category: str | StrategyCategory | None) -> PromptTemplate:
        cat_key = normalize_category(category)
        if cat_key in self._templates:
            return self._templates[cat_key]
        return self._templates[StrategyCategory.GENERAL.value]

    def register_template(self, template: PromptTemplate) -> None:
        self._templates[template.category.value] = template

    def render(
        self,
        category: str | StrategyCategory | None,
        catalog_context: str,
        operators_table: str,
        candidate_count: int = 10,
        region: str = "USA",
        universe: str = "TOP3000",
        generation: int = 0,
        learning_feedback: str = "No prior generation data. Explore baseline anomalies.",
        winning_examples: list[dict[str, Any]] | None = None,
        anti_patterns: list[dict[str, Any]] | None = None,
        template_text: str | None = None,
    ) -> str:
        """Renders complete prompt text with all variables populated."""
        template = self.get_template(category)
        active_template_text = template_text if template_text is not None else template.template_text

        # Format winning examples
        examples = winning_examples if winning_examples is not None else template.default_examples
        ex_str = ""
        if examples:
            lines = []
            for ex in examples:
                h = ex.get("hypothesis", "Empirically verified factor")
                e = ex.get("expression", "")
                s = ex.get("sharpe", "")
                sharpe_text = f" (Sharpe: {s})" if s else ""
                lines.append(f"- Hypothesis: {h}\n  Expression: `{e}`{sharpe_text}")
            ex_str = "\n".join(lines)
        else:
            ex_str = "None available yet."

        # Format anti patterns
        antis = anti_patterns if anti_patterns is not None else template.anti_patterns
        anti_str = ""
        if antis:
            lines = []
            for a in antis:
                f = a.get("flaw", "Failed test")
                e = a.get("expression", "")
                lines.append(f"- Warning: {f}\n  Faulty Expression: `{e}`")
            anti_str = "\n".join(lines)
        else:
            anti_str = "Avoid exceeding 8 operators or 3 data fields."

        return active_template_text.format(
            base_system_role=template.system_role,
            strategy_guidance=template.strategy_guidance,
            catalog_context=catalog_context,
            operators_table=operators_table,
            generation=generation,
            learning_feedback=learning_feedback,
            winning_examples=ex_str,
            anti_patterns=anti_str,
            candidate_count=candidate_count,
            region=region,
            universe=universe,
        )


# --- P1-01: Evidence-Grounded LLM Reflection System & Prompts ---

ALLOWED_CITATION_FIELDS: frozenset[str] = frozenset({
    "expression",
    "sharpe",
    "fitness",
    "turnover",
    "fields",
    "operator_count",
    "sharpe_decay",
    "after_cost_sharpe",
})

REFLECTION_SYSTEM_PROMPT = """\
You are an expert quantitative research review committee and statistical auditor.
Your job is to objectively analyze the empirical results from the latest alpha generation batch \
and extract evidence-grounded hypotheses, anti-patterns, and actionable recommendations.

CRITICAL METHODOLOGICAL PRINCIPLES:
1. CAUSAL HUMILITY: Do not assert unconditional causality. Every claim is an empirical hypothesis \
based on observed sample data until experimentally validated.
2. STRICT EVIDENCE GROUNDING: Every hypothesis, anti-pattern, and recommendation MUST include at \
least one explicit citation to a specific candidate in the evidence batch, naming the candidate ID \
and the exact field observed (e.g. candidate_id, field="expression", field="sharpe", etc.).
3. CITATION VALIDATION: You may ONLY cite candidates and fields present in the provided evidence \
tables. Permitted citation fields are: expression, sharpe, fitness, turnover, fields, operator_count, \
sharpe_decay, after_cost_sharpe. Never cite candidate IDs or fields not in the batch.
4. HOLDOUT PROTECTION: Holdout performance metrics are strictly confidential and intentionally omitted. \
Do not speculate or extrapolate beyond the provided in-sample and validation metrics.
5. NO HALLUCINATED METRICS: Authoritative performance metrics are deterministically computed by the \
evaluation engine. Do not synthesize or hallucinate alternative metrics.

You must respond with valid JSON strictly conforming to the requested schema.
"""

REFLECTION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "Executive summary of what was learned from this generation's candidate batch.",
        },
        "hypotheses": {
            "type": "array",
            "description": "Explanations of empirical patterns observed in the evidence batch. Explicitly treated as unverified hypotheses.",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "confidence": {
                        "type": "string",
                        "enum": [
                            "HIGH_CONFIDENCE_HYPOTHESIS",
                            "PLAUSIBLE_HYPOTHESIS",
                            "SPECULATIVE_EXPLORATION",
                        ],
                    },
                    "citations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "candidate_id": {"type": "integer"},
                                "field": {"type": "string"},
                                "observed_value": {
                                    "type": ["string", "number", "array", "null"],
                                },
                            },
                            "required": ["candidate_id", "field"],
                        },
                        "minItems": 1,
                    },
                },
                "required": ["claim", "confidence", "citations"],
            },
        },
        "recommendations": {
            "type": "array",
            "description": "Actionable prompt guidance adjustments for future alpha discovery rounds.",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "rationale": {"type": "string"},
                    "citations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "candidate_id": {"type": "integer"},
                                "field": {"type": "string"},
                                "observed_value": {
                                    "type": ["string", "number", "array", "null"],
                                },
                            },
                            "required": ["candidate_id", "field"],
                        },
                        "minItems": 1,
                    },
                },
                "required": ["action", "rationale", "citations"],
            },
        },
        "anti_patterns": {
            "type": "array",
            "description": "Underperforming patterns or failure modes to explicitly caution against.",
            "items": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                    "flaw": {"type": "string"},
                    "citations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "candidate_id": {"type": "integer"},
                                "field": {"type": "string"},
                                "observed_value": {
                                    "type": ["string", "number", "array", "null"],
                                },
                            },
                            "required": ["candidate_id", "field"],
                        },
                        "minItems": 1,
                    },
                },
                "required": ["expression", "flaw", "citations"],
            },
        },
    },
    "required": ["summary", "hypotheses", "recommendations", "anti_patterns"],
}


def render_reflection_evidence_prompt(
    session_id: int,
    generation: int,
    eligible_candidates: list[dict[str, Any]],
    factual_stats: dict[str, Any],
) -> str:
    """Formats deterministic database evidence into structured text tables for LLM reflection.

    Strictly excludes holdout metrics. Candidates must be eligible and verified.
    """
    lines = [
        f"=== RESEARCH BATCH EVIDENCE REPORT (Session {session_id}, Generation {generation}) ===",
        "",
        "--- AUTHORITATIVE BATCH PERFORMANCE STATISTICS (CALCULATED FROM DB) ---",
        f"Eligible Batch Size: {factual_stats.get('batch_size', len(eligible_candidates))}",
        f"Average In-Sample/Validation Sharpe: {factual_stats.get('avg_sharpe', 0.0):.3f}",
        f"Best In-Sample/Validation Sharpe: {factual_stats.get('best_sharpe', 0.0):.3f}",
        f"Validation Pass Rate: {factual_stats.get('valid_rate', 1.0) * 100:.1f}%",
    ]

    op_stats = factual_stats.get("operator_stats")
    if op_stats:
        lines.append("")
        lines.append("Authoritative Operator Statistics (P1-09 Controlled Attribution):")
        lines.append("NOTE: These figures reflect observational statistical associations from validation data, not proven causal drivers.")
        for op in op_stats:
            shrunk = op.get("shrunk_lift")
            confidence = op.get("confidence_level", "UNKNOWN")
            evidence = op.get("evidence_status", "UNKNOWN")
            line = (
                f"- Operator '{op.get('name')}': count={op.get('appearances')}, "
                f"avg_sharpe={op.get('avg_sharpe', 0.0):.3f}"
            )
            if shrunk is not None:
                line += f", shrunk_lift={shrunk:.4f}, confidence={confidence}, evidence={evidence}"
            lines.append(line)

    lines.append("")
    lines.append("--- ELIGIBLE RESEARCH CANDIDATE EVIDENCE ---")
    lines.append("NOTE: All citations MUST use the Candidate IDs listed below.")
    lines.append(
        "Permitted citation fields: expression, sharpe, fitness, turnover, fields, "
        "operator_count, sharpe_decay, after_cost_sharpe."
    )
    lines.append("Holdout metrics are strictly protected and withheld.")
    lines.append("")

    for cand in eligible_candidates:
        cid = cand.get("candidate_id")
        expr = cand.get("expression", "")
        hyp = cand.get("hypothesis", "")
        sharpe = cand.get("sharpe", 0.0)
        fitness = cand.get("fitness", 0.0)
        turnover = cand.get("turnover", 0.0)
        op_count = cand.get("operator_count", 0)
        fields_used = cand.get("fields", [])
        decay = cand.get("sharpe_decay", 0.0)
        after_cost = cand.get("after_cost_sharpe", 0.0)

        lines.append(f"[Candidate ID: {cid}]")
        lines.append(f"  Expression: {expr}")
        lines.append(f"  Stated Hypothesis: {hyp}")
        lines.append(
            f"  Metrics: sharpe={sharpe:.3f}, fitness={fitness:.3f}, turnover={turnover:.3f}, "
            f"operator_count={op_count}, sharpe_decay={decay:.3f}, after_cost_sharpe={after_cost:.3f}"
        )
        lines.append(f"  Fields: {fields_used}")
        lines.append("")

    lines.append("--- REQUIRED OUTPUT ---")
    lines.append(
        "Generate a structured reflection JSON matching the required schema. Ensure every hypothesis, "
        "recommendation, and anti-pattern has at least one valid citation to the above Candidate IDs."
    )
    return "\n".join(lines)
