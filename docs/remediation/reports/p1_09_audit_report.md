# Issue P1-09 Audit Report: Operator Attribution & Confounder Control

**Issue**: P1-09 — Operator attribution uses naive mean Sharpe / confounders  
**Date**: 2026-10-13  
**Status**: AUDITED — AWAITING USER APPROVAL  
**Repository Working Tree**: `C:\Users\LENOVO\Desktop\alpha-harness`

---

## 1. Executive Summary & Problem Diagnosis

In `alpha_harness.learning.memory` and `alpha_harness.learning.loop`, operator efficacy is calculated by `compute_operator_stats()` to guide prompt evolution, few-shot generation, and LLM reflection. 

The current implementation suffers from three critical statistical flaws:
1. **Fragile Regex Extraction**: Operator names are parsed via regex (`re.findall(r"([a-z_][a-z0-9_]*)\s*\(", expr.lower())`) rather than through the AST parser and `OperatorCatalog`.
2. **Naive Uniform Credit Assignment (Omitted Confounders)**: When an alpha expression achieves a high or low Sharpe, 100% of that Sharpe is uniformly assigned to every operator in the expression. This ignores major confounding variables:
   - **Data field quality**: An expression with a stellar alternative dataset field looks high-performing regardless of operator choice.
   - **Operator co-occurrence / collinearity**: A neutral operator paired with a strong operator gets false positive credit.
   - **Expression complexity**: Deeper expressions with many operators distribute unearned credit broadly.
   - **Generation / regime baselines**: High overall market/sample performance inflates operator averages.
3. **Absence of Shrinkage / Small-Sample Instability**:
   - `avg_sharpe = sum(sharpes) / len(sharpes)` with a minimal threshold of `appearances >= 2`.
   - An operator appearing twice in lucky alphas with Sharpe 2.2 is ranked higher than a proven operator appearing 25 times with average Sharpe 1.4.
   - The flawed top/worst 3 operators are immediately injected into prompt strategy rules (`preferred_operators`, `avoid_operators`), creating negative feedback loops in subsequent generations.

---

## 2. Current Implementation Analysis

### 2.1 Current Code in `alpha_harness.learning.memory` (Lines 285–362)

```python
async def compute_operator_stats(
    self, session_id: int, category: str | None = None
) -> dict[str, OperatorEfficacy]:
    """Aggregates operator-level Sharpe lift and win rates."""
    ...
    # Simple token extraction for operator names
    import re
    ops = re.findall(r"([a-z_][a-z0-9_]*)\s*\(", expr.lower())
    for op in set(ops):
        if op not in stats_map:
            stats_map[op] = OperatorEfficacy(name=op)
        item = stats_map[op]
        item.appearances += 1
        item.sharpe_values.append(s_val)
        if s_val >= 1.25:
            item.winner_count += 1
        elif s_val < 0.5:
            item.loser_count += 1

    for item in stats_map.values():
        if item.sharpe_values:
            item.avg_sharpe = round(sum(item.sharpe_values) / len(item.sharpe_values), 3)

    return stats_map
```

### 2.2 Flaws & Failure Modes

| Flaw | Mechanism | Consequence in Research Loop |
|---|---|---|
| **Fragile Regex** | `re.findall(r"([a-z_][a-z0-9_]*)\s*\(")` | Can pick up non-operator tokens; does not validate against `OperatorCatalog`; ignores AST nesting hierarchy. |
| **Full Confounding** | `s_val` attributed identically to all `set(ops)` | An alpha `rank(ts_decay_linear(high_alpha_field, 10))` credits both `rank` and `ts_decay_linear` identically, even if `rank` did nothing or `ts_decay_linear` was the true driver. |
| **No Complexity Adjustment** | Alphas with 10 operators distribute credit across all 10 | Operators in complex overfitted alphas get artificial credit boosts. |
| **No Empirical Bayes Shrinkage** | Raw arithmetic mean | Sample size of $N=2$ can outrank $N=30$. In noisy alpha searches, high variance causes random noise to be codified into `preferred_operators`. |
| **Permissive Ranking Threshold** | `appearances >= 2` in `loop.py` | Prompt strategies and LLM prompts are populated with low-confidence noise operators. |

---

## 3. Proposed Remediation Architecture for P1-09

### 3.1 Robust AST Operator Extraction
Replace regex with AST parsing leveraging `fastexpr.parse()` and validation against `OperatorCatalog` (P1-06):
- Traverse AST expression nodes to extract verified FastExpr operator identifiers.
- Track operator position, nesting depth, and parameter lookbacks.
- Ignore invalid tokens, dataset field names, or unparseable text.

### 3.2 Controlled Marginal Attribution (Deconfounding)
To isolate an operator's true marginal effect from field quality, generation baseline, and complexity:
1. **Category / Generation Baseline Adjustment**:
   $$\Delta \text{Sharpe}_i = \text{Sharpe}_i - \overline{\text{Sharpe}}_{\text{batch}}$$
   Centering removes session-wide and generation-wide performance drift.
2. **Complexity Weighting**:
   Weight the contribution of each operator inversely by the number of distinct operators $M$ in the expression, or estimate a regularized linear model (Ridge regression / marginal decomposition) where:
   $$\Delta \text{Sharpe}_i \approx \sum_{j} \beta_j \cdot X_{ij} + \epsilon_i$$
   where $X_{ij} = 1$ if operator $j$ is present in candidate $i$.
3. **Field Confounder Adjustment**:
   Where candidate data field quality is known, baseline adjust by the data field's historical performance.

### 3.3 Empirical Bayes / James-Stein Shrinkage
Rather than ranking raw sample averages, shrink estimated operator lifts toward the prior mean (0.0 lift):
$$\text{Shrunk Lift}_j = \frac{N_j}{N_j + k} \cdot \overline{\text{Lift}}_j$$
where:
- $N_j$ is the number of appearances of operator $j$.
- $k$ is the shrinkage prior parameter (e.g., $k = 3.0$ or $k = 5.0$).
- When $N_j = 1$, the operator lift is shrunk heavily toward 0.
- When $N_j \gg k$, the operator lift reflects its true empirical average.

### 3.4 Confidence Intervals & Statistical Gates
Calculate standard error and confidence bounds:
$$\text{SE}_j = \frac{\sigma_j}{\sqrt{N_j}}$$
$$\text{LCB}_j = \text{Shrunk Lift}_j - 1.96 \cdot \text{SE}_j$$
- **Preferred Operators**: Rank by Lower Confidence Bound (conservative estimate of positive lift).
- **Avoided Operators**: Rank by Upper Confidence Bound (conservative estimate of negative lift).
- **Minimum Sample Gate**: Require $N \ge 3$ (configurable, default $\ge 3$) before an operator can be marked as `preferred` or `avoid` in evolved strategies.
- **Confidence Rating**: Categorize operator attribution confidence into `HIGH` ($N \ge 10$), `MEDIUM` ($N \ge 5$), `LOW` ($N \ge 3$), `INSUFFICIENT` ($N < 3$).

### 3.5 Extended `OperatorEfficacy` Schema
```python
@dataclass(slots=True)
class OperatorEfficacy:
    name: str
    appearances: int = 0
    winner_count: int = 0
    loser_count: int = 0
    avg_sharpe: float = 0.0
    sharpe_values: list[float] = field(default_factory=list)
    # P1-09 Controlled attribution extensions:
    marginal_lift: float = 0.0
    shrunk_lift: float = 0.0
    std_error: float = 0.0
    t_stat: float = 0.0
    confidence_level: str = "INSUFFICIENT"  # HIGH | MEDIUM | LOW | INSUFFICIENT
    is_statistically_significant: bool = False
```

### 3.6 Downstream Integration with `SelfLearningEngine`
- In `loop.py`: Update `best_ops` and `worst_ops` to filter by `appearances >= 3` and sort by `shrunk_lift` (or LCB).
- In `prompts/library.py`: Present shrunk lift, sample count, and confidence level in `factual_stats["operator_stats"]` to prevent LLM hallucinations.
- In `ResearchMemory`: Provide backward-compatible getters while deprecating naive uniform Sharpe attribution.

---

## 4. Proposed Test Matrix for P1-09

A dedicated test suite `backend/tests/test_p1_09_operator_attribution.py` will verify:
1. `test_ast_operator_extraction_vs_regex`: Verifies accurate AST extraction over fragile regex.
2. `test_operator_credit_deconfounding`: Verifies that a neutral operator paired with a superstar operator does not receive full unadjusted credit.
3. `test_empirical_bayes_shrinkage`: Verifies that an operator with $N=1$ and high Sharpe is shrunk toward zero and does not outrank an established operator with $N=20$.
4. `test_complexity_penalty`: Verifies that operators in bloated expressions have their credit discounted.
5. `test_minimum_sample_size_gate`: Verifies operators with $N < 3$ are not selected as `preferred_operators` or `avoid_operators`.
6. `test_confidence_level_categorization`: Verifies correct assignment of `HIGH`, `MEDIUM`, `LOW`, and `INSUFFICIENT`.
7. `test_backward_compatibility`: Verifies `OperatorEfficacy.avg_sharpe` and `win_rate` properties remain intact for legacy consumers.
8. `test_prompt_factual_stats_integration`: Verifies LLM prompt reflection consumes deconfounded stats.

---

## 5. Next Steps

- **Awaiting User Review**: In accordance with the remediation backlog workflow, P1-09 will not be implemented until user approval is received.
- Once approved, implementation will proceed directly in `backend/src/alpha_harness/learning/memory.py`, `backend/src/alpha_harness/learning/loop.py`, and `backend/tests/test_p1_09_operator_attribution.py`.
