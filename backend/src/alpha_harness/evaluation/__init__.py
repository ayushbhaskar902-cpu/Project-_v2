"""Alpha evaluation and robustness testing module."""

from .metrics import (
    EvaluationEngine,
    EvaluationScorecard,
    compute_composite_score,
    evaluate_pnl_series,
)
from .robustness import (
    RobustnessEngine,
    RobustnessTestPlan,
    RobustnessTestResult,
)

__all__ = [
    "EvaluationEngine",
    "EvaluationScorecard",
    "RobustnessEngine",
    "RobustnessTestPlan",
    "RobustnessTestResult",
    "compute_composite_score",
    "evaluate_pnl_series",
]
