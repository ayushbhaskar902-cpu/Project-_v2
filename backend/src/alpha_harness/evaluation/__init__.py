"""Alpha evaluation and robustness testing module."""

from __future__ import annotations

from .metrics import (
    EvaluationEngine,
    EvaluationScorecard,
    HoldoutChronologyError,
    HoldoutImmutabilityError,
    HoldoutIndeterminacyError,
    HoldoutOverlapError,
    InsufficientDataError,
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
    "HoldoutChronologyError",
    "HoldoutImmutabilityError",
    "HoldoutIndeterminacyError",
    "HoldoutOverlapError",
    "InsufficientDataError",
    "RobustnessEngine",
    "RobustnessTestPlan",
    "RobustnessTestResult",
    "compute_composite_score",
    "evaluate_pnl_series",
]
