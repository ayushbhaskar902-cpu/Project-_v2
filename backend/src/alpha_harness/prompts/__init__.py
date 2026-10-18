"""Prompt strategy library for alpha research."""

from __future__ import annotations

from .library import (
    BASE_SYSTEM_ROLE,
    FAST_EXPR_RULES,
    SEED_TEMPLATES,
    PromptLibrary,
    PromptTemplate,
    StrategyCategory,
    normalize_category,
)

__all__ = [
    "BASE_SYSTEM_ROLE",
    "FAST_EXPR_RULES",
    "SEED_TEMPLATES",
    "PromptLibrary",
    "PromptTemplate",
    "StrategyCategory",
    "normalize_category",
]
