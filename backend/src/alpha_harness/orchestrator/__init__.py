"""Multi-LLM researcher orchestration module."""

from __future__ import annotations

from .researchers import (
    DEFAULT_RESEARCHER_ROSTER,
    ITEMS_PER_CORE,
    ROUND_CAPACITY,
    SLOT_CAP,
    MultiLLMOrchestrator,
    ResearcherConfig,
)

__all__ = [
    "DEFAULT_RESEARCHER_ROSTER",
    "ITEMS_PER_CORE",
    "ROUND_CAPACITY",
    "SLOT_CAP",
    "MultiLLMOrchestrator",
    "ResearcherConfig",
]
