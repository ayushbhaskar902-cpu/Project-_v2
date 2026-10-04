"""Multi-LLM researcher orchestration module."""

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
