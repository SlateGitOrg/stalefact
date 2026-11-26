"""The four memory strategies and a factory for them."""

from __future__ import annotations

from collections.abc import Mapping

from stalefact.memory.base import Memory
from stalefact.memory.hybrid import HybridMemory
from stalefact.memory.ledger import FactLedgerMemory
from stalefact.memory.rolling import RollingSummaryMemory
from stalefact.memory.window import WindowMemory
from stalefact.model import MemoryModel

STRATEGIES = ("window", "rolling", "ledger", "hybrid")


def build_memory(
    strategy: str,
    model: MemoryModel,
    budget: int,
    importance: Mapping[str, int] | None = None,
) -> Memory:
    """A fresh memory of the named strategy. `budget` sizes the rolling summary's compaction."""
    if strategy == "window":
        return WindowMemory()
    if strategy == "rolling":
        return RollingSummaryMemory(model, budget)
    if strategy == "ledger":
        return FactLedgerMemory(model, importance)
    if strategy == "hybrid":
        return HybridMemory(model, importance)
    raise ValueError(f"unknown strategy {strategy!r}; choose from {', '.join(STRATEGIES)}")


__all__ = [
    "STRATEGIES",
    "FactLedgerMemory",
    "HybridMemory",
    "Memory",
    "RollingSummaryMemory",
    "WindowMemory",
    "build_memory",
]
