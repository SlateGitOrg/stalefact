"""Hybrid memory: the ledger render plus a recent window, splitting the budget 70/30."""

from __future__ import annotations

from collections.abc import Mapping

from stalefact.dialogue import Turn
from stalefact.memory.ledger import FactLedgerMemory
from stalefact.memory.window import WindowMemory
from stalefact.model import MemoryModel

LEDGER_SHARE = 0.7


class HybridMemory:
    """The ledger gets 70% of the budget and a sliding window the other 30%."""

    name = "hybrid"

    def __init__(self, model: MemoryModel, importance: Mapping[str, int] | None = None) -> None:
        self.ledger = FactLedgerMemory(model, importance)
        self.window = WindowMemory()

    def ingest(self, turn: Turn) -> None:
        self.ledger.ingest(turn)
        self.window.ingest(turn)

    def render(self, budget: int) -> str:
        ledger_budget = round(budget * LEDGER_SHARE)
        parts = [
            self.ledger.render(ledger_budget),
            self.window.render(budget - ledger_budget),
        ]
        return "\n".join(p for p in parts if p)
