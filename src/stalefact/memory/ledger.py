"""Fact-ledger memory: the current value and the history of every (subject, slot)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from stalefact.dialogue import Turn
from stalefact.grammar import DEFAULT_IMPORTANCE
from stalefact.model import Fact, MemoryModel
from stalefact.tokens import count_tokens


@dataclass
class LedgerEntry:
    """Values of one (subject, slot), oldest first; the last one is current."""

    history: list[str] = field(default_factory=list)
    updated: int = 0

    @property
    def current(self) -> str:
        return self.history[-1]


class FactLedgerMemory:
    """Extracts facts per turn and keeps the supersession history.

    * assert and correct set the current value (a repeat of the current value only refreshes its
      recency); the old value stays in the history;
    * retract drops the current value and restores the previous one; a retraction with no earlier
      value on record is ignored, because deleting the only known value would lose a fact the
      retraction never referred to.
    """

    name = "ledger"

    def __init__(self, model: MemoryModel, importance: Mapping[str, int] | None = None) -> None:
        self.model = model
        self.importance = dict(DEFAULT_IMPORTANCE if importance is None else importance)
        self._entries: dict[tuple[str, str], LedgerEntry] = {}
        self._clock = 0

    def ingest(self, turn: Turn) -> None:
        for fact in self.model.extract_facts(turn):
            self.apply(fact)

    def apply(self, fact: Fact) -> None:
        """Update the ledger with one extracted fact."""
        self._clock += 1
        key = (fact.subject, fact.slot)
        entry = self._entries.get(key)
        if fact.kind == "retract":
            if entry is not None and len(entry.history) > 1:
                entry.history.pop()
                entry.updated = self._clock
            return
        if entry is None:
            self._entries[key] = LedgerEntry([fact.value], self._clock)
            return
        if entry.current != fact.value:
            entry.history.append(fact.value)
        entry.updated = self._clock

    def current(self, subject: str, slot: str) -> str | None:
        """The current value of a (subject, slot), if the ledger has one."""
        entry = self._entries.get((subject, slot))
        return entry.current if entry else None

    def history(self, subject: str, slot: str) -> list[str]:
        """Every value recorded for a (subject, slot), oldest first."""
        entry = self._entries.get((subject, slot))
        return list(entry.history) if entry else []

    def lines(self) -> list[str]:
        """All `slot: value` lines ordered by slot importance, then recency."""
        ranked = sorted(
            self._entries.items(),
            key=lambda item: (-self.importance.get(item[0][1], 0), -item[1].updated),
        )
        return [f"{slot}: {entry.current}" for (_, slot), entry in ranked]

    def render(self, budget: int) -> str:
        kept: list[str] = []
        remaining = budget
        for line in self.lines():
            cost = count_tokens(line)
            if cost <= remaining:
                kept.append(line)
                remaining -= cost
        return "\n".join(kept)
