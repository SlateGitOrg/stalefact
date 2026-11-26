"""Rolling-summary memory: old turns are folded into a model-written summary."""

from __future__ import annotations

from stalefact.dialogue import Turn
from stalefact.memory.base import turn_line
from stalefact.model import MemoryModel
from stalefact.tokens import count_tokens, fit_recent

SUMMARY_SHARE = 0.65


class RollingSummaryMemory:
    """Recent turns stay verbatim; when summary plus buffer exceed the budget, the oldest turns and
    the prior summary are summarised together into at most SUMMARY_SHARE of the budget."""

    name = "rolling"

    def __init__(self, model: MemoryModel, budget: int) -> None:
        self.model = model
        self.budget = budget
        self._summary: list[str] = []
        self._summary_tokens = 0
        self._buffer: list[tuple[str, int]] = []
        self._buffer_tokens = 0

    def ingest(self, turn: Turn) -> None:
        line = turn_line(turn)
        cost = count_tokens(line)
        self._buffer.append((line, cost))
        self._buffer_tokens += cost
        if self._summary_tokens + self._buffer_tokens > self.budget:
            self._compact()

    def _compact(self) -> None:
        summary_budget = int(self.budget * SUMMARY_SHARE)
        recent_cap = self.budget - summary_budget
        folded: list[str] = []
        while self._buffer and self._buffer_tokens > recent_cap:
            line, cost = self._buffer.pop(0)
            self._buffer_tokens -= cost
            folded.append(line)
        text = "\n".join([*self._summary, *folded])
        reply = self.model.summarise(text, summary_budget)
        lines = fit_recent([ln for ln in reply.splitlines() if ln.strip()], summary_budget)
        self._summary = lines
        self._summary_tokens = sum(count_tokens(ln) for ln in lines)

    def render(self, budget: int) -> str:
        lines = [*self._summary, *(line for line, _ in self._buffer)]
        return "\n".join(fit_recent(lines, budget))
