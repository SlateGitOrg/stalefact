"""Sliding-window memory: the most recent turns that fit."""

from __future__ import annotations

from collections import deque

from stalefact.dialogue import Turn
from stalefact.memory.base import turn_line
from stalefact.tokens import count_tokens

MAX_TURNS_KEPT = 400


class WindowMemory:
    """Keeps raw turns and renders the longest recent run whose token count fits the budget."""

    name = "window"

    def __init__(self) -> None:
        self._lines: deque[tuple[str, int]] = deque(maxlen=MAX_TURNS_KEPT)

    def ingest(self, turn: Turn) -> None:
        line = turn_line(turn)
        self._lines.append((line, count_tokens(line)))

    def render(self, budget: int) -> str:
        kept: list[str] = []
        used = 0
        for line, cost in reversed(self._lines):
            if used + cost > budget:
                break
            kept.append(line)
            used += cost
        return "\n".join(reversed(kept))
