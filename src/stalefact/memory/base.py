"""The common interface of the four memory strategies."""

from __future__ import annotations

from typing import Protocol

from stalefact.dialogue import Turn


class Memory(Protocol):
    """A conversation memory: feed it turns, then ask for a context within a token budget."""

    name: str

    def ingest(self, turn: Turn) -> None:
        """Take in the next turn of the conversation."""
        ...

    def render(self, budget: int) -> str:
        """Context text of at most `budget` tokens (see stalefact.tokens)."""
        ...


def turn_line(turn: Turn) -> str:
    """How a raw turn appears inside a context."""
    return f"{turn.speaker}: {turn.text}"
