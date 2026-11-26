"""Dialogue turns and their JSONL storage."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel


class Turn(BaseModel):
    """One utterance in one call of one claim conversation."""

    conversation_id: str
    call_no: int
    turn_no: int
    speaker: str
    text: str
    timestamp: datetime

    @property
    def key(self) -> tuple[str, int, int]:
        """Identity of the turn: (conversation, call, turn)."""
        return (self.conversation_id, self.call_no, self.turn_no)


@dataclass
class Conversation:
    """All turns of one claim, grouped by call."""

    conversation_id: str
    calls: dict[int, list[Turn]] = field(default_factory=dict)

    @property
    def call_numbers(self) -> list[int]:
        return sorted(self.calls)

    def turns(self) -> list[Turn]:
        return [t for c in self.call_numbers for t in self.calls[c]]


def write_turns(path: Path, turns: Iterable[Turn]) -> None:
    """Write turns as one JSON object per line."""
    with path.open("w", encoding="utf-8") as handle:
        for turn in turns:
            handle.write(turn.model_dump_json() + "\n")


def load_turns(path: Path) -> list[Turn]:
    """Read turns from a JSONL file."""
    turns: list[Turn] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                turns.append(Turn.model_validate_json(line))
    return turns


def group_conversations(turns: Iterable[Turn]) -> dict[str, Conversation]:
    """Group turns by conversation and call, each call ordered by turn number."""
    grouped: dict[str, dict[int, list[Turn]]] = defaultdict(lambda: defaultdict(list))
    for turn in turns:
        grouped[turn.conversation_id][turn.call_no].append(turn)
    result: dict[str, Conversation] = {}
    for cid in sorted(grouped):
        calls = {n: sorted(ts, key=lambda t: t.turn_no) for n, ts in grouped[cid].items()}
        result[cid] = Conversation(cid, calls)
    return result
