"""A seeded dataset on disk: turns, ground truth and the planted extraction misses."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel

from stalefact.dialogue import Conversation, Turn, group_conversations, load_turns, write_turns
from stalefact.truth import Truth

TURNS_FILE = "turns.jsonl"
TRUTH_FILE = "truth.json"
MISSES_FILE = "planted_misses.yaml"

MissKey = tuple[str, int, int]


class PlantedMiss(BaseModel):
    """A fact-bearing turn the offline extractor is made to miss."""

    conversation_id: str
    call_no: int
    turn_no: int
    slot: str
    kind: str

    @property
    def key(self) -> MissKey:
        return (self.conversation_id, self.call_no, self.turn_no)


@dataclass
class Dataset:
    """Everything `stalefact seed` writes."""

    turns: list[Turn]
    truth: Truth
    misses: list[PlantedMiss]

    @property
    def conversations(self) -> dict[str, Conversation]:
        return group_conversations(self.turns)

    @property
    def miss_keys(self) -> set[MissKey]:
        return {m.key for m in self.misses}


def write_dataset(out_dir: Path, data: Dataset) -> None:
    """Write turns.jsonl, truth.json and planted_misses.yaml into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    write_turns(out_dir / TURNS_FILE, data.turns)
    (out_dir / TRUTH_FILE).write_text(data.truth.model_dump_json(), encoding="utf-8")
    header = (
        "# Fact-bearing turns the offline extractor fails on (about 5% of them).\n"
        "# A fact on this list that is never restated cannot enter the ledger.\n"
    )
    body = yaml.safe_dump(
        {"misses": [m.model_dump() for m in data.misses]},
        sort_keys=False,
        default_flow_style=None,
        width=120,
    )
    (out_dir / MISSES_FILE).write_text(header + body, encoding="utf-8")


def load_planted_misses(path: Path) -> list[PlantedMiss]:
    """Read planted_misses.yaml."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [PlantedMiss.model_validate(item) for item in raw.get("misses", [])]


def load_dataset(data_dir: Path) -> Dataset:
    """Load a dataset written by write_dataset."""
    turns_path = data_dir / TURNS_FILE
    if not turns_path.exists():
        raise FileNotFoundError(f"{turns_path} not found; run `stalefact seed` first")
    truth = Truth.model_validate(json.loads((data_dir / TRUTH_FILE).read_text(encoding="utf-8")))
    misses = load_planted_misses(data_dir / MISSES_FILE)
    return Dataset(load_turns(turns_path), truth, misses)
