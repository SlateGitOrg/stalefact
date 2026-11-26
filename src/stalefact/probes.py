"""Probes asked after every call, and the four-way scoring of their answers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from stalefact.grammar import SLOT_BY_NAME
from stalefact.truth import ConversationTruth


class Outcome(StrEnum):
    """How one answer compares with the value in force at probe time."""

    CORRECT = "correct"
    STALE = "stale"
    MISSING = "missing"
    WRONG = "wrong"


OUTCOMES = tuple(Outcome)
MISSING_ANSWERS = frozenset({"", "unknown"})


@dataclass(frozen=True)
class Probe:
    """One question about one slot at the end of one call."""

    conversation_id: str
    call_no: int
    slot: str
    question: str
    gold: str
    earlier: frozenset[str]
    age: int
    changed: bool


def _normal(text: str) -> str:
    return " ".join(text.lower().split())


def classify(answer: str, gold: str, earlier: frozenset[str] | set[str]) -> Outcome:
    """Score an answer.

    * correct: equals the value in force at probe time;
    * stale: equals a value the slot held or was said to hold earlier (including a retracted one);
    * missing: the model said it does not know;
    * wrong: anything else.
    """
    given = _normal(answer)
    if given in MISSING_ANSWERS:
        return Outcome.MISSING
    if given == _normal(gold):
        return Outcome.CORRECT
    if given in {_normal(v) for v in earlier}:
        return Outcome.STALE
    return Outcome.WRONG


def build_probes(truth: ConversationTruth) -> dict[int, list[Probe]]:
    """Probes for every call: one per slot that has a value in force at the end of that call."""
    by_call: dict[int, list[Probe]] = {}
    for call_no in range(1, truth.n_calls + 1):
        probes: list[Probe] = []
        for slot in truth.slots_stated_by(call_no):
            gold = truth.gold_at(slot, call_no)
            assert gold is not None
            probes.append(
                Probe(
                    conversation_id=truth.conversation_id,
                    call_no=call_no,
                    slot=slot,
                    question=SLOT_BY_NAME[slot].question,
                    gold=gold,
                    earlier=frozenset(truth.earlier_values(slot, call_no)),
                    age=truth.age_at(slot, call_no),
                    changed=truth.changed_by(slot, call_no),
                )
            )
        by_call[call_no] = probes
    return by_call
