"""Seeded generator for synthetic home and motor claim conversations.

Every conversation has 3-8 calls of 10-30 turns about 14 claim slots. Patterns planted on purpose:

* 30% of conversations change the payout account or the loss address in a later call;
* 15% contain a same-call correction ("the 12th, sorry, the 14th");
* 5% contain a change that is retracted again ("ignore that, keep the old account");
* about 40% of facts are stated once, in call 1, and never repeated;
* 30% of turns are small-talk distractors;
* the spouse's phone number is a near-duplicate of the claimant's;
* 5% of fact-bearing turns are put on a planted extraction-miss list.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta

from stalefact.dataset import Dataset, PlantedMiss
from stalefact.dialogue import Turn
from stalefact.grammar import SLOT_BY_NAME, SLOTS, SlotSpec
from stalefact.truth import ConversationTruth, FactEvent, Truth

CHANGE_FRACTION = 0.30
CORRECTION_FRACTION = 0.15
RETRACTION_FRACTION = 0.05
DISTRACTOR_FRACTION = 0.30
MISS_FRACTION = 0.05
ONCE_PROBABILITY = 0.52
RESTATE_PROBABILITY = 0.30
PREFIX_PROBABILITY = 0.25
CALL_COUNT_WEIGHTS = {3: 1.0, 4: 1.0, 5: 1.0, 6: 1.5, 7: 2.0, 8: 2.5}

CHANGE_SLOTS = ("payout_account", "loss_address")
CORRECTION_SLOTS = ("loss_date", "promised_callback_date", "deductible")

SMALL_TALK = {
    "claimant": (
        "Thanks for taking my call, I know you are busy.",
        "It has been a stressful month, honestly.",
        "The weather has been awful here lately.",
        "My phone has been acting up all week.",
        "I should check my bank app later today.",
        "The kids are home from school so it is a bit noisy.",
        "My neighbour keeps asking how the claim is going.",
        "Sorry about the background noise.",
        "I was on hold with the council earlier.",
        "The road outside is still flooded, if you can believe it.",
        "I have not slept well since it happened.",
        "Is it always this slow in the afternoons?",
        "We were hoping to get this wrapped up before the holidays.",
        "My sister says I should have called sooner.",
        "I will look for the paperwork after we hang up.",
        "Our dog has been barking at the postman all morning.",
    ),
    "handler": (
        "No problem at all, take your time.",
        "I know this is a stressful situation.",
        "Thanks for your patience with me.",
        "It is quite busy on our side this week.",
        "Hope the rest of your day goes well.",
        "That sounds like a rough week.",
        "Appreciate you getting back to me.",
        "Let me know if the line drops.",
        "Happy to go through it again if that helps.",
        "The system is a bit slow today, bear with me.",
    ),
}
PROCEDURAL = {
    "handler": (
        "Let me pull up your file.",
        "Which account should the payout go to?",
        "Could you confirm the best number to reach you?",
        "I will note that on the claim.",
        "Let me read that back to you.",
        "Is there anything else you need from us today?",
        "I will send a summary by email after this call.",
        "Can you spell that for me?",
        "One moment while I check the system.",
        "What is the date of the loss again?",
        "I need to check one thing with the repairer first.",
        "Let me update the notes before we carry on.",
    ),
    "claimant": (
        "Yes, that is right.",
        "Okay, understood.",
        "Go ahead, I am listening.",
        "I can send photos if that helps.",
        "Can you repeat that, please?",
        "That works for me.",
        "I have the documents in front of me now.",
        "Please do, thank you.",
    ),
}
PREFIXES = ("Okay.", "Right.", "Let me check my notes.", "One moment.", "Sure.", "Yes.")


@dataclass
class _Item:
    slot: str
    kind: str  # assert | restate | correct | retract
    value: str
    value_after: str
    speaker: str
    text: str


def _expected_count(n: int, fraction: float) -> int:
    return round(n * fraction)


def _choose_patterns(rng: random.Random, n: int) -> tuple[set[int], set[int], set[int]]:
    """Exact-count conversation indices for the change, retraction and correction patterns."""
    indices = list(range(n))
    changers = set(rng.sample(indices, _expected_count(n, CHANGE_FRACTION)))
    rest = [i for i in indices if i not in changers]
    retractors = set(rng.sample(rest, _expected_count(n, RETRACTION_FRACTION)))
    correctors = set(rng.sample(indices, _expected_count(n, CORRECTION_FRACTION)))
    return changers, retractors, correctors


def _interleave(rng: random.Random, chains: list[list[_Item]]) -> list[_Item]:
    """Random merge of chains that keeps the order inside each chain."""
    pending = [list(c) for c in chains if c]
    merged: list[_Item] = []
    while pending:
        chain = rng.choice(pending)
        merged.append(chain.pop(0))
        pending = [c for c in pending if c]
    return merged


def _render_item(rng: random.Random, spec: SlotSpec, kind: str, value: str, old: str) -> str:
    if kind == "assert":
        template = rng.choice(spec.assert_templates)
        text = template.format(v=value)
    elif kind == "restate":
        template = rng.choice(spec.restate_templates)
        text = template.format(v=value)
    elif kind == "correct":
        template = rng.choice(spec.correct_templates)
        text = template.format(**spec.correct_fields(old, value))
    else:
        text = rng.choice(spec.retract_templates)
    if rng.random() < PREFIX_PROBABILITY:
        text = f"{rng.choice(PREFIXES)} {text}"
    return text


def _call_lengths(rng: random.Random, n_calls: int) -> list[int]:
    return [rng.randint(18, 30)] + [rng.randint(10, 30) for _ in range(n_calls - 1)]


def _build_conversation(
    rng: random.Random,
    cid: str,
    pattern: str | None,
    corrected: bool,
    base_time: datetime,
) -> tuple[list[Turn], ConversationTruth]:
    claim_type = rng.choice(("home", "motor"))
    n_calls = rng.choices(list(CALL_COUNT_WEIGHTS), weights=list(CALL_COUNT_WEIGHTS.values()))[0]

    values: dict[str, str] = {}
    for spec in SLOTS:
        values[spec.name] = spec.sample(rng, values, claim_type)

    patterns: list[str] = []
    change_slot = change_call = None
    new_value = ""
    if pattern in ("change", "retract"):
        change_slot = rng.choice(CHANGE_SLOTS)
        change_call = rng.randint(2, n_calls)
        new_value = SLOT_BY_NAME[change_slot].alter(rng, values[change_slot], values, claim_type)
        patterns.append(f"{pattern}:{change_slot}")
    correct_slot = None
    corrected_value = ""
    if corrected:
        correct_slot = rng.choice(CORRECTION_SLOTS)
        corrected_value = SLOT_BY_NAME[correct_slot].alter(
            rng, values[correct_slot], values, claim_type
        )
        patterns.append(f"correct:{correct_slot}")

    pattern_slots = {s for s in (change_slot, correct_slot) if s}
    once = {
        s.name
        for s in SLOTS
        if s.first_call == 1 and s.name not in pattern_slots and rng.random() < ONCE_PROBABILITY
    }

    def in_force(slot: str, call: int) -> str:
        if slot == change_slot and change_call is not None and call >= change_call:
            return values[slot] if pattern == "retract" else new_value
        if slot == correct_slot:
            return corrected_value
        return values[slot]

    per_call: dict[int, list[list[_Item]]] = {c: [] for c in range(1, n_calls + 1)}
    for spec in SLOTS:
        slot = spec.name
        first = spec.first_call
        if slot == correct_slot:
            per_call[first].append(
                [
                    _Item(slot, "assert", values[slot], values[slot], spec.speaker, ""),
                    _Item(slot, "correct", corrected_value, corrected_value, spec.speaker, ""),
                ]
            )
        else:
            per_call[first].append(
                [_Item(slot, "assert", values[slot], values[slot], spec.speaker, "")]
            )
        if slot == change_slot and change_call is not None:
            chain = [_Item(slot, "correct", new_value, new_value, spec.speaker, "")]
            if pattern == "retract":
                chain.append(_Item(slot, "retract", "", values[slot], spec.speaker, ""))
            per_call[change_call].append(chain)
        if slot in once:
            continue
        for call in range(first + 1, n_calls + 1):
            if call == change_call and slot == change_slot:
                continue
            if rng.random() < RESTATE_PROBABILITY:
                value = in_force(slot, call)
                per_call[call].append([_Item(slot, "restate", value, value, "handler", "")])

    turns: list[Turn] = []
    events: list[FactEvent] = []
    clock = base_time
    lengths = _call_lengths(rng, n_calls)
    for call in range(1, n_calls + 1):
        n = lengths[call - 1]
        distractors = round(DISTRACTOR_FRACTION * n)
        chains = per_call[call]
        capacity = n - distractors
        total = sum(len(c) for c in chains)
        droppable = [c for c in chains if len(c) == 1 and c[0].kind == "restate"]
        rng.shuffle(droppable)
        while total > capacity and droppable:
            chains.remove(droppable.pop())
            total -= 1
        if total > capacity:  # only when mandatory facts alone do not fit
            n = min(30, math.ceil(total / (1 - DISTRACTOR_FRACTION)) + 1)
            distractors = round(DISTRACTOR_FRACTION * n)
        items = _interleave(rng, chains)
        kinds = (
            ["item"] * len(items)
            + ["small"] * distractors
            + ["proc"] * (n - len(items) - distractors)
        )
        rng.shuffle(kinds)
        queue = list(items)
        clock += timedelta(days=rng.randint(1, 9), minutes=rng.randint(0, 300))
        for turn_no, role in enumerate(kinds, start=1):
            clock += timedelta(seconds=rng.randint(8, 70))
            if role == "item":
                item = queue.pop(0)
                spec = SLOT_BY_NAME[item.slot]
                item.text = _render_item(rng, spec, item.kind, item.value, values[item.slot])
                speaker = item.speaker
                text = item.text
                events.append(
                    FactEvent(
                        call_no=call,
                        turn_no=turn_no,
                        slot=item.slot,
                        subject=spec.subject,
                        kind=item.kind,
                        value=item.value,
                        value_after=item.value_after,
                    )
                )
            else:
                speaker = rng.choice(("claimant", "handler"))
                pool = (SMALL_TALK if role == "small" else PROCEDURAL)[speaker]
                text = rng.choice(pool)
            turns.append(
                Turn(
                    conversation_id=cid,
                    call_no=call,
                    turn_no=turn_no,
                    speaker=speaker,
                    text=text,
                    timestamp=clock,
                )
            )

    truth = ConversationTruth(
        conversation_id=cid,
        claim_type=claim_type,
        n_calls=n_calls,
        patterns=patterns,
        once_slots=sorted(once),
        events=events,
    )
    return turns, truth


def generate(seed: int = 7, n_conversations: int = 120) -> Dataset:
    """Generate the full synthetic dataset for a seed."""
    rng = random.Random(seed)
    changers, retractors, correctors = _choose_patterns(rng, n_conversations)
    base = datetime(2026, 1, 5, 9, 0, 0)
    all_turns: list[Turn] = []
    conversation_truths: list[ConversationTruth] = []
    for index in range(n_conversations):
        cid = f"C{index + 1:04d}"
        pattern = "change" if index in changers else "retract" if index in retractors else None
        corrected = index in correctors
        start = base + timedelta(days=rng.randint(0, 150), minutes=rng.randint(0, 420))
        turns, truth = _build_conversation(rng, cid, pattern, corrected, start)
        all_turns.extend(turns)
        conversation_truths.append(truth)

    fact_turns = [(t.conversation_id, e) for t in conversation_truths for e in t.events]
    chosen: list[int] = []
    for kind in ("assert", "restate", "correct", "retract"):
        stratum = [i for i, (_, e) in enumerate(fact_turns) if e.kind == kind]
        count = max(1, _expected_count(len(stratum), MISS_FRACTION)) if stratum else 0
        chosen.extend(rng.sample(stratum, count))
    chosen.sort(
        key=lambda i: (fact_turns[i][0], fact_turns[i][1].call_no, fact_turns[i][1].turn_no)
    )
    misses = [
        PlantedMiss(
            conversation_id=fact_turns[i][0],
            call_no=fact_turns[i][1].call_no,
            turn_no=fact_turns[i][1].turn_no,
            slot=fact_turns[i][1].slot,
            kind=fact_turns[i][1].kind,
        )
        for i in chosen
    ]
    return Dataset(all_turns, Truth(seed=seed, conversations=conversation_truths), misses)
