"""The claim-call utterance grammar: slots, their phrasings and the regexes that read them back.

The synthetic generator renders every fact-bearing turn from the templates below, and the offline
model reads facts back out of text by compiling those same templates into regular expressions. That
coupling is deliberate: the offline extractor is a stand-in for a competent extraction model, so the
experiment measures what memory strategies forget and supersede, not how well a regex parses.
"""

from __future__ import annotations

import random
import re
import string
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache
from typing import Any

MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
STREETS = (
    "Oak",
    "Elm",
    "Maple",
    "Harbour",
    "Willow",
    "Cedar",
    "Kings",
    "Mill",
    "Station",
    "Orchard",
    "Bridge",
    "Hill",
    "Church",
    "Victoria",
    "Park",
)
STREET_TYPES = ("Street", "Road", "Lane", "Avenue", "Close")
SHOPS = (
    "Harbour Body Works",
    "Apex Collision Centre",
    "Northgate Auto Repair",
    "Fenwick Roofing",
    "Calder Restoration",
    "Millbank Motors",
)
ADJUSTERS = (
    "Dana Whitfield",
    "Marcus Ellery",
    "Priya Raman",
    "Helen Okafor",
    "Jonas Lindqvist",
    "Rosa Delgado",
    "Callum Brightwell",
    "Mei Tanaka",
)
CONTACT_METHODS = ("email", "phone", "text message", "post")
YES_NO = ("yes", "no")
DEDUCTIBLES = ("$250", "$500", "$750", "$1000")

ORDINAL_PATTERN = r"\d{1,2}(?:st|nd|rd|th)"


def ordinal(n: int) -> str:
    """12 -> '12th', 21 -> '21st'."""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _alternation(options: tuple[str, ...]) -> str:
    ordered = sorted(options, key=len, reverse=True)
    return "(?:" + "|".join(re.escape(o) for o in ordered) + ")"


Context = dict[str, str]
Sampler = Callable[[random.Random, Context, str], str]


@dataclass(frozen=True)
class SlotSpec:
    """One fact slot of a claim: how it is asked, said, corrected and ranked."""

    name: str
    subject: str
    question: str
    speaker: str
    first_call: int
    importance: int
    pattern: str
    sample: Sampler
    assert_templates: tuple[str, ...]
    restate_templates: tuple[str, ...]
    correct_templates: tuple[str, ...] = ()
    retract_templates: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def alter(self, rng: random.Random, old: str, ctx: Context, kind: str) -> str:
        """A different, plausible value for this slot (used for changes and corrections)."""
        if self.name == "loss_date":
            day, month = old.split(" ", 1)
            choices = [d for d in range(1, 29) if d != int(day)]
            return f"{rng.choice(choices)} {month}"
        for _ in range(100):
            candidate = self.sample(rng, ctx, kind)
            if candidate != old:
                return candidate
        raise RuntimeError(f"could not draw a new value for {self.name}")

    def correct_fields(self, old: str, new: str) -> dict[str, str]:
        """Template fields for rendering a correction of old into new."""
        if self.name == "loss_date":
            old_day, month = old.split(" ", 1)
            new_day, _ = new.split(" ", 1)
            return {
                "d_old": ordinal(int(old_day)),
                "d_new": ordinal(int(new_day)),
                "month": month,
            }
        return {"old": old, "new": new}

    def compose_correct(self, groups: dict[str, str]) -> str:
        """The corrected value recovered from a matched correction template."""
        if self.name == "loss_date":
            day = int(re.match(r"\d+", groups["d_new"]).group())  # type: ignore[union-attr]
            return f"{day} {groups['month']}"
        return groups["new"]


def _digits(rng: random.Random, n: int) -> str:
    return "".join(str(rng.randint(0, 9)) for _ in range(n))


def _sample_phone(rng: random.Random, ctx: Context, kind: str) -> str:
    return f"555-01{_digits(rng, 2)}"


def _sample_spouse_phone(rng: random.Random, ctx: Context, kind: str) -> str:
    base = ctx.get("claimant_phone", "555-0100")
    head, tail = base[:-2], int(base[-2:])
    return f"{head}{(tail + rng.randint(1, 9)) % 100:02d}"


SLOTS: tuple[SlotSpec, ...] = (
    SlotSpec(
        name="claim_number",
        subject="claim",
        question="What is the claim number?",
        speaker="handler",
        first_call=1,
        importance=7,
        pattern=r"CLM-\d{6}",
        sample=lambda rng, ctx, kind: f"CLM-{rng.randint(100000, 999999)}",
        assert_templates=(
            "Your claim number is {v}.",
            "For reference, the claim number is {v}.",
        ),
        restate_templates=("As a reminder, the claim number is {v}.",),
    ),
    SlotSpec(
        name="loss_date",
        subject="claim",
        question="When did the loss happen?",
        speaker="claimant",
        first_call=1,
        importance=8,
        pattern=r"\d{1,2} " + _alternation(MONTHS),
        sample=lambda rng, ctx, kind: f"{rng.randint(1, 28)} {rng.choice(MONTHS[:10])}",
        assert_templates=(
            "The loss happened on {v}.",
            "It was on {v} that this happened.",
        ),
        restate_templates=("Just confirming the loss date is {v}, is that right?",),
        correct_templates=("The loss date: the {d_old}, sorry, the {d_new} of {month}.",),
    ),
    SlotSpec(
        name="loss_address",
        subject="claim",
        question="What is the loss address?",
        speaker="claimant",
        first_call=1,
        importance=8,
        pattern=r"\d{1,3} " + _alternation(STREETS) + " " + _alternation(STREET_TYPES),
        sample=lambda rng, ctx, kind: (
            f"{rng.randint(1, 199)} {rng.choice(STREETS)} {rng.choice(STREET_TYPES)}"
        ),
        assert_templates=(
            "The damage is at {v}.",
            "The address of the loss is {v}.",
        ),
        restate_templates=("Confirming the loss address, {v}, correct?",),
        correct_templates=("Correction, the loss address is now {new}, not {old}.",),
        retract_templates=(
            "Ignore that, keep the old address.",
            "Scratch that, keep the old address.",
        ),
    ),
    SlotSpec(
        name="payout_account",
        subject="claim",
        question="Which account should the payout go to?",
        speaker="claimant",
        first_call=1,
        importance=10,
        pattern=r"\d{4}",
        sample=lambda rng, ctx, kind: str(rng.randint(1000, 9999)),
        assert_templates=(
            "Please pay into the account ending {v}.",
            "My bank account for the payout ends in {v}.",
        ),
        restate_templates=("Confirming the payout goes to the account ending {v}.",),
        correct_templates=("I changed banks, so the payout account now ends in {new}, not {old}.",),
        retract_templates=(
            "Ignore that, keep the old account.",
            "Scratch that, keep the old account.",
        ),
    ),
    SlotSpec(
        name="contact_preference",
        subject="claimant",
        question="How does the claimant prefer to be contacted?",
        speaker="claimant",
        first_call=1,
        importance=5,
        pattern=_alternation(CONTACT_METHODS),
        sample=lambda rng, ctx, kind: rng.choice(CONTACT_METHODS),
        assert_templates=(
            "I prefer to be contacted by {v}.",
            "The best way to reach me is {v}.",
        ),
        restate_templates=("I have you down as preferring {v}.",),
    ),
    SlotSpec(
        name="promised_callback_date",
        subject="adjuster",
        question="By when was the callback promised?",
        speaker="handler",
        first_call=2,
        importance=9,
        pattern=_alternation(WEEKDAYS),
        sample=lambda rng, ctx, kind: rng.choice(WEEKDAYS),
        assert_templates=(
            "I will call you back by {v}.",
            "Expect a call from me by {v}.",
        ),
        restate_templates=("A reminder that I promised a callback by {v}.",),
        correct_templates=("Sorry, I now owe you the callback by {new}, not {old}.",),
    ),
    SlotSpec(
        name="repair_shop",
        subject="claim",
        question="Which repairer is doing the work?",
        speaker="handler",
        first_call=2,
        importance=6,
        pattern=_alternation(SHOPS),
        sample=lambda rng, ctx, kind: rng.choice(SHOPS[:3] if kind == "motor" else SHOPS[3:]),
        assert_templates=(
            "The approved repairer is {v}.",
            "We have booked the repairs with {v}.",
        ),
        restate_templates=("As agreed, the repairs are with {v}.",),
    ),
    SlotSpec(
        name="deductible",
        subject="claim",
        question="What is the deductible?",
        speaker="handler",
        first_call=1,
        importance=7,
        pattern=_alternation(DEDUCTIBLES),
        sample=lambda rng, ctx, kind: rng.choice(DEDUCTIBLES),
        assert_templates=(
            "Your deductible on this claim is {v}.",
            "The excess you pay is {v}.",
        ),
        restate_templates=("Just to repeat, the deductible is {v}.",),
        correct_templates=("Correction, the deductible is {new}, not {old}.",),
    ),
    SlotSpec(
        name="rental_authorized",
        subject="claim",
        question="Is a rental authorized?",
        speaker="handler",
        first_call=2,
        importance=4,
        pattern=_alternation(YES_NO),
        sample=lambda rng, ctx, kind: rng.choice(YES_NO),
        assert_templates=(
            "Rental authorized: {v}.",
            "As for a rental vehicle, authorized: {v}.",
        ),
        restate_templates=("To confirm, rental authorized is {v}.",),
    ),
    SlotSpec(
        name="injury_reported",
        subject="claim",
        question="Was an injury reported?",
        speaker="claimant",
        first_call=1,
        importance=6,
        pattern=_alternation(YES_NO),
        sample=lambda rng, ctx, kind: rng.choice(YES_NO),
        assert_templates=(
            "Injuries reported: {v}.",
            "Regarding injuries, anyone hurt: {v}.",
        ),
        restate_templates=("Noting again, injury reported is {v}.",),
    ),
    SlotSpec(
        name="police_report_number",
        subject="claim",
        question="What is the police report number?",
        speaker="claimant",
        first_call=1,
        importance=5,
        pattern=r"PR-\d{5}",
        sample=lambda rng, ctx, kind: f"PR-{rng.randint(10000, 99999)}",
        assert_templates=(
            "The police gave me report number {v}.",
            "My police report number is {v}.",
        ),
        restate_templates=("I have the police report number as {v}.",),
    ),
    SlotSpec(
        name="adjuster_name",
        subject="adjuster",
        question="Who is the adjuster?",
        speaker="handler",
        first_call=1,
        importance=6,
        pattern=_alternation(ADJUSTERS),
        sample=lambda rng, ctx, kind: rng.choice(ADJUSTERS),
        assert_templates=(
            "I am your adjuster, {v}.",
            "My name is {v} and I will be handling the adjustment.",
        ),
        restate_templates=("Your adjuster is still {v}.",),
    ),
    SlotSpec(
        name="claimant_phone",
        subject="claimant",
        question="What is the claimant's phone number?",
        speaker="claimant",
        first_call=1,
        importance=9,
        pattern=r"\d{3}-\d{4}",
        sample=_sample_phone,
        assert_templates=(
            "You can reach me on {v}.",
            "My mobile is {v}.",
        ),
        restate_templates=("Confirming the claimant phone number {v}.",),
    ),
    SlotSpec(
        name="spouse_phone",
        subject="spouse",
        question="What is the spouse's phone number?",
        speaker="claimant",
        first_call=1,
        importance=3,
        pattern=r"\d{3}-\d{4}",
        sample=_sample_spouse_phone,
        assert_templates=(
            "My husband's number is {v}.",
            "If I do not answer, try my partner on {v}.",
        ),
        restate_templates=("The backup contact on file is the spouse at {v}.",),
    ),
)

SLOT_BY_NAME: dict[str, SlotSpec] = {s.name: s for s in SLOTS}
SLOT_BY_QUESTION: dict[str, SlotSpec] = {s.question: s for s in SLOTS}
DEFAULT_IMPORTANCE: dict[str, int] = {s.name: s.importance for s in SLOTS}


def render_template(template: str, **fields: str) -> str:
    """Fill a template's fields."""
    return template.format(**fields)


@dataclass(frozen=True)
class ParsedFact:
    """A fact read out of a piece of text."""

    subject: str
    slot: str
    value: str
    kind: str  # assert | correct | retract


def _compile(template: str, spec: SlotSpec) -> re.Pattern[str]:
    parts: list[str] = []
    for literal, name, _, _ in string.Formatter().parse(template):
        parts.append(re.escape(literal))
        if name is None:
            continue
        if name in ("v", "old", "new"):
            parts.append(f"(?P<{name}>{spec.pattern})")
        elif name in ("d_old", "d_new"):
            parts.append(f"(?P<{name}>{ORDINAL_PATTERN})")
        elif name == "month":
            parts.append(f"(?P<month>{_alternation(MONTHS)})")
        else:
            raise ValueError(f"unknown template field {name!r}")
    return re.compile("".join(parts))


@dataclass(frozen=True)
class _Reader:
    spec: SlotSpec
    kind: str
    regex: re.Pattern[str]


def _build_readers() -> tuple[_Reader, ...]:
    readers: list[_Reader] = []
    for spec in SLOTS:
        for template in spec.assert_templates + spec.restate_templates:
            readers.append(_Reader(spec, "assert", _compile(template, spec)))
        for template in spec.correct_templates:
            readers.append(_Reader(spec, "correct", _compile(template, spec)))
        for template in spec.retract_templates:
            readers.append(_Reader(spec, "retract", _compile(template, spec)))
    return tuple(readers)


_READERS = _build_readers()
_LEDGER_LINE = re.compile(r"^([a-z_]+): (.+)$")


@cache
def parse_text(text: str) -> tuple[ParsedFact, ...]:
    """Every fact the grammar can read from text (a turn, a summary sentence or a ledger line)."""
    found: list[ParsedFact] = []
    for reader in _READERS:
        match = reader.regex.search(text)
        if match is None:
            continue
        spec = reader.spec
        if reader.kind == "retract":
            value = ""
        elif reader.kind == "correct":
            value = spec.compose_correct(match.groupdict())
        else:
            value = match.group("v")
        found.append(ParsedFact(spec.subject, spec.name, value, reader.kind))
    return tuple(found)


@cache
def parse_context_line(line: str) -> tuple[ParsedFact, ...]:
    """Like parse_text, but also reads a ledger line of the form 'slot: value'."""
    ledger = _LEDGER_LINE.match(line)
    if ledger and ledger.group(1) in SLOT_BY_NAME:
        spec = SLOT_BY_NAME[ledger.group(1)]
        return (ParsedFact(spec.subject, spec.name, ledger.group(2), "assert"),)
    return parse_text(line)
