import random

import pytest

from stalefact.grammar import (
    SLOT_BY_NAME,
    SLOTS,
    ordinal,
    parse_context_line,
    parse_text,
    render_template,
)


def test_there_are_fourteen_slots():
    assert len(SLOTS) == 14
    assert len({s.name for s in SLOTS}) == 14
    assert len({s.question for s in SLOTS}) == 14


@pytest.mark.parametrize(
    ("n", "text"), [(1, "1st"), (2, "2nd"), (3, "3rd"), (12, "12th"), (21, "21st"), (14, "14th")]
)
def test_ordinal(n, text):
    assert ordinal(n) == text


@pytest.mark.parametrize("spec", SLOTS, ids=lambda s: s.name)
def test_every_template_reads_back_what_was_rendered(spec):
    rng = random.Random(5)
    ctx: dict[str, str] = {"claimant_phone": "555-0142"}
    old = spec.sample(rng, ctx, "motor")
    new = spec.alter(rng, old, ctx, "motor")
    for template in spec.assert_templates + spec.restate_templates:
        found = parse_text(render_template(template, v=old))
        assert [(f.slot, f.kind, f.value) for f in found] == [(spec.name, "assert", old)]
    for template in spec.correct_templates:
        text = template.format(**spec.correct_fields(old, new))
        found = parse_text(text)
        assert [(f.slot, f.kind, f.value) for f in found] == [(spec.name, "correct", new)]
    for template in spec.retract_templates:
        found = parse_text(template)
        assert [(f.slot, f.kind, f.value) for f in found] == [(spec.name, "retract", "")]


def test_the_12th_sorry_the_14th_reads_as_a_correction_to_the_14th():
    (fact,) = parse_text("The loss date: the 12th, sorry, the 14th of March.")
    assert (fact.slot, fact.kind, fact.value) == ("loss_date", "correct", "14 March")


def test_small_talk_with_cue_words_is_not_a_fact():
    assert parse_text("My phone has been acting up all week.") == ()
    assert parse_text("Which account should the payout go to?") == ()


def test_spouse_phone_is_never_read_as_the_claimant_phone():
    (fact,) = parse_text("My husband's number is 555-0187.")
    assert fact.slot == "spouse_phone"
    (fact,) = parse_text("You can reach me on 555-0185.")
    assert fact.slot == "claimant_phone"


def test_ledger_lines_are_parsed_by_slot_name():
    (fact,) = parse_context_line("payout_account: 4417")
    assert (fact.slot, fact.value, fact.kind) == ("payout_account", "4417", "assert")
    assert parse_context_line("claimant: hello there") == ()


def test_spouse_phone_sample_is_a_near_duplicate_of_the_claimant_phone():
    rng = random.Random(1)
    ctx = {"claimant_phone": "555-0142"}
    spouse = SLOT_BY_NAME["spouse_phone"].sample(rng, ctx, "home")
    assert spouse != "555-0142"
    assert spouse[:6] == "555-01"
