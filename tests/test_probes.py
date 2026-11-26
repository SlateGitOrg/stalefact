import pytest

from stalefact.probes import Outcome, build_probes, classify
from stalefact.truth import ConversationTruth, FactEvent


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("5521", Outcome.CORRECT),
        (" 5521 ", Outcome.CORRECT),
        ("4417", Outcome.STALE),
        ("unknown", Outcome.MISSING),
        ("Unknown", Outcome.MISSING),
        ("", Outcome.MISSING),
        ("9999", Outcome.WRONG),
    ],
)
def test_classify_hand_made_fixtures(answer, expected):
    assert classify(answer, gold="5521", earlier=frozenset({"4417", "3030"})) is expected


def test_a_value_mentioned_earlier_is_stale_not_credited():
    assert classify("3030", "5521", {"4417", "3030"}) is Outcome.STALE


def event(call, turn, kind, value, after, slot="payout_account"):
    return FactEvent(
        call_no=call,
        turn_no=turn,
        slot=slot,
        subject="claim",
        kind=kind,
        value=value,
        value_after=after,
    )


def make_truth():
    return ConversationTruth(
        conversation_id="T1",
        claim_type="home",
        n_calls=4,
        events=[
            event(1, 3, "assert", "4417", "4417"),
            event(2, 5, "restate", "4417", "4417"),
            event(3, 2, "correct", "5521", "5521"),
            event(3, 9, "retract", "", "4417"),
            event(2, 8, "assert", "Friday", "Friday", slot="promised_callback_date"),
        ],
    )


def test_gold_is_the_value_in_force_at_the_end_of_each_call():
    truth = make_truth()
    assert [truth.gold_at("payout_account", c) for c in (1, 2, 3, 4)] == [
        "4417",
        "4417",
        "4417",
        "4417",
    ]
    assert truth.gold_at("promised_callback_date", 1) is None
    assert truth.gold_at("promised_callback_date", 2) == "Friday"


def test_a_retracted_value_counts_as_stale_after_the_retraction():
    probes = build_probes(make_truth())
    probe = next(p for p in probes[3] if p.slot == "payout_account")
    assert probe.gold == "4417"
    assert probe.earlier == frozenset({"5521"})
    assert probe.changed is True
    assert classify("5521", probe.gold, probe.earlier) is Outcome.STALE


def test_probes_cover_only_slots_stated_by_each_call_and_track_age():
    probes = build_probes(make_truth())
    assert [p.slot for p in probes[1]] == ["payout_account"]
    assert sorted(p.slot for p in probes[2]) == ["payout_account", "promised_callback_date"]
    by_slot = {p.slot: p for p in probes[4]}
    assert by_slot["payout_account"].age == 2  # 4417 was last stated by the restate in call 2
    assert by_slot["promised_callback_date"].age == 2
