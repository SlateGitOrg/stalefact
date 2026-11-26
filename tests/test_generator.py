from collections import Counter

from stalefact.dataset import load_dataset, write_dataset
from stalefact.generator import SMALL_TALK, generate
from stalefact.grammar import SLOT_BY_NAME, parse_text
from stalefact.truth import STATING_KINDS


def test_conversation_shape(dataset):
    assert len(dataset.truth.conversations) == 120
    per_call: Counter[tuple[str, int]] = Counter(
        (t.conversation_id, t.call_no) for t in dataset.turns
    )
    assert all(10 <= n <= 30 for n in per_call.values())
    calls_per_conv = Counter(cid for cid, _ in per_call)
    assert all(3 <= n <= 8 for n in calls_per_conv.values())
    assert 14_000 <= len(dataset.turns) <= 17_000


def test_exact_pattern_counts(dataset):
    kinds = Counter(p.split(":")[0] for c in dataset.truth.conversations for p in c.patterns)
    assert kinds == {"change": 36, "correct": 18, "retract": 6}  # 30%, 15%, 5% of 120
    for c in dataset.truth.conversations:
        for p in c.patterns:
            kind, slot = p.split(":")
            if kind in ("change", "retract"):
                assert slot in ("payout_account", "loss_address")


def test_thirty_percent_of_turns_are_small_talk(dataset):
    fact_keys = {
        (c.conversation_id, e.call_no, e.turn_no)
        for c in dataset.truth.conversations
        for e in c.events
    }
    small = {s for pool in SMALL_TALK.values() for s in pool}
    distractors = sum(1 for t in dataset.turns if t.text in small and t.key not in fact_keys)
    assert 0.28 <= distractors / len(dataset.turns) <= 0.32


def test_about_forty_percent_of_facts_are_stated_once_in_call_one(dataset):
    once = sum(len(c.once_slots) for c in dataset.truth.conversations)
    assert 0.37 <= once / (14 * 120) <= 0.43
    for c in dataset.truth.conversations:
        for slot in c.once_slots:
            events = c.slot_events(slot)
            assert len(events) == 1
            assert events[0].call_no == 1


def test_five_percent_of_fact_turns_are_planted_misses(dataset):
    facts = sum(len(c.events) for c in dataset.truth.conversations)
    assert 0.048 <= len(dataset.misses) / facts <= 0.056
    kinds = {m.kind for m in dataset.misses}
    assert {"correct", "retract"} <= kinds  # the failure path is exercised
    keys = {
        (c.conversation_id, e.call_no, e.turn_no)
        for c in dataset.truth.conversations
        for e in c.events
    }
    assert dataset.miss_keys <= keys


def test_grammar_reads_every_fact_turn_and_nothing_else(dataset):
    events = {
        (c.conversation_id, e.call_no, e.turn_no): e
        for c in dataset.truth.conversations
        for e in c.events
    }
    for turn in dataset.turns:
        found = parse_text(turn.text)
        event = events.get(turn.key)
        if event is None:
            assert found == (), turn.text
        else:
            expected_kind = "assert" if event.kind == "restate" else event.kind
            assert [(f.slot, f.kind, f.value) for f in found] == [
                (event.slot, expected_kind, event.value)
            ]


def test_truth_values_in_force_are_consistent(dataset):
    for c in dataset.truth.conversations:
        for slot in SLOT_BY_NAME:
            events = c.slot_events(slot)
            assert events, f"{c.conversation_id} never states {slot}"
            for e in events:
                if e.kind == "retract":
                    assert e.value == ""
                    earlier = [x.value for x in events if x.kind in STATING_KINDS]
                    assert e.value_after in earlier


def test_retraction_conversations_end_on_the_original_value(dataset):
    retract = [
        c for c in dataset.truth.conversations if any(p.startswith("retract") for p in c.patterns)
    ]
    assert len(retract) == 6
    for c in retract:
        slot = (
            c.patterns[0].split(":")[1]
            if c.patterns[0].startswith("retract")
            else c.patterns[1].split(":")[1]
        )
        events = c.slot_events(slot)
        first = events[0].value
        assert [e.kind for e in events if e.kind in ("correct", "retract")] == [
            "correct",
            "retract",
        ]
        assert c.gold_at(slot, c.n_calls) == first


def test_spouse_phone_differs_from_the_claimant_phone_by_the_last_digits(dataset):
    for c in dataset.truth.conversations:
        claimant = c.slot_events("claimant_phone")[0].value
        spouse = c.slot_events("spouse_phone")[0].value
        assert claimant != spouse
        assert claimant[:6] == spouse[:6]


def test_same_seed_same_data_and_different_seed_differs():
    a, b, c = generate(11, 12), generate(11, 12), generate(12, 12)
    assert [t.text for t in a.turns] == [t.text for t in b.turns]
    assert a.truth == b.truth
    assert [t.text for t in a.turns] != [t.text for t in c.turns]


def test_dataset_roundtrips_through_disk(tmp_path, small_dataset):
    write_dataset(tmp_path, small_dataset)
    loaded = load_dataset(tmp_path)
    assert [t.model_dump() for t in loaded.turns] == [t.model_dump() for t in small_dataset.turns]
    assert loaded.truth == small_dataset.truth
    assert loaded.misses == small_dataset.misses
    text = (tmp_path / "planted_misses.yaml").read_text()
    assert text.startswith("# Fact-bearing turns") and "misses:" in text
