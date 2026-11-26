import numpy as np

from stalefact.evaluate import (
    DEFAULT_BUDGETS,
    EvalResult,
    age_bucket,
    evaluate,
    seed_sensitivity,
)
from stalefact.model import FakeMemoryModel
from stalefact.probes import Outcome, build_probes
from stalefact.stats import bootstrap_ratios
from stalefact.truth import STATING_KINDS

STRATEGY_NAMES = ("window", "rolling", "ledger", "hybrid")


def test_ledger_retention_at_256_is_at_least_90_percent(full_eval):
    assert full_eval.stats("ledger", 256).retention.point >= 0.90


def test_ledger_stale_rate_on_changed_slots_is_at_most_5_percent(full_eval):
    changed = full_eval.stats("ledger", 256, lambda r: r.changed)
    assert changed.n > 200
    assert changed.stale.point <= 0.05
    assert changed.stale.point > 0  # the missed corrections make the failure path visible


def test_rolling_summary_stale_rate_on_changed_slots_is_at_least_30_percent(full_eval):
    changed = full_eval.stats("rolling", 256, lambda r: r.changed)
    assert changed.stale.point >= 0.30


def test_window_retention_on_facts_older_than_two_calls_is_at_most_30_percent(full_eval):
    old = full_eval.stats("window", 256, lambda r: r.age > 2)
    assert old.n > 500
    assert old.retention.point <= 0.30


def test_retention_does_not_decrease_as_the_budget_grows(full_eval):
    for strategy in STRATEGY_NAMES:
        values = [full_eval.stats(strategy, b).retention.point for b in DEFAULT_BUDGETS]
        for smaller, larger in zip(values, values[1:], strict=False):
            assert larger >= smaller - 0.01, (strategy, values)


def test_every_render_stayed_within_its_budget(full_eval):
    for (strategy, budget), rows in full_eval.results.items():
        assert rows, (strategy, budget)
        assert max(r.tokens for r in rows) <= budget, (strategy, budget)


def test_ledger_misses_are_exactly_planted_misses_that_were_never_restated(dataset, full_eval):
    miss_keys = dataset.miss_keys
    truths = dataset.truth.by_id()
    expected = set()
    for cid, truth in truths.items():
        for call_no, probes in build_probes(truth).items():
            for probe in probes:
                stated = [
                    e for e in truth.slot_events(probe.slot, call_no) if e.kind in STATING_KINDS
                ]
                if all((cid, e.call_no, e.turn_no) in miss_keys for e in stated):
                    expected.add((cid, call_no, probe.slot))
    assert expected, "the planted misses should leave some facts unrecoverable"
    for budget in (256, 512, 1024):
        observed = {
            (r.conversation_id, r.call_no, r.slot)
            for r in full_eval.probes("ledger", budget)
            if r.outcome is Outcome.MISSING
        }
        assert observed == expected
        assert len(observed) == len(expected)


def test_a_missed_fact_that_is_restated_is_not_missing(dataset, full_eval):
    miss_keys = dataset.miss_keys
    truths = dataset.truth.by_id()
    restated_after_miss = 0
    for m in dataset.misses:
        truth = truths[m.conversation_id]
        later = [
            e
            for e in truth.slot_events(m.slot)
            if e.kind in STATING_KINDS
            and (e.call_no, e.turn_no) > (m.call_no, m.turn_no)
            and (m.conversation_id, e.call_no, e.turn_no) not in miss_keys
        ]
        never_changed = not truth.changed_by(m.slot, truth.n_calls)
        if later and m.kind == "assert" and never_changed:
            restated_after_miss += 1
            call = later[0].call_no
            row = next(
                r
                for r in full_eval.probes("ledger", 256)
                if (r.conversation_id, r.call_no, r.slot) == (m.conversation_id, call, m.slot)
            )
            assert row.outcome is Outcome.CORRECT
    assert restated_after_miss > 0


def test_claimant_phone_probe_never_returns_the_spouse_phone_from_the_ledger(dataset, full_eval):
    truths = dataset.truth.by_id()
    rows = [r for r in full_eval.probes("ledger", 256) if r.slot == "claimant_phone"]
    assert len(rows) > 300
    for r in rows:
        spouse = {e.value for e in truths[r.conversation_id].slot_events("spouse_phone")}
        assert r.answer not in spouse
    assert any(r.outcome is Outcome.CORRECT for r in rows)


def test_probe_outcomes_partition_and_rates_sum_to_one(full_eval):
    for strategy in STRATEGY_NAMES:
        s = full_eval.stats(strategy, 512)
        total = s.retention.point + s.stale.point + s.missing.point + s.wrong.point
        assert abs(total - 1.0) < 1e-9


def test_bootstrap_is_reproducible_and_contains_the_point_estimate(full_eval):
    again = EvalResult(
        full_eval.conversation_ids,
        full_eval.strategies,
        full_eval.budgets,
        full_eval.results,
        seed=full_eval.seed,
        n_resamples=1000,
    )
    other = EvalResult(
        full_eval.conversation_ids,
        full_eval.strategies,
        full_eval.budgets,
        full_eval.results,
        seed=full_eval.seed + 1,
    )
    for strategy in STRATEGY_NAMES:
        for budget in DEFAULT_BUDGETS:
            a, b = full_eval.stats(strategy, budget), again.stats(strategy, budget)
            assert a == b
            for interval in (a.retention, a.stale, a.missing, a.per_100_tokens):
                assert interval.low <= interval.point <= interval.high
    assert full_eval.stats("window", 256).retention != other.stats("window", 256).retention


def test_bootstrap_ratios_on_a_known_case():
    numerators = np.array([[1.0], [1.0], [0.0], [0.0]])
    denominators = np.array([1.0, 1.0, 1.0, 1.0])
    point, low, high = bootstrap_ratios(numerators, denominators, seed=3, n_resamples=1000)
    assert point[0] == 0.5
    assert 0.0 <= low[0] < 0.5 < high[0] <= 1.0
    again = bootstrap_ratios(numerators, denominators, seed=3, n_resamples=1000)
    assert (point, low, high) == tuple(again)


def test_ledger_is_far_more_token_efficient_than_the_window(full_eval):
    ledger = full_eval.stats("ledger", 1024).per_100_tokens.point
    window = full_eval.stats("window", 1024).per_100_tokens.point
    assert ledger > 4 * window


def test_age_buckets_and_changed_breakdown_cover_every_probe(full_eval):
    ages = full_eval.by_age("window", 256)
    assert sum(s.n for s in ages.values()) == full_eval.stats("window", 256).n
    changes = full_eval.by_change("rolling", 256)
    assert changes["changed"].n + changes["unchanged"].n == full_eval.stats("rolling", 256).n
    assert [age_bucket(a) for a in (0, 1, 2, 3, 7)] == ["0", "1", "2", "3+", "3+"]


def test_evaluate_on_a_subset_with_the_fake_model(small_dataset):
    model = FakeMemoryModel(small_dataset.miss_keys)
    ids = sorted(small_dataset.conversations)[:4]
    result = evaluate(small_dataset, model, ["ledger"], [256], conversation_ids=ids)
    assert {r.conversation_id for r in result.probes("ledger", 256)} == set(ids)


def test_rolling_stale_rate_holds_up_across_seeds_but_not_at_a_fixed_threshold():
    """The summary parameters were tuned on seed 7; other seeds show the spread (see README)."""
    rows = seed_sensitivity([1, 2, 3], budget=256)
    assert [r.seed for r in rows] == [1, 2, 3]
    mean_rolling = sum(r.rolling_stale for r in rows) / len(rows)
    assert 0.20 <= mean_rolling <= 0.40
    for row in rows:
        assert row.changed_probes > 150
        assert row.rolling_stale > 0.15
        assert row.rolling_stale - row.ledger_stale > 0.15
        assert row.ledger_retention >= 0.90


def test_most_of_the_rolling_stale_rate_at_large_budgets_comes_from_the_first_match_answerer(
    dataset, full_eval
):
    """Both old and new values sit in the summary, oldest first; 'first' reads the old one."""
    last = evaluate(
        dataset,
        FakeMemoryModel(dataset.miss_keys, answer_policy="last"),
        ["rolling", "ledger"],
        [256, 512, 1024],
    )
    for budget in (256, 512, 1024):
        first_rate = full_eval.stats("rolling", budget, lambda r: r.changed).stale.point
        last_rate = last.stats("rolling", budget, lambda r: r.changed).stale.point
        assert last_rate < first_rate
    assert last.stats("rolling", 1024, lambda r: r.changed).stale.point < 0.10
    assert full_eval.stats("rolling", 1024, lambda r: r.changed).stale.point > 0.80
    # the ledger holds one line per slot, so the answer policy cannot matter to it
    assert (
        last.stats("ledger", 256).retention.point == full_eval.stats("ledger", 256).retention.point
    )
