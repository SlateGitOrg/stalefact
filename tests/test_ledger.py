from conftest import make_turn

from stalefact.memory.ledger import FactLedgerMemory
from stalefact.model import Fact, FakeMemoryModel
from stalefact.tokens import count_tokens


def ingest(ledger, *texts, call_no=1):
    for i, text in enumerate(texts, start=1):
        ledger.ingest(make_turn(text, call_no=call_no, turn_no=i))


def test_address_v2_replaces_v1_in_the_render_and_history_keeps_v1():
    ledger = FactLedgerMemory(FakeMemoryModel())
    ingest(ledger, "The damage is at 14 Oak Street.")
    ingest(ledger, "Correction, the loss address is now 9 Mill Lane, not 14 Oak Street.", call_no=2)
    assert "loss_address: 9 Mill Lane" in ledger.render(100)
    assert "14 Oak Street" not in ledger.render(100)
    assert ledger.history("claim", "loss_address") == ["14 Oak Street", "9 Mill Lane"]


def test_the_12th_sorry_the_14th_stores_the_14th():
    ledger = FactLedgerMemory(FakeMemoryModel())
    ingest(
        ledger,
        "The loss happened on 12 March.",
        "The loss date: the 12th, sorry, the 14th of March.",
    )
    assert ledger.current("claim", "loss_date") == "14 March"
    assert ledger.history("claim", "loss_date") == ["12 March", "14 March"]


def test_a_retraction_restores_the_previous_payout_account():
    ledger = FactLedgerMemory(FakeMemoryModel())
    ingest(
        ledger,
        "Please pay into the account ending 4417.",
        "I changed banks, so the payout account now ends in 5521, not 4417.",
    )
    assert ledger.current("claim", "payout_account") == "5521"
    ingest(ledger, "Ignore that, keep the old account.", call_no=2)
    assert ledger.current("claim", "payout_account") == "4417"
    assert ledger.history("claim", "payout_account") == ["4417"]


def test_a_retraction_with_nothing_earlier_is_ignored():
    ledger = FactLedgerMemory(FakeMemoryModel())
    ingest(ledger, "Please pay into the account ending 4417.", "Ignore that, keep the old account.")
    assert ledger.current("claim", "payout_account") == "4417"


def test_restating_the_current_value_does_not_grow_the_history():
    ledger = FactLedgerMemory(FakeMemoryModel())
    ingest(ledger, "My mobile is 555-0142.", "Confirming the claimant phone number 555-0142.")
    assert ledger.history("claimant", "claimant_phone") == ["555-0142"]


def test_render_orders_by_importance_then_recency():
    ledger = FactLedgerMemory(FakeMemoryModel(), importance={"payout_account": 5, "deductible": 5})
    ledger.apply(Fact("claim", "payout_account", "4417", "assert"))
    ledger.apply(Fact("claim", "deductible", "$500", "assert"))
    ledger.apply(Fact("claim", "repair_shop", "Fenwick Roofing", "assert"))
    ledger.apply(Fact("claim", "payout_account", "4417", "assert"))  # refresh recency
    lines = ledger.render(1000).splitlines()
    assert lines == ["payout_account: 4417", "deductible: $500", "repair_shop: Fenwick Roofing"]


def test_render_fills_the_budget_with_the_most_important_lines_first():
    ledger = FactLedgerMemory(FakeMemoryModel())
    ledger.apply(Fact("claim", "payout_account", "4417", "assert"))  # 8 tokens
    ledger.apply(Fact("spouse", "spouse_phone", "555-0187", "assert"))  # 12 tokens
    rendered = ledger.render(10)
    assert rendered == "payout_account: 4417"
    assert count_tokens(rendered) <= 10


def test_a_missed_correction_leaves_the_old_value_in_the_ledger():
    first = make_turn("Please pay into the account ending 4417.", turn_no=1)
    changed = make_turn(
        "I changed banks, so the payout account now ends in 5521, not 4417.", call_no=2, turn_no=3
    )
    ledger = FactLedgerMemory(FakeMemoryModel(misses=[changed.key]))
    ledger.ingest(first)
    ledger.ingest(changed)
    assert ledger.current("claim", "payout_account") == "4417"


def test_claimant_phone_stays_separate_from_the_spouse_phone():
    ledger = FactLedgerMemory(FakeMemoryModel())
    ingest(ledger, "You can reach me on 555-0142.", "My husband's number is 555-0147.")
    assert ledger.current("claimant", "claimant_phone") == "555-0142"
    assert ledger.current("spouse", "spouse_phone") == "555-0147"
