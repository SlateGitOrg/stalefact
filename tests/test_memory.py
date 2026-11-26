import pytest
from conftest import make_turn

from stalefact.memory import STRATEGIES, build_memory
from stalefact.memory.hybrid import HybridMemory
from stalefact.memory.rolling import RollingSummaryMemory
from stalefact.memory.window import WindowMemory
from stalefact.model import FakeMemoryModel
from stalefact.tokens import count_tokens


def test_window_keeps_only_the_most_recent_turns_that_fit():
    window = WindowMemory()
    for i in range(1, 6):
        window.ingest(make_turn(f"word{i} stuff.", turn_no=i))
    # each line "claimant: wordN stuff." is 6 tokens (claimant : word N stuff .)
    rendered = window.render(12)
    assert rendered.splitlines() == ["claimant: word4 stuff.", "claimant: word5 stuff."]
    assert window.render(5) == ""


class CountingModel(FakeMemoryModel):
    def __init__(self):
        super().__init__()
        self.calls: list[tuple[str, int]] = []

    def summarise(self, text, budget):
        self.calls.append((text, budget))
        return super().summarise(text, budget)


def test_rolling_summary_compacts_only_when_over_budget_and_stays_within_it():
    model = CountingModel()
    memory = RollingSummaryMemory(model, budget=60)
    memory.ingest(make_turn("Your claim number is CLM-123456.", turn_no=1, speaker="handler"))
    assert model.calls == []  # still under budget: nothing summarised
    for i in range(2, 12):
        memory.ingest(make_turn(f"Thanks for waiting, number {i}.", turn_no=i))
    assert model.calls, "buffer went over budget, so the model must have been asked to summarise"
    assert all(budget < 60 for _, budget in model.calls)
    assert count_tokens(memory.render(60)) <= 60


def test_rolling_summary_folds_the_prior_summary_into_the_next_one():
    model = CountingModel()
    memory = RollingSummaryMemory(model, budget=60)
    for i in range(1, 16):
        memory.ingest(make_turn(f"The loss happened on {i} March.", turn_no=i))
    assert len(model.calls) >= 2
    second_text = model.calls[1][0]
    assert "March" in second_text and len(second_text.splitlines()) > 1


def test_hybrid_splits_the_budget_seventy_thirty():
    model = FakeMemoryModel()
    hybrid = HybridMemory(model)
    hybrid.ingest(make_turn("Please pay into the account ending 4417.", turn_no=1))
    hybrid.ingest(make_turn("My mobile is 555-0142.", turn_no=2))
    for i in range(3, 12):
        hybrid.ingest(make_turn("Thanks for your patience with me.", turn_no=i, speaker="handler"))
    rendered = hybrid.render(100)
    lines = rendered.splitlines()
    assert lines[:2] == ["payout_account: 4417", "claimant_phone: 555-0142"]
    ledger_part = [ln for ln in lines if ln.startswith(("payout_account", "claimant_phone"))]
    window_part = [ln for ln in lines if ln.startswith("handler:")]
    assert count_tokens("\n".join(ledger_part)) <= 70
    assert count_tokens("\n".join(window_part)) <= 30
    assert window_part, "the window share must hold recent turns"


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("budget", [128, 256, 512, 1024])
def test_render_never_exceeds_the_budget_after_any_turn(strategy, budget, dataset, model):
    conversations = dataset.conversations
    for cid in sorted(conversations)[:3]:
        memory = build_memory(strategy, model, budget)
        for turn in conversations[cid].turns():
            memory.ingest(turn)
            assert count_tokens(memory.render(budget)) <= budget


def test_unknown_strategy_is_rejected(model):
    with pytest.raises(ValueError, match="unknown strategy"):
        build_memory("telepathy", model, 128)
