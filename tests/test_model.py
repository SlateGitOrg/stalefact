import json
import os

import httpx
import pytest
from conftest import make_turn

from stalefact.model import (
    FakeMemoryModel,
    ModelError,
    OpenAICompatModel,
    salience_scores,
    split_sentences,
)
from stalefact.tokens import count_tokens


def test_split_sentences_drops_speaker_labels():
    text = "claimant: Okay. The loss happened on 14 March.\nhandler: Noted!"
    assert split_sentences(text) == ["Okay.", "The loss happened on 14 March.", "Noted!"]


def test_rare_words_score_higher_than_common_ones():
    scores = salience_scores(["the the the the", "the claim is open", "zebra quartz 9917"])
    assert scores[2] > scores[1] > scores[0]


def test_summarise_keeps_high_salience_sentences_in_original_order_within_budget():
    model = FakeMemoryModel()
    text = "\n".join(
        [
            "claimant: Thanks, thanks, thanks.",
            "claimant: The loss happened on 14 March.",
            "handler: Thanks, thanks, thanks.",
            "handler: Your claim number is CLM-482913.",
        ]
    )
    summary = model.summarise(text, 24)
    assert summary.splitlines() == [
        "The loss happened on 14 March.",
        "Your claim number is CLM-482913.",
    ]
    assert count_tokens(summary) <= 24
    tiny = model.summarise(text, 8)
    assert count_tokens(tiny) <= 8


def test_summarise_is_deterministic():
    text = "\n".join(f"claimant: Sentence number {i} about thing {i * 7}." for i in range(30))
    model = FakeMemoryModel()
    assert model.summarise(text, 60) == model.summarise(text, 60)


def test_extract_facts_skips_planted_misses_only():
    turn = make_turn("Please pay into the account ending 4417.")
    assert [f.value for f in FakeMemoryModel().extract_facts(turn)] == ["4417"]
    assert FakeMemoryModel(misses=[turn.key]).extract_facts(turn) == []


def test_answer_returns_the_first_matching_line_in_context_order():
    context = "\n".join(
        [
            "claimant: My phone has been acting up.",
            "claimant: Please pay into the account ending 4417.",
            "claimant: I changed banks, so the payout account now ends in 5521, not 4417.",
        ]
    )
    model = FakeMemoryModel()
    assert model.answer("Which account should the payout go to?", context) == "4417"
    assert model.answer("What is the deductible?", context) == "unknown"
    assert model.answer("Is this a question?", context) == "unknown"


def test_answer_reads_ledger_lines_and_ignores_retractions():
    context = "payout_account: 4417\nclaimant: Ignore that, keep the old account."
    assert FakeMemoryModel().answer("Which account should the payout go to?", context) == "4417"


def _mock_client(reply: str, seen: list[httpx.Request]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": reply}}]})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_openai_adapter_extracts_facts_from_a_fenced_json_reply():
    seen: list[httpx.Request] = []
    reply = (
        '```json\n{"facts": [{"subject": "claim", "slot": "payout_account", "value": "5521", '
        '"kind": "correct"}, {"slot": "made_up", "value": "x", "kind": "assert"}]}\n```'
    )
    model = OpenAICompatModel("http://llm.test/v1/", "k", "m", client=_mock_client(reply, seen))
    facts = model.extract_facts(make_turn("anything"))
    assert [(f.slot, f.value, f.kind) for f in facts] == [("payout_account", "5521", "correct")]
    request = seen[0]
    assert str(request.url) == "http://llm.test/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer k"
    assert json.loads(request.content)["model"] == "m"


def test_openai_adapter_answer_and_summarise():
    seen: list[httpx.Request] = []
    model = OpenAICompatModel("http://llm.test", "k", "m", client=_mock_client("4417.", seen))
    assert model.answer("Which account should the payout go to?", "ctx") == "4417"
    long_reply = "\n".join(f"Sentence {i} is here." for i in range(20))
    model = OpenAICompatModel("http://llm.test", "k", "m", client=_mock_client(long_reply, seen))
    summary = model.summarise("text", 30)
    assert count_tokens(summary) <= 30
    assert summary.splitlines()[-1] == "Sentence 19 is here."


def test_openai_adapter_rejects_non_json_fact_replies():
    model = OpenAICompatModel("http://llm.test", "k", "m", client=_mock_client("no json", []))
    with pytest.raises(ModelError):
        model.extract_facts(make_turn("anything"))


def test_from_env_requires_all_three_variables_and_reads_them():
    with pytest.raises(ModelError, match="STALEFACT_LLM_API_KEY"):
        OpenAICompatModel.from_env({"STALEFACT_LLM_BASE_URL": "u", "STALEFACT_LLM_MODEL": "m"})
    env = {
        "STALEFACT_LLM_BASE_URL": "http://x/v1",
        "STALEFACT_LLM_API_KEY": "key",
        "STALEFACT_LLM_MODEL": "mod",
    }
    model = OpenAICompatModel.from_env(env)
    assert (model.base_url, model.api_key, model.model) == ("http://x/v1", "key", "mod")


@pytest.mark.live
@pytest.mark.skipif(os.environ.get("STALEFACT_LIVE") != "1", reason="set STALEFACT_LIVE=1")
def test_live_model_answers_a_simple_question():
    model = OpenAICompatModel.from_env()
    answer = model.answer(
        "Which account should the payout go to?", "payout_account: 4417\nclaim_number: CLM-123456"
    )
    assert "4417" in answer


def test_last_answer_policy_prefers_the_latest_matching_line():
    context = "\n".join(
        [
            "claimant: Please pay into the account ending 4417.",
            "claimant: I changed banks, so the payout account now ends in 5521, not 4417.",
        ]
    )
    question = "Which account should the payout go to?"
    assert FakeMemoryModel(answer_policy="first").answer(question, context) == "4417"
    assert FakeMemoryModel(answer_policy="last").answer(question, context) == "5521"
    with pytest.raises(ValueError, match="answer_policy"):
        FakeMemoryModel(answer_policy="middle")
