from stalefact.tokens import count_tokens, fit_recent, tokenize, total_tokens


def test_tokenize_splits_letters_digits_and_punctuation():
    assert tokenize("payout_account: 4417") == ["payout", "_", "account", ":", "4", "4", "1", "7"]


def test_count_is_deterministic_and_ignores_whitespace():
    assert count_tokens("a  b\n c") == 3
    assert count_tokens("CLM-482913") == 8  # CLM, -, six digits


def test_fit_recent_keeps_the_newest_lines_that_fit():
    lines = ["one two", "three four", "five six"]  # 2 tokens each
    assert fit_recent(lines, 4) == ["three four", "five six"]
    assert fit_recent(lines, 1) == []
    assert total_tokens(fit_recent(lines, 100)) == 6
