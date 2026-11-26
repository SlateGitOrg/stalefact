"""Deterministic token counting.

The counter is a regular-expression approximation of a subword tokenizer: each run of letters is one
token, every digit is its own token (as in several common tokenizers) and every other non-space
character, including an underscore, is one token. It approximates real tokenizers; absolute counts
differ from any particular model's, but it is stable across platforms, needs no model files and
gives every strategy the same yardstick.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

_TOKEN_RE = re.compile(r"[A-Za-z]+|\d|[^\sA-Za-z\d]")


def tokenize(text: str) -> list[str]:
    """Split text into the approximate tokens used for every budget in this project."""
    return _TOKEN_RE.findall(text)


def count_tokens(text: str) -> int:
    """Number of approximate tokens in text."""
    return len(_TOKEN_RE.findall(text))


def fit_recent(lines: Sequence[str], budget: int) -> list[str]:
    """Longest suffix of lines whose combined token count is within budget."""
    kept: list[str] = []
    used = 0
    for line in reversed(lines):
        cost = count_tokens(line)
        if used + cost > budget:
            break
        kept.append(line)
        used += cost
    kept.reverse()
    return kept


def total_tokens(lines: Iterable[str]) -> int:
    """Sum of token counts over lines (whitespace between lines is free)."""
    return sum(count_tokens(line) for line in lines)
