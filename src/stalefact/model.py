"""The model behind the memory strategies: a protocol, a deterministic fake and an HTTP adapter."""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import httpx

from stalefact.dataset import MISSES_FILE, MissKey, load_planted_misses
from stalefact.dialogue import Turn
from stalefact.grammar import (
    SLOT_BY_NAME,
    SLOT_BY_QUESTION,
    parse_context_line,
    parse_text,
)
from stalefact.tokens import count_tokens

FACT_KINDS = ("assert", "correct", "retract")
UNKNOWN = "unknown"
_SPEAKER_PREFIX = re.compile(r"^(?:claimant|handler): ")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True)
class Fact:
    """A fact extracted from a turn."""

    subject: str
    slot: str
    value: str
    kind: str  # assert | correct | retract


class MemoryModel(Protocol):
    """What a memory strategy needs from a language model."""

    def summarise(self, text: str, budget: int) -> str:
        """Compress text into at most `budget` tokens."""
        ...

    def extract_facts(self, turn: Turn) -> list[Fact]:
        """Facts stated, corrected or retracted in one turn."""
        ...

    def answer(self, question: str, context: str) -> str:
        """Answer a question from the context only, or return 'unknown'."""
        ...


def split_sentences(text: str) -> list[str]:
    """Sentences of a multi-line text, with leading 'speaker: ' labels removed."""
    sentences: list[str] = []
    for line in text.splitlines():
        line = _SPEAKER_PREFIX.sub("", line.strip())
        sentences.extend(s.strip() for s in _SENTENCE_SPLIT.split(line) if s.strip())
    return sentences


def salience_scores(sentences: list[str]) -> list[float]:
    """TF-IDF salience of each sentence, treating sentences as the documents.

    A sentence scores the mean tf*idf weight of its words, so rare words (numbers, names) raise a
    sentence's score and sheer length does not.
    """
    words = [_WORD.findall(s.lower()) for s in sentences]
    document_frequency: Counter[str] = Counter()
    for ws in words:
        document_frequency.update(set(ws))
    n = len(sentences)
    scores: list[float] = []
    for ws in words:
        if not ws:
            scores.append(0.0)
            continue
        counts = Counter(ws)
        weight = sum(
            tf * (math.log((1 + n) / (1 + document_frequency[w])) + 1.0) for w, tf in counts.items()
        )
        scores.append(weight / len(ws))
    return scores


@lru_cache(maxsize=8)
def _slot_values(context: str) -> dict[str, list[str]]:
    """For each slot, the values stated on the context lines that mention it, in context order."""
    values: dict[str, list[str]] = {}
    for line in context.splitlines():
        for fact in parse_context_line(line.strip()):
            if fact.kind != "retract":
                values.setdefault(fact.slot, []).append(fact.value)
    return values


ANSWER_POLICIES = ("first", "last")


class FakeMemoryModel:
    """Deterministic offline model.

    `summarise` is extractive: it keeps the sentences with the highest TF-IDF salience, in their
    original order, until the budget is full. `extract_facts` reads the generator's utterance
    grammar and misses exactly the turns on the planted-miss list. `answer` finds the slot in the
    context and returns the value on the first line that states it, in context order
    (`answer_policy="first"`, the default). `answer_policy="last"` returns the value on the last
    such line instead; it exists to measure how much of a strategy's stale rate comes from the
    answerer preferring the earliest mention rather than from what the memory kept.
    """

    def __init__(self, misses: Iterable[MissKey] = (), answer_policy: str = "first") -> None:
        if answer_policy not in ANSWER_POLICIES:
            raise ValueError(f"answer_policy must be one of {ANSWER_POLICIES}")
        self.misses: frozenset[MissKey] = frozenset(misses)
        self.answer_policy = answer_policy

    @classmethod
    def from_data_dir(cls, data_dir: Path, answer_policy: str = "first") -> FakeMemoryModel:
        """Build the fake with the planted misses stored next to a seeded dataset."""
        misses = load_planted_misses(data_dir / MISSES_FILE)
        return cls((m.key for m in misses), answer_policy)

    def summarise(self, text: str, budget: int) -> str:
        sentences = split_sentences(text)
        scores = salience_scores(sentences)
        order = sorted(range(len(sentences)), key=lambda i: (-scores[i], -i))
        chosen: list[int] = []
        remaining = budget
        for i in order:
            cost = count_tokens(sentences[i])
            if cost <= remaining:
                chosen.append(i)
                remaining -= cost
        return "\n".join(sentences[i] for i in sorted(chosen))

    def extract_facts(self, turn: Turn) -> list[Fact]:
        if turn.key in self.misses:
            return []
        return [Fact(f.subject, f.slot, f.value, f.kind) for f in parse_text(turn.text)]

    def answer(self, question: str, context: str) -> str:
        spec = SLOT_BY_QUESTION.get(question)
        if spec is None:
            return UNKNOWN
        values = _slot_values(context).get(spec.name)
        if not values:
            return UNKNOWN
        return values[0] if self.answer_policy == "first" else values[-1]


class ModelError(RuntimeError):
    """The remote model returned an unusable response."""


class OpenAICompatModel:
    """Adapter for any OpenAI-compatible chat-completions endpoint.

    Configured with STALEFACT_LLM_BASE_URL, STALEFACT_LLM_API_KEY and STALEFACT_LLM_MODEL.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        client: httpx.Client | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._client = client or httpx.Client(timeout=timeout)

    @classmethod
    def from_env(
        cls, env: Mapping[str, str] | None = None, client: httpx.Client | None = None
    ) -> OpenAICompatModel:
        env = os.environ if env is None else env
        missing = [
            name
            for name in ("STALEFACT_LLM_BASE_URL", "STALEFACT_LLM_API_KEY", "STALEFACT_LLM_MODEL")
            if not env.get(name)
        ]
        if missing:
            raise ModelError("set " + ", ".join(missing) + " to use the OpenAI-compatible model")
        return cls(
            env["STALEFACT_LLM_BASE_URL"],
            env["STALEFACT_LLM_API_KEY"],
            env["STALEFACT_LLM_MODEL"],
            client=client,
        )

    def _chat(self, system: str, user: str) -> str:
        response = self._client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        response.raise_for_status()
        try:
            return str(response.json()["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, ValueError) as exc:
            raise ModelError(f"unexpected response shape: {response.text[:200]}") from exc

    def summarise(self, text: str, budget: int) -> str:
        words = max(1, int(budget * 0.7))
        system = (
            "You compress insurance-claim call transcripts into a running summary. Keep concrete "
            f"facts (numbers, dates, names, promises). Use at most {words} words, one sentence "
            "per line, no commentary."
        )
        reply = self._chat(system, text)
        lines = [line.strip() for line in reply.splitlines() if line.strip()]
        kept: list[str] = []
        used = 0
        for line in reversed(lines):
            used += count_tokens(line)
            if used > budget:
                break
            kept.append(line)
        return "\n".join(reversed(kept))

    def extract_facts(self, turn: Turn) -> list[Fact]:
        slots = ", ".join(sorted(SLOT_BY_NAME))
        system = (
            "Extract claim facts from one utterance. Reply with JSON only: "
            '{"facts": [{"subject": str, "slot": str, "value": str, '
            '"kind": "assert"|"correct"|"retract"}]}. '
            f"Allowed slots: {slots}. Use kind correct when the speaker replaces an earlier value "
            "and retract when they withdraw the last statement (value empty). "
            'Return {"facts": []} when there is nothing to extract.'
        )
        reply = self._chat(system, f"{turn.speaker}: {turn.text}")
        return _parse_fact_json(reply)

    def answer(self, question: str, context: str) -> str:
        system = (
            "Answer from the claim notes only. Reply with the value alone, no sentence. If the "
            f"notes do not say, reply exactly {UNKNOWN}."
        )
        reply = self._chat(system, f"Notes:\n{context}\n\nQuestion: {question}")
        return reply.strip().strip(".").strip() or UNKNOWN


def _parse_fact_json(reply: str) -> list[Fact]:
    text = reply.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ModelError(f"model did not return JSON: {reply[:200]}") from exc
    facts: list[Fact] = []
    for item in payload.get("facts", []):
        slot = str(item.get("slot", ""))
        kind = str(item.get("kind", ""))
        if slot not in SLOT_BY_NAME or kind not in FACT_KINDS:
            continue
        subject = str(item.get("subject") or SLOT_BY_NAME[slot].subject)
        facts.append(Fact(subject, slot, str(item.get("value", "")), kind))
    return facts
