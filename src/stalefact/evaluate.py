"""Run every strategy at every budget and score the probes."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from stalefact.dataset import Dataset
from stalefact.dialogue import Conversation
from stalefact.generator import generate
from stalefact.memory import STRATEGIES, build_memory
from stalefact.model import FakeMemoryModel, MemoryModel
from stalefact.probes import OUTCOMES, Outcome, Probe, build_probes, classify
from stalefact.stats import N_RESAMPLES, bootstrap_ratios
from stalefact.tokens import count_tokens
from stalefact.truth import ConversationTruth

DEFAULT_BUDGETS = (128, 256, 512, 1024)
AGE_BUCKETS = ("0", "1", "2", "3+")
BOOTSTRAP_SEED = 20260101


class BudgetExceeded(RuntimeError):
    """A strategy rendered more tokens than the budget allows."""


@dataclass(slots=True)
class ProbeResult:
    """One scored answer."""

    strategy: str
    budget: int
    conversation_id: str
    call_no: int
    slot: str
    gold: str
    answer: str
    outcome: Outcome
    age: int
    changed: bool
    tokens: int


@dataclass(frozen=True)
class Interval:
    """A point estimate with its 95% bootstrap interval."""

    point: float
    low: float
    high: float

    def as_dict(self) -> dict[str, float]:
        return {"point": self.point, "low": self.low, "high": self.high}


@dataclass(frozen=True)
class CellStats:
    """Rates for one strategy at one budget, over a set of probes."""

    n: int
    retention: Interval
    stale: Interval
    missing: Interval
    wrong: Interval
    per_100_tokens: Interval
    mean_tokens: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "retention": self.retention.as_dict(),
            "stale": self.stale.as_dict(),
            "missing": self.missing.as_dict(),
            "wrong": self.wrong.as_dict(),
            "retention_per_100_tokens": self.per_100_tokens.as_dict(),
            "mean_tokens": self.mean_tokens,
        }


def age_bucket(age: int) -> str:
    """Bucket label for calls since a fact was last stated."""
    return str(age) if age < 3 else "3+"


def run_conversation(
    strategy: str,
    budgets: Sequence[int],
    conversation: Conversation,
    truth: ConversationTruth,
    model: MemoryModel,
    importance: Mapping[str, int] | None = None,
    probes: dict[int, list[Probe]] | None = None,
) -> dict[int, list[ProbeResult]]:
    """Replay one conversation through a strategy; probe after every call at every budget.

    Strategies whose state does not depend on the budget are fed once and rendered per budget;
    the rolling summary is rebuilt per budget because its compaction depends on it.
    """
    probes = probes if probes is not None else build_probes(truth)
    budget_dependent = strategy == "rolling"
    groups = [[b] for b in budgets] if budget_dependent else [list(budgets)]
    results: dict[int, list[ProbeResult]] = {b: [] for b in budgets}
    for group in groups:
        memory = build_memory(strategy, model, group[0], importance)
        for call_no in conversation.call_numbers:
            for turn in conversation.calls[call_no]:
                memory.ingest(turn)
            for budget in group:
                context = memory.render(budget)
                tokens = count_tokens(context)
                if tokens > budget:
                    raise BudgetExceeded(
                        f"{strategy} rendered {tokens} tokens for budget {budget} "
                        f"in {conversation.conversation_id} call {call_no}"
                    )
                for probe in probes[call_no]:
                    answer = model.answer(probe.question, context)
                    results[budget].append(
                        ProbeResult(
                            strategy=strategy,
                            budget=budget,
                            conversation_id=probe.conversation_id,
                            call_no=call_no,
                            slot=probe.slot,
                            gold=probe.gold,
                            answer=answer,
                            outcome=classify(answer, probe.gold, probe.earlier),
                            age=probe.age,
                            changed=probe.changed,
                            tokens=tokens,
                        )
                    )
    return results


@dataclass
class EvalResult:
    """All probe results of an evaluation, with bootstrap statistics on demand."""

    conversation_ids: list[str]
    strategies: tuple[str, ...]
    budgets: tuple[int, ...]
    results: dict[tuple[str, int], list[ProbeResult]] = field(default_factory=dict)
    seed: int = BOOTSTRAP_SEED
    n_resamples: int = N_RESAMPLES

    def probes(
        self,
        strategy: str,
        budget: int,
        where: Callable[[ProbeResult], bool] | None = None,
    ) -> list[ProbeResult]:
        rows = self.results[(strategy, budget)]
        return [r for r in rows if where(r)] if where else rows

    def stats(
        self,
        strategy: str,
        budget: int,
        where: Callable[[ProbeResult], bool] | None = None,
    ) -> CellStats:
        """Rates with 95% bootstrap intervals over conversations for the selected probes."""
        rows = self.probes(strategy, budget, where)
        index = {cid: i for i, cid in enumerate(self.conversation_ids)}
        n_conv = len(self.conversation_ids)
        outcome_index = {o: i for i, o in enumerate(OUTCOMES)}
        counts = np.zeros((n_conv, len(OUTCOMES)))
        probes_per_conv = np.zeros(n_conv)
        tokens_per_conv = np.zeros(n_conv)
        for row in rows:
            i = index[row.conversation_id]
            counts[i, outcome_index[row.outcome]] += 1
            probes_per_conv[i] += 1
            tokens_per_conv[i] += row.tokens
        correct = counts[:, outcome_index[Outcome.CORRECT]]
        numerators = np.column_stack([counts, correct])
        denominators = np.column_stack(
            [np.repeat(probes_per_conv[:, None], len(OUTCOMES), axis=1), tokens_per_conv / 100.0]
        )
        point, low, high = bootstrap_ratios(numerators, denominators, self.seed, self.n_resamples)

        def interval(column: int) -> Interval:
            return Interval(float(point[column]), float(low[column]), float(high[column]))

        total = int(probes_per_conv.sum())
        return CellStats(
            n=total,
            retention=interval(outcome_index[Outcome.CORRECT]),
            stale=interval(outcome_index[Outcome.STALE]),
            missing=interval(outcome_index[Outcome.MISSING]),
            wrong=interval(outcome_index[Outcome.WRONG]),
            per_100_tokens=interval(len(OUTCOMES)),
            mean_tokens=float(tokens_per_conv.sum() / total) if total else 0.0,
        )

    def by_age(self, strategy: str, budget: int) -> dict[str, CellStats]:
        """Stats per age bucket (calls since the value in force was last stated)."""
        return {
            bucket: self.stats(strategy, budget, lambda r, b=bucket: age_bucket(r.age) == b)
            for bucket in AGE_BUCKETS
        }

    def by_change(self, strategy: str, budget: int) -> dict[str, CellStats]:
        """Stats for probes of slots that had been changed by probe time, and for the rest."""
        return {
            "changed": self.stats(strategy, budget, lambda r: r.changed),
            "unchanged": self.stats(strategy, budget, lambda r: not r.changed),
        }

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready summary of every cell and its breakdowns."""
        cells: list[dict[str, Any]] = []
        for strategy in self.strategies:
            for budget in self.budgets:
                cells.append(
                    {
                        "strategy": strategy,
                        "budget": budget,
                        "overall": self.stats(strategy, budget).as_dict(),
                        "by_age": {
                            k: v.as_dict() for k, v in self.by_age(strategy, budget).items()
                        },
                        "by_change": {
                            k: v.as_dict() for k, v in self.by_change(strategy, budget).items()
                        },
                    }
                )
        return {
            "conversations": len(self.conversation_ids),
            "bootstrap": {"resamples": self.n_resamples, "seed": self.seed, "level": 0.95},
            "cells": cells,
        }


def evaluate(
    dataset: Dataset,
    model: MemoryModel,
    strategies: Iterable[str] = STRATEGIES,
    budgets: Iterable[int] = DEFAULT_BUDGETS,
    importance: Mapping[str, int] | None = None,
    conversation_ids: Sequence[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> EvalResult:
    """Run every strategy at every budget over the dataset's conversations."""
    strategies = tuple(strategies)
    budgets = tuple(budgets)
    conversations = dataset.conversations
    truths = dataset.truth.by_id()
    ids = list(conversation_ids) if conversation_ids is not None else sorted(conversations)
    probes = {cid: build_probes(truths[cid]) for cid in ids}
    result = EvalResult(ids, strategies, budgets)
    for strategy in strategies:
        for budget in budgets:
            result.results[(strategy, budget)] = []
        for cid in ids:
            per_budget = run_conversation(
                strategy, budgets, conversations[cid], truths[cid], model, importance, probes[cid]
            )
            for budget, rows in per_budget.items():
                result.results[(strategy, budget)].extend(rows)
        if progress:
            progress(strategy)
    return result


@dataclass(frozen=True)
class SeedRow:
    """Stale and retention rates of rolling and ledger memory on one generator seed."""

    seed: int
    changed_probes: int
    rolling_stale: float
    ledger_stale: float
    ledger_retention: float


def seed_sensitivity(
    seeds: Iterable[int], budget: int = 256, n_conversations: int = 120
) -> list[SeedRow]:
    """Regenerate the data for each seed and measure the stale rate on changed slots.

    The rolling-summary parameters were tuned on the default seed; running other seeds shows how
    much of its stale rate is specific to that draw of synthetic data.
    """
    rows: list[SeedRow] = []
    for seed in seeds:
        data = generate(seed, n_conversations)
        model = FakeMemoryModel(data.miss_keys)
        result = evaluate(data, model, ("rolling", "ledger"), (budget,))
        rolling = result.stats("rolling", budget, lambda r: r.changed)
        ledger = result.stats("ledger", budget, lambda r: r.changed)
        overall = result.stats("ledger", budget)
        rows.append(
            SeedRow(
                seed,
                rolling.n,
                rolling.stale.point,
                ledger.stale.point,
                overall.retention.point,
            )
        )
    return rows
