"""Command-line interface: seed, run, eval and trace."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import box
from rich.console import Console
from rich.table import Table

from stalefact.dataset import Dataset, load_dataset, write_dataset
from stalefact.evaluate import (
    AGE_BUCKETS,
    CellStats,
    EvalResult,
    evaluate,
    run_conversation,
    seed_sensitivity,
)
from stalefact.generator import generate
from stalefact.grammar import SLOT_BY_NAME
from stalefact.memory import STRATEGIES, build_memory
from stalefact.model import FakeMemoryModel, MemoryModel, ModelError, OpenAICompatModel
from stalefact.probes import Outcome, build_probes
from stalefact.tokens import count_tokens

app = typer.Typer(
    help="Compare conversation-memory strategies on what they retain and what they leave stale.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _load(data_dir: Path) -> Dataset:
    try:
        return load_dataset(data_dir)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc


def _model(kind: str, data_dir: Path, answer_policy: str = "first") -> MemoryModel:
    if kind == "fake":
        try:
            return FakeMemoryModel.from_data_dir(data_dir, answer_policy)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(2) from exc
    if kind == "openai":
        try:
            return OpenAICompatModel.from_env()
        except ModelError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
    console.print(f"[red]unknown model {kind!r}; use fake or openai[/red]")
    raise typer.Exit(2)


def _pct(x: float) -> str:
    return f"{x * 100:5.1f}%"


def _ci(interval_low: float, interval_high: float) -> str:
    return f"[{interval_low * 100:.1f}, {interval_high * 100:.1f}]"


def _parse_budgets(text: str) -> list[int]:
    try:
        budgets = [int(part) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise typer.BadParameter("budgets must be comma-separated integers") from exc
    if not budgets or any(b <= 0 for b in budgets):
        raise typer.BadParameter("budgets must be positive integers")
    return budgets


@app.command()
def seed(
    out: Path = typer.Option(Path("data"), "--out", help="Output directory."),
    seed_value: int = typer.Option(7, "--seed", help="Generator seed."),
    conversations: int = typer.Option(120, "--conversations", min=10, help="Claim count."),
) -> None:
    """Generate synthetic claim conversations, ground truth and the planted-miss list."""
    data = generate(seed_value, conversations)
    write_dataset(out, data)
    turns = len(data.turns)
    facts = sum(len(c.events) for c in data.truth.conversations)
    once = sum(len(c.once_slots) for c in data.truth.conversations)
    patterns: dict[str, int] = {}
    for conv in data.truth.conversations:
        for pattern in conv.patterns:
            name = pattern.split(":")[0]
            patterns[name] = patterns.get(name, 0) + 1
    console.print(f"wrote {out}/: {conversations} conversations, {turns} turns, {facts} fact turns")
    console.print(
        "patterns: "
        + ", ".join(f"{n} {k}" for k, n in sorted(patterns.items()))
        + f"; {once} facts stated once; {len(data.misses)} planted extraction misses"
    )


@app.command()
def run(
    strategy: str = typer.Option(..., "--strategy", help=f"One of {', '.join(STRATEGIES)}."),
    budget: int = typer.Option(256, "--budget", min=16, help="Memory budget in tokens."),
    conversation: str = typer.Option(..., "--conversation", help="Conversation id, e.g. C0001."),
    data_dir: Path = typer.Option(Path("data"), "--data", help="Seeded data directory."),
    model_kind: str = typer.Option("fake", "--model", help="fake or openai."),
    show_context: bool = typer.Option(False, "--show-context", help="Print the final context."),
) -> None:
    """Replay one conversation through one strategy and score the probes after every call."""
    if strategy not in STRATEGIES:
        console.print(
            f"[red]unknown strategy {strategy!r}; choose from {', '.join(STRATEGIES)}[/red]"
        )
        raise typer.Exit(2)
    data = _load(data_dir)
    conversations = data.conversations
    if conversation not in conversations:
        console.print(f"[red]conversation {conversation} not found[/red]")
        raise typer.Exit(1)
    model = _model(model_kind, data_dir)
    truth = data.truth.by_id()[conversation]
    results = run_conversation(strategy, [budget], conversations[conversation], truth, model)[
        budget
    ]
    table = Table(title=f"{conversation} / {strategy} / budget {budget}", box=box.SIMPLE_HEAD)
    for column in ("call", "tokens", "correct", "stale", "missing", "wrong"):
        table.add_column(column, justify="right")
    for call_no in sorted({r.call_no for r in results}):
        rows = [r for r in results if r.call_no == call_no]
        counts = {o: sum(1 for r in rows if r.outcome == o) for o in Outcome}
        table.add_row(
            str(call_no),
            str(rows[0].tokens),
            *(
                str(counts[o])
                for o in (Outcome.CORRECT, Outcome.STALE, Outcome.MISSING, Outcome.WRONG)
            ),
        )
    console.print(table)
    total = len(results)
    correct = sum(1 for r in results if r.outcome == Outcome.CORRECT)
    console.print(f"retention {correct}/{total} = {correct / total:.3f}")
    if show_context:
        memory = build_memory(strategy, model, budget)
        for turn in conversations[conversation].turns():
            memory.ingest(turn)
        context = memory.render(budget)
        console.print(f"\nfinal context ({count_tokens(context)} tokens):")
        console.print(context, markup=False, highlight=False)


def _main_table(result: EvalResult) -> Table:
    table = Table(
        title=f"Strategy x budget ({len(result.conversation_ids)} conversations, 95% CI)",
        box=box.SIMPLE_HEAD,
    )
    for column in (
        "strategy",
        "budget",
        "tokens",
        "retention",
        "retention CI",
        "stale",
        "missing",
        "wrong",
        "ret/100 tok",
    ):
        table.add_column(column, justify="left" if column == "strategy" else "right")
    for strategy in result.strategies:
        for budget in result.budgets:
            s = result.stats(strategy, budget)
            table.add_row(
                strategy,
                str(budget),
                f"{s.mean_tokens:.0f}",
                _pct(s.retention.point),
                _ci(s.retention.low, s.retention.high),
                _pct(s.stale.point),
                _pct(s.missing.point),
                _pct(s.wrong.point),
                f"{s.per_100_tokens.point:.3f}",
            )
    return table


def _breakdown_tables(result: EvalResult, budget: int) -> list[Table]:
    ages = Table(title=f"Retention by fact age at budget {budget}", box=box.SIMPLE_HEAD)
    ages.add_column("strategy")
    for bucket in AGE_BUCKETS:
        ages.add_column(f"age {bucket}", justify="right")
    changes = Table(title=f"Changed vs unchanged slots at budget {budget}", box=box.SIMPLE_HEAD)
    for column in ("strategy", "changed n", "changed ret", "changed stale", "unchanged ret"):
        changes.add_column(column, justify="left" if column == "strategy" else "right")
    for strategy in result.strategies:
        by_age: dict[str, CellStats] = result.by_age(strategy, budget)
        ages.add_row(strategy, *(_pct(by_age[b].retention.point) for b in AGE_BUCKETS))
        by_change = result.by_change(strategy, budget)
        changes.add_row(
            strategy,
            str(by_change["changed"].n),
            _pct(by_change["changed"].retention.point),
            _pct(by_change["changed"].stale.point),
            _pct(by_change["unchanged"].retention.point),
        )
    return [ages, changes]


@app.command("eval")
def eval_command(
    budgets: str = typer.Option("128,256,512,1024", "--budgets", help="Comma-separated tokens."),
    data_dir: Path = typer.Option(Path("data"), "--data", help="Seeded data directory."),
    model_kind: str = typer.Option("fake", "--model", help="fake or openai."),
    breakdown_budget: int = typer.Option(256, "--breakdown-budget", help="Budget for breakdowns."),
    answer_policy: str = typer.Option(
        "first", "--answer-policy", help="Offline answerer: first or last matching line."
    ),
    json_out: Path | None = typer.Option(None, "--json", help="Also write the results as JSON."),
) -> None:
    """Run every strategy at every budget and report retention, stale, missing and wrong rates."""
    budget_list = _parse_budgets(budgets)
    data = _load(data_dir)
    model = _model(model_kind, data_dir, answer_policy)
    result = evaluate(data, model, STRATEGIES, budget_list)
    console.print(_main_table(result))
    if breakdown_budget in budget_list:
        for table in _breakdown_tables(result, breakdown_budget):
            console.print(table)
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        console.print(f"wrote {json_out}")


@app.command()
def trace(
    conversation: str = typer.Argument(..., help="Conversation id, e.g. C0001."),
    slot: str = typer.Option(..., "--slot", help="Slot name, e.g. payout_account."),
    budget: int = typer.Option(256, "--budget", min=16, help="Memory budget in tokens."),
    data_dir: Path = typer.Option(Path("data"), "--data", help="Seeded data directory."),
    model_kind: str = typer.Option("fake", "--model", help="fake or openai."),
) -> None:
    """Show a slot's value history and what each strategy answered at each probe."""
    if slot not in SLOT_BY_NAME:
        console.print(f"[red]unknown slot {slot!r}; choose from {', '.join(SLOT_BY_NAME)}[/red]")
        raise typer.Exit(2)
    data = _load(data_dir)
    conversations = data.conversations
    if conversation not in conversations:
        console.print(f"[red]conversation {conversation} not found[/red]")
        raise typer.Exit(1)
    truth = data.truth.by_id()[conversation]
    events = truth.slot_events(slot)
    if not events:
        console.print(f"[red]{conversation} never mentions {slot}[/red]")
        raise typer.Exit(1)
    model = _model(model_kind, data_dir)

    history = Table(title=f"{conversation}: {slot} history", box=box.SIMPLE_HEAD)
    for column in ("call", "turn", "event", "stated", "in force after"):
        history.add_column(column, justify="right" if column in ("call", "turn") else "left")
    for e in events:
        history.add_row(str(e.call_no), str(e.turn_no), e.kind, e.value or "-", e.value_after)
    console.print(history)

    probes = build_probes(truth)
    answers: dict[str, dict[int, tuple[str, Outcome]]] = {}
    for strategy in STRATEGIES:
        rows = run_conversation(
            strategy, [budget], conversations[conversation], truth, model, probes=probes
        )[budget]
        answers[strategy] = {r.call_no: (r.answer, r.outcome) for r in rows if r.slot == slot}
    table = Table(title=f"Answers at budget {budget}", box=box.SIMPLE_HEAD)
    table.add_column("after call", justify="right")
    table.add_column("gold")
    for strategy in STRATEGIES:
        table.add_column(strategy)
    for call_no in range(1, truth.n_calls + 1):
        probe = next((p for p in probes[call_no] if p.slot == slot), None)
        if probe is None:
            continue
        cells = []
        for strategy in STRATEGIES:
            answer, outcome = answers[strategy][call_no]
            cells.append(f"{answer} ({outcome.value})")
        table.add_row(str(call_no), probe.gold, *cells)
    console.print(table)


@app.command()
def sensitivity(
    seeds: str = typer.Option(
        "1,2,3,4,5,6,7,8", "--seeds", help="Comma-separated generator seeds."
    ),
    budget: int = typer.Option(256, "--budget", min=16, help="Memory budget in tokens."),
    conversations: int = typer.Option(120, "--conversations", min=10, help="Claims per seed."),
) -> None:
    """Stale rate on changed slots (rolling vs ledger) across generator seeds."""
    seed_list = _parse_budgets(seeds)
    rows = seed_sensitivity(seed_list, budget, conversations)
    table = Table(title=f"Changed-slot stale rate by seed at budget {budget}", box=box.SIMPLE_HEAD)
    for column in ("seed", "changed probes", "rolling stale", "ledger stale", "ledger retention"):
        table.add_column(column, justify="right")
    for row in rows:
        table.add_row(
            str(row.seed),
            str(row.changed_probes),
            _pct(row.rolling_stale),
            _pct(row.ledger_stale),
            _pct(row.ledger_retention),
        )
    n = len(rows)
    table.add_row(
        "mean",
        "",
        _pct(sum(r.rolling_stale for r in rows) / n),
        _pct(sum(r.ledger_stale for r in rows) / n),
        _pct(sum(r.ledger_retention for r in rows) / n),
    )
    console.print(table)
