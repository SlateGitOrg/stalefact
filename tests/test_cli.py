import json

import pytest
from _cli import output_lines, plain
from typer.testing import CliRunner

from stalefact.cli import app

runner = CliRunner()


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("data")
    result = runner.invoke(
        app, ["seed", "--out", str(path), "--seed", "3", "--conversations", "12"]
    )
    assert result.exit_code == 0, result.output
    return path


def test_seed_writes_the_three_files(data_dir):
    assert {p.name for p in data_dir.iterdir()} == {
        "turns.jsonl",
        "truth.json",
        "planted_misses.yaml",
    }


def test_seed_reports_the_planted_patterns(tmp_path):
    result = runner.invoke(app, ["seed", "--out", str(tmp_path), "--conversations", "20"])
    assert "20 conversations" in plain(result)
    assert "6 change" in plain(result) and "3 correct" in plain(result)
    assert "1 retract" in plain(result)


def test_run_prints_a_row_per_call(data_dir):
    result = runner.invoke(
        app,
        [
            "run",
            "--strategy",
            "ledger",
            "--budget",
            "256",
            "--conversation",
            "C0001",
            "--data",
            str(data_dir),
            "--show-context",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "retention" in plain(result)
    assert "final context" in plain(result)
    assert "payout_account:" in plain(result)


def test_run_rejects_unknown_strategy_and_conversation(data_dir):
    bad = runner.invoke(
        app, ["run", "--strategy", "nope", "--conversation", "C0001", "--data", str(data_dir)]
    )
    assert bad.exit_code == 2
    missing = runner.invoke(
        app, ["run", "--strategy", "window", "--conversation", "C9999", "--data", str(data_dir)]
    )
    assert missing.exit_code == 1


def test_eval_prints_the_table_and_writes_json(data_dir, tmp_path):
    out = tmp_path / "report.json"
    result = runner.invoke(
        app, ["eval", "--budgets", "128,256", "--data", str(data_dir), "--json", str(out)]
    )
    assert result.exit_code == 0, result.output
    for strategy in ("window", "rolling", "ledger", "hybrid"):
        assert strategy in plain(result)
    report = json.loads(out.read_text())
    assert len(report["cells"]) == 8
    assert report["bootstrap"]["resamples"] == 1000
    cell = report["cells"][0]
    assert {"overall", "by_age", "by_change"} <= set(cell)


def test_trace_shows_history_and_each_strategys_answer(data_dir):
    result = runner.invoke(
        app, ["trace", "C0002", "--slot", "payout_account", "--data", str(data_dir)]
    )
    assert result.exit_code == 0, result.output
    assert "history" in plain(result)
    for strategy in ("window", "rolling", "ledger", "hybrid"):
        assert strategy in plain(result)
    assert "(correct)" in plain(result)


def test_trace_rejects_unknown_slot(data_dir):
    result = runner.invoke(app, ["trace", "C0001", "--slot", "nope", "--data", str(data_dir)])
    assert result.exit_code == 2


def test_commands_explain_a_missing_data_directory(tmp_path):
    result = runner.invoke(app, ["eval", "--data", str(tmp_path / "absent")])
    assert result.exit_code == 1
    assert "stalefact seed" in plain(result)


def test_sensitivity_prints_one_row_per_seed_and_a_mean():
    result = runner.invoke(app, ["sensitivity", "--seeds", "1,2", "--conversations", "10"])
    assert result.exit_code == 0, result.output
    lines = [ln.split() for ln in output_lines(result)]
    assert [ln[0] for ln in lines if ln and ln[0] in ("1", "2", "mean")] == ["1", "2", "mean"]


def test_eval_rejects_an_unknown_answer_policy(data_dir):
    result = runner.invoke(app, ["eval", "--data", str(data_dir), "--answer-policy", "middle"])
    assert result.exit_code == 2
