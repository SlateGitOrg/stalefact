from __future__ import annotations

from datetime import datetime

import pytest

from stalefact.dataset import Dataset
from stalefact.dialogue import Turn
from stalefact.evaluate import EvalResult, evaluate
from stalefact.generator import generate
from stalefact.model import FakeMemoryModel


def make_turn(
    text: str,
    call_no: int = 1,
    turn_no: int = 1,
    conversation_id: str = "T0001",
    speaker: str = "claimant",
) -> Turn:
    return Turn(
        conversation_id=conversation_id,
        call_no=call_no,
        turn_no=turn_no,
        speaker=speaker,
        text=text,
        timestamp=datetime(2026, 1, 5, 9, 0, 0),
    )


@pytest.fixture(scope="session")
def dataset() -> Dataset:
    return generate(7, 120)


@pytest.fixture(scope="session")
def model(dataset: Dataset) -> FakeMemoryModel:
    return FakeMemoryModel(dataset.miss_keys)


@pytest.fixture(scope="session")
def full_eval(dataset: Dataset, model: FakeMemoryModel) -> EvalResult:
    return evaluate(dataset, model)


@pytest.fixture(scope="session")
def small_dataset() -> Dataset:
    return generate(3, 12)
