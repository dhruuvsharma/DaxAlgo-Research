from __future__ import annotations

from dataclasses import dataclass

import pytest

from run_study import _partition_categories


@dataclass(frozen=True)
class _Fold:
    train_example_ids: tuple[str, ...]
    validation_example_ids: tuple[str, ...]


@dataclass(frozen=True)
class _Holdout:
    example_ids: tuple[str, ...]


class _Split:
    def __init__(self, *, outer: tuple[str, ...] = ("sealed",)) -> None:
        self._folds = {
            1: _Fold(("train-a", "train-b"), ("selected",)),
            2: _Fold((), ("forward-a",)),
            3: _Fold((), ("forward-b",)),
        }
        self.outer_holdout = _Holdout(outer)

    def fold(self, number: int) -> _Fold:
        return self._folds[number]


def test_partition_categories_admit_only_declared_development_ids() -> None:
    categories = _partition_categories(_Split())

    assert categories == {
        "in_sample_train_replay": frozenset(("train-a", "train-b")),
        "selected_validation": frozenset(("selected",)),
        "untouched_development_forward": frozenset(("forward-a", "forward-b")),
    }
    assert "sealed" not in set().union(*categories.values())


def test_partition_categories_reject_outer_holdout_overlap() -> None:
    with pytest.raises(ValueError, match="outer holdout"):
        _partition_categories(_Split(outer=("forward-b",)))
