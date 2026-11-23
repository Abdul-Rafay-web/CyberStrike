from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ml.dataset import Dataset
from ml.metrics import ConfusionMatrix, confusion
from ml.naive_bayes import NaiveBayes


def stratified_folds(labels: np.ndarray, folds: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Split row numbers into `folds` groups that each keep the class mix of the whole."""
    if folds < 2:
        raise ValueError("need at least 2 folds")
    groups: list[list[int]] = [[] for _ in range(folds)]
    for label in np.unique(labels):
        rows = np.flatnonzero(labels == label)
        rng.shuffle(rows)
        for group, chunk in zip(groups, np.array_split(rows, folds), strict=True):
            group.extend(int(r) for r in chunk)
    return [np.array(sorted(group), dtype=np.int64) for group in groups]


def train_validation_test_split(
    labels: np.ndarray, train_share: float, validation_share: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three disjoint sets of row numbers, each with the same class mix."""
    if train_share <= 0 or validation_share <= 0 or train_share + validation_share >= 1:
        raise ValueError("the three shares must each be positive and add up to 1")
    train: list[int] = []
    validation: list[int] = []
    test: list[int] = []
    for label in np.unique(labels):
        rows = np.flatnonzero(labels == label)
        rng.shuffle(rows)
        a = int(round(len(rows) * train_share))
        b = a + int(round(len(rows) * validation_share))
        train.extend(int(r) for r in rows[:a])
        validation.extend(int(r) for r in rows[a:b])
        test.extend(int(r) for r in rows[b:])
    return (
        np.array(sorted(train), dtype=np.int64),
        np.array(sorted(validation), dtype=np.int64),
        np.array(sorted(test), dtype=np.int64),
    )


@dataclass(frozen=True, slots=True)
class Spread:
    mean: float
    std: float


@dataclass(frozen=True, slots=True)
class CrossValResult:
    folds: tuple[ConfusionMatrix, ...]

    def _spread(self, pick: Callable[[ConfusionMatrix], float]) -> Spread:
        values = [pick(fold) for fold in self.folds]
        return Spread(statistics.fmean(values), statistics.pstdev(values))

    @property
    def accuracy(self) -> Spread:
        return self._spread(lambda m: m.accuracy)

    @property
    def precision(self) -> Spread:
        return self._spread(lambda m: m.precision)

    @property
    def recall(self) -> Spread:
        return self._spread(lambda m: m.recall)

    @property
    def f1(self) -> Spread:
        return self._spread(lambda m: m.f1)

    @property
    def false_alarm_rate(self) -> Spread:
        return self._spread(lambda m: m.false_alarm_rate)

    @property
    def pooled(self) -> ConfusionMatrix:
        total = self.folds[0]
        for fold in self.folds[1:]:
            total = total + fold
        return total


def cross_validate(
    data: Dataset,
    folds: int,
    smoothing: float,
    rng: np.random.Generator,
    threshold: float = 0.5,
) -> CrossValResult:
    """K fold cross validation: every fold takes one turn as the test set."""
    parts = stratified_folds(data.labels, folds, rng)
    results: list[ConfusionMatrix] = []
    for held_out in parts:
        keep = np.setdiff1d(np.arange(len(data)), held_out)
        train, test = data.take(keep), data.take(held_out)
        model = NaiveBayes(smoothing).fit(train.features, train.labels, data.sizes)
        results.append(confusion(test.labels, model.predict(test.features, threshold)))
    return CrossValResult(tuple(results))


def leave_one_out(data: Dataset, smoothing: float, threshold: float = 0.5) -> ConfusionMatrix:
    """Leave one out cross validation: train on everything except one row, test on that row."""
    total = ConfusionMatrix(0, 0, 0, 0)
    everything = np.arange(len(data))
    for row in everything:
        train = data.take(everything[everything != row])
        model = NaiveBayes(smoothing).fit(train.features, train.labels, data.sizes)
        held_out = data.take(np.array([row]))
        total = total + confusion(held_out.labels, model.predict(held_out.features, threshold))
    return total


def best_threshold(
    truth: np.ndarray, attack_probability: np.ndarray, grid: np.ndarray | None = None
) -> float:
    """The alert threshold with the best F1 score on the data you give it."""
    options = np.linspace(0.05, 0.95, 19) if grid is None else grid
    scores = [
        confusion(truth, (attack_probability >= threshold).astype(np.int64)).f1
        for threshold in options
    ]
    return float(options[int(np.argmax(scores))])

