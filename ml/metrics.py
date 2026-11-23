from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _ratio(top: float, bottom: float) -> float:
    return top / bottom if bottom else 0.0


@dataclass(frozen=True, slots=True)
class ConfusionMatrix:
    """Four counts that describe every kind of right and wrong answer."""

    tp: int  # attack, and we raised an alert
    fp: int  # normal, but we raised an alert (a false alarm)
    tn: int  # normal, and we stayed quiet
    fn: int  # attack, but we stayed quiet (a miss)

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def accuracy(self) -> float:
        return _ratio(self.tp + self.tn, self.total)

    @property
    def precision(self) -> float:
        return _ratio(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        return _ratio(self.tp, self.tp + self.fn)

    @property
    def false_alarm_rate(self) -> float:
        return _ratio(self.fp, self.fp + self.tn)

    @property
    def f1(self) -> float:
        return _ratio(2 * self.precision * self.recall, self.precision + self.recall)

    def __add__(self, other: ConfusionMatrix) -> ConfusionMatrix:
        return ConfusionMatrix(
            self.tp + other.tp, self.fp + other.fp, self.tn + other.tn, self.fn + other.fn
        )


def confusion(truth: np.ndarray, predicted: np.ndarray) -> ConfusionMatrix:
    if len(truth) != len(predicted):
        raise ValueError("truth and predicted must have the same length")
    actual = truth == 1
    alerted = predicted == 1
    return ConfusionMatrix(
        tp=int((actual & alerted).sum()),
        fp=int((~actual & alerted).sum()),
        tn=int((~actual & ~alerted).sum()),
        fn=int((actual & ~alerted).sum()),
    )


def majority_baseline(train_labels: np.ndarray, test_labels: np.ndarray) -> ConfusionMatrix:
    """What you get by always answering with the most common class in the training data."""
    guess = int(np.argmax(np.bincount(train_labels)))
    return confusion(test_labels, np.full(len(test_labels), guess, dtype=np.int64))
