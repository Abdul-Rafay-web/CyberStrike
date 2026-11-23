import numpy as np
import pytest

from ml.metrics import ConfusionMatrix, confusion, majority_baseline


def test_confusion_counts() -> None:
    truth = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    guess = np.array([1, 1, 0, 1, 0, 0, 0, 0])
    assert confusion(truth, guess) == ConfusionMatrix(tp=2, fp=1, tn=4, fn=1)


def test_every_rate_matches_its_definition() -> None:
    matrix = ConfusionMatrix(tp=30, fp=10, tn=50, fn=10)
    assert matrix.total == 100
    assert matrix.accuracy == pytest.approx(0.8)
    assert matrix.precision == pytest.approx(0.75)
    assert matrix.recall == pytest.approx(0.75)
    assert matrix.false_alarm_rate == pytest.approx(10 / 60)
    assert matrix.f1 == pytest.approx(0.75)


def test_empty_cases_do_not_divide_by_zero() -> None:
    empty = ConfusionMatrix(0, 0, 0, 0)
    assert empty.accuracy == empty.precision == empty.recall == empty.f1 == 0.0
    assert empty.false_alarm_rate == 0.0


def test_matrices_add() -> None:
    assert ConfusionMatrix(1, 2, 3, 4) + ConfusionMatrix(4, 3, 2, 1) == ConfusionMatrix(5, 5, 5, 5)


def test_majority_baseline_never_alerts_when_attacks_are_rare() -> None:
    labels = np.array([0] * 80 + [1] * 20)
    matrix = majority_baseline(labels, labels)
    assert matrix.accuracy == pytest.approx(0.8)
    assert matrix.recall == 0.0


def test_mismatched_lengths_are_refused() -> None:
    with pytest.raises(ValueError):
        confusion(np.array([1, 0]), np.array([1]))

