import numpy as np
import pytest

from ml.crossval import (
    best_threshold,
    cross_validate,
    leave_one_out,
    stratified_folds,
    train_validation_test_split,
)
from ml.synthetic_logs import generate_dataset
from utils.rng import STREAM_DATA, make_rng


def _labels() -> np.ndarray:
    return np.array([0] * 80 + [1] * 20)


def test_folds_cover_every_row_exactly_once() -> None:
    folds = stratified_folds(_labels(), 5, make_rng(1, STREAM_DATA))
    joined = np.concatenate(folds)
    assert sorted(joined.tolist()) == list(range(100))
    assert len(set(joined.tolist())) == 100


def test_every_fold_keeps_the_class_mix() -> None:
    labels = _labels()
    for fold in stratified_folds(labels, 5, make_rng(1, STREAM_DATA)):
        assert len(fold) == 20
        assert labels[fold].sum() == 4


def test_folds_are_repeatable() -> None:
    first = stratified_folds(_labels(), 5, make_rng(3, STREAM_DATA))
    second = stratified_folds(_labels(), 5, make_rng(3, STREAM_DATA))
    assert all((a == b).all() for a, b in zip(first, second, strict=True))


def test_at_least_two_folds() -> None:
    with pytest.raises(ValueError):
        stratified_folds(_labels(), 1, make_rng(1, STREAM_DATA))


def test_three_way_split_is_disjoint_complete_and_stratified() -> None:
    labels = _labels()
    rng = make_rng(1, STREAM_DATA)
    train, validation, test = train_validation_test_split(labels, 0.6, 0.2, rng)
    together = np.concatenate([train, validation, test])
    assert sorted(together.tolist()) == list(range(100))
    assert (len(train), len(validation), len(test)) == (60, 20, 20)
    assert labels[train].sum() == 12 and labels[validation].sum() == 4 and labels[test].sum() == 4


def test_bad_split_shares_are_refused() -> None:
    for train, validation in ((0.0, 0.2), (0.6, 0.0), (0.7, 0.4)):
        with pytest.raises(ValueError):
            train_validation_test_split(_labels(), train, validation, make_rng(1, STREAM_DATA))


def test_cross_validation_tests_every_row_once() -> None:
    data = generate_dataset(500, 0.2, make_rng(1, STREAM_DATA))
    result = cross_validate(data, 5, 1.0, make_rng(2, STREAM_DATA))
    assert len(result.folds) == 5
    assert result.pooled.total == 500
    assert 0.5 < result.accuracy.mean <= 1.0
    assert result.accuracy.std >= 0.0


def test_cross_validation_is_repeatable() -> None:
    data = generate_dataset(500, 0.2, make_rng(1, STREAM_DATA))
    first = cross_validate(data, 5, 1.0, make_rng(2, STREAM_DATA))
    second = cross_validate(data, 5, 1.0, make_rng(2, STREAM_DATA))
    assert first.folds == second.folds


def test_leave_one_out_tests_each_row_once() -> None:
    data = generate_dataset(120, 0.25, make_rng(1, STREAM_DATA))
    assert leave_one_out(data, 1.0).total == 120


def test_best_threshold_finds_a_perfect_split() -> None:
    truth = np.array([0, 0, 0, 1, 1, 1])
    probability = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    threshold = best_threshold(truth, probability)
    assert 0.3 < threshold <= 0.7

