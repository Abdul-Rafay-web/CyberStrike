import numpy as np
import pytest

from ml.naive_bayes import NaiveBayes


def _tiny() -> tuple[np.ndarray, np.ndarray]:
    # one feature with two values. Class 0 rows: 0, 0, 1. Class 1 rows: 1, 1.
    features = np.array([[0], [0], [1], [1], [1]])
    labels = np.array([0, 0, 0, 1, 1])
    return features, labels


def test_matches_a_hand_calculation_with_smoothing() -> None:
    features, labels = _tiny()
    model = NaiveBayes(1.0).fit(features, labels, sizes=[2])
    # prior 4/7 and 3/7. P(1|class 0) = 2/5. P(1|class 1) = 3/4.
    class_0 = 4 / 7 * 2 / 5
    class_1 = 3 / 7 * 3 / 4
    proba = model.predict_proba(np.array([[1]]))
    assert proba[0, 1] == pytest.approx(class_1 / (class_0 + class_1))
    assert proba[0].sum() == pytest.approx(1.0)


def test_without_smoothing_an_unseen_value_rules_a_class_out() -> None:
    features, labels = _tiny()
    model = NaiveBayes(0.0).fit(features, labels, sizes=[2])
    proba = model.predict_proba(np.array([[0]]))
    assert proba[0, 1] == 0.0
    assert proba[0, 0] == 1.0


def test_smoothing_stops_one_unseen_value_from_vetoing_all_other_evidence() -> None:
    # Feature A is strong evidence of attack. Feature B value 2 was never seen in an attack.
    features = np.array([[1, 0]] * 10 + [[0, 0]] * 10 + [[0, 2]])
    labels = np.array([1] * 10 + [0] * 11)
    record = np.array([[1, 2]])
    assert NaiveBayes(0.0).fit(features, labels, sizes=[2, 3]).predict(record)[0] == 0
    assert NaiveBayes(1.0).fit(features, labels, sizes=[2, 3]).predict(record)[0] == 1


def test_smoothing_keeps_every_score_finite() -> None:
    features, labels = _tiny()
    model = NaiveBayes(1.0).fit(features, labels, sizes=[2])
    assert np.isfinite(model.log_scores(np.array([[0], [1]]))).all()


def test_impossible_rows_fall_back_to_the_prior() -> None:
    features = np.array([[0], [0], [1]])
    labels = np.array([0, 0, 1])
    model = NaiveBayes(0.0).fit(features, labels, sizes=[3])
    unseen = np.array([[2]])
    assert model.impossible_rows(unseen) == 1
    assert model.predict_proba(unseen)[0] == pytest.approx([2 / 3, 1 / 3])
    assert NaiveBayes(1.0).fit(features, labels, sizes=[3]).impossible_rows(unseen) == 0


def test_probabilities_always_add_up_to_one() -> None:
    rng = np.random.default_rng(0)
    features = rng.integers(0, 4, size=(300, 3))
    labels = rng.integers(0, 2, size=300)
    model = NaiveBayes(1.0).fit(features, labels, sizes=[4, 4, 4])
    proba = model.predict_proba(rng.integers(0, 4, size=(50, 3)))
    assert proba.sum(axis=1) == pytest.approx(np.ones(50))
    assert (proba >= 0).all()


def test_vectorised_scores_match_a_slow_reference() -> None:
    rng = np.random.default_rng(1)
    sizes = [3, 5, 2]
    features = np.stack([rng.integers(0, s, size=400) for s in sizes], axis=1)
    labels = rng.integers(0, 2, size=400)
    model = NaiveBayes(1.0).fit(features, labels, sizes=sizes)
    sample = np.stack([rng.integers(0, s, size=20) for s in sizes], axis=1)
    fast = model.log_scores(sample)
    for row, record in enumerate(sample):
        for label in (0, 1):
            members = features[labels == label]
            score = np.log((len(members) + 1) / (len(features) + 2))
            for column, size in enumerate(sizes):
                seen = int((members[:, column] == record[column]).sum())
                score += np.log((seen + 1) / (len(members) + size))
            assert fast[row, label] == pytest.approx(score)


def test_threshold_moves_the_decision() -> None:
    features, labels = _tiny()
    model = NaiveBayes(1.0).fit(features, labels, sizes=[2])
    rows = np.array([[0], [1]])
    assert model.predict(rows, threshold=0.0).tolist() == [1, 1]
    assert model.predict(rows, threshold=1.01).tolist() == [0, 0]


def test_three_classes_use_the_highest_score() -> None:
    features = np.array([[0], [0], [1], [1], [2], [2]])
    labels = np.array([0, 0, 1, 1, 2, 2])
    model = NaiveBayes(1.0).fit(features, labels, sizes=[3], n_classes=3)
    assert model.predict(np.array([[0], [1], [2]])).tolist() == [0, 1, 2]


def test_saving_and_loading_keeps_the_predictions() -> None:
    rng = np.random.default_rng(2)
    features = rng.integers(0, 3, size=(200, 2))
    labels = rng.integers(0, 2, size=200)
    model = NaiveBayes(0.5).fit(features, labels, sizes=[3, 3])
    clone = NaiveBayes.from_dict(model.to_dict())
    assert np.allclose(model.log_scores(features), clone.log_scores(features))
    assert clone.smoothing == 0.5


def test_bad_inputs_are_refused() -> None:
    features, labels = _tiny()
    with pytest.raises(ValueError):
        NaiveBayes(-1.0)
    with pytest.raises(ValueError):
        NaiveBayes().fit(features, np.zeros(5, dtype=np.int64), sizes=[2])
    with pytest.raises(ValueError):
        NaiveBayes().fit(features, labels, sizes=[1])
    with pytest.raises(ValueError):
        NaiveBayes().fit(features, labels, sizes=[2, 2])
    with pytest.raises(RuntimeError):
        NaiveBayes().predict(features)
    model = NaiveBayes().fit(features, labels, sizes=[2])
    with pytest.raises(ValueError):
        model.predict(np.array([[5]]))
    with pytest.raises(ValueError):
        model.predict(np.array([[0, 0]]))

