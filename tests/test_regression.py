import numpy as np
import pytest

from core.generator import generate_network
from ml.regression import (
    MAP_FEATURE_NAMES,
    LinearRegression,
    extract_map_features,
    fit_difficulty_surrogate,
)


def test_linear_regression_recovers_exact_linear_plane() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(200, 3))
    true_w = np.array([2.5, -1.2, 0.8])
    true_b = 4.0
    y = x @ true_w + true_b

    model = LinearRegression().fit(x, y)
    assert model.is_fitted
    assert model.weights == pytest.approx(true_w, abs=1e-9)
    assert model.bias == pytest.approx(true_b, abs=1e-9)
    assert model.r_squared(x, y) == pytest.approx(1.0, abs=1e-9)
    assert model.rmse(x, y) == pytest.approx(0.0, abs=1e-9)

    clone = LinearRegression.from_dict(model.to_dict())
    assert np.allclose(model.predict(x), clone.predict(x))


def test_extract_map_features_matches_schema_length() -> None:
    net = generate_network(1, size=15)
    feats = extract_map_features(net)
    assert feats.shape == (len(MAP_FEATURE_NAMES),)
    assert feats[0] == 15.0
    assert feats[1] == float(len(net.edges()))
    assert feats[2] >= 1.0


def test_difficulty_surrogate_generalizes_to_held_out_maps() -> None:
    surrogate = fit_difficulty_surrogate(num_maps=60, trials_per_map=6)
    assert surrogate.turns_report.test_r2 > 0.1
    assert surrogate.turns_report.test_rmse < 8.0
    assert 0.0 <= surrogate.predict_win_rate(generate_network(99, size=15)) <= 1.0
    assert surrogate.predict_turns(generate_network(99, size=15)) >= 1.0


def test_linear_regression_rejects_bad_inputs() -> None:
    with pytest.raises(RuntimeError):
        LinearRegression().predict(np.ones((2, 2)))
    with pytest.raises(ValueError):
        LinearRegression().fit(np.ones((5, 2)), np.ones(4))
    model = LinearRegression().fit(np.ones((5, 2)), np.ones(5))
    with pytest.raises(ValueError):
        model.predict(np.ones((2, 3)))

