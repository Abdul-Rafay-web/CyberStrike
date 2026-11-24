from pathlib import Path

import pytest

from evo.localsearch import (
    SeedSplit,
    default_seed_split,
    hill_climb,
    load_tuned_weights,
    save_tuned_weights,
    simulated_annealing,
    weight_neighbors,
)
from evo.plots import plot_local_search_curve
from search.evaluation import DEFAULT_WEIGHTS, EvalWeights

PNG = b"\x89PNG"


def _quadratic_peak(w: EvalWeights) -> float:
    # Smooth synthetic peak at (patch=30, firewall=50, exits=2.0, honeypot=14)
    return -(
        ((w.patch - 30.0) / 5.0) ** 2
        + ((w.firewall - 50.0) / 10.0) ** 2
        + ((w.exits - 2.0) / 0.5) ** 2
        + ((w.honeypot - 14.0) / 4.0) ** 2
    )


def test_seed_split_enforces_disjoint_train_and_test_seeds() -> None:
    split = default_seed_split(train_count=20, test_count=20)
    assert set(split.train_seeds).isdisjoint(split.test_seeds)
    assert min(split.test_seeds) >= 1000

    with pytest.raises(ValueError):
        SeedSplit(train_seeds=(0, 1, 2), test_seeds=(2, 3, 4))


def test_weight_neighbors_nudges_one_weight_at_a_time() -> None:
    nbrs = weight_neighbors(DEFAULT_WEIGHTS)
    assert len(nbrs) == 8
    assert DEFAULT_WEIGHTS not in nbrs


def test_hill_climb_finds_exact_optimum_on_synthetic_surface() -> None:
    res = hill_climb(_quadratic_peak, restarts=2, max_iterations=15, seed=7)
    assert res.best_score > res.default_score
    assert res.best_score == pytest.approx(0.0, abs=1e-6)
    assert res.best_weights == EvalWeights(patch=30.0, firewall=50.0, exits=2.0, honeypot=14.0)


def test_simulated_annealing_improves_over_initial_weights() -> None:
    res = simulated_annealing(
        _quadratic_peak, max_iterations=40, initial_temp=5.0, cooling=0.9, seed=7
    )
    assert res.best_score >= res.default_score
    assert len(res.history) == 41


def test_tuned_weights_save_and_load_round_trip(tmp_path: Path) -> None:
    w = EvalWeights(patch=25.0, firewall=60.0, exits=1.5, honeypot=14.0)
    path = tmp_path / "tuned.json"
    save_tuned_weights(w, path, train_score=720.0, default_test_score=700.0, tuned_test_score=715.0)
    assert load_tuned_weights(path) == w


def test_local_search_curve_plot_creates_png(tmp_path: Path) -> None:
    res = hill_climb(_quadratic_peak, restarts=1, max_iterations=8, seed=1)
    out = tmp_path / "local_search_curve.png"
    plot_local_search_curve(res.history, 680.0, 710.0, out)
    assert out.exists() and out.read_bytes()[:4] == PNG and out.stat().st_size > 2000

