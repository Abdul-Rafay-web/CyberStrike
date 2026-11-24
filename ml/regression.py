from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from agents.greedy_attacker import GreedyAttacker
from agents.random_agents import RandomAttacker
from agents.simple_defenders import StaticDefender
from config import CONFIG, AppConfig
from core.game import play_game
from core.generator import generate_network
from core.network import Network, bfs_distances
from core.rules import Outcome
from core.state import new_game
from csp.model import k_cheapest_routes
from search.problem import problem_from_state
from search.uninformed import uniform_cost
from utils.rng import STREAM_AGENTS, STREAM_SENSORS, make_rng

MAP_FEATURE_NAMES: tuple[str, ...] = (
    "machines",
    "links",
    "mean_hardness",
    "cheapest_cost",
    "route_count",
    "entry_jewel_hops",
    "avg_degree",
)


def extract_map_features(network: Network) -> np.ndarray:
    """Extract numeric structural features of a network for fast difficulty regression."""
    size = network.size
    links = len(network.edges())
    mean_hardness = sum(network.hardness) / size
    ucs = uniform_cost(problem_from_state(new_game(network)))
    cheapest_cost = float(ucs.cost if ucs.cost is not None else size * 3)
    routes = k_cheapest_routes(network, network.entry, k=5)
    route_count = float(len(routes))
    hops = float(bfs_distances(network.adjacency, network.entry)[network.jewel])
    avg_degree = (2.0 * links) / size
    return np.array(
        [
            float(size),
            float(links),
            mean_hardness,
            cheapest_cost,
            route_count,
            hops,
            avg_degree,
        ],
        dtype=np.float64,
    )


def simulate_map_difficulty(
    network: Network,
    trials: int = CONFIG.evo.regression_trials,
    base_seed: int = 0,
    config: AppConfig = CONFIG,
) -> tuple[float, float]:
    """Measure `(greedy_turns_to_breach, random_attacker_win_rate)` on `network`.

    Greedy A* vs StaticDefender gives a deterministic, low-noise measure of attacker turns
    to breach; `trials` seeded RandomAttacker vs StaticDefender games measure baseline balance.
    """
    if trials < 1:
        raise ValueError("trials must be at least 1")
    greedy_res = play_game(
        network,
        GreedyAttacker(),
        StaticDefender(),
        make_rng(base_seed, STREAM_SENSORS),
        config,
    )
    wins = 0
    for i in range(trials):
        seed = base_seed * 31 + i
        res = play_game(
            network,
            RandomAttacker(make_rng(seed, STREAM_AGENTS)),
            StaticDefender(),
            make_rng(seed, STREAM_SENSORS),
            config,
        )
        wins += res.outcome is Outcome.ATTACKER_WINS
    return float(greedy_res.turns), wins / trials


@dataclass(frozen=True, slots=True)
class RegressionReport:
    """Held-out evaluation metrics for a fitted linear regression model."""

    train_r2: float
    test_r2: float
    train_rmse: float
    test_rmse: float


class LinearRegression:
    """Ordinary least-squares linear regression implemented directly in NumPy."""

    def __init__(self) -> None:
        self._weights: np.ndarray = np.zeros(0, dtype=np.float64)
        self._bias: float = 0.0
        self._fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def weights(self) -> np.ndarray:
        return self._weights.copy()

    @property
    def bias(self) -> float:
        return self._bias

    def fit(self, features: np.ndarray, targets: np.ndarray) -> LinearRegression:
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if x.ndim != 2 or len(x) == 0:
            raise ValueError("features must be a non-empty 2D array")
        if y.ndim != 1 or len(y) != len(x):
            raise ValueError("targets must be a 1D array with one entry per row")
        design = np.hstack([x, np.ones((len(x), 1), dtype=np.float64)])
        coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        self._weights = coeffs[:-1].astype(np.float64)
        self._bias = float(coeffs[-1])
        self._fitted = True
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("fit the model before predicting")
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self._weights):
            raise ValueError("features has the wrong number of columns")
        result: np.ndarray = x @ self._weights + self._bias
        return result

    def predict_network(self, network: Network) -> float:
        feats = extract_map_features(network).reshape(1, -1)
        return float(self.predict(feats)[0])

    def rmse(self, features: np.ndarray, targets: np.ndarray) -> float:
        preds = self.predict(features)
        y = np.asarray(targets, dtype=np.float64)
        return float(np.sqrt(np.mean((y - preds) ** 2)))

    def r_squared(self, features: np.ndarray, targets: np.ndarray) -> float:
        preds = self.predict(features)
        y = np.asarray(targets, dtype=np.float64)
        ss_res = float(np.sum((y - preds) ** 2))
        ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return 1.0 - (ss_res / ss_tot)

    def to_dict(self) -> dict[str, Any]:
        return {"weights": self._weights.tolist(), "bias": self._bias}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LinearRegression:
        model = cls()
        model._weights = np.array(payload["weights"], dtype=np.float64)
        model._bias = float(payload["bias"])
        model._fitted = True
        return model


@dataclass(frozen=True, slots=True)
class DifficultySurrogate:
    """Paired linear regressors predicting attacker turns to breach and attacker win rate."""

    turns_model: LinearRegression
    win_rate_model: LinearRegression
    turns_report: RegressionReport
    win_rate_report: RegressionReport
    test_true_turns: tuple[float, ...]
    test_pred_turns: tuple[float, ...]

    def predict_turns(self, network: Network) -> float:
        return max(1.0, self.turns_model.predict_network(network))

    def predict_win_rate(self, network: Network) -> float:
        raw = self.win_rate_model.predict_network(network)
        return min(1.0, max(0.0, raw))


def fit_difficulty_surrogate(
    num_maps: int = CONFIG.evo.regression_maps,
    trials_per_map: int = CONFIG.evo.regression_trials,
    test_fraction: float = 0.2,
    seed_offset: int = 0,
    config: AppConfig = CONFIG,
) -> DifficultySurrogate:
    """Simulate `num_maps` varied random maps, fit OLS regressors, and evaluate on held-out maps."""
    if num_maps < 10:
        raise ValueError("need at least 10 maps to fit and validate regression")
    sizes = (10, 12, 15, 18, 20)
    feats_list: list[np.ndarray] = []
    turns_list: list[float] = []
    win_list: list[float] = []

    for i in range(num_maps):
        seed = seed_offset + i
        size = sizes[i % len(sizes)]
        net = generate_network(seed, size=size, config=config)
        feats_list.append(extract_map_features(net))
        mean_turns, win_rate = simulate_map_difficulty(
            net, trials=trials_per_map, base_seed=seed, config=config
        )
        turns_list.append(mean_turns)
        win_list.append(win_rate)

    x_all = np.vstack(feats_list)
    y_turns = np.array(turns_list, dtype=np.float64)
    y_wins = np.array(win_list, dtype=np.float64)

    split = max(1, int(round(num_maps * (1.0 - test_fraction))))
    x_train, x_test = x_all[:split], x_all[split:]
    yt_train, yt_test = y_turns[:split], y_turns[split:]
    yw_train, yw_test = y_wins[:split], y_wins[split:]

    turns_model = LinearRegression().fit(x_train, yt_train)
    win_model = LinearRegression().fit(x_train, yw_train)

    turns_report = RegressionReport(
        train_r2=turns_model.r_squared(x_train, yt_train),
        test_r2=turns_model.r_squared(x_test, yt_test),
        train_rmse=turns_model.rmse(x_train, yt_train),
        test_rmse=turns_model.rmse(x_test, yt_test),
    )
    win_report = RegressionReport(
        train_r2=win_model.r_squared(x_train, yw_train),
        test_r2=win_model.r_squared(x_test, yw_test),
        train_rmse=win_model.rmse(x_train, yw_train),
        test_rmse=win_model.rmse(x_test, yw_test),
    )
    pred_turns = tuple(float(v) for v in turns_model.predict(x_test))
    true_turns = tuple(float(v) for v in yt_test)

    return DifficultySurrogate(
        turns_model=turns_model,
        win_rate_model=win_model,
        turns_report=turns_report,
        win_rate_report=win_report,
        test_true_turns=true_turns,
        test_pred_turns=pred_turns,
    )

