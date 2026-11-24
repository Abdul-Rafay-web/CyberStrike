from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from agents.minimax_attacker import MinimaxAttacker
from agents.simple_defenders import ReactiveDefender
from config import CONFIG, AppConfig, LocalSearchConfig, SensorConfig
from core.game import play_game
from core.generator import generate_network
from core.rules import Outcome
from search.evaluation import DEFAULT_WEIGHTS, EvalWeights
from utils.rng import STREAM_LOCAL_SEARCH, STREAM_SENSORS, make_rng

PERFECT_SENSORS = replace(CONFIG, sensors=SensorConfig(detection_rate=1.0, false_alarm_rate=0.0))

ObjectiveFn = Callable[[EvalWeights], float]


@dataclass(frozen=True, slots=True)
class SeedSplit:
    """Strictly disjoint training and test seed ranges enforced at construction."""

    train_seeds: tuple[int, ...]
    test_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.train_seeds or not self.test_seeds:
            raise ValueError("train_seeds and test_seeds must both be non-empty")
        if set(self.train_seeds) & set(self.test_seeds):
            raise ValueError("train_seeds and test_seeds must be completely disjoint")


def default_seed_split(
    train_count: int = CONFIG.local_search.train_seeds_count,
    test_count: int = CONFIG.local_search.test_seeds_count,
    config: AppConfig = CONFIG,
) -> SeedSplit:
    """Build disjoint dev (0..) and test (1000..) seed splits from `config.seeds`."""
    dev_start = config.seeds.dev_seed_start
    test_start = config.seeds.test_seed_start
    train = tuple(range(dev_start, dev_start + train_count))
    test = tuple(range(test_start, test_start + test_count))
    return SeedSplit(train_seeds=train, test_seeds=test)


@dataclass(frozen=True, slots=True)
class SearchStep:
    """Telemetry recorded at each local search iteration."""

    iteration: int
    restart: int
    current_score: float
    best_score: float
    weights: EvalWeights


@dataclass(frozen=True, slots=True)
class LocalSearchResult:
    """Best weights found, score history, and evaluation count."""

    best_weights: EvalWeights
    best_score: float
    default_score: float
    history: tuple[SearchStep, ...]
    evaluations: int


def _clamp_weights(patch: float, firewall: float, exits: float, honeypot: float) -> EvalWeights:
    return EvalWeights(
        patch=max(0.0, min(100.0, round(patch, 3))),
        firewall=max(0.0, min(150.0, round(firewall, 3))),
        exits=max(0.0, min(20.0, round(exits, 3))),
        honeypot=max(0.0, min(60.0, round(honeypot, 3))),
    )


def weight_neighbors(
    weights: EvalWeights,
    step_sizes: tuple[float, float, float, float] = CONFIG.local_search.step_sizes,
) -> list[EvalWeights]:
    """Generate up to 8 neighbouring weight vectors by nudging one weight up or down."""
    dp, df, de, dh = step_sizes
    raw = [
        _clamp_weights(weights.patch + dp, weights.firewall, weights.exits, weights.honeypot),
        _clamp_weights(weights.patch - dp, weights.firewall, weights.exits, weights.honeypot),
        _clamp_weights(weights.patch, weights.firewall + df, weights.exits, weights.honeypot),
        _clamp_weights(weights.patch, weights.firewall - df, weights.exits, weights.honeypot),
        _clamp_weights(weights.patch, weights.firewall, weights.exits + de, weights.honeypot),
        _clamp_weights(weights.patch, weights.firewall, weights.exits - de, weights.honeypot),
        _clamp_weights(weights.patch, weights.firewall, weights.exits, weights.honeypot + dh),
        _clamp_weights(weights.patch, weights.firewall, weights.exits, weights.honeypot - dh),
    ]
    unique: list[EvalWeights] = []
    for cand in raw:
        if cand != weights and cand not in unique:
            unique.append(cand)
    return unique


@dataclass(frozen=True, slots=True)
class WeightEvalSummary:
    """Mean utility, win rate, and turns measured for one `EvalWeights` vector."""

    score: float
    win_rate: float
    mean_turns: float
    mean_utility: float


def evaluate_weights_on_seeds(
    weights: EvalWeights,
    seeds: Sequence[int],
    *,
    depth: int = CONFIG.local_search.eval_depth,
    time_budget: float = CONFIG.local_search.eval_time_budget,
) -> WeightEvalSummary:
    """Score `weights` by playing MinimaxAttacker vs ReactiveDefender on `seeds`."""
    if not seeds:
        raise ValueError("seeds cannot be empty")
    wins = 0
    total_turns = 0
    total_utility = 0
    for seed in seeds:
        net = generate_network(seed)
        res = play_game(
            net,
            MinimaxAttacker(max_depth=depth, time_budget=time_budget, weights=weights),
            ReactiveDefender(),
            make_rng(seed, STREAM_SENSORS),
            PERFECT_SENSORS,
        )
        wins += res.outcome is Outcome.ATTACKER_WINS
        total_turns += res.turns
        total_utility += res.utility

    n = len(seeds)
    win_rate = wins / n
    mean_turns = total_turns / n
    mean_utility = total_utility / n
    return WeightEvalSummary(
        score=mean_utility,
        win_rate=win_rate,
        mean_turns=mean_turns,
        mean_utility=mean_utility,
    )


def hill_climb(
    objective: ObjectiveFn,
    *,
    initial: EvalWeights = DEFAULT_WEIGHTS,
    restarts: int = CONFIG.local_search.restarts,
    max_iterations: int = CONFIG.local_search.max_iterations,
    step_sizes: tuple[float, float, float, float] = CONFIG.local_search.step_sizes,
    seed: int = CONFIG.seeds.master_seed,
) -> LocalSearchResult:
    """Steepest-ascent hill climbing with random restarts over `EvalWeights`."""
    rng = make_rng(seed, STREAM_LOCAL_SEARCH)
    cache: dict[EvalWeights, float] = {}

    def score_of(w: EvalWeights) -> float:
        if w not in cache:
            cache[w] = float(objective(w))
        return cache[w]

    default_score = score_of(initial)
    global_best_w = initial
    global_best_score = default_score
    history: list[SearchStep] = []
    step_idx = 0

    for restart in range(max(1, restarts)):
        if restart == 0:
            current_w = initial
        else:
            current_w = _clamp_weights(
                float(rng.uniform(5.0, 45.0)),
                float(rng.uniform(10.0, 80.0)),
                float(rng.uniform(0.0, 5.0)),
                float(rng.uniform(2.0, 25.0)),
            )
        current_score = score_of(current_w)
        if current_score > global_best_score:
            global_best_score = current_score
            global_best_w = current_w

        history.append(SearchStep(step_idx, restart, current_score, global_best_score, current_w))
        step_idx += 1

        for _ in range(max_iterations):
            neighbors = weight_neighbors(current_w, step_sizes)
            best_nbr = current_w
            best_nbr_score = current_score
            for cand in neighbors:
                s = score_of(cand)
                if s > best_nbr_score + 1e-9:
                    best_nbr_score = s
                    best_nbr = cand

            if best_nbr == current_w:
                break

            current_w = best_nbr
            current_score = best_nbr_score
            if current_score > global_best_score:
                global_best_score = current_score
                global_best_w = current_w

            history.append(
                SearchStep(step_idx, restart, current_score, global_best_score, current_w)
            )
            step_idx += 1

    return LocalSearchResult(
        best_weights=global_best_w,
        best_score=global_best_score,
        default_score=default_score,
        history=tuple(history),
        evaluations=len(cache),
    )


def simulated_annealing(
    objective: ObjectiveFn,
    *,
    initial: EvalWeights = DEFAULT_WEIGHTS,
    max_iterations: int = CONFIG.local_search.max_iterations,
    step_sizes: tuple[float, float, float, float] = CONFIG.local_search.step_sizes,
    initial_temp: float = CONFIG.local_search.sa_initial_temp,
    cooling: float = CONFIG.local_search.sa_cooling,
    seed: int = CONFIG.seeds.master_seed,
) -> LocalSearchResult:
    """Simulated annealing over `EvalWeights` allowing occasional uphill/downhill escapes."""
    rng = make_rng(seed + 99, STREAM_LOCAL_SEARCH)
    cache: dict[EvalWeights, float] = {}

    def score_of(w: EvalWeights) -> float:
        if w not in cache:
            cache[w] = float(objective(w))
        return cache[w]

    current_w = initial
    current_score = score_of(current_w)
    default_score = current_score
    best_w = current_w
    best_score = current_score
    history: list[SearchStep] = [SearchStep(0, 0, current_score, best_score, current_w)]

    temp = max(1e-3, initial_temp)
    for it in range(1, max_iterations + 1):
        neighbors = weight_neighbors(current_w, step_sizes)
        cand = neighbors[int(rng.integers(0, len(neighbors)))]
        cand_score = score_of(cand)
        delta = cand_score - current_score
        if delta >= 0.0 or float(rng.random()) < math.exp(delta / temp):
            current_w = cand
            current_score = cand_score
            if current_score > best_score:
                best_score = current_score
                best_w = current_w
        history.append(SearchStep(it, 0, current_score, best_score, current_w))
        temp = max(1e-3, temp * cooling)

    return LocalSearchResult(
        best_weights=best_w,
        best_score=best_score,
        default_score=default_score,
        history=tuple(history),
        evaluations=len(cache),
    )


def save_tuned_weights(
    weights: EvalWeights,
    path: Path,
    *,
    train_score: float,
    default_test_score: float,
    tuned_test_score: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "weights": {
            "patch": weights.patch,
            "firewall": weights.firewall,
            "exits": weights.exits,
            "honeypot": weights.honeypot,
        },
        "train_score": round(train_score, 3),
        "default_test_score": round(default_test_score, 3),
        "tuned_test_score": round(tuned_test_score, 3),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_tuned_weights(path: Path) -> EvalWeights:
    payload = json.loads(path.read_text(encoding="utf-8"))
    w = payload["weights"]
    return EvalWeights(
        patch=float(w["patch"]),
        firewall=float(w["firewall"]),
        exits=float(w["exits"]),
        honeypot=float(w["honeypot"]),
    )


def tune_eval_weights(
    split: SeedSplit | None = None,
    ls_cfg: LocalSearchConfig = CONFIG.local_search,
    seed: int = CONFIG.seeds.master_seed,
) -> tuple[LocalSearchResult, WeightEvalSummary, WeightEvalSummary]:
    """Tune `EvalWeights` strictly on `split.train_seeds`, then evaluate once on `test_seeds`."""
    resolved = (
        split
        if split is not None
        else default_seed_split(ls_cfg.train_seeds_count, ls_cfg.test_seeds_count)
    )

    def train_obj(w: EvalWeights) -> float:
        return evaluate_weights_on_seeds(
            w,
            resolved.train_seeds,
            depth=ls_cfg.eval_depth,
            time_budget=ls_cfg.eval_time_budget,
        ).score

    result = hill_climb(
        train_obj,
        initial=DEFAULT_WEIGHTS,
        restarts=ls_cfg.restarts,
        max_iterations=ls_cfg.max_iterations,
        step_sizes=ls_cfg.step_sizes,
        seed=seed,
    )
    default_test = evaluate_weights_on_seeds(
        DEFAULT_WEIGHTS,
        resolved.test_seeds,
        depth=ls_cfg.eval_depth,
        time_budget=ls_cfg.eval_time_budget,
    )
    tuned_test = evaluate_weights_on_seeds(
        result.best_weights,
        resolved.test_seeds,
        depth=ls_cfg.eval_depth,
        time_budget=ls_cfg.eval_time_budget,
    )
    return result, default_test, tuned_test
