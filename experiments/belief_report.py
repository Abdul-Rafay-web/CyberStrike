from __future__ import annotations

from collections import defaultdict

import numpy as np

from agents.belief_defender import BeliefReactiveDefender, load_sensor_rates
from agents.greedy_attacker import GreedyAttacker
from agents.minimax_attacker import MinimaxAttacker
from agents.simple_defenders import StaticDefender
from config import CONFIG, BeliefConfig, SensorConfig
from core.actions import Action
from core.game import play_game
from core.generator import generate_network
from core.network import Network
from core.observation import observe_defender
from core.rules import Outcome
from core.sensors import active_machine
from core.state import GameState, Side
from ml.pipeline import load_sensor, train_sensor
from ml.sensor import ClassifierSensor
from prob.bayesnet import exact_enumeration_posterior
from prob.belief import BeliefTracker
from prob.blanket import markov_blanket
from prob.plots import (
    TrackerSummaryRow,
    TurnBeliefPoint,
    plot_belief_by_turn,
    plot_tracker_comparison,
)
from utils.paths import DATA_DIR, RESULTS_DIR
from utils.rng import STREAM_SENSORS, make_rng


def _build_six_node_network() -> Network:
    """Small 6-machine network with multiple paths for exact enumeration verification."""
    return Network.from_edges(
        6,
        [(0, 1), (0, 2), (1, 3), (2, 3), (1, 4), (2, 4), (3, 5), (4, 5)],
        hardness=[1, 1, 2, 2, 1, 3],
        entry=0,
        jewel=5,
    )


def _last_alert_distribution(alerts: tuple[bool, ...]) -> tuple[float, ...]:
    """Baseline that splits mass evenly across alerting machines (or uniform if none fire)."""
    flagged = [i for i, flag in enumerate(alerts) if flag]
    size = len(alerts)
    if not flagged:
        return tuple(1.0 / size for _ in range(size))
    share = 1.0 / len(flagged)
    return tuple(share if i in flagged else 0.0 for i in range(size))


def _top_k_contains(dist: tuple[float, ...], target: int, k: int) -> bool:
    ranked = sorted(range(len(dist)), key=lambda i: (-dist[i], i))[:k]
    return target in ranked


def verify_exact_enumeration_on_six_nodes(rates: SensorConfig) -> tuple[float, float]:
    """Compare `BeliefTracker` against brute-force exact enumeration on a 6-machine map."""
    net = _build_six_node_network()
    alert_history = [
        (False, True, False, False, False, False),
        (False, False, False, True, False, False),
        (False, False, False, True, False, True),
        (False, False, False, False, False, True),
    ]
    exact = exact_enumeration_posterior(net, alert_history, rates)

    exact_cfg = BeliefConfig(stay_base=CONFIG.belief.stay_base, floor=0.0)
    tracker_exact = BeliefTracker(net, rates, config=exact_cfg)
    tracker_floor = BeliefTracker(net, rates, config=CONFIG.belief)
    for alerts in alert_history:
        tracker_exact.predict()
        tracker_exact.update(alerts)
        tracker_floor.predict()
        tracker_floor.update(alerts)

    max_err_exact = max(
        abs(a - b) for a, b in zip(exact, tracker_exact.probabilities, strict=True)
    )
    max_err_floor = max(
        abs(a - b) for a, b in zip(exact, tracker_floor.probabilities, strict=True)
    )
    return max_err_exact, max_err_floor


def _run_eval_game(
    network: Network,
    seed: int,
    rates: SensorConfig,
    sensor: ClassifierSensor,
    by_turn: dict[int, dict[str, list[float]]],
    all_belief: list[tuple[float, float, float]],
    all_alert: list[tuple[float, float, float]],
    all_uniform: list[tuple[float, float, float]],
) -> None:
    size = network.size
    tracker = BeliefTracker(network, rates)
    sensor_rng = make_rng(seed, STREAM_SENSORS)

    def on_step(before: GameState, action: Action, after: GameState) -> None:
        del action
        if before.side is Side.ATTACKER and after.side is Side.DEFENDER:
            alerts = sensor.alerts(after, sensor_rng)
            obs = observe_defender(after, alerts, CONFIG.game)
            belief = tracker.step(obs)
            truth = active_machine(after)

            last_alert = _last_alert_distribution(alerts)
            uniform_p = 1.0 / size

            b_prob = belief[truth]
            b_top1 = float(_top_k_contains(belief, truth, 1))
            b_top3 = float(_top_k_contains(belief, truth, 3))

            la_prob = last_alert[truth]
            la_top1 = float(_top_k_contains(last_alert, truth, 1))
            la_top3 = float(_top_k_contains(last_alert, truth, 3))

            bucket = by_turn[after.turn]
            bucket["belief"].append(b_prob)
            bucket["uniform"].append(uniform_p)
            bucket["last_alert"].append(la_prob)
            bucket["top1"].append(b_top1)
            bucket["top3"].append(b_top3)
            bucket["entropy"].append(tracker.entropy)

            all_belief.append((b_prob, b_top1, b_top3))
            all_alert.append((la_prob, la_top1, la_top3))
            all_uniform.append((uniform_p, 1.0 / size, min(1.0, 3.0 / size)))

    play_game(
        network,
        GreedyAttacker(),
        StaticDefender(),
        make_rng(seed + 10_000, STREAM_SENSORS),
        CONFIG,
        on_step=on_step,
        sensor=sensor,
    )


def evaluate_tracker(
    games: int = CONFIG.belief.eval_games,
) -> tuple[list[TurnBeliefPoint], list[TrackerSummaryRow]]:
    """Measure Bayesian belief accuracy per turn over seeded games using the Naive Bayes sensor."""
    model_path = DATA_DIR / "sensor_model.json"
    trained = load_sensor(model_path) if model_path.exists() else train_sensor()
    rates = trained.rates
    sensor = ClassifierSensor.from_trained(trained)

    by_turn: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    all_belief: list[tuple[float, float, float]] = []
    all_alert: list[tuple[float, float, float]] = []
    all_uniform: list[tuple[float, float, float]] = []

    for seed in range(games):
        network = generate_network(seed)
        _run_eval_game(
            network,
            seed,
            rates,
            sensor,
            by_turn,
            all_belief,
            all_alert,
            all_uniform,
        )

    points: list[TurnBeliefPoint] = []
    for turn in sorted(by_turn):
        bucket = by_turn[turn]
        if len(bucket["belief"]) < 5:
            continue
        points.append(
            TurnBeliefPoint(
                turn=turn,
                samples=len(bucket["belief"]),
                belief_true_prob=float(np.mean(bucket["belief"])),
                uniform_true_prob=float(np.mean(bucket["uniform"])),
                last_alert_true_prob=float(np.mean(bucket["last_alert"])),
                top1_hit_rate=float(np.mean(bucket["top1"])),
                top3_hit_rate=float(np.mean(bucket["top3"])),
                mean_entropy=float(np.mean(bucket["entropy"])),
            )
        )

    def _summarize(name: str, rows: list[tuple[float, float, float]]) -> TrackerSummaryRow:
        arr = np.array(rows, dtype=np.float64)
        return TrackerSummaryRow(
            name=name,
            mean_true_prob=float(arr[:, 0].mean()),
            top1_hit_rate=float(arr[:, 1].mean()),
            top3_hit_rate=float(arr[:, 2].mean()),
        )

    summary = [
        _summarize("Bayesian belief", all_belief),
        _summarize("Last-alert guess", all_alert),
        _summarize("Uniform (1/N)", all_uniform),
    ]
    return points, summary


def main() -> None:
    rates = load_sensor_rates()
    print(
        f"using Phase 5 measured sensor rates: detection_rate={rates.detection_rate:.3f}, "
        f"false_alarm_rate={rates.false_alarm_rate:.3f}\n"
    )

    err_exact, err_floor = verify_exact_enumeration_on_six_nodes(rates)
    print("exact enumeration check on 6-machine map (4 turns):")
    print(f"  max |BeliefTracker(floor=0) - Exact|     = {err_exact:.3e}")
    print(f"  max |BeliefTracker(floor=1e-6) - Exact|  = {err_floor:.3e}\n")

    six_net = _build_six_node_network()
    mb = markov_blanket(six_net, 1)
    print(
        f"causal Markov blanket on 6-machine map for machine 1: "
        f"parents={sorted(mb.parents)}, children={sorted(mb.children)}, "
        f"co_parents={sorted(mb.co_parents)}\n"
    )

    points, summary = evaluate_tracker(CONFIG.belief.eval_games)
    print(f"belief tracking across {CONFIG.belief.eval_games} seeded games (Naive Bayes sensor):")
    print(f"{'method':<20}{'P(true)':>10}{'Top-1':>10}{'Top-3':>10}")
    for row in summary:
        print(
            f"{row.name:<20}{row.mean_true_prob:>10.3f}"
            f"{row.top1_hit_rate:>10.1%}{row.top3_hit_rate:>10.1%}"
        )

    print("\nper-turn probability on the true machine:")
    print(f"{'turn':>4}{'n':>6}{'belief':>10}{'last-alert':>12}{'uniform':>10}{'top-3':>9}{'H(bits)':>9}")
    for pt in points:
        print(
            f"{pt.turn:>4}{pt.samples:>6}{pt.belief_true_prob:>10.3f}"
            f"{pt.last_alert_true_prob:>12.3f}{pt.uniform_true_prob:>10.3f}"
            f"{pt.top3_hit_rate:>9.1%}{pt.mean_entropy:>9.2f}"
        )

    # Quick matchup check: BeliefReactiveDefender vs StaticDefender under noisy ClassifierSensor
    model_path = DATA_DIR / "sensor_model.json"
    trained = load_sensor(model_path) if model_path.exists() else train_sensor()
    sensor = ClassifierSensor.from_trained(trained)
    def_wins_static = 0
    def_wins_belief = 0
    sample_games = 30
    for seed in range(sample_games):
        net = generate_network(seed)
        r_static = play_game(
            net,
            MinimaxAttacker(max_depth=3, time_budget=0.5),
            StaticDefender(),
            make_rng(seed, STREAM_SENSORS),
            CONFIG,
            sensor=sensor,
        )
        r_belief = play_game(
            net,
            MinimaxAttacker(max_depth=3, time_budget=0.5),
            BeliefReactiveDefender(rates=rates),
            make_rng(seed, STREAM_SENSORS),
            CONFIG,
            sensor=sensor,
        )
        def_wins_static += r_static.outcome is Outcome.DEFENDER_WINS
        def_wins_belief += r_belief.outcome is Outcome.DEFENDER_WINS
    print(
        f"\nminimax depth-3 vs defenders over {sample_games} noisy-sensor games: "
        f"StaticDefender wins {def_wins_static}/{sample_games}, "
        f"BeliefReactiveDefender wins {def_wins_belief}/{sample_games}"
    )

    plot_belief_by_turn(points, RESULTS_DIR / "belief_by_turn.png")
    plot_tracker_comparison(summary, RESULTS_DIR / "belief_comparison.png")
    print("\nsaved results/belief_by_turn.png and results/belief_comparison.png")


if __name__ == "__main__":
    main()
