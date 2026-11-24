from __future__ import annotations

import numpy as np

from agents.belief_csp_defender import BeliefCSPDefender
from agents.belief_defender import load_sensor_rates
from agents.greedy_attacker import GreedyAttacker
from agents.random_agents import RandomAttacker
from agents.simple_defenders import StaticDefender
from config import CONFIG, SensorConfig
from core.game import play_game
from core.generator import generate_network
from core.rules import Outcome
from core.state import Stock
from csp.model import build_defense_csp
from csp.plots import CSPBenchRow, plot_csp_backtracks, plot_csp_defender_matchup
from csp.solver import ALL_SOLVER_CONFIGS, solve_csp
from ml.pipeline import load_sensor, train_sensor
from ml.sensor import ClassifierSensor
from utils.paths import DATA_DIR, RESULTS_DIR
from utils.rng import STREAM_AGENTS, STREAM_SENSORS, make_rng


def benchmark_solvers(
    sizes: tuple[int, ...] = CONFIG.csp.bench_sizes,
    instances: int = CONFIG.csp.bench_instances,
) -> list[CSPBenchRow]:
    """Benchmark all four CSP solver configurations on `instances` maps per size."""
    stock = Stock(
        CONFIG.game.stock.patches,
        CONFIG.game.stock.firewalls,
        CONFIG.game.stock.honeypots,
    )
    rows: list[CSPBenchRow] = []

    for size in sizes:
        problems = [
            build_defense_csp(generate_network(seed, size), stock) for seed in range(instances)
        ]
        for cfg in ALL_SOLVER_CONFIGS:
            results = [solve_csp(problem, cfg) for problem in problems]
            solved_count = sum(1 for r in results if r.stats.solved)
            rows.append(
                CSPBenchRow(
                    map_size=size,
                    solver_name=cfg.name,
                    mean_assignments=float(np.mean([r.stats.assignments_tried for r in results])),
                    mean_backtracks=float(np.mean([r.stats.backtracks for r in results])),
                    mean_ms=float(np.mean([r.stats.seconds * 1000.0 for r in results])),
                    solve_rate=solved_count / len(results),
                )
            )
    return rows


def evaluate_defenders(
    games: int = CONFIG.csp.matchup_games,
    rates: SensorConfig | None = None,
) -> list[tuple[str, float, float]]:
    """Compare BeliefCSPDefender against StaticDefender over `games` seeded maps."""
    model_path = DATA_DIR / "sensor_model.json"
    trained = load_sensor(model_path) if model_path.exists() else train_sensor()
    resolved_rates = rates if rates is not None else trained.rates
    sensor = ClassifierSensor.from_trained(trained)

    summary: list[tuple[str, float, float]] = []
    for attacker_label, use_random in (("vs Random Attacker", True), ("vs Greedy A*", False)):
        static_wins = 0
        csp_wins = 0
        for seed in range(games):
            net = generate_network(seed)
            atk_static = (
                RandomAttacker(make_rng(seed, STREAM_AGENTS)) if use_random else GreedyAttacker()
            )
            atk_csp = (
                RandomAttacker(make_rng(seed, STREAM_AGENTS)) if use_random else GreedyAttacker()
            )
            res_static = play_game(
                net,
                atk_static,
                StaticDefender(),
                make_rng(seed, STREAM_SENSORS),
                CONFIG,
                sensor=sensor,
            )
            res_csp = play_game(
                net,
                atk_csp,
                BeliefCSPDefender(rates=resolved_rates),
                make_rng(seed, STREAM_SENSORS),
                CONFIG,
                sensor=sensor,
            )
            static_wins += res_static.outcome is Outcome.DEFENDER_WINS
            csp_wins += res_csp.outcome is Outcome.DEFENDER_WINS
        summary.append((attacker_label, static_wins / games, csp_wins / games))
    return summary


def main() -> None:
    rates = load_sensor_rates()
    print(
        f"CSP solver benchmark: {CONFIG.csp.bench_instances} instances per map size "
        f"{CONFIG.csp.bench_sizes}"
    )
    print(
        f"{'size':>5}{'solver':>14}{'assignments':>14}{'backtracks':>13}{'ms':>10}{'solved':>9}"
    )
    rows = benchmark_solvers()
    for row in rows:
        print(
            f"{row.map_size:>5}{row.solver_name:>14}{row.mean_assignments:>14.1f}"
            f"{row.mean_backtracks:>13.1f}{row.mean_ms:>10.2f}{row.solve_rate:>9.0%}"
        )

    print(f"\ndefender matchup over {CONFIG.csp.matchup_games} seeded maps (Naive Bayes sensor):")
    print(f"{'matchup':<22}{'StaticDefender':>16}{'BeliefCSPDefender':>20}")
    matchups = evaluate_defenders(CONFIG.csp.matchup_games, rates=rates)
    for label, static_rate, csp_rate in matchups:
        print(f"{label:<22}{static_rate:>16.1%}{csp_rate:>20.1%}")

    plot_csp_backtracks(rows, RESULTS_DIR / "csp_backtracks.png")
    plot_csp_defender_matchup(matchups, RESULTS_DIR / "csp_defender_matchup.png")
    print("\nsaved results/csp_backtracks.png and results/csp_defender_matchup.png")


if __name__ == "__main__":
    main()
