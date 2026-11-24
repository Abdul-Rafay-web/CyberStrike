from __future__ import annotations

import numpy as np

from config import CONFIG
from core.generator import generate_network
from evo.genetic import evolve_maps, save_evolved_map
from evo.plots import plot_ga_fitness_and_balance, plot_regression_fit
from ml.regression import fit_difficulty_surrogate, simulate_map_difficulty
from utils.paths import DATA_DIR, RESULTS_DIR


def main() -> None:
    evo = CONFIG.evo
    print(
        f"fitting linear regression difficulty surrogate on {evo.regression_maps} maps "
        f"({evo.regression_trials} trials/map)..."
    )
    surrogate = fit_difficulty_surrogate(
        num_maps=evo.regression_maps,
        trials_per_map=evo.regression_trials,
    )
    print(
        f"turns predictor:    train R^2={surrogate.turns_report.train_r2:.3f}, "
        f"held-out R^2={surrogate.turns_report.test_r2:.3f}, "
        f"held-out RMSE={surrogate.turns_report.test_rmse:.2f} turns"
    )
    print(
        f"win-rate predictor: train R^2={surrogate.win_rate_report.train_r2:.3f}, "
        f"held-out R^2={surrogate.win_rate_report.test_r2:.3f}, "
        f"held-out RMSE={surrogate.win_rate_report.test_rmse:.3f}"
    )

    print(
        f"\nevolving {CONFIG.game.default_machines}-machine maps: "
        f"pop={evo.population_size}, gens={evo.generations}..."
    )
    result = evolve_maps(
        surrogate, size=CONFIG.game.default_machines, seed=CONFIG.seeds.master_seed
    )
    first = result.history[0]
    last = result.history[-1]
    print(
        f"generation 0:  mean_fitness={first.mean_fitness:.3f}, "
        f"best_fitness={first.best_fitness:.3f}"
    )
    print(
        f"generation {last.generation}: mean_fitness={last.mean_fitness:.3f}, "
        f"best_fitness={last.best_fitness:.3f}, "
        f"pred_win_rate={last.best_predicted_win_rate:.1%}"
    )

    # Export top evolved maps to data/maps/
    maps_dir = DATA_DIR / "maps"
    export_n = min(evo.export_count, len(result.best_networks))
    for i in range(export_n):
        net = result.best_networks[i]
        save_evolved_map(
            net,
            maps_dir / f"evolved_{i + 1:02d}.json",
            name=f"EVOLVED_SECTOR_{i + 1:02d}",
            fitness=result.best_scores[i],
            predicted_win_rate=surrogate.predict_win_rate(net),
        )
    print(f"exported {export_n} evolved maps to data/maps/")

    # Measure true simulated win-rate balance on random maps vs top evolved maps (held-out trials)
    eval_trials = 20
    random_wins = [
        simulate_map_difficulty(
            generate_network(1000 + i, size=CONFIG.game.default_machines),
            trials=eval_trials,
            base_seed=5000 + i,
        )[1]
        for i in range(10)
    ]
    evolved_sample = result.best_networks[: min(10, len(result.best_networks))]
    evolved_wins = [
        simulate_map_difficulty(net, trials=eval_trials, base_seed=6000 + i)[1]
        for i, net in enumerate(evolved_sample)
    ]

    rand_err = float(np.mean([abs(w - 0.5) for w in random_wins]))
    evo_err = float(np.mean([abs(w - 0.5) for w in evolved_wins]))
    print(
        f"\nbalance check ({eval_trials} games/map): "
        f"random mean win={np.mean(random_wins):.1%} (|w-50%|={rand_err:.3f}), "
        f"evolved mean win={np.mean(evolved_wins):.1%} (|w-50%|={evo_err:.3f})"
    )

    plot_ga_fitness_and_balance(
        result.history, random_wins, evolved_wins, RESULTS_DIR / "ga_fitness.png"
    )
    plot_regression_fit(
        surrogate.test_true_turns,
        surrogate.test_pred_turns,
        surrogate.turns_report.test_r2,
        RESULTS_DIR / "regression_fit.png",
    )
    print("saved results/ga_fitness.png and results/regression_fit.png")


if __name__ == "__main__":
    main()
