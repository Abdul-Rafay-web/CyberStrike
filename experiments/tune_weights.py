from __future__ import annotations

from agents.coach import analyze_defender_game
from agents.greedy_attacker import GreedyAttacker
from agents.random_agents import RandomDefender
from config import CONFIG
from core.actions import Action
from core.game import play_game
from core.generator import generate_network
from core.state import GameState
from evo.localsearch import (
    default_seed_split,
    save_tuned_weights,
    tune_eval_weights,
)
from evo.plots import plot_local_search_curve
from utils.paths import DATA_DIR, RESULTS_DIR
from utils.rng import STREAM_AGENTS, STREAM_SENSORS, make_rng


def main() -> None:
    split = default_seed_split()
    print(
        f"tuning Minimax EvalWeights via hill climbing with random restarts "
        f"(train seeds {split.train_seeds[0]}..{split.train_seeds[-1]}, "
        f"held-out test seeds {split.test_seeds[0]}..{split.test_seeds[-1]})..."
    )
    result, default_test, tuned_test = tune_eval_weights(split)

    print(
        f"train: default score={result.default_score:.2f} -> "
        f"best score={result.best_score:.2f} ({result.evaluations} evals)"
    )
    print(f"best weights: {result.best_weights}")
    print(
        f"held-out test ({len(split.test_seeds)} unseen seeds): "
        f"default utility={default_test.mean_utility:.2f} (win={default_test.win_rate:.1%}, "
        f"turns={default_test.mean_turns:.2f}) vs "
        f"tuned utility={tuned_test.mean_utility:.2f} (win={tuned_test.win_rate:.1%}, "
        f"turns={tuned_test.mean_turns:.2f})"
    )

    save_tuned_weights(
        result.best_weights,
        DATA_DIR / "tuned_weights.json",
        train_score=result.best_score,
        default_test_score=default_test.mean_utility,
        tuned_test_score=tuned_test.mean_utility,
    )
    plot_local_search_curve(
        result.history,
        default_test.mean_utility,
        tuned_test.mean_utility,
        RESULTS_DIR / "local_search_curve.png",
    )

    # Post-game Coach demonstration on a finished game
    recorded: list[tuple[GameState, Action]] = []

    def record(before: GameState, action: Action, after: GameState) -> None:
        del after
        recorded.append((before, action))

    play_game(
        generate_network(1001),
        GreedyAttacker(),
        RandomDefender(make_rng(1001, STREAM_AGENTS)),
        make_rng(1001, STREAM_SENSORS),
        CONFIG,
        on_step=record,
    )
    report = analyze_defender_game(recorded)
    print(f"\npost-game coach ({report.decisions_reviewed} Defender turns reviewed):")
    print(f"  {report.headline}")
    for tip in report.advice:
        print(f"  - {tip.summary}")

    print("\nsaved data/tuned_weights.json and results/local_search_curve.png")


if __name__ == "__main__":
    main()

