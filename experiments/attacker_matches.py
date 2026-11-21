from __future__ import annotations

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, replace

from agents.base import AttackerAgent, DefenderAgent
from agents.greedy_attacker import GreedyAttacker
from agents.minimax_attacker import MinimaxAttacker
from agents.simple_defenders import PassiveDefender, ReactiveDefender, StaticDefender
from config import CONFIG, AppConfig, SensorConfig
from core.game import play_game
from core.generator import generate_network
from core.rules import Outcome
from utils.rng import STREAM_SENSORS, make_rng

PERFECT_SENSORS = replace(CONFIG, sensors=SensorConfig(detection_rate=1.0, false_alarm_rate=0.0))


@dataclass(frozen=True, slots=True)
class MatchRow:
    attacker: str
    defender: str
    win_rate: float
    mean_turns: float
    seconds_per_game: float


def play_matches(
    attacker_name: str,
    make_attacker: Callable[[], AttackerAgent],
    defender_name: str,
    make_defender: Callable[[], DefenderAgent],
    games: int,
    config: AppConfig = CONFIG,
) -> MatchRow:
    wins = 0
    turns = 0
    started = time.perf_counter()
    for seed in range(games):
        result = play_game(
            generate_network(seed),
            make_attacker(),
            make_defender(),
            make_rng(seed, STREAM_SENSORS),
            config,
        )
        wins += result.outcome is Outcome.ATTACKER_WINS
        turns += result.turns
    elapsed = time.perf_counter() - started
    return MatchRow(attacker_name, defender_name, wins / games, turns / games, elapsed / games)


def main(games: int = 40) -> None:
    attackers: list[tuple[str, Callable[[], AttackerAgent]]] = [
        ("greedy A*", GreedyAttacker),
        ("minimax depth 4", lambda: MinimaxAttacker(max_depth=4, time_budget=1.0)),
    ]
    defenders: list[tuple[str, Callable[[], DefenderAgent], AppConfig]] = [
        ("passive", PassiveDefender, CONFIG),
        ("static", StaticDefender, CONFIG),
        ("reactive, perfect sensors", ReactiveDefender, PERFECT_SENSORS),
    ]
    print(f"{'attacker':<17}{'defender':<27}{'wins':>6}{'turns':>8}{'sec/game':>10}")
    for dname, make_defender, config in defenders:
        for aname, make_attacker in attackers:
            row = play_matches(aname, make_attacker, dname, make_defender, games, config)
            print(
                f"{row.attacker:<17}{row.defender:<27}{row.win_rate:>6.0%}"
                f"{row.mean_turns:>8.1f}{row.seconds_per_game:>10.2f}"
            )


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)

