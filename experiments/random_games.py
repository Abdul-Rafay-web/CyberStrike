from __future__ import annotations

from dataclasses import dataclass

from agents.random_agents import RandomAttacker, RandomDefender
from config import CONFIG, AppConfig
from core.game import play_game
from core.generator import generate_network
from core.rules import Outcome
from utils.rng import STREAM_AGENTS, STREAM_DEFENDER, STREAM_SENSORS, make_rng


@dataclass(frozen=True, slots=True)
class RandomGamesSummary:
    games: int
    attacker_wins: int
    defender_wins: int
    mean_turns: float

    def report(self) -> str:
        rate = self.attacker_wins / self.games if self.games else 0.0
        return (
            f"{self.games} games | attacker won {self.attacker_wins} ({rate:.1%}) | "
            f"defender won {self.defender_wins} | average turns {self.mean_turns:.1f}"
        )


def run_random_games(
    count: int, size: int | None = None, config: AppConfig = CONFIG
) -> RandomGamesSummary:
    """Play random attacker against random defender on seeds 0 to count minus 1."""
    attacker_wins = 0
    total_turns = 0
    for seed in range(count):
        network = generate_network(seed, size, config)
        result = play_game(
            network,
            RandomAttacker(make_rng(seed, STREAM_AGENTS)),
            RandomDefender(make_rng(seed, STREAM_DEFENDER)),
            make_rng(seed, STREAM_SENSORS),
            config,
        )
        total_turns += result.turns
        if result.outcome is Outcome.ATTACKER_WINS:
            attacker_wins += 1
    return RandomGamesSummary(
        games=count,
        attacker_wins=attacker_wins,
        defender_wins=count - attacker_wins,
        mean_turns=total_turns / count if count else 0.0,
    )