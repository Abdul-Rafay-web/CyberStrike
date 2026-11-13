from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from agents.base import AttackerAgent, DefenderAgent
from config import CONFIG, AppConfig
from core.actions import Action
from core.network import Network
from core.observation import observe_attacker, observe_defender
from core.rules import Outcome, apply, is_terminal, outcome, utility
from core.sensors import sample_alerts
from core.state import GameState, Side, new_game

StepCallback = Callable[[GameState, Action, GameState], None]


@dataclass(frozen=True, slots=True)
class GameResult:
    outcome: Outcome
    turns: int
    plies: int
    utility: int


def play_game(
    network: Network,
    attacker: AttackerAgent,
    defender: DefenderAgent,
    sensor_rng: np.random.Generator,
    config: AppConfig = CONFIG,
    on_step: StepCallback | None = None,
) -> GameResult:
    """Play one full game and return the result. Players only ever see their observation."""
    game = config.game
    state = new_game(network, game)
    plies = 0
    max_plies = 2 * game.turn_limit + 2

    while not is_terminal(state, game):
        if plies > max_plies:
            raise RuntimeError("the game did not finish, so the rules have a bug")
        if state.side is Side.ATTACKER:
            action = attacker.choose_action(observe_attacker(state, game))
        else:
            alerts = sample_alerts(state, sensor_rng, config.sensors)
            action = defender.choose_action(observe_defender(state, alerts, game))
        following = apply(state, action, game)
        if on_step is not None:
            on_step(state, action, following)
        state = following
        plies += 1

    return GameResult(
        outcome=outcome(state, game),
        turns=min(state.turn, game.turn_limit),
        plies=plies,
        utility=utility(state, game),
    )