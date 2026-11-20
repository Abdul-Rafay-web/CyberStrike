from __future__ import annotations

from config import CONFIG, GameConfig
from core.actions import Action, Advance
from core.observation import AttackerObservation
from search.astar import astar
from search.problem import problem_from_observation


class GreedyAttacker:
    """Plans the cheapest route with A star every turn and takes its first step."""

    name = "greedy A star attacker"

    def __init__(self, config: GameConfig = CONFIG.game) -> None:
        self._config = config

    def choose_action(self, observation: AttackerObservation) -> Action:
        options = observation.legal_actions
        if len(options) == 1:
            return options[0]
        result = astar(problem_from_observation(observation, self._config))
        if not result.found or len(result.path) < 2:
            return options[0]
        step = Advance(result.path[1])
        if step not in options:
            raise RuntimeError("the planner proposed a move that is not legal")
        return step