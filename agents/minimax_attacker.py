from __future__ import annotations

from config import CONFIG, GameConfig, SearchConfig
from core.actions import Action
from core.observation import AttackerObservation
from search.evaluation import EvalWeights
from search.minimax import SearchReport, iterative_search
from search.model import state_from_observation


class MinimaxAttacker:
    """Looks ahead several moves, assuming the defender always replies as well as it can."""

    name = "minimax attacker"

    def __init__(
        self,
        max_depth: int = CONFIG.search.max_depth_plies,
        time_budget: float = CONFIG.search.time_budget_seconds,
        pruning: bool = True,
        weights: EvalWeights | None = None,
        game: GameConfig = CONFIG.game,
        search: SearchConfig = CONFIG.search,
    ) -> None:
        self._max_depth = max_depth
        self._time_budget = time_budget
        self._pruning = pruning
        self._weights = weights
        self._game = game
        self._search = search
        self.last_report: SearchReport | None = None

    def choose_action(self, observation: AttackerObservation) -> Action:
        options = observation.legal_actions
        if len(options) == 1:
            return options[0]
        state = state_from_observation(observation, self._game)
        report = iterative_search(
            state,
            self._max_depth,
            self._time_budget,
            self._pruning,
            self._game,
            self._search,
            self._weights,
        )
        self.last_report = report
        if report.action not in options:
            raise RuntimeError("the search proposed a move that is not legal")
        return report.action

