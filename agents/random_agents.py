from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from core.actions import Action
from core.observation import AttackerObservation, DefenderObservation


def _pick(rng: np.random.Generator, options: Sequence[Action]) -> Action:
    if not options:
        raise ValueError("there are no legal actions to choose from")
    return options[int(rng.integers(0, len(options)))]


class RandomAttacker:
    name = "random attacker"

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng

    def choose_action(self, observation: AttackerObservation) -> Action:
        return _pick(self._rng, observation.legal_actions)


class RandomDefender:
    name = "random defender"

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng

    def choose_action(self, observation: DefenderObservation) -> Action:
        return _pick(self._rng, observation.legal_actions)