from __future__ import annotations

from core.actions import Action, Pass
from core.observation import DefenderObservation


class PassiveDefender:
    """Does nothing every turn. A useful yardstick for how fast an attacker can be."""

    name = "passive defender"

    def choose_action(self, observation: DefenderObservation) -> Action:
        return Pass()