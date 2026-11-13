from __future__ import annotations

from typing import Protocol

from core.actions import Action
from core.observation import AttackerObservation, DefenderObservation


class AttackerAgent(Protocol):
    name: str

    def choose_action(self, observation: AttackerObservation) -> Action: ...


class DefenderAgent(Protocol):
    name: str

    def choose_action(self, observation: DefenderObservation) -> Action: ...