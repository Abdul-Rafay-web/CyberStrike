from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from config import CONFIG, SensorConfig
from core.state import GameState


class Sensor(Protocol):
    """Anything that turns the hidden game state into one noisy alert flag per machine."""

    def alerts(self, state: GameState, rng: np.random.Generator) -> tuple[bool, ...]: ...


def active_machine(state: GameState) -> int:
    """The machine where the attacker is busy right now."""
    return state.attacker_pos if state.breach_target is None else state.breach_target


def sample_alerts(
    state: GameState, rng: np.random.Generator, config: SensorConfig = CONFIG.sensors
) -> tuple[bool, ...]:
    """One noisy alert flag per machine. The attacker's machine alerts most often."""
    size = state.network.size
    probability = np.full(size, config.false_alarm_rate)
    probability[active_machine(state)] = config.detection_rate
    fired = rng.random(size) < probability
    return tuple(bool(flag) for flag in fired)


@dataclass(frozen=True, slots=True)
class RateSensor:
    """The simple sensor: fixed detection and false alarm chances, no log records involved."""

    config: SensorConfig = CONFIG.sensors

    def alerts(self, state: GameState, rng: np.random.Generator) -> tuple[bool, ...]:
        return sample_alerts(state, rng, self.config)
