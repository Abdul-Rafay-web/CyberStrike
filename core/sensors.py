from __future__ import annotations

import numpy as np

from config import CONFIG, SensorConfig
from core.state import GameState


def sample_alerts(
    state: GameState, rng: np.random.Generator, config: SensorConfig = CONFIG.sensors
) -> tuple[bool, ...]:
    """One noisy alert flag per machine. The attacker's machine alerts most often."""
    size = state.network.size
    active = state.attacker_pos if state.breach_target is None else state.breach_target
    probability = np.full(size, config.false_alarm_rate)
    probability[active] = config.detection_rate
    fired = rng.random(size) < probability
    return tuple(bool(flag) for flag in fired)