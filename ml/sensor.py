from __future__ import annotations

import numpy as np

from core.sensors import active_machine
from core.state import GameState
from ml.naive_bayes import NaiveBayes
from ml.pipeline import TrainedSensor
from ml.synthetic_logs import DEFAULT_TRAFFIC, TrafficModel


class ClassifierSensor:
    """Every machine produces a log record each turn. A trained classifier decides which to flag.

    Harmless machines produce normal traffic. The machine the attacker is working on
    produces attack traffic. The alert is the classifier's opinion, so it can be wrong.
    """

    def __init__(
        self,
        model: NaiveBayes,
        threshold: float,
        traffic: TrafficModel = DEFAULT_TRAFFIC,
    ) -> None:
        self._model = model
        self._threshold = threshold
        self._traffic = traffic
        self.last_probabilities: tuple[float, ...] = ()

    @classmethod
    def from_trained(
        cls, trained: TrainedSensor, traffic: TrafficModel = DEFAULT_TRAFFIC
    ) -> ClassifierSensor:
        return cls(trained.model, trained.threshold, traffic)

    def alerts(self, state: GameState, rng: np.random.Generator) -> tuple[bool, ...]:
        size = state.network.size
        records = self._traffic.sample(0, size, rng)
        records[active_machine(state)] = self._traffic.sample(1, 1, rng)[0]
        attack_probability = self._model.predict_proba(records)[:, 1]
        self.last_probabilities = tuple(float(p) for p in attack_probability)
        return tuple(bool(flag) for flag in attack_probability >= self._threshold)

