from __future__ import annotations

from config import CONFIG, BeliefConfig, GameConfig, SensorConfig
from core.actions import Action, Firewall, Honeypot, Pass, Patch
from core.network import Edge, Network
from core.observation import DefenderObservation
from ml.pipeline import load_sensor, train_sensor
from prob.belief import BeliefTracker
from search.heuristics import UNREACHABLE
from search.problem import build_problem
from search.uninformed import uniform_cost
from utils.paths import DATA_DIR


def load_sensor_rates() -> SensorConfig:
    """Load Phase 5's measured sensor rates from disk, training once if not cached."""
    model_path = DATA_DIR / "sensor_model.json"
    if model_path.exists():
        return load_sensor(model_path).rates
    return train_sensor().rates


class BeliefSession:
    """UI-facing wrapper around `BeliefTracker` so UI code never imports `prob` directly."""

    def __init__(
        self,
        network: Network,
        rates: SensorConfig | None = None,
        game: GameConfig = CONFIG.game,
        belief_cfg: BeliefConfig = CONFIG.belief,
    ) -> None:
        resolved_rates = rates if rates is not None else CONFIG.sensors
        self._tracker = BeliefTracker(
            network,
            resolved_rates,
            game=game,
            config=belief_cfg,
        )

    @property
    def probabilities(self) -> tuple[float, ...]:
        return self._tracker.probabilities

    def top_k(self, k: int = 3) -> tuple[tuple[int, float], ...]:
        return self._tracker.top_k(k)

    def step(self, observation: DefenderObservation) -> tuple[float, ...]:
        return self._tracker.step(observation)

    def reset(self) -> None:
        self._tracker.reset()


def create_belief_session(
    network: Network, rates: SensorConfig | None = None
) -> BeliefSession:
    """Factory for a fresh live belief tracker session on `network`."""
    return BeliefSession(network, rates=rates)


class BeliefReactiveDefender:
    """Defender powered by `BeliefTracker` that blocks expected intruder routes under noise.

    Unlike `ReactiveDefender` (which passes whenever multiple alerts fire), this agent
    tracks the posterior belief over all machines and chooses the legal patch, firewall,
    or honeypot that maximizes belief-weighted delay toward the Crown Jewel.
    """

    name = "belief reactive defender"

    def __init__(
        self,
        rates: SensorConfig | None = None,
        game: GameConfig = CONFIG.game,
        belief_cfg: BeliefConfig = CONFIG.belief,
    ) -> None:
        self._rates = rates
        self._game = game
        self._belief_cfg = belief_cfg
        self._tracker: BeliefTracker | None = None
        self.last_belief: tuple[float, ...] = ()

    def _ensure_tracker(self, observation: DefenderObservation) -> BeliefTracker:
        if self._tracker is None or self._tracker.network is not observation.network:
            rates = self._rates if self._rates is not None else load_sensor_rates()
            self._tracker = BeliefTracker(
                observation.network,
                rates,
                game=self._game,
                config=self._belief_cfg,
            )
        return self._tracker

    def choose_action(self, observation: DefenderObservation) -> Action:
        tracker = self._ensure_tracker(observation)
        belief = tracker.step(observation)
        self.last_belief = belief

        suspects = tracker.top_k(3)
        base_expected = self._expected_cost(
            observation, suspects, observation.patched, observation.firewalls
        )
        best: Action = Pass()
        best_gain = 0.0

        for action in observation.legal_actions:
            if isinstance(action, Patch):
                after = self._expected_cost(
                    observation,
                    suspects,
                    observation.patched | {action.machine},
                    observation.firewalls,
                )
                gain = after - base_expected
            elif isinstance(action, Firewall):
                after = self._expected_cost(
                    observation,
                    suspects,
                    observation.patched,
                    observation.firewalls | {(action.a, action.b)},
                )
                gain = after - base_expected
            elif isinstance(action, Honeypot):
                # Reward placing a trap on a high-belief unpatched neighbour ahead of the intruder
                gain = belief[action.machine] * self._game.honeypot_delay_turns
            else:
                continue

            if gain > best_gain + 1e-9:
                best, best_gain = action, gain

        return best

    def _expected_cost(
        self,
        observation: DefenderObservation,
        suspects: tuple[tuple[int, float], ...],
        patched: frozenset[int],
        firewalls: frozenset[Edge],
    ) -> float:
        total_weight = sum(prob for _, prob in suspects)
        if total_weight <= 0.0:
            return 0.0
        expected = 0.0
        for machine, prob in suspects:
            problem = build_problem(
                observation.network,
                machine,
                None,
                0,
                frozenset({machine}),
                patched,
                firewalls,
                self._game,
            )
            result = uniform_cost(problem)
            cost = float(UNREACHABLE if result.cost is None else result.cost)
            expected += (prob / total_weight) * cost
        return expected

