from __future__ import annotations

import math
from collections.abc import Collection, Sequence

import numpy as np

from config import CONFIG, BeliefConfig, GameConfig, SensorConfig
from core.network import Edge, Network, make_edge
from core.observation import DefenderObservation


def _effective_hardness(
    network: Network, machine: int, patched: Collection[int], game: GameConfig
) -> int:
    base = network.hardness[machine]
    return base + game.patch_hardness_bonus if machine in patched else base


def transition_matrix(
    network: Network,
    patched: Collection[int] = (),
    firewalls: Collection[Edge] = (),
    game: GameConfig = CONFIG.game,
    belief_cfg: BeliefConfig = CONFIG.belief,
) -> np.ndarray:
    """Row-stochastic matrix T where T[i, j] = P(X_t = j | X_{t-1} = i).

    Harder (or patched) machines hold the attacker with higher stay probability,
    and outbound movement prefers open neighbours with lower effective hardness.
    Firewalled links have zero transition probability.
    """
    if not 0.0 <= belief_cfg.stay_base <= 1.0:
        raise ValueError("stay_base must lie in [0, 1]")
    size = network.size
    eff = [_effective_hardness(network, m, patched, game) for m in range(size)]
    matrix = np.zeros((size, size), dtype=np.float64)

    for src in range(size):
        open_nbrs = [
            nbr for nbr in network.neighbours(src) if make_edge(src, nbr) not in firewalls
        ]
        if not open_nbrs:
            matrix[src, src] = 1.0
            continue
        stay = min(1.0, max(0.0, (eff[src] - 1.0 + belief_cfg.stay_base) / eff[src]))
        matrix[src, src] = stay
        move_mass = 1.0 - stay
        weights = np.array([1.0 / eff[nbr] for nbr in open_nbrs], dtype=np.float64)
        weights /= weights.sum()
        for nbr, weight in zip(open_nbrs, weights, strict=True):
            matrix[src, nbr] = move_mass * float(weight)

    return matrix


def single_alert_likelihood(
    size: int, machine: int, alerted: bool, rates: SensorConfig
) -> np.ndarray:
    """Likelihood P(A_machine = alerted | X = i) for each hypothesis machine i."""
    if not 0 <= machine < size:
        raise ValueError(f"machine {machine} out of range 0..{size - 1}")
    d = rates.detection_rate
    f = rates.false_alarm_rate
    p_hit = d if alerted else (1.0 - d)
    p_other = f if alerted else (1.0 - f)
    likelihood = np.full(size, p_other, dtype=np.float64)
    likelihood[machine] = p_hit
    return likelihood


def observation_log_likelihood(alerts: Sequence[bool], rates: SensorConfig) -> np.ndarray:
    """Log-likelihood log P(A_0..A_{N-1} = alerts | X = i) for each machine i.

    Handles exact 0.0 and 1.0 sensor rates cleanly (returning -inf for impossible hypotheses)
    and stays numerically stable for arbitrarily large networks.
    """
    size = len(alerts)
    d = rates.detection_rate
    f = rates.false_alarm_rate
    if not (0.0 <= d <= 1.0 and 0.0 <= f <= 1.0):
        raise ValueError("sensor rates must lie in [0, 1]")

    flags = [bool(flag) for flag in alerts]
    n_true = sum(flags)
    n_false = size - n_true

    log_d = math.log(d) if d > 0.0 else -math.inf
    log_1_d = math.log(1.0 - d) if d < 1.0 else -math.inf
    log_f = math.log(f) if f > 0.0 else -math.inf
    log_1_f = math.log(1.0 - f) if f < 1.0 else -math.inf

    out = np.empty(size, dtype=np.float64)
    for i, fired in enumerate(flags):
        if fired:
            p_self = log_d
            other_true = n_true - 1
            other_false = n_false
        else:
            p_self = log_1_d
            other_true = n_true
            other_false = n_false - 1

        if math.isinf(p_self):
            out[i] = -math.inf
            continue
        if other_true > 0 and math.isinf(log_f):
            out[i] = -math.inf
            continue
        if other_false > 0 and math.isinf(log_1_f):
            out[i] = -math.inf
            continue

        total = p_self
        if other_true > 0:
            total += other_true * log_f
        if other_false > 0:
            total += other_false * log_1_f
        out[i] = total
    return out


def observation_likelihood(alerts: Sequence[bool], rates: SensorConfig) -> np.ndarray:
    """Exact joint likelihood P(A_0..A_{N-1} = alerts | X = i) for every machine i."""
    log_lik = observation_log_likelihood(alerts, rates)
    return np.exp(log_lik)


class BeliefTracker:
    """Forward Bayesian belief filter over the intruder's active machine.

    Maintains a probability vector b_t(i) = P(X_t = i | A_{1:t}) across all N machines.
    Each turn performs a Markov prediction step followed by a Bayesian alert update,
    with an exact certainty override when a honeypot triggers and a tiny probability floor
    to prevent permanent zero lock-out under noisy sensors.
    """

    def __init__(
        self,
        network: Network,
        rates: SensorConfig = CONFIG.sensors,
        *,
        game: GameConfig = CONFIG.game,
        config: BeliefConfig = CONFIG.belief,
        initial: Sequence[float] | None = None,
    ) -> None:
        if config.floor < 0.0:
            raise ValueError("belief floor cannot be negative")
        if config.floor * network.size >= 1.0:
            raise ValueError("belief floor is too large for the network size")
        self.network = network
        self.rates = rates
        self.game = game
        self.config = config
        self._belief = np.zeros(network.size, dtype=np.float64)
        self.reset(initial)

    def reset(self, initial: Sequence[float] | None = None) -> None:
        """Reset belief to certainty at `network.entry` or to a supplied distribution."""
        if initial is None:
            self._belief = np.zeros(self.network.size, dtype=np.float64)
            self._belief[self.network.entry] = 1.0
            return
        arr = np.asarray(initial, dtype=np.float64)
        if arr.ndim != 1 or len(arr) != self.network.size:
            raise ValueError(f"initial belief must have length {self.network.size}")
        if (arr < 0.0).any() or not np.isfinite(arr).all():
            raise ValueError("initial belief values must be finite and non-negative")
        total = float(arr.sum())
        if total <= 0.0:
            raise ValueError("initial belief must have positive sum")
        self._belief = arr / total

    def copy(self) -> BeliefTracker:
        return BeliefTracker(
            self.network,
            self.rates,
            game=self.game,
            config=self.config,
            initial=self.probabilities,
        )

    @property
    def probabilities(self) -> tuple[float, ...]:
        return tuple(float(p) for p in self._belief)

    @property
    def array(self) -> np.ndarray:
        return self._belief.copy()

    @property
    def most_likely(self) -> int:
        return int(np.argmax(self._belief))

    @property
    def entropy(self) -> float:
        """Shannon entropy of the current belief in bits."""
        positive = self._belief[self._belief > 0.0]
        return float(-np.sum(positive * np.log2(positive)))

    def prob_at(self, machine: int) -> float:
        return float(self._belief[machine])

    def top_k(self, k: int = 3) -> tuple[tuple[int, float], ...]:
        """Return the top-k (machine, probability) pairs sorted descending by belief."""
        if k <= 0:
            return ()
        ranked = sorted(
            range(self.network.size),
            key=lambda m: (-float(self._belief[m]), m),
        )
        return tuple((m, float(self._belief[m])) for m in ranked[:k])

    def _apply_floor_and_normalize(self, raw: np.ndarray) -> np.ndarray:
        total = float(raw.sum())
        if total <= 0.0 or not math.isfinite(total):
            # Fallback safely to the previous belief if an impossible observation occurred
            normalized = self._belief.copy()
        else:
            normalized = raw / total
        if self.config.floor > 0.0:
            normalized = np.maximum(normalized, self.config.floor)
            normalized /= float(normalized.sum())
        return normalized

    def predict(
        self,
        patched: Collection[int] = (),
        firewalls: Collection[Edge] = (),
    ) -> tuple[float, ...]:
        """Advance the belief by one attacker movement step: b'_t = b_{t-1} @ T."""
        trans = transition_matrix(self.network, patched, firewalls, self.game, self.config)
        predicted = self._belief @ trans
        self._belief = self._apply_floor_and_normalize(predicted)
        return self.probabilities

    def update(
        self,
        alerts: Sequence[bool],
        intruder_known_at: int | None = None,
    ) -> tuple[float, ...]:
        """Incorporate a full alert vector (or exact honeypot trigger) via Bayes' rule."""
        if intruder_known_at is not None:
            if not 0 <= intruder_known_at < self.network.size:
                raise ValueError(
                    f"intruder_known_at {intruder_known_at} out of range 0..{self.network.size - 1}"
                )
            self._belief = np.zeros(self.network.size, dtype=np.float64)
            self._belief[intruder_known_at] = 1.0
            return self.probabilities

        if len(alerts) != self.network.size:
            raise ValueError(f"expected {self.network.size} alert flags, got {len(alerts)}")

        log_lik = observation_log_likelihood(alerts, self.rates)
        with np.errstate(divide="ignore"):
            log_prior = np.where(self._belief > 0.0, np.log(self._belief), -math.inf)
        log_post = log_prior + log_lik
        finite = np.isfinite(log_post)
        if not finite.any():
            # All hypotheses had zero probability; retain prior safely with floor
            self._belief = self._apply_floor_and_normalize(self._belief)
            return self.probabilities

        peak = float(log_post[finite].max())
        weights = np.where(finite, np.exp(log_post - peak), 0.0)
        self._belief = self._apply_floor_and_normalize(weights)
        return self.probabilities

    def update_single(self, machine: int, alerted: bool) -> tuple[float, ...]:
        """Bayes update conditioning on a single machine's alert flag."""
        lik = single_alert_likelihood(self.network.size, machine, alerted, self.rates)
        self._belief = self._apply_floor_and_normalize(self._belief * lik)
        return self.probabilities

    def step(self, observation: DefenderObservation) -> tuple[float, ...]:
        """Run one full predict-then-correct cycle for a Defender turn."""
        if observation.intruder_known_at is not None:
            return self.update(observation.alerts, intruder_known_at=observation.intruder_known_at)
        self.predict(observation.patched, observation.firewalls)
        return self.update(observation.alerts)
