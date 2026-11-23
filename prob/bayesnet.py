from __future__ import annotations

import itertools
from collections.abc import Collection, Sequence

import numpy as np

from config import CONFIG, BeliefConfig, GameConfig, SensorConfig
from core.network import Edge, Network
from prob.belief import observation_likelihood, single_alert_likelihood, transition_matrix


def _initial_distribution(network: Network, initial: Sequence[float] | None) -> np.ndarray:
    if initial is None:
        dist = np.zeros(network.size, dtype=np.float64)
        dist[network.entry] = 1.0
        return dist
    arr = np.asarray(initial, dtype=np.float64)
    if arr.ndim != 1 or len(arr) != network.size:
        raise ValueError(f"initial distribution must have length {network.size}")
    total = float(arr.sum())
    if total <= 0.0 or (arr < 0.0).any():
        raise ValueError("initial distribution must be non-negative with positive sum")
    return arr / total


def exact_joint_probability(
    network: Network,
    trajectory: Sequence[int],
    alert_history: Sequence[Sequence[bool]],
    rates: SensorConfig,
    *,
    initial: Sequence[float] | None = None,
    patched_history: Sequence[Collection[int]] | None = None,
    firewalls_history: Sequence[Collection[Edge]] | None = None,
    game: GameConfig = CONFIG.game,
    belief_cfg: BeliefConfig = CONFIG.belief,
) -> float:
    """Exact joint probability P(X_0..X_T = trajectory, A_1..A_T = alert_history)."""
    steps = len(alert_history)
    if len(trajectory) != steps + 1:
        raise ValueError(f"trajectory length must be {steps + 1}, got {len(trajectory)}")
    init = _initial_distribution(network, initial)
    prob = float(init[trajectory[0]])
    if prob == 0.0:
        return 0.0

    for t in range(steps):
        patched = () if patched_history is None else patched_history[t]
        firewalls = () if firewalls_history is None else firewalls_history[t]
        trans = transition_matrix(network, patched, firewalls, game, belief_cfg)
        lik = observation_likelihood(alert_history[t], rates)
        prob *= float(trans[trajectory[t], trajectory[t + 1]]) * float(lik[trajectory[t + 1]])
        if prob == 0.0:
            return 0.0
    return prob


def exact_enumeration_posterior(
    network: Network,
    alert_history: Sequence[Sequence[bool]],
    rates: SensorConfig,
    *,
    initial: Sequence[float] | None = None,
    patched_history: Sequence[Collection[int]] | None = None,
    firewalls_history: Sequence[Collection[Edge]] | None = None,
    game: GameConfig = CONFIG.game,
    belief_cfg: BeliefConfig = CONFIG.belief,
) -> tuple[float, ...]:
    """Brute-force exact marginal P(X_T | A_1..A_T) by enumerating all N^(T+1) hidden paths.

    Used on small maps (e.g., 6 machines, up to 4-5 turns) to prove that `BeliefTracker`
    computes the exact posterior of the Dynamic Bayesian Network.
    """
    steps = len(alert_history)
    size = network.size
    init = _initial_distribution(network, initial)

    # Precompute per-step local CPTs (transition matrix & sensor likelihoods) once
    transitions = [
        transition_matrix(
            network,
            () if patched_history is None else patched_history[t],
            () if firewalls_history is None else firewalls_history[t],
            game,
            belief_cfg,
        )
        for t in range(steps)
    ]
    likelihoods = [observation_likelihood(alerts, rates) for alerts in alert_history]

    marginal = np.zeros(size, dtype=np.float64)
    for trajectory in itertools.product(range(size), repeat=steps + 1):
        weight = float(init[trajectory[0]])
        if weight == 0.0:
            continue
        for t in range(steps):
            weight *= float(transitions[t][trajectory[t], trajectory[t + 1]])
            if weight == 0.0:
                break
            weight *= float(likelihoods[t][trajectory[t + 1]])
            if weight == 0.0:
                break
        marginal[trajectory[-1]] += weight

    total = float(marginal.sum())
    if total <= 0.0:
        raise ValueError("evidence has zero total probability under exact enumeration")
    marginal /= total
    return tuple(float(p) for p in marginal)


def exact_single_evidence_posterior(
    prior: Sequence[float],
    evidence: dict[int, bool],
    rates: SensorConfig,
) -> tuple[float, ...]:
    """Exact Bayes posterior P(X | {A_m = flag}) for any subset of observed machine alerts."""
    size = len(prior)
    posterior = np.asarray(prior, dtype=np.float64).copy()
    if posterior.ndim != 1 or size == 0:
        raise ValueError("prior must be a non-empty 1D sequence")
    for machine, alerted in evidence.items():
        posterior *= single_alert_likelihood(size, machine, alerted, rates)
    total = float(posterior.sum())
    if total <= 0.0:
        raise ValueError("evidence has zero total probability")
    posterior /= total
    return tuple(float(p) for p in posterior)


def dbn_conditional_given_blanket(
    network: Network,
    target_step: int,
    prev_state: int,
    next_state: int,
    alerts_at_t: Sequence[bool],
    rates: SensorConfig,
    *,
    earlier_state: int | None = None,
    later_alerts: Sequence[bool] | None = None,
    game: GameConfig = CONFIG.game,
    belief_cfg: BeliefConfig = CONFIG.belief,
) -> tuple[float, ...]:
    """Compute P(X_t | X_{t-1}, X_{t+1}, A_t, [optional outside-blanket evidence]) exactly.

    By the Markov blanket theorem in the unrolled DBN, `MB(X_t) = {X_{t-1}, X_{t+1}, A_t}`.
    Conditioning on additional variables outside `MB(X_t)` (such as `X_{t-2}` or `A_{t+1}`)
    must leave the conditional distribution of `X_t` completely unchanged.
    """
    if target_step < 1:
        raise ValueError("target_step must be at least 1")
    size = network.size
    trans = transition_matrix(network, (), (), game, belief_cfg)
    lik_t = observation_likelihood(alerts_at_t, rates)
    lik_next = (
        np.ones(size, dtype=np.float64)
        if later_alerts is None
        else observation_likelihood(later_alerts, rates)
    )
    prior_prev = (
        1.0 if earlier_state is None else float(trans[earlier_state, prev_state])
    )
    if prior_prev == 0.0:
        raise ValueError("conditioning event has zero probability")

    unnormalized = np.zeros(size, dtype=np.float64)
    for x_t in range(size):
        unnormalized[x_t] = (
            prior_prev
            * float(trans[prev_state, x_t])
            * float(lik_t[x_t])
            * float(trans[x_t, next_state])
            * float(lik_next[next_state])
        )
    total = float(unnormalized.sum())
    if total <= 0.0:
        raise ValueError("conditioning event has zero total probability")
    unnormalized /= total
    return tuple(float(p) for p in unnormalized)

