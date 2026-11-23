import pytest

from config import CONFIG, BeliefConfig, SensorConfig
from core.network import Network
from prob.bayesnet import (
    dbn_conditional_given_blanket,
    exact_enumeration_posterior,
    exact_joint_probability,
    exact_single_evidence_posterior,
)
from prob.belief import BeliefTracker
from tests.factories import diamond_network


def _six_machine_network() -> Network:
    return Network.from_edges(
        6,
        [(0, 1), (0, 2), (1, 3), (2, 3), (1, 4), (2, 4), (3, 5), (4, 5)],
        hardness=[1, 1, 2, 2, 1, 3],
        entry=0,
        jewel=5,
    )


def test_tracker_matches_exact_enumeration_on_six_machine_map() -> None:
    net = _six_machine_network()
    rates = SensorConfig(detection_rate=0.725, false_alarm_rate=0.119)
    alert_history = [
        (False, True, False, False, False, False),
        (False, False, False, True, False, False),
        (False, False, False, True, False, True),
        (False, False, False, False, True, False),
    ]
    exact = exact_enumeration_posterior(net, alert_history, rates)

    # Zero floor: matches brute-force exact enumeration to machine precision
    tracker_exact = BeliefTracker(
        net, rates, config=BeliefConfig(stay_base=CONFIG.belief.stay_base, floor=0.0)
    )
    for alerts in alert_history:
        tracker_exact.predict()
        tracker_exact.update(alerts)
    assert tracker_exact.probabilities == pytest.approx(exact, abs=1e-12)

    # Production floor (1e-6): still matches exact enumeration within 1e-4
    tracker_floor = BeliefTracker(net, rates, config=CONFIG.belief)
    for alerts in alert_history:
        tracker_floor.predict()
        tracker_floor.update(alerts)
    assert tracker_floor.probabilities == pytest.approx(exact, abs=1e-4)


def test_tracker_matches_exact_enumeration_with_dynamic_patches_and_firewalls() -> None:
    net = _six_machine_network()
    rates = SensorConfig(detection_rate=0.725, false_alarm_rate=0.119)
    alert_history = [
        (False, True, False, False, False, False),
        (False, True, False, False, True, False),
        (False, False, False, False, True, True),
    ]
    patched_history = [frozenset(), frozenset({1}), frozenset({1, 4})]
    firewalls_history = [frozenset(), frozenset({(1, 3)}), frozenset({(1, 3), (2, 4)})]

    exact = exact_enumeration_posterior(
        net,
        alert_history,
        rates,
        patched_history=patched_history,
        firewalls_history=firewalls_history,
    )

    tracker = BeliefTracker(
        net, rates, config=BeliefConfig(stay_base=CONFIG.belief.stay_base, floor=0.0)
    )
    for alerts, patched, firewalls in zip(
        alert_history, patched_history, firewalls_history, strict=True
    ):
        tracker.predict(patched=patched, firewalls=firewalls)
        tracker.update(alerts)

    assert tracker.probabilities == pytest.approx(exact, abs=1e-12)


def test_exact_single_evidence_matches_playbook_worked_example() -> None:
    rates = SensorConfig(detection_rate=0.8, false_alarm_rate=0.1)
    post = exact_single_evidence_posterior((0.5, 0.5), {0: True}, rates)
    assert post == pytest.approx((8 / 9, 1 / 9), abs=1e-12)


def test_exact_joint_probability_sums_to_marginal_evidence() -> None:
    net = diamond_network()
    rates = SensorConfig(detection_rate=0.8, false_alarm_rate=0.1)
    alerts = [(False, True, False, False)]
    total = sum(
        exact_joint_probability(net, (x0, x1), alerts, rates)
        for x0 in range(net.size)
        for x1 in range(net.size)
    )
    assert total > 0.0
    for x1 in range(net.size):
        joint_x1 = sum(
            exact_joint_probability(net, (x0, x1), alerts, rates) for x0 in range(net.size)
        )
        post = exact_enumeration_posterior(net, alerts, rates)
        assert joint_x1 / total == pytest.approx(post[x1], abs=1e-12)


def test_dbn_markov_blanket_conditional_independence_holds_numerically() -> None:
    net = _six_machine_network()
    rates = SensorConfig(detection_rate=0.725, false_alarm_rate=0.119)
    alerts_t = (False, False, False, True, False, False)

    # P(X_t | MB(X_t)) where MB(X_t) = {X_{t-1}=1, X_{t+1}=5, A_t=alerts_t}
    base = dbn_conditional_given_blanket(
        net, target_step=2, prev_state=1, next_state=5, alerts_at_t=alerts_t, rates=rates
    )

    # Conditioning on outside-blanket variables X_{t-2}=0 and A_{t+1} must not change P(X_t)
    with_outside_a = dbn_conditional_given_blanket(
        net,
        target_step=2,
        prev_state=1,
        next_state=5,
        alerts_at_t=alerts_t,
        rates=rates,
        earlier_state=0,
        later_alerts=(True, True, False, False, False, True),
    )
    with_outside_b = dbn_conditional_given_blanket(
        net,
        target_step=2,
        prev_state=1,
        next_state=5,
        alerts_at_t=alerts_t,
        rates=rates,
        earlier_state=4,
        later_alerts=(False, False, False, False, False, False),
    )

    assert with_outside_a == pytest.approx(base, abs=1e-12)
    assert with_outside_b == pytest.approx(base, abs=1e-12)

