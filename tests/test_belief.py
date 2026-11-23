import numpy as np
import pytest

from agents.belief_defender import BeliefReactiveDefender
from agents.greedy_attacker import GreedyAttacker
from agents.simple_defenders import StaticDefender
from config import CONFIG, BeliefConfig, SensorConfig
from core.actions import Action
from core.game import play_game
from core.generator import generate_network
from core.network import Network
from core.observation import observe_defender
from core.sensors import active_machine
from core.state import GameState, Side
from ml.pipeline import train_sensor
from ml.sensor import ClassifierSensor
from prob.belief import BeliefTracker, observation_likelihood, transition_matrix
from tests.factories import chain_network, diamond_network
from utils.rng import STREAM_SENSORS, make_rng


def _two_machine_network() -> Network:
    return Network.from_edges(2, [(0, 1)], hardness=[1, 1], entry=0, jewel=1)


def test_hand_worked_single_alert_example_matches_to_four_decimals() -> None:
    # Playbook Page 16 worked example:
    # 2 machines, prior 0.5 each, d = 0.8, f = 0.1, alert fires on machine 0:
    # P(A | alert on A) = (0.5 * 0.8) / (0.5 * 0.8 + 0.5 * 0.1) = 0.4 / 0.45 = 0.888888...
    rates = SensorConfig(detection_rate=0.8, false_alarm_rate=0.1)
    tracker = BeliefTracker(
        _two_machine_network(),
        rates,
        config=BeliefConfig(stay_base=0.25, floor=0.0),
        initial=(0.5, 0.5),
    )
    posterior = tracker.update_single(0, alerted=True)
    assert posterior[0] == pytest.approx(8 / 9, abs=1e-4)
    assert posterior[1] == pytest.approx(1 / 9, abs=1e-4)
    assert sum(posterior) == pytest.approx(1.0)


def test_hand_worked_full_alert_vector_matches_to_four_decimals() -> None:
    # Observing BOTH (A alerts, B quiet) with d=0.8, f=0.1:
    # L(A) = 0.8 * 0.9 = 0.72, L(B) = 0.1 * 0.2 = 0.02 -> P(A) = 72 / 74 = 0.972973
    rates = SensorConfig(detection_rate=0.8, false_alarm_rate=0.1)
    tracker = BeliefTracker(
        _two_machine_network(),
        rates,
        config=BeliefConfig(stay_base=0.25, floor=0.0),
        initial=(0.5, 0.5),
    )
    posterior = tracker.update((True, False))
    assert posterior[0] == pytest.approx(72 / 74, abs=1e-4)
    assert posterior[1] == pytest.approx(2 / 74, abs=1e-4)
    assert sum(posterior) == pytest.approx(1.0)


def test_transition_matrix_is_row_stochastic_and_respects_hardness_and_firewalls() -> None:
    net = diamond_network()
    trans = transition_matrix(net)
    assert trans.shape == (4, 4)
    assert np.allclose(trans.sum(axis=1), 1.0)
    assert (trans >= 0.0).all()

    # From entry (0), machine 1 (hardness 1) is softer than machine 2 (hardness 2)
    assert trans[0, 1] > trans[0, 2]
    # Non-neighbour 3 has zero direct transition from 0
    assert trans[0, 3] == 0.0

    # Patching machine 1 increases stay probability on 1 and lowers inbound preference from 0
    trans_patched = transition_matrix(net, patched={1})
    assert trans_patched[1, 1] > trans[1, 1]
    assert trans_patched[0, 1] < trans[0, 1]

    # Firewalling link (0, 1) sets transition between 0 and 1 to zero
    trans_fw = transition_matrix(net, firewalls={(0, 1)})
    assert trans_fw[0, 1] == 0.0
    assert trans_fw[1, 0] == 0.0
    assert np.allclose(trans_fw.sum(axis=1), 1.0)


def test_belief_starts_at_entry_always_sums_to_one_and_keeps_floor() -> None:
    net = generate_network(3)
    rates = SensorConfig(detection_rate=0.725, false_alarm_rate=0.119)
    tracker = BeliefTracker(net, rates)
    assert tracker.prob_at(net.entry) == pytest.approx(1.0)
    assert sum(tracker.probabilities) == pytest.approx(1.0)

    # Even after repeated silence or contradictory alerts, belief stays normalized and > 0
    for _ in range(8):
        tracker.predict()
        probs = tracker.update((False,) * net.size)
        assert sum(probs) == pytest.approx(1.0)
        assert min(probs) >= CONFIG.belief.floor * 0.99


def test_floor_allows_recovery_from_false_alarms() -> None:
    net = chain_network()
    rates = SensorConfig(detection_rate=0.95, false_alarm_rate=0.05)
    tracker = BeliefTracker(net, rates, initial=(0.0, 0.0, 1.0))
    # Force floor via predict then strong evidence on machine 0
    tracker.predict()
    for _ in range(4):
        tracker.update((True, False, False))
    assert tracker.most_likely == 0
    assert tracker.prob_at(0) > 0.8


def test_honeypot_trigger_sets_exact_certainty() -> None:
    net = diamond_network()
    tracker = BeliefTracker(net, SensorConfig(0.725, 0.119))
    tracker.predict()
    probs = tracker.update((False, False, False, False), intruder_known_at=2)
    assert probs[2] == pytest.approx(1.0)
    assert probs[0] == 0.0
    assert tracker.most_likely == 2
    assert tracker.entropy == pytest.approx(0.0)


def test_consistent_alerts_reduce_entropy_and_rank_top_k() -> None:
    net = generate_network(7)
    tracker = BeliefTracker(net, SensorConfig(0.725, 0.119))
    tracker.predict()
    initial_entropy = tracker.entropy

    target = net.neighbours(net.entry)[0]
    alerts = tuple(i == target for i in range(net.size))
    tracker.update(alerts)
    assert tracker.entropy < initial_entropy
    top3 = tracker.top_k(3)
    assert len(top3) == 3
    assert top3[0][0] == target
    assert top3[0][1] >= top3[1][1] >= top3[2][1]


def test_observation_likelihood_handles_perfect_sensors_cleanly() -> None:
    perfect = SensorConfig(detection_rate=1.0, false_alarm_rate=0.0)
    lik = observation_likelihood((False, True, False), perfect)
    assert lik.tolist() == [0.0, 1.0, 0.0]


def _collect_game_belief_probs(
    seed: int,
    sensor: ClassifierSensor,
    rates: SensorConfig,
    belief_probs: list[float],
    uniform_probs: list[float],
) -> None:
    net = generate_network(seed)
    tracker = BeliefTracker(net, rates)
    rng = make_rng(seed, STREAM_SENSORS)

    def on_step(before: GameState, action: Action, after: GameState) -> None:
        del before, action
        if after.side is Side.DEFENDER:
            alerts = sensor.alerts(after, rng)
            obs = observe_defender(after, alerts, CONFIG.game)
            b = tracker.step(obs)
            belief_probs.append(b[active_machine(after)])
            uniform_probs.append(1.0 / net.size)

    play_game(
        net,
        GreedyAttacker(),
        StaticDefender(),
        make_rng(seed + 500, STREAM_SENSORS),
        CONFIG,
        on_step=on_step,
        sensor=sensor,
    )


def test_belief_tracker_beats_uniform_baseline_over_seeded_games() -> None:
    trained = train_sensor()
    sensor = ClassifierSensor.from_trained(trained)
    belief_probs: list[float] = []
    uniform_probs: list[float] = []

    for seed in range(25):
        _collect_game_belief_probs(seed, sensor, trained.rates, belief_probs, uniform_probs)

    assert float(np.mean(belief_probs)) > float(np.mean(uniform_probs)) * 3.0


def test_belief_reactive_defender_plays_legal_moves_and_tracks_belief() -> None:
    trained = train_sensor()
    sensor = ClassifierSensor.from_trained(trained)
    defender = BeliefReactiveDefender(rates=trained.rates)
    for seed in range(5):
        net = generate_network(seed)
        result = play_game(
            net,
            GreedyAttacker(),
            defender,
            make_rng(seed, STREAM_SENSORS),
            CONFIG,
            sensor=sensor,
        )
        assert result.turns >= 1
        assert len(defender.last_belief) == net.size
        assert sum(defender.last_belief) == pytest.approx(1.0)


def test_belief_tracker_rejects_invalid_inputs() -> None:
    net = diamond_network()
    with pytest.raises(ValueError):
        BeliefTracker(net, config=BeliefConfig(floor=-0.1))
    with pytest.raises(ValueError):
        BeliefTracker(net, initial=(0.5, 0.5))
    tracker = BeliefTracker(net)
    with pytest.raises(ValueError):
        tracker.update((True, False))
    with pytest.raises(ValueError):
        tracker.update((False, False, False, False), intruder_known_at=99)
