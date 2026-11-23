import numpy as np

from agents.greedy_attacker import GreedyAttacker
from agents.minimax_attacker import MinimaxAttacker
from agents.simple_defenders import ReactiveDefender, StaticDefender
from config import CONFIG, SensorConfig
from core.actions import Advance
from core.game import play_game
from core.generator import generate_network
from core.rules import apply
from core.sensors import RateSensor, sample_alerts
from core.state import new_game
from ml.pipeline import train_sensor
from ml.sensor import ClassifierSensor
from tests.factories import chain_network, diamond_network
from utils.rng import STREAM_SENSORS, make_rng


def test_rate_sensor_matches_the_plain_function() -> None:
    state = new_game(diamond_network())
    config = SensorConfig(0.7, 0.2)
    assert RateSensor(config).alerts(state, make_rng(1, STREAM_SENSORS)) == sample_alerts(
        state, make_rng(1, STREAM_SENSORS), config
    )


def test_game_with_an_explicit_rate_sensor_matches_the_default() -> None:
    network = generate_network(3)
    plain = play_game(network, GreedyAttacker(), StaticDefender(), make_rng(3, STREAM_SENSORS))
    explicit = play_game(
        network,
        GreedyAttacker(),
        StaticDefender(),
        make_rng(3, STREAM_SENSORS),
        sensor=RateSensor(CONFIG.sensors),
    )
    assert plain == explicit


def test_classifier_sensor_gives_one_flag_per_machine() -> None:
    sensor = ClassifierSensor.from_trained(train_sensor())
    state = new_game(generate_network(1))
    assert len(sensor.alerts(state, make_rng(1, STREAM_SENSORS))) == state.network.size
    assert len(sensor.last_probabilities) == state.network.size
    assert all(0.0 <= p <= 1.0 for p in sensor.last_probabilities)


def test_classifier_sensor_is_repeatable() -> None:
    sensor = ClassifierSensor.from_trained(train_sensor())
    state = new_game(diamond_network())
    assert sensor.alerts(state, make_rng(4, STREAM_SENSORS)) == sensor.alerts(
        state, make_rng(4, STREAM_SENSORS)
    )


def test_long_run_rates_match_what_was_measured_on_the_test_set() -> None:
    trained = train_sensor()
    sensor = ClassifierSensor.from_trained(trained)
    state = new_game(diamond_network())
    rng = make_rng(5, STREAM_SENSORS)
    draws = np.array([sensor.alerts(state, rng) for _ in range(3000)])
    detection = draws[:, 0].mean()
    false_alarm = draws[:, 1:].mean()
    assert abs(detection - trained.rates.detection_rate) < 0.06
    assert abs(false_alarm - trained.rates.false_alarm_rate) < 0.04


def test_alerts_follow_the_machine_being_breached() -> None:
    trained = train_sensor()
    sensor = ClassifierSensor.from_trained(trained)
    mid_breach = apply(new_game(chain_network()), Advance(1))
    rng = make_rng(6, STREAM_SENSORS)
    draws = np.array([sensor.alerts(mid_breach, rng) for _ in range(2000)])
    assert draws[:, 1].mean() > draws[:, 0].mean() + 0.3
    assert draws[:, 1].mean() > draws[:, 2].mean() + 0.3


def test_full_games_run_with_the_classifier_sensor() -> None:
    sensor = ClassifierSensor.from_trained(train_sensor())
    for seed in range(5):
        result = play_game(
            generate_network(seed),
            MinimaxAttacker(max_depth=3, time_budget=1.0),
            ReactiveDefender(),
            make_rng(seed, STREAM_SENSORS),
            CONFIG,
            sensor=sensor,
        )
        assert result.turns >= 1

