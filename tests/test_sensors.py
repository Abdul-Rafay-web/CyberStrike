from dataclasses import replace

from config import SensorConfig
from core.actions import Advance
from core.rules import apply
from core.sensors import sample_alerts
from core.state import new_game
from tests.factories import chain_network, diamond_network
from utils.rng import STREAM_SENSORS, make_rng


def test_perfect_sensors_flag_only_the_attacker_machine() -> None:
    perfect = SensorConfig(detection_rate=1.0, false_alarm_rate=0.0)
    state = new_game(diamond_network())
    assert sample_alerts(state, make_rng(1, STREAM_SENSORS), perfect) == (True, False, False, False)


def test_alerts_follow_the_machine_being_breached() -> None:
    perfect = SensorConfig(detection_rate=1.0, false_alarm_rate=0.0)
    mid_breach = apply(new_game(chain_network()), Advance(1))
    assert sample_alerts(mid_breach, make_rng(1, STREAM_SENSORS), perfect) == (False, True, False)


def test_silent_sensors_never_fire() -> None:
    silent = SensorConfig(detection_rate=0.0, false_alarm_rate=0.0)
    state = new_game(diamond_network())
    assert not any(sample_alerts(state, make_rng(1, STREAM_SENSORS), silent))


def test_same_seed_gives_the_same_alerts() -> None:
    state = new_game(diamond_network())
    first = sample_alerts(state, make_rng(5, STREAM_SENSORS))
    second = sample_alerts(state, make_rng(5, STREAM_SENSORS))
    assert first == second


def test_long_run_rates_match_the_settings() -> None:
    config = SensorConfig(detection_rate=0.8, false_alarm_rate=0.1)
    state = replace(new_game(diamond_network()))
    rng = make_rng(3, STREAM_SENSORS)
    draws = [sample_alerts(state, rng, config) for _ in range(4000)]
    hit_rate = sum(d[0] for d in draws) / len(draws)
    false_rate = sum(d[1] for d in draws) / len(draws)
    assert abs(hit_rate - 0.8) < 0.03
    assert abs(false_rate - 0.1) < 0.03