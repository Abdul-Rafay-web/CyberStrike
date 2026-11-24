from dataclasses import replace

import pytest

from agents.belief_csp_defender import BeliefCSPDefender
from agents.greedy_attacker import GreedyAttacker
from agents.random_agents import RandomAttacker
from agents.simple_defenders import StaticDefender
from config import CONFIG
from core.actions import Pass
from core.game import play_game
from core.generator import generate_network
from core.observation import observe_defender
from core.rules import Outcome
from core.state import Side, Stock, new_game
from ml.pipeline import train_sensor
from ml.sensor import ClassifierSensor
from tests.factories import diamond_network
from utils.rng import STREAM_AGENTS, STREAM_SENSORS, make_rng


def test_belief_csp_defender_always_returns_legal_actions_and_records_plan() -> None:
    trained = train_sensor()
    sensor = ClassifierSensor.from_trained(trained)
    for seed in range(10):
        net = generate_network(seed)
        defender = BeliefCSPDefender(rates=trained.rates)
        res = play_game(
            net,
            GreedyAttacker(),
            defender,
            make_rng(seed, STREAM_SENSORS),
            CONFIG,
            sensor=sensor,
        )
        assert res.turns >= 1
        assert len(defender.last_belief) == net.size
        assert sum(defender.last_belief) == pytest.approx(1.0)


def test_belief_csp_defender_passes_when_stock_is_empty() -> None:
    state = replace(new_game(diamond_network()), side=Side.DEFENDER, stock=Stock(0, 0, 0))
    obs = observe_defender(state, (False, False, False, False))
    defender = BeliefCSPDefender(rates=CONFIG.sensors)
    assert defender.choose_action(obs) == Pass()
    assert defender.last_plan == ()


def test_belief_csp_defender_beats_static_defender_over_seeded_games() -> None:
    trained = train_sensor()
    sensor = ClassifierSensor.from_trained(trained)
    static_wins = 0
    csp_wins = 0
    games = 40
    for seed in range(games):
        net = generate_network(seed)
        r_static = play_game(
            net,
            RandomAttacker(make_rng(seed, STREAM_AGENTS)),
            StaticDefender(),
            make_rng(seed, STREAM_SENSORS),
            CONFIG,
            sensor=sensor,
        )
        r_csp = play_game(
            net,
            RandomAttacker(make_rng(seed, STREAM_AGENTS)),
            BeliefCSPDefender(rates=trained.rates),
            make_rng(seed, STREAM_SENSORS),
            CONFIG,
            sensor=sensor,
        )
        static_wins += r_static.outcome is Outcome.DEFENDER_WINS
        csp_wins += r_csp.outcome is Outcome.DEFENDER_WINS
    assert csp_wins > static_wins
