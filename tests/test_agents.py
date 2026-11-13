from dataclasses import replace

import pytest

from agents.random_agents import RandomAttacker, RandomDefender
from core.observation import DefenderObservation, observe_attacker, observe_defender
from core.state import Side, new_game
from tests.factories import diamond_network
from utils.rng import STREAM_AGENTS, make_rng


def _defender_view() -> DefenderObservation:
    state = replace(new_game(diamond_network()), side=Side.DEFENDER)
    return observe_defender(state, (False,) * 4)


def test_random_attacker_picks_a_legal_action() -> None:
    obs = observe_attacker(new_game(diamond_network()))
    agent = RandomAttacker(make_rng(1, STREAM_AGENTS))
    for _ in range(20):
        assert agent.choose_action(obs) in obs.legal_actions


def test_random_defender_picks_a_legal_action() -> None:
    obs = _defender_view()
    agent = RandomDefender(make_rng(1, STREAM_AGENTS))
    for _ in range(50):
        assert agent.choose_action(obs) in obs.legal_actions


def test_same_seed_gives_the_same_choices() -> None:
    obs = _defender_view()
    first = RandomDefender(make_rng(9, STREAM_AGENTS))
    second = RandomDefender(make_rng(9, STREAM_AGENTS))
    assert [first.choose_action(obs) for _ in range(10)] == [
        second.choose_action(obs) for _ in range(10)
    ]


def test_choosing_from_nothing_is_an_error() -> None:
    finished = replace(new_game(diamond_network()), turn=99)
    obs = observe_attacker(finished)
    with pytest.raises(ValueError):
        RandomAttacker(make_rng(1, STREAM_AGENTS)).choose_action(obs)