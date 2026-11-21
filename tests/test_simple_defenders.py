from dataclasses import replace

from agents.simple_defenders import PassiveDefender, ReactiveDefender, StaticDefender
from core.actions import Pass, Patch
from core.observation import DefenderObservation, observe_defender
from core.state import GameState, Side, new_game
from experiments.scenarios import defended_state
from tests.factories import diamond_network


def _view(state: GameState, alerts: tuple[bool, ...]) -> DefenderObservation:
    return observe_defender(replace(state, side=Side.DEFENDER), alerts)


def test_passive_defender_always_passes() -> None:
    obs = _view(new_game(diamond_network()), (False,) * 4)
    assert PassiveDefender().choose_action(obs) == Pass()


def test_static_defender_always_picks_a_legal_move() -> None:
    for seed in range(40):
        obs = _view(defended_state(seed), (False,) * 15)
        assert StaticDefender().choose_action(obs) in obs.legal_actions


def test_static_defender_starts_with_the_busiest_machine() -> None:
    state = new_game(diamond_network())
    obs = _view(state, (False,) * 4)
    action = StaticDefender().choose_action(obs)
    assert isinstance(action, Patch)
    assert action.machine in (1, 2, 3)


def test_reactive_defender_waits_when_it_cannot_find_the_intruder() -> None:
    obs = _view(new_game(diamond_network()), (False, False, False, False))
    assert ReactiveDefender().choose_action(obs) == Pass()
    confused = _view(new_game(diamond_network()), (True, True, False, False))
    assert ReactiveDefender().choose_action(confused) == Pass()


def test_reactive_defender_hits_the_intruders_route() -> None:
    obs = _view(new_game(diamond_network()), (True, False, False, False))
    action = ReactiveDefender().choose_action(obs)
    assert action == Patch(3)

