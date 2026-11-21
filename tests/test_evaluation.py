from dataclasses import replace

import pytest

from core.actions import Advance
from core.rules import apply
from core.state import Side, new_game
from experiments.scenarios import defended_state
from search.evaluation import analyze_routes, evaluate
from tests.factories import chain_network, diamond_network


def test_route_info_on_a_chain() -> None:
    route = analyze_routes(new_game(chain_network()))
    assert route.cost == 5
    assert route.path() == (0, 1, 2)
    assert route.exits() == 1
    assert route.reachable


def test_a_breach_in_progress_forces_the_first_step() -> None:
    mid_breach = apply(new_game(diamond_network()), Advance(2))
    route = analyze_routes(mid_breach)
    assert route.path() == (0, 2, 3)
    assert route.cost == 2


def test_evaluation_matches_a_hand_calculation() -> None:
    state = new_game(chain_network())
    # Unopposed the attacker finishes on turn 5, which scores 1000 minus 50.
    # Threats: 2 exposed machines, 3 patches (2 usable) and 2 firewalls (2 usable).
    # 950 - 20*2 - 40*2 - 1*(5-1 exits) - 10*(2 turns of honeypot risk) = 806
    assert evaluate(state, analyze_routes(state)) == pytest.approx(806.0)


def test_a_route_that_cannot_finish_in_time_is_a_certain_loss() -> None:
    state = replace(new_game(chain_network()), turn=24)
    assert evaluate(state, analyze_routes(state)) == -750.0


def test_waiting_for_the_defender_costs_one_extra_turn() -> None:
    attacker_to_move = new_game(chain_network())
    defender_to_move = replace(attacker_to_move, side=Side.DEFENDER)
    gap = evaluate(attacker_to_move, analyze_routes(attacker_to_move)) - evaluate(
        defender_to_move, analyze_routes(defender_to_move)
    )
    assert gap == pytest.approx(10.0)


def test_an_estimate_never_exceeds_the_unopposed_score() -> None:
    for seed in range(40):
        state = defended_state(seed)
        route = analyze_routes(state)
        finish = state.turn + route.cost - 1
        assert evaluate(state, route) <= 1000 - 10 * finish

