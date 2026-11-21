from dataclasses import replace

import pytest

from core.actions import Advance
from core.generator import generate_network
from core.rules import legal_actions
from core.state import GameState, new_game
from experiments.scenarios import defended_state
from search.evaluation import analyze_routes
from search.minimax import iterative_search, search_to_depth
from search.model import ordered_attacker_moves
from tests.factories import diamond_network


def _positions() -> list[GameState]:
    states: list[GameState] = []
    for seed in range(30):
        states.append(new_game(generate_network(seed)))
        states.append(defended_state(seed))
    return states


def test_alpha_beta_and_minimax_agree_on_the_move_and_the_value() -> None:
    for state in _positions():
        plain = search_to_depth(state, 3, pruning=False)
        pruned = search_to_depth(state, 3, pruning=True)
        assert plain.action == pruned.action
        assert plain.value == pytest.approx(pruned.value)


def test_alpha_beta_agrees_with_a_margin_too() -> None:
    for state in _positions():
        plain = search_to_depth(state, 3, pruning=False, margin=25.0)
        pruned = search_to_depth(state, 3, pruning=True, margin=25.0)
        assert plain.action == pruned.action


def test_alpha_beta_expands_fewer_nodes() -> None:
    plain_total = 0
    pruned_total = 0
    for state in _positions():
        plain = search_to_depth(state, 3, pruning=False)
        pruned = search_to_depth(state, 3, pruning=True)
        assert pruned.nodes <= plain.nodes
        plain_total += plain.nodes
        pruned_total += pruned.nodes
    assert pruned_total * 2 < plain_total


def test_plain_minimax_never_cuts_off() -> None:
    state = new_game(generate_network(1))
    assert search_to_depth(state, 3, pruning=False).cutoffs == 0
    assert search_to_depth(state, 3, pruning=True).cutoffs > 0


def test_a_sure_win_is_taken() -> None:
    state = replace(
        new_game(diamond_network()), attacker_pos=1, breached=frozenset({0, 1})
    )
    report = search_to_depth(state, 2)
    assert report.action == Advance(3)
    assert report.value == 990.0


def test_a_lost_position_still_races_for_the_jewel() -> None:
    # The jewel has only two links, so two firewalls can seal it off.
    report = iterative_search(new_game(diamond_network()), max_depth=4, time_budget=2.0)
    assert report.value < 0
    assert report.action == Advance(1)


def test_a_large_margin_always_chooses_the_cheapest_route() -> None:
    for state in _positions():
        cheapest = ordered_attacker_moves(state, analyze_routes(state))[0]
        assert search_to_depth(state, 2, margin=1e9).action == cheapest


def test_iterative_search_returns_a_legal_move_within_its_budget() -> None:
    state = defended_state(4)
    report = iterative_search(state, max_depth=6, time_budget=0.05)
    assert report.action in legal_actions(state)
    assert report.seconds < 1.0


def test_iterative_search_reaches_the_requested_depth_when_time_allows() -> None:
    report = iterative_search(new_game(generate_network(2)), max_depth=4, time_budget=5.0)
    assert report.depth == 4


def test_a_zero_budget_falls_back_to_the_cheapest_route() -> None:
    state = new_game(generate_network(5))
    report = iterative_search(state, max_depth=6, time_budget=0.0)
    assert report.action == ordered_attacker_moves(state, analyze_routes(state))[0]


def test_the_trace_shows_pruning_only_when_pruning_is_on() -> None:
    state = new_game(diamond_network())
    plain: list[str] = []
    pruned: list[str] = []
    search_to_depth(state, 3, pruning=False, trace=plain)
    search_to_depth(state, 3, pruning=True, trace=pruned)
    assert not any("pruned" in line for line in plain)
    assert any("pruned" in line for line in pruned)
    assert len(pruned) < len(plain)

