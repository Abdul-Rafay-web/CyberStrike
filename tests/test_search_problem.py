from dataclasses import replace

from core.actions import Advance
from core.observation import observe_attacker
from core.rules import apply
from core.state import new_game
from search.problem import problem_from_observation, problem_from_state
from tests.factories import chain_network, diamond_network


def test_entry_costs_follow_hardness_and_ownership() -> None:
    problem = problem_from_state(new_game(chain_network()))
    assert problem.entry_cost == (1, 2, 3)
    assert problem.start == 0
    assert problem.goal == 2


def test_patches_raise_costs_and_owned_machines_cost_one() -> None:
    state = replace(
        new_game(chain_network()), patched=frozenset({2}), breached=frozenset({0, 1})
    )
    problem = problem_from_state(state)
    assert problem.entry_cost == (1, 1, 5)


def test_firewalls_remove_links() -> None:
    state = replace(new_game(diamond_network()), firewalls=frozenset({(0, 2)}))
    problem = problem_from_state(state)
    assert problem.adjacency[0] == (1,)
    assert problem.adjacency[2] == (3,)


def test_a_breach_in_progress_costs_only_the_turns_left() -> None:
    mid_breach = apply(new_game(chain_network()), Advance(1))
    assert mid_breach.breach_target == 1
    problem = problem_from_state(mid_breach)
    assert problem.entry_cost[1] == mid_breach.breach_left == 1


def test_state_and_observation_give_the_same_problem() -> None:
    state = replace(new_game(diamond_network()), firewalls=frozenset({(0, 2)}))
    assert problem_from_state(state) == problem_from_observation(observe_attacker(state))


def test_path_cost_skips_the_starting_machine() -> None:
    problem = problem_from_state(new_game(chain_network()))
    assert problem.path_cost((0, 1, 2)) == 5
    assert problem.path_cost((0,)) == 0