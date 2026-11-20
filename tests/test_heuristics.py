from dataclasses import replace

import pytest

from core.generator import generate_network
from core.state import new_game
from experiments.scenarios import defended_state
from search.astar import astar
from search.heuristics import cost_to_goal
from search.problem import RouteProblem, problem_from_state
from search.uninformed import uniform_cost


def test_cost_to_goal_on_a_tiny_line() -> None:
    adjacency = ((1,), (0, 2), (1,))
    assert cost_to_goal(adjacency, (1, 2, 3), goal=2) == (5, 3, 0)


def _problems() -> list[RouteProblem]:
    problems = []
    for seed in range(40):
        problems.append(problem_from_state(new_game(generate_network(seed))))
        problems.append(problem_from_state(defended_state(seed)))
    return problems


def test_heuristic_never_overestimates() -> None:
    for problem in _problems():
        for machine in range(problem.size):
            result = uniform_cost(replace(problem, start=machine))
            if result.cost is not None:
                assert problem.heuristic[machine] <= result.cost


def test_heuristic_is_consistent() -> None:
    for problem in _problems():
        assert problem.heuristic[problem.goal] == 0
        for machine in range(problem.size):
            for nxt in problem.adjacency[machine]:
                bound = problem.entry_cost[nxt] + problem.heuristic[nxt]
                assert problem.heuristic[machine] <= bound


def test_heuristic_is_exact_when_nothing_is_defended() -> None:
    problem = problem_from_state(new_game(generate_network(5)))
    assert problem.heuristic[problem.start] == uniform_cost(problem).cost


def test_a_wrong_sized_heuristic_is_rejected() -> None:
    problem = problem_from_state(new_game(generate_network(1)))
    with pytest.raises(ValueError):
        astar(problem, heuristic=(0, 0))