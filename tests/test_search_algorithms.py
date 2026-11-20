from dataclasses import replace

from core.generator import generate_network
from core.rules import is_terminal
from core.state import new_game
from experiments.scenarios import defended_state
from search.astar import astar
from search.problem import RouteProblem, problem_from_state
from search.uninformed import bfs, dfs, uniform_cost
from tests.factories import chain_network, diamond_network


def _diamond_problem() -> RouteProblem:
    return problem_from_state(new_game(diamond_network()))


def test_every_algorithm_finds_a_valid_route_on_the_diamond() -> None:
    problem = _diamond_problem()
    for algorithm in (bfs, dfs, uniform_cost, astar):
        result = algorithm(problem)
        assert result.found
        assert result.path[0] == problem.start
        assert result.path[-1] == problem.goal
        assert result.cost == problem.path_cost(result.path)


def test_uniform_cost_and_astar_choose_the_cheaper_route() -> None:
    problem = _diamond_problem()
    assert uniform_cost(problem).path == (0, 1, 3)
    assert uniform_cost(problem).cost == 2
    assert astar(problem).cost == 2


def test_bfs_minimises_links_not_cost() -> None:
    # Machine 1 is a short but expensive detour. Machines 2 and 3 are a longer cheap one.
    problem = RouteProblem(
        adjacency=((1, 2), (0, 4), (0, 3), (2, 4), (1, 3)),
        entry_cost=(1, 9, 1, 1, 1),
        start=0,
        goal=4,
        heuristic=(0, 0, 0, 0, 0),
    )
    assert bfs(problem).path == (0, 1, 4)
    assert bfs(problem).cost == 10
    assert uniform_cost(problem).path == (0, 2, 3, 4)
    assert uniform_cost(problem).cost == 3


def test_no_route_means_not_found() -> None:
    cut_off = RouteProblem(
        adjacency=((), ()), entry_cost=(1, 1), start=0, goal=1, heuristic=(0, 0)
    )
    for algorithm in (bfs, dfs, uniform_cost, astar):
        result = algorithm(cut_off)
        assert not result.found
        assert result.path == ()


def test_astar_with_a_zero_guess_behaves_like_uniform_cost() -> None:
    problem = problem_from_state(new_game(generate_network(8)))
    zero = (0,) * problem.size
    assert astar(problem, zero).cost == uniform_cost(problem).cost
    assert astar(problem, zero).stats.expanded == uniform_cost(problem).stats.expanded


def test_astar_matches_uniform_cost_on_one_hundred_maps() -> None:
    for seed in range(100):
        for state in (new_game(generate_network(seed)), defended_state(seed)):
            problem = problem_from_state(state)
            assert astar(problem).cost == uniform_cost(problem).cost


def test_astar_expands_no_more_machines_than_uniform_cost() -> None:
    total_astar = 0
    total_uniform = 0
    for seed in range(100):
        problem = problem_from_state(defended_state(seed))
        assert astar(problem).stats.expanded <= uniform_cost(problem).stats.expanded
        total_astar += astar(problem).stats.expanded
        total_uniform += uniform_cost(problem).stats.expanded
    assert total_astar < total_uniform


def test_defended_states_always_have_a_route() -> None:
    for seed in range(100):
        state = defended_state(seed)
        assert not is_terminal(state)
        assert uniform_cost(problem_from_state(state)).found


def test_stats_count_work() -> None:
    result = astar(problem_from_state(new_game(chain_network())))
    assert result.stats.expanded >= 3
    assert result.stats.generated >= result.stats.expanded - 1
    assert result.stats.seconds >= 0.0


def test_replace_works_on_a_problem() -> None:
    problem = _diamond_problem()
    assert replace(problem, start=1).start == 1