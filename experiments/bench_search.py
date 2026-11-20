from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from core.generator import generate_network
from core.state import new_game
from experiments.scenarios import defended_state
from search.astar import astar
from search.counters import SearchResult
from search.problem import RouteProblem, problem_from_state
from search.uninformed import bfs, dfs, uniform_cost

Algorithm = Callable[[RouteProblem], SearchResult]

ALGORITHMS: dict[str, Algorithm] = {
    "BFS": bfs,
    "DFS": dfs,
    "Uniform cost": uniform_cost,
    "A star": astar,
}


@dataclass(frozen=True, slots=True)
class BenchRow:
    name: str
    optimal_rate: float
    mean_cost: float
    mean_expanded: float
    mean_microseconds: float


def build_problems(maps: int, defended: bool, size: int | None = None) -> list[RouteProblem]:
    problems: list[RouteProblem] = []
    for seed in range(maps):
        state = defended_state(seed, size) if defended else new_game(generate_network(seed, size))
        problems.append(problem_from_state(state))
    return problems


def run_benchmark(problems: list[RouteProblem]) -> list[BenchRow]:
    best_costs: list[int] = []
    for problem in problems:
        reference = uniform_cost(problem)
        if reference.cost is None:
            raise ValueError("every benchmark problem must have a route")
        best_costs.append(reference.cost)

    rows: list[BenchRow] = []
    for name, algorithm in ALGORITHMS.items():
        optimal = 0
        total_cost = 0
        total_expanded = 0
        total_seconds = 0.0
        for problem, best in zip(problems, best_costs, strict=True):
            result = algorithm(problem)
            if result.cost is None:
                raise ValueError(f"{name} found no route on a solvable problem")
            optimal += result.cost == best
            total_cost += result.cost
            total_expanded += result.stats.expanded
            total_seconds += result.stats.seconds
        count = len(problems)
        rows.append(
            BenchRow(
                name,
                optimal / count,
                total_cost / count,
                total_expanded / count,
                1e6 * total_seconds / count,
            )
        )
    return rows


def format_table(title: str, rows: list[BenchRow]) -> str:
    lines = [
        title,
        f"{'algorithm':<14}{'optimal':>9}{'mean cost':>11}{'expanded':>10}{'microsec':>10}",
    ]
    for row in rows:
        lines.append(
            f"{row.name:<14}{row.optimal_rate:>9.0%}{row.mean_cost:>11.2f}"
            f"{row.mean_expanded:>10.1f}{row.mean_microseconds:>10.0f}"
        )
    return "\n".join(lines)


def main() -> None:
    for defended, title in ((False, "no defenses"), (True, "random defenses placed")):
        rows = run_benchmark(build_problems(100, defended))
        print(format_table(f"100 maps, {title}", rows))
        print()


if __name__ == "__main__":
    main()