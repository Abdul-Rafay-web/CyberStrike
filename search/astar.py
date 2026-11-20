from __future__ import annotations

import heapq
from collections.abc import Sequence

from search.counters import Counters, SearchResult, found_result, missing_result
from search.problem import RouteProblem


def best_first_search(problem: RouteProblem, heuristic: Sequence[int]) -> SearchResult:
    """Always expand the machine with the smallest cost so far plus guess still to go."""
    if len(heuristic) != problem.size:
        raise ValueError("the heuristic needs one value per machine")
    counters = Counters()
    start = problem.start
    cheapest: dict[int, int] = {start: 0}
    parent: dict[int, int | None] = {start: None}
    closed: set[int] = set()
    frontier = [(heuristic[start], heuristic[start], start)]
    counters.generated += 1

    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current in closed:
            continue
        closed.add(current)
        counters.expanded += 1
        if current == problem.goal:
            return found_result(problem, parent, counters)
        spent = cheapest[current]
        for nxt in problem.adjacency[current]:
            if nxt in closed:
                continue
            total = spent + problem.entry_cost[nxt]
            if nxt not in cheapest or total < cheapest[nxt]:
                cheapest[nxt] = total
                parent[nxt] = current
                counters.generated += 1
                heapq.heappush(frontier, (total + heuristic[nxt], heuristic[nxt], nxt))
    return missing_result(counters)


def astar(problem: RouteProblem, heuristic: Sequence[int] | None = None) -> SearchResult:
    """A star search. By default it uses the relaxed problem guess stored in the problem."""
    return best_first_search(problem, problem.heuristic if heuristic is None else heuristic)