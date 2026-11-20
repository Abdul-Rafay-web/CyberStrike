from __future__ import annotations

from collections import deque

from search.astar import best_first_search
from search.counters import Counters, SearchResult, found_result, missing_result
from search.problem import RouteProblem


def bfs(problem: RouteProblem) -> SearchResult:
    """Breadth first: fewest links, ignores how hard each machine is."""
    counters = Counters()
    parent: dict[int, int | None] = {problem.start: None}
    frontier = deque([problem.start])
    counters.generated += 1
    while frontier:
        current = frontier.popleft()
        counters.expanded += 1
        if current == problem.goal:
            return found_result(problem, parent, counters)
        for nxt in problem.adjacency[current]:
            if nxt not in parent:
                parent[nxt] = current
                counters.generated += 1
                frontier.append(nxt)
    return missing_result(counters)


def dfs(problem: RouteProblem) -> SearchResult:
    """Depth first: dives as deep as it can. Finds a route, not a good one."""
    counters = Counters()
    parent: dict[int, int | None] = {}
    stack: list[tuple[int, int | None]] = [(problem.start, None)]
    counters.generated += 1
    while stack:
        current, came_from = stack.pop()
        if current in parent:
            continue
        parent[current] = came_from
        counters.expanded += 1
        if current == problem.goal:
            return found_result(problem, parent, counters)
        for nxt in reversed(problem.adjacency[current]):
            if nxt not in parent:
                counters.generated += 1
                stack.append((nxt, current))
    return missing_result(counters)


def uniform_cost(problem: RouteProblem) -> SearchResult:
    """Cheapest route with no guess about the future. Same as A star with a guess of zero."""
    return best_first_search(problem, (0,) * problem.size)