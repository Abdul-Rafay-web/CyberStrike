from __future__ import annotations

import heapq
from collections.abc import Sequence

UNREACHABLE = 10**6


def cost_to_goal(
    adjacency: Sequence[Sequence[int]], entry_cost: Sequence[int], goal: int
) -> tuple[int, ...]:
    """Cheapest total entry cost from every machine to the goal (a search run backwards).

    The graph may be one way: adjacency[a] lists the machines you can step to from a.
    """
    size = len(adjacency)
    arrivals: list[list[int]] = [[] for _ in range(size)]
    for source, targets in enumerate(adjacency):
        for target in targets:
            arrivals[target].append(source)

    best = [UNREACHABLE] * size
    best[goal] = 0
    queue = [(0, goal)]
    while queue:
        spent, machine = heapq.heappop(queue)
        if spent > best[machine]:
            continue
        for source in arrivals[machine]:
            candidate = spent + entry_cost[machine]
            if candidate < best[source]:
                best[source] = candidate
                heapq.heappush(queue, (candidate, source))
    return tuple(best)
