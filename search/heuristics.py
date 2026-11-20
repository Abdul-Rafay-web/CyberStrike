from __future__ import annotations

import heapq
from collections.abc import Sequence

UNREACHABLE = 10**6


def cost_to_goal(
    adjacency: Sequence[Sequence[int]], entry_cost: Sequence[int], goal: int
) -> tuple[int, ...]:
    """Cheapest total entry cost from every machine to the goal (a search run backwards)."""
    best = [UNREACHABLE] * len(adjacency)
    best[goal] = 0
    queue = [(0, goal)]
    while queue:
        spent, machine = heapq.heappop(queue)
        if spent > best[machine]:
            continue
        for other in adjacency[machine]:
            candidate = spent + entry_cost[machine]
            if candidate < best[other]:
                best[other] = candidate
                heapq.heappush(queue, (candidate, other))
    return tuple(best)