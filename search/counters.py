from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass

from search.problem import RouteProblem


@dataclass(frozen=True, slots=True)
class SearchStats:
    expanded: int
    generated: int
    seconds: float


@dataclass(frozen=True, slots=True)
class SearchResult:
    path: tuple[int, ...]
    cost: int | None
    stats: SearchStats

    @property
    def found(self) -> bool:
        return self.cost is not None


class Counters:
    """Tallies the work a search does. Start one at the top of every algorithm."""

    def __init__(self) -> None:
        self.expanded = 0
        self.generated = 0
        self._started = time.perf_counter()

    def finish(self) -> SearchStats:
        return SearchStats(self.expanded, self.generated, time.perf_counter() - self._started)


def found_result(
    problem: RouteProblem, parent: Mapping[int, int | None], counters: Counters
) -> SearchResult:
    steps = [problem.goal]
    while (before := parent[steps[-1]]) is not None:
        steps.append(before)
    path = tuple(reversed(steps))
    return SearchResult(path, problem.path_cost(path), counters.finish())


def missing_result(counters: Counters) -> SearchResult:
    return SearchResult((), None, counters.finish())