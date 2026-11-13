from __future__ import annotations

from collections import deque
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass

Edge = tuple[int, int]


def make_edge(a: int, b: int) -> Edge:
    """Return the link between a and b with smaller number first."""
    return (a, b) if a < b else (b, a)

def bfs_distances(
    adjacency: Sequence[Iterable[int]], start: int, blocked: Collection[int] = ()
) -> list[int]:
    """Fewest links from start to every machine. A value of -1 means unreachable."""
    distance = [-1] * len(adjacency)
    if start in blocked:
        return distance
    distance[start] = 0
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for nxt in adjacency[current]:
            if distance[nxt] == -1 and nxt not in blocked:
                distance[nxt] = distance[current] + 1
                queue.append(nxt)
    return distance

@dataclass(frozen=True, slots=True)
class Network:
    adjacency: tuple[tuple[int, ...], ...]
    hardness: tuple[int, ...]
    entry: int
    jewel: int

    def __post_init__(self) -> None:
        size = len(self.adjacency)
        if len(self.hardness) != size:
            raise ValueError("hardness needs one value per machine")
        if any(h < 1 for h in self.hardness):
            raise ValueError("hardness must be at least 1")
        if not (0 <= self.entry < size and 0 <= self.jewel < size):
            raise ValueError("entry and jewel must be on the network")
        if self.entry == self.jewel:
            raise ValueError("entry and jewel must be different machines")
        for machine, neighbors in enumerate(self.adjacency):
            for other in neighbors:
                if other == machine:
                    raise ValueError(f"machine: {machine} links to itself")
                if not 0 <= other < size:
                    raise ValueError(f"machine: {machine} links to unknown machine {other}")
                if machine not in self.adjacency[other]:
                    raise ValueError(f"link {machine} to {other} is one way")

    @property
    def size(self) -> int:
        return len(self.adjacency)

    def neighbours(self, machine: int) -> tuple[int, ...]:
        return self.adjacency[machine]

    def has_edge(self, a: int, b: int) -> bool:
        return b in self.adjacency[a]

    def edges(self) -> list[Edge]:
        return [(a, b) for a, others in enumerate(self.adjacency) for b in others if a < b]

    def is_connected(self) -> bool:
        return -1 not in bfs_distances(self.adjacency, self.entry)

    @classmethod
    def from_edges(
        cls,
        size: int,
        edges: Iterable[Edge],
        hardness: Sequence[int],
        entry: int,
        jewel: int,
    ) -> Network:
        neighbours: list[set[int]] = [set() for _ in range(size)]
        for a, b in edges:
            if not (0 <= a < size and 0 <= b < size):
                raise ValueError(f"link {a} to {b} uses an unknown machine")
            neighbours[a].add(b)
            neighbours[b].add(a)
        adjacency = tuple(tuple(sorted(group)) for group in neighbours)
        return cls(adjacency, tuple(hardness), entry, jewel)
