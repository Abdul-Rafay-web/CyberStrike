from __future__ import annotations

from itertools import combinations

from core.network import Edge, Network, bfs_distances


def chokepoints(network: Network) -> frozenset[int]:
    """Machines that every route from the entry to the jewel must pass through."""
    found: set[int] = set()
    for machine in range(network.size):
        if machine in (network.entry, network.jewel):
            continue
        distances = bfs_distances(network.adjacency, network.entry, blocked={machine})
        if distances[network.jewel] == -1:
            found.add(machine)
    return frozenset(found)


def _entry_reaches_jewel(network: Network, removed: frozenset[Edge]) -> bool:
    seen = {network.entry}
    stack = [network.entry]
    while stack:
        current = stack.pop()
        if current == network.jewel:
            return True
        for other in network.adjacency[current]:
            link = (current, other) if current < other else (other, current)
            if other not in seen and link not in removed:
                seen.add(other)
                stack.append(other)
    return False


def can_be_cut_off(network: Network, links: int) -> bool:
    """True if cutting some `links` cables would separate the entry from the jewel."""
    if min(len(network.neighbours(network.entry)), len(network.neighbours(network.jewel))) <= links:
        return True
    return any(
        not _entry_reaches_jewel(network, frozenset(removed))
        for removed in combinations(network.edges(), links)
    )
