from __future__ import annotations

from core.network import Network, bfs_distances


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