from __future__ import annotations

from core.network import Network


def chain_network() -> Network:
    """Machines 0, 1, 2 in a line. Entry is 0 and the jewel is 2."""
    return Network.from_edges(3, [(0, 1), (1, 2)], hardness=[1, 2, 3], entry=0, jewel=2)


def diamond_network() -> Network:
    """Two separate routes from entry 0 to jewel 3: through 1 or through 2."""
    return Network.from_edges(
        4, [(0, 1), (0, 2), (1, 3), (2, 3)], hardness=[1, 1, 2, 1], entry=0, jewel=3
    )