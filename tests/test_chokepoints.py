from core.chokepoints import chokepoints
from core.network import bfs_distances
from tests.factories import chain_network, diamond_network


def test_middle_of_a_chain_is_a_chokepoint() -> None:
    assert chokepoints(chain_network()) == frozenset({1})


def test_diamond_has_no_chokepoint() -> None:
    assert chokepoints(diamond_network()) == frozenset()


def test_bfs_distances_counts_links() -> None:
    assert bfs_distances(chain_network().adjacency, 0) == [0, 1, 2]


def test_bfs_marks_unreachable_machines_and_respects_blocked() -> None:
    assert bfs_distances(chain_network().adjacency, 0, blocked={1}) == [0, -1, -1]
    assert bfs_distances(chain_network().adjacency, 1, blocked={1}) == [-1, -1, -1]