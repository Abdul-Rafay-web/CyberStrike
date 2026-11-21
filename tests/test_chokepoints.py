from core.chokepoints import can_be_cut_off, chokepoints
from core.network import Network, bfs_distances
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


def test_a_machine_with_two_links_can_be_cut_off_by_two_firewalls() -> None:
    assert can_be_cut_off(diamond_network(), 2)
    assert not can_be_cut_off(diamond_network(), 1)


def test_a_chain_is_cut_off_by_a_single_firewall() -> None:
    assert can_be_cut_off(chain_network(), 1)


def test_a_well_connected_pair_survives_two_cuts() -> None:
    net = Network.from_edges(
        5,
        [(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4)],
        hardness=[1, 1, 1, 1, 1],
        entry=0,
        jewel=4,
    )
    assert not can_be_cut_off(net, 2)
    assert can_be_cut_off(net, 3)
