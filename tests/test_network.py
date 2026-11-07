import pytest

from core.network import Network, make_edge
from tests.factories import chain_network, diamond_network


def test_make_edge_puts_smaller_number_first() -> None:
    assert make_edge(5, 2) == (2, 5)
    assert make_edge(2, 5) == (2, 5)


def test_neighbours_are_sorted_and_two_way() -> None:
    net = diamond_network()
    assert net.neighbours(0) == (1, 2)
    assert net.neighbours(3) == (1, 2)
    assert net.has_edge(1, 3) and net.has_edge(3, 1)


def test_edges_lists_every_cable_once() -> None:
    assert chain_network().edges() == [(0, 1), (1, 2)]


def test_connected_and_disconnected() -> None:
    assert diamond_network().is_connected()
    split = Network.from_edges(4, [(0, 1), (2, 3)], hardness=[1, 1, 1, 1], entry=0, jewel=3)
    assert not split.is_connected()


def test_one_way_link_is_rejected() -> None:
    with pytest.raises(ValueError, match="one way"):
        Network(adjacency=((1,), ()), hardness=(1, 1), entry=0, jewel=1)


def test_entry_and_jewel_must_differ() -> None:
    with pytest.raises(ValueError, match="different"):
        Network.from_edges(2, [(0, 1)], hardness=[1, 1], entry=0, jewel=0)


def test_hardness_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        Network.from_edges(2, [(0, 1)], hardness=[1, 0], entry=0, jewel=1)