import pytest

from config import CONFIG
from core.chokepoints import chokepoints
from core.generator import generate_network
from core.network import bfs_distances

SIZES = (10, 15, 20, 25)


def test_same_seed_gives_the_same_map() -> None:
    assert generate_network(7) == generate_network(7)


def test_different_seeds_give_different_maps() -> None:
    maps = {generate_network(seed) for seed in range(10)}
    assert len(maps) >= 8


@pytest.mark.parametrize("size", SIZES)
def test_every_generated_map_is_valid(size: int) -> None:
    gen = CONFIG.generator
    game = CONFIG.game
    for seed in range(60):
        net = generate_network(seed, size)
        assert net.size == size
        assert net.is_connected()
        assert chokepoints(net) == frozenset()
        assert bfs_distances(net.adjacency, net.entry)[net.jewel] >= gen.min_entry_jewel_hops
        assert all(game.min_hardness <= h <= game.max_hardness for h in net.hardness)
        assert net.hardness[net.entry] == game.min_hardness
        assert net.hardness[net.jewel] == game.max_hardness
        assert len(net.edges()) == round(size * gen.edge_factor)


def test_size_must_be_inside_the_allowed_range() -> None:
    with pytest.raises(ValueError):
        generate_network(1, 5)
    with pytest.raises(ValueError):
        generate_network(1, 40)