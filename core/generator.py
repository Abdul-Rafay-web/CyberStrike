from __future__ import annotations

import numpy as np

from config import CONFIG, AppConfig
from core.chokepoints import chokepoints
from core.network import Edge, Network, bfs_distances, make_edge
from utils.rng import STREAM_MAP, make_rng


class GenerationError(RuntimeError):
    """Raised when no valid map was found within the allowed attempts."""


def generate_network(seed: int, size: int | None = None, config: AppConfig = CONFIG) -> Network:
    """Build one random map. The same seed and size always give the same map."""
    game = config.game
    size = game.default_machines if size is None else size
    if not game.min_machines <= size <= game.max_machines:
        raise ValueError(
            f"size must be between {game.min_machines} and {game.max_machines}, got {size}"
        )
    rng = make_rng(seed, STREAM_MAP)
    for _ in range(config.generator.max_attempts):
        candidate = _try_generate(rng, size, config)
        if candidate is not None:
            return candidate
    raise GenerationError(
        f"no valid map for seed {seed} and size {size} after "
        f"{config.generator.max_attempts} attempts"
    )


def _random_connected_edges(rng: np.random.Generator, size: int, edge_factor: float) -> set[Edge]:
    order = [int(m) for m in rng.permutation(size)]
    edges: set[Edge] = set()
    for i in range(1, size):
        parent = order[int(rng.integers(0, i))]
        edges.add(make_edge(order[i], parent))
    target = min(round(size * edge_factor), size * (size - 1) // 2)
    while len(edges) < target:
        a = int(rng.integers(0, size))
        b = int(rng.integers(0, size))
        if a != b:
            edges.add(make_edge(a, b))
    return edges


def _adjacency(size: int, edges: set[Edge]) -> list[list[int]]:
    adjacency: list[list[int]] = [[] for _ in range(size)]
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    return adjacency


def _try_generate(rng: np.random.Generator, size: int, config: AppConfig) -> Network | None:
    game, gen = config.game, config.generator
    edges = _random_connected_edges(rng, size, gen.edge_factor)

    entry = int(rng.integers(0, size))
    distance = bfs_distances(_adjacency(size, edges), entry)
    far = [m for m in range(size) if distance[m] >= gen.min_entry_jewel_hops]
    if not far:
        return None
    jewel = far[int(rng.integers(0, len(far)))]

    levels = np.arange(game.min_hardness, game.max_hardness + 1)
    weights = np.array(gen.hardness_weights, dtype=float)
    if len(weights) != len(levels):
        raise ValueError("hardness_weights needs one weight per hardness level")
    drawn = rng.choice(levels, size=size, p=weights / weights.sum())
    hardness = [int(h) for h in drawn]
    hardness[entry] = game.min_hardness
    hardness[jewel] = game.max_hardness

    network = Network.from_edges(size, edges, hardness, entry, jewel)
    if chokepoints(network):
        return None
    return network