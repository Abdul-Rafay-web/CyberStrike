from __future__ import annotations

import numpy as np

from config import CONFIG, AppConfig
from core.chokepoints import can_be_cut_off, chokepoints
from core.network import Edge, Network, bfs_distances, make_edge


def all_edge_pairs(size: int) -> tuple[Edge, ...]:
    """Canonical ordering of every possible undirected edge on `size` machines."""
    return tuple((a, b) for a in range(size) for b in range(a + 1, size))


def _components(size: int, edges: set[Edge]) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(size)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    seen: set[int] = set()
    groups: list[list[int]] = []
    for node in range(size):
        if node in seen:
            continue
        comp: list[int] = []
        stack = [node]
        seen.add(node)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        groups.append(comp)
    return groups


def _build_adj(size: int, edges: set[Edge]) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(size)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    return adj


def repair_network_edges(
    size: int,
    edges: set[Edge],
    hardness: list[int],
    entry: int,
    jewel: int,
    rng: np.random.Generator,
    config: AppConfig = CONFIG,
) -> tuple[set[Edge], list[int], int, int]:
    """Repair a candidate graph so it is connected, fair, and satisfies CyberStrike map rules.

    Guarantees:
      - Connected graph (reconnecting disconnected components after crossover/mutation)
      - Entry != Jewel with valid hardness endpoints (`min_hardness` at entry, `max_hardness` jewel)
      - Entry-to-jewel distance >= `min_entry_jewel_hops` whenever achievable
      - No single-machine chokepoints and not cut off by `stock.firewalls` links
    """
    game = config.game
    gen = config.generator

    # 1. Clamp hardness and fix entry/jewel values
    if entry == jewel:
        jewel = (entry + size // 2) % size
    repaired_h = [max(game.min_hardness, min(game.max_hardness, int(h))) for h in hardness]
    repaired_h[entry] = game.min_hardness
    repaired_h[jewel] = game.max_hardness

    repaired_edges: set[Edge] = {
        make_edge(a, b) for a, b in edges if 0 <= a < size and 0 <= b < size and a != b
    }

    # 2. Reconnect all disconnected components
    comps = _components(size, repaired_edges)
    while len(comps) > 1:
        c0 = comps[0]
        c1 = comps[1]
        a = c0[int(rng.integers(0, len(c0)))]
        b = c1[int(rng.integers(0, len(c1)))]
        repaired_edges.add(make_edge(a, b))
        comps = _components(size, repaired_edges)

    # 3. Ensure entry-jewel separation >= min_entry_jewel_hops if a direct shortcut exists
    direct = make_edge(entry, jewel)
    if direct in repaired_edges:
        trial = repaired_edges - {direct}
        if len(_components(size, trial)) == 1:
            repaired_edges = trial

    dist = bfs_distances(_build_adj(size, repaired_edges), entry)
    if dist[jewel] < gen.min_entry_jewel_hops:
        far = [m for m in range(size) if dist[m] >= gen.min_entry_jewel_hops and m != entry]
        if far:
            # Move jewel to a sufficiently distant node and preserve hardness convention
            repaired_h[jewel] = int(rng.integers(game.min_hardness, game.max_hardness + 1))
            jewel = far[int(rng.integers(0, len(far)))]
            repaired_h[jewel] = game.max_hardness

    # 4. Eliminate single-node chokepoints and 2-firewall cut vulnerabilities
    min_deg = game.stock.firewalls + 1
    for endpoint in (entry, jewel):
        adj = _build_adj(size, repaired_edges)
        while len(adj[endpoint]) < min_deg:
            candidates = [
                m
                for m in range(size)
                if m != endpoint
                and m not in adj[endpoint]
                and not (endpoint in (entry, jewel) and m in (entry, jewel))
            ]
            if not candidates:
                break
            other = candidates[int(rng.integers(0, len(candidates)))]
            repaired_edges.add(make_edge(endpoint, other))
            adj = _build_adj(size, repaired_edges)

    for _ in range(size * 3):
        net = Network.from_edges(size, repaired_edges, repaired_h, entry, jewel)
        if not chokepoints(net) and not can_be_cut_off(net, game.stock.firewalls):
            break
        # Add a non-shortcut edge connecting lower-degree nodes or bypassing chokepoints
        adj = _build_adj(size, repaired_edges)
        chokes = chokepoints(net)
        added = False
        if chokes:
            c = next(iter(chokes))
            d_blocked = bfs_distances(adj, entry, blocked={c})
            side_a = [m for m in range(size) if d_blocked[m] != -1 and m != c]
            side_b = [m for m in range(size) if d_blocked[m] == -1 and m != c]
            pairs = [
                make_edge(u, v)
                for u in side_a
                for v in side_b
                if make_edge(u, v) not in repaired_edges
                and make_edge(u, v) != make_edge(entry, jewel)
            ]
            if pairs:
                repaired_edges.add(pairs[int(rng.integers(0, len(pairs)))])
                added = True
        if not added:
            possible = [
                make_edge(u, v)
                for u in range(size)
                for v in range(u + 1, size)
                if make_edge(u, v) not in repaired_edges
                and make_edge(u, v) != make_edge(entry, jewel)
            ]
            if not possible:
                break
            repaired_edges.add(possible[int(rng.integers(0, len(possible)))])

    # 5. Prevent overly dense cliques while keeping validity intact
    max_edges = min(size * (size - 1) // 2, round(size * 2.1))
    if len(repaired_edges) > max_edges:
        edge_list = list(repaired_edges)
        rng.shuffle(edge_list)
        for edge in edge_list:
            if len(repaired_edges) <= max_edges:
                break
            candidate_edges = repaired_edges - {edge}
            if len(_components(size, candidate_edges)) != 1:
                continue
            cand_net = Network.from_edges(size, candidate_edges, repaired_h, entry, jewel)
            if not chokepoints(cand_net) and not can_be_cut_off(cand_net, game.stock.firewalls):
                repaired_edges = candidate_edges

    return repaired_edges, repaired_h, entry, jewel
