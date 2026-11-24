from __future__ import annotations

import heapq
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from typing import Any

from config import CONFIG, GameConfig
from core.actions import Action, Firewall, Honeypot, Patch
from core.network import Edge, Network, make_edge
from core.state import Stock
from csp.arc import BinaryConstraint


@dataclass(frozen=True, slots=True)
class RouteRequirement:
    """One route to the jewel that must contain at least one assigned patch or firewall."""

    path: tuple[int, ...]
    patch_targets: frozenset[int]
    firewall_targets: frozenset[Edge]

    def covered_by(self, action: Action) -> bool:
        if isinstance(action, Patch):
            return action.machine in self.patch_targets
        if isinstance(action, Firewall):
            return (action.a, action.b) in self.firewall_targets
        return False


@dataclass(frozen=True)
class CSPProblem:
    """General finite-domain CSP with binary constraints and optional route-coverage constraints."""

    variables: tuple[str, ...]
    domains: dict[str, tuple[Any, ...]]
    binary_constraints: dict[tuple[str, str], BinaryConstraint]
    neighbors: dict[str, tuple[str, ...]]
    route_requirements: tuple[RouteRequirement, ...] = ()
    cover_variables: frozenset[str] = field(default_factory=frozenset)
    value_scores: dict[tuple[str, Any], float] = field(default_factory=dict)

    def is_consistent_pair(self, var_a: str, val_a: Any, var_b: str, val_b: Any) -> bool:
        fn = self.binary_constraints.get((var_a, var_b))
        return True if fn is None else fn(val_a, val_b)

    def is_valid_assignment(self, assignment: dict[str, Any]) -> bool:
        if len(assignment) != len(self.variables):
            return False
        for (var_a, var_b), fn in self.binary_constraints.items():
            if not fn(assignment[var_a], assignment[var_b]):
                return False
        for req in self.route_requirements:
            if not any(
                isinstance(assignment[v], Action) and req.covered_by(assignment[v])
                for v in self.cover_variables
            ):
                return False
        return True


def _dijkstra_path(
    network: Network,
    start: int,
    goal: int,
    entry_cost: Sequence[int],
    blocked_edges: Collection[Edge],
    blocked_nodes: Collection[int],
) -> tuple[int, tuple[int, ...]] | None:
    if start in blocked_nodes or goal in blocked_nodes:
        return None
    best: dict[int, int] = {start: 0}
    parent: dict[int, int] = {}
    queue: list[tuple[int, int]] = [(0, start)]

    while queue:
        spent, node = heapq.heappop(queue)
        if spent > best[node]:
            continue
        if node == goal:
            path = [goal]
            while path[-1] != start:
                path.append(parent[path[-1]])
            path.reverse()
            return spent, tuple(path)
        for nxt in network.neighbours(node):
            if nxt in blocked_nodes or make_edge(node, nxt) in blocked_edges:
                continue
            cand = spent + entry_cost[nxt]
            if cand < best.get(nxt, 10**9):
                best[nxt] = cand
                parent[nxt] = node
                heapq.heappush(queue, (cand, nxt))
    return None


def k_cheapest_routes(
    network: Network,
    start: int,
    k: int = 3,
    *,
    patched: Collection[int] = (),
    firewalls: Collection[Edge] = (),
    game: GameConfig = CONFIG.game,
) -> tuple[tuple[int, ...], ...]:
    """Return up to `k` cheapest distinct simple routes from `start` to `network.jewel` (Yen's)."""
    if k <= 0 or start == network.jewel:
        return ()
    entry_cost = [
        network.hardness[m] + (game.patch_hardness_bonus if m in patched else 0)
        for m in range(network.size)
    ]
    base_blocked = set(firewalls)
    first = _dijkstra_path(network, start, network.jewel, entry_cost, base_blocked, ())
    if first is None:
        return ()

    accepted: list[tuple[int, ...]] = [first[1]]
    candidates: list[tuple[int, tuple[int, ...]]] = []
    seen_candidates: set[tuple[int, ...]] = {first[1]}

    for _ in range(1, k):
        prev_path = accepted[-1]
        for i in range(len(prev_path) - 1):
            spur_node = prev_path[i]
            root_path = prev_path[: i + 1]
            removed_edges = set(base_blocked)
            for existing in accepted:
                if len(existing) > i + 1 and existing[: i + 1] == root_path:
                    removed_edges.add(make_edge(existing[i], existing[i + 1]))
            removed_nodes = set(root_path[:-1])
            spur = _dijkstra_path(
                network, spur_node, network.jewel, entry_cost, removed_edges, removed_nodes
            )
            if spur is None:
                continue
            total_path = root_path[:-1] + spur[1]
            if total_path in seen_candidates:
                continue
            seen_candidates.add(total_path)
            total_cost = sum(entry_cost[m] for m in total_path[1:])
            heapq.heappush(candidates, (total_cost, total_path))

        if not candidates:
            break
        _, next_path = heapq.heappop(candidates)
        accepted.append(next_path)

    return tuple(accepted)


def _neq(a: Any, b: Any) -> bool:
    return bool(a != b)


def _patch_honeypot_disjoint(p_act: Any, h_act: Any) -> bool:
    assert isinstance(p_act, Patch) and isinstance(h_act, Honeypot)
    return p_act.machine != h_act.machine


def _honeypot_patch_disjoint(h_act: Any, p_act: Any) -> bool:
    assert isinstance(h_act, Honeypot) and isinstance(p_act, Patch)
    return h_act.machine != p_act.machine


def _add_binary(
    constraints: dict[tuple[str, str], BinaryConstraint],
    adj: dict[str, set[str]],
    var_a: str,
    var_b: str,
    fn_ab: BinaryConstraint,
    fn_ba: BinaryConstraint,
) -> None:
    constraints[(var_a, var_b)] = fn_ab
    constraints[(var_b, var_a)] = fn_ba
    adj[var_a].add(var_b)
    adj[var_b].add(var_a)


def _route_requirement(path: tuple[int, ...]) -> RouteRequirement:
    nodes = frozenset(path[1:])
    edges = frozenset(make_edge(a, b) for a, b in zip(path, path[1:], strict=False))
    return RouteRequirement(path=path, patch_targets=nodes, firewall_targets=edges)


def build_defense_csp(
    network: Network,
    stock: Stock,
    *,
    patched: frozenset[int] = frozenset(),
    firewalls: frozenset[Edge] = frozenset(),
    honeypots: frozenset[int] = frozenset(),
    belief: Sequence[float] | None = None,
    routes_to_cover: int = CONFIG.csp.routes_to_cover,
    game: GameConfig = CONFIG.game,
) -> CSPProblem:
    """Build the Defender resource-placement CSP for the current network, stock, and belief.

    Variables:
      - P1..Pk for available patches (domain: unpatched, non-honeypot machines)
      - F1..Fk for available firewalls (domain: unfirewalled edges)
      - H1..Hk for available honeypots (domain: interior unpatched, non-honeypot machines)
    Constraints:
      - All different within each defense type (P, F, H)
      - No honeypot on entry or crown jewel
      - No honeypot on a patched machine
      - Each of the top `routes_to_cover` cheapest routes holds at least one patch or firewall
    """
    patch_actions = tuple(
        Patch(m) for m in range(network.size) if m not in patched and m not in honeypots
    )
    firewall_actions = tuple(
        Firewall(a, b) for a, b in network.edges() if (a, b) not in firewalls
    )
    honeypot_actions = tuple(
        Honeypot(m)
        for m in range(network.size)
        if m not in (network.entry, network.jewel) and m not in patched and m not in honeypots
    )

    n_patches = min(stock.patches, len(patch_actions))
    n_firewalls = min(stock.firewalls, len(firewall_actions))
    n_honeypots = min(stock.honeypots, len(honeypot_actions))

    # Keep total (patches + honeypots) feasible across available machines
    available_for_ph = len(patch_actions)
    if n_patches + n_honeypots > available_for_ph:
        n_honeypots = max(0, available_for_ph - n_patches)

    p_vars = tuple(f"P{i}" for i in range(1, n_patches + 1))
    f_vars = tuple(f"F{i}" for i in range(1, n_firewalls + 1))
    h_vars = tuple(f"H{i}" for i in range(1, n_honeypots + 1))
    variables = h_vars + f_vars + p_vars

    domains: dict[str, tuple[Any, ...]] = {}
    for v in p_vars:
        domains[v] = patch_actions
    for v in f_vars:
        domains[v] = firewall_actions
    for v in h_vars:
        domains[v] = honeypot_actions

    constraints: dict[tuple[str, str], BinaryConstraint] = {}
    adj: dict[str, set[str]] = {v: set() for v in variables}

    # 1. All-different within each type
    for group in (p_vars, f_vars, h_vars):
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                _add_binary(constraints, adj, group[i], group[j], _neq, _neq)

    # 2. No honeypot on a patched machine
    for pv in p_vars:
        for hv in h_vars:
            _add_binary(
                constraints,
                adj,
                pv,
                hv,
                _patch_honeypot_disjoint,
                _honeypot_patch_disjoint,
            )

    # 3. Determine routes from top-belief origin(s) (or entry) to the Crown Jewel
    belief_vec = (
        tuple(float(b) for b in belief)
        if belief is not None and len(belief) == network.size
        else tuple(1.0 if m == network.entry else 0.0 for m in range(network.size))
    )
    top_origins = sorted(range(network.size), key=lambda m: (-belief_vec[m], m))
    primary_origin = next((m for m in top_origins if m != network.jewel), network.entry)
    raw_routes = k_cheapest_routes(
        network,
        primary_origin,
        routes_to_cover,
        patched=patched,
        firewalls=firewalls,
        game=game,
    )
    if len(raw_routes) < routes_to_cover and primary_origin != network.entry:
        fallback_routes = k_cheapest_routes(
            network,
            network.entry,
            routes_to_cover,
            patched=patched,
            firewalls=firewalls,
            game=game,
        )
        merged = list(raw_routes)
        for path in fallback_routes:
            if path not in merged:
                merged.append(path)
            if len(merged) == routes_to_cover:
                break
        raw_routes = tuple(merged)

    # Keep only routes that are not already covered by an existing patch and are coverable
    cover_vars = frozenset(p_vars + f_vars)
    valid_patch_set = {a.machine for a in patch_actions}
    valid_fw_set = {(a.a, a.b) for a in firewall_actions}
    requirements: list[RouteRequirement] = []
    for path in raw_routes:
        if any(m in patched for m in path[1:]):
            continue
        req = _route_requirement(path)
        can_patch = bool(p_vars) and bool(req.patch_targets & valid_patch_set)
        can_fw = bool(f_vars) and bool(req.firewall_targets & valid_fw_set)
        if can_patch or can_fw:
            requirements.append(req)
    max_coverable = len(cover_vars)
    if len(requirements) > max_coverable:
        requirements = requirements[:max_coverable]

    # 4. Belief-guided value scores for LCV ordering
    node_route_hits: dict[int, float] = {}
    edge_route_hits: dict[Edge, float] = {}
    for rank, path in enumerate(raw_routes):
        weight = 1.0 / (rank + 1)
        for step_idx, node in enumerate(path[1:], start=1):
            # Earlier nodes along the attacker's route + jewel itself are high priority
            urgency = 1.4 if node == network.jewel else 1.0 / (0.6 + 0.35 * step_idx)
            node_route_hits[node] = node_route_hits.get(node, 0.0) + weight * urgency
        for step_idx, (a, b) in enumerate(zip(path, path[1:], strict=False), start=1):
            edge = make_edge(a, b)
            urgency = 1.5 if network.jewel in (a, b) else 1.0 / (0.5 + 0.4 * step_idx)
            edge_route_hits[edge] = edge_route_hits.get(edge, 0.0) + weight * urgency

    # Propagate belief one step ahead to estimate where the intruder is about to step
    next_step_threat: dict[int, float] = {}
    for m in range(network.size):
        prob_m = belief_vec[m]
        if prob_m <= 1e-4:
            continue
        open_nbrs = [n for n in network.neighbours(m) if make_edge(m, n) not in firewalls]
        if not open_nbrs:
            continue
        share = prob_m / len(open_nbrs)
        for n in open_nbrs:
            next_step_threat[n] = next_step_threat.get(n, 0.0) + share

    value_scores: dict[tuple[str, Any], float] = {}
    for pv in p_vars:
        for p_act in patch_actions:
            m = p_act.machine
            score = (
                3.0 * node_route_hits.get(m, 0.0)
                + 2.0 * next_step_threat.get(m, 0.0)
                + 1.5 * belief_vec[m]
                + (1.2 if m == network.jewel else 0.0)
            )
            value_scores[(pv, p_act)] = score
    for fv in f_vars:
        for f_act in firewall_actions:
            edge = (f_act.a, f_act.b)
            endpoint_threat = belief_vec[f_act.a] + belief_vec[f_act.b]
            score = (
                4.0 * edge_route_hits.get(edge, 0.0)
                + 2.5 * endpoint_threat
                + (1.5 if network.jewel in edge else 0.0)
            )
            value_scores[(fv, f_act)] = score
    for hv in h_vars:
        for h_act in honeypot_actions:
            m = h_act.machine
            score = 3.5 * next_step_threat.get(m, 0.0) + 2.0 * node_route_hits.get(m, 0.0)
            value_scores[(hv, h_act)] = score

    neighbors = {v: tuple(sorted(nbrs)) for v, nbrs in adj.items()}
    return CSPProblem(
        variables=variables,
        domains=domains,
        binary_constraints=constraints,
        neighbors=neighbors,
        route_requirements=tuple(requirements),
        cover_variables=cover_vars,
        value_scores=value_scores,
    )


def map_coloring_csp(
    regions: Sequence[str],
    borders: Sequence[tuple[str, str]],
    colors: Sequence[str],
) -> CSPProblem:
    """Build a classic map-coloring CSP (used for unit tests and the 4-region docs dry run)."""
    variables = tuple(regions)
    domains: dict[str, tuple[Any, ...]] = {r: tuple(colors) for r in variables}
    constraints: dict[tuple[str, str], BinaryConstraint] = {}
    adj: dict[str, set[str]] = {r: set() for r in variables}
    for a, b in borders:
        _add_binary(constraints, adj, a, b, _neq, _neq)
    neighbors = {v: tuple(sorted(nbrs)) for v, nbrs in adj.items()}
    return CSPProblem(
        variables=variables,
        domains=domains,
        binary_constraints=constraints,
        neighbors=neighbors,
    )
