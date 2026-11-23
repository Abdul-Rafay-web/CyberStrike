from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

from core.network import Network, bfs_distances

NodeT = TypeVar("NodeT")


@dataclass(frozen=True, slots=True)
class MarkovBlanket(Generic[NodeT]):
    """Parents, children, and children's other parents (co-parents) of a node in a DAG."""

    node: NodeT
    parents: frozenset[NodeT]
    children: frozenset[NodeT]
    co_parents: frozenset[NodeT]

    @property
    def members(self) -> frozenset[NodeT]:
        return self.parents | self.children | self.co_parents


class BayesNetDAG(Generic[NodeT]):
    """Directed acyclic graph supporting parent, child, co-parent, and Markov blanket queries."""

    def __init__(self, nodes: Iterable[NodeT] = ()) -> None:
        self._parents: dict[NodeT, set[NodeT]] = {}
        self._children: dict[NodeT, set[NodeT]] = {}
        for node in nodes:
            self.add_node(node)

    def add_node(self, node: NodeT) -> None:
        self._parents.setdefault(node, set())
        self._children.setdefault(node, set())

    def add_edge(self, parent: NodeT, child: NodeT) -> None:
        if parent == child:
            raise ValueError(f"self-loop not allowed on node {parent!r}")
        self.add_node(parent)
        self.add_node(child)
        if child in self._children[parent]:
            return
        # Reject any edge that would introduce a directed cycle
        if self._reaches(child, parent):
            raise ValueError(f"adding edge {parent!r} -> {child!r} would create a cycle")
        self._children[parent].add(child)
        self._parents[child].add(parent)

    def _reaches(self, start: NodeT, target: NodeT) -> bool:
        seen: set[NodeT] = {start}
        stack = [start]
        while stack:
            current = stack.pop()
            if current == target:
                return True
            for nxt in self._children.get(current, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return False

    @property
    def nodes(self) -> frozenset[NodeT]:
        return frozenset(self._parents)

    def parents(self, node: NodeT) -> frozenset[NodeT]:
        self._require(node)
        return frozenset(self._parents[node])

    def children(self, node: NodeT) -> frozenset[NodeT]:
        self._require(node)
        return frozenset(self._children[node])

    def co_parents(self, node: NodeT) -> frozenset[NodeT]:
        """Other parents of this node's children (excluding the node itself)."""
        self._require(node)
        spouses: set[NodeT] = set()
        for child in self._children[node]:
            for parent in self._parents[child]:
                if parent != node:
                    spouses.add(parent)
        return frozenset(spouses)

    def markov_blanket(self, node: NodeT) -> MarkovBlanket[NodeT]:
        """Return parents, children, and children's other parents for `node`."""
        self._require(node)
        return MarkovBlanket(
            node=node,
            parents=self.parents(node),
            children=self.children(node),
            co_parents=self.co_parents(node),
        )

    def topological_order(self) -> tuple[NodeT, ...]:
        """Return nodes in a valid topological order (parents before children)."""
        in_degree = {node: len(pars) for node, pars in self._parents.items()}
        ready = deque(node for node, deg in in_degree.items() if deg == 0)
        ordered: list[NodeT] = []
        while ready:
            current = ready.popleft()
            ordered.append(current)
            for child in self._children[current]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    ready.append(child)
        if len(ordered) != len(self._parents):
            raise RuntimeError("graph contains a cycle")
        return tuple(ordered)

    def _require(self, node: NodeT) -> None:
        if node not in self._parents:
            raise KeyError(f"unknown node {node!r}")


def network_causal_dag(network: Network) -> BayesNetDAG[int]:
    """Orient a CyberStrike network into an acyclic causal progression DAG from entry outward.

    Every edge (u, v) is directed from the machine closer to `network.entry` to the one further
    away, breaking equal-hop ties by smaller machine index. Because `(hop_distance, machine_id)`
    strictly increases along every edge, the resulting graph is guaranteed acyclic.
    """
    distances = bfs_distances(network.adjacency, network.entry)
    dag: BayesNetDAG[int] = BayesNetDAG(range(network.size))
    for u, v in network.edges():
        key_u = (distances[u], u)
        key_v = (distances[v], v)
        if key_u < key_v:
            dag.add_edge(u, v)
        else:
            dag.add_edge(v, u)
    return dag


def markov_blanket(network: Network, machine: int) -> MarkovBlanket[int]:
    """Markov blanket of `machine` in the network's causal progression Bayesian network.

    Returns parents, children, and the children's other parents (co-parents).
    """
    if not 0 <= machine < network.size:
        raise ValueError(f"machine {machine} out of range 0..{network.size - 1}")
    return network_causal_dag(network).markov_blanket(machine)


def dbn_dag(network_size: int, steps: int) -> BayesNetDAG[str]:
    """Unrolled Dynamic Bayesian Network for `steps` turns on `network_size` machines.

    Variables:
      - `X_t` for t in 0..steps (hidden attacker active machine at turn t)
      - `A_{t}_{m}` for t in 1..steps, m in 0..network_size-1 (alert flag at machine m on turn t)
    Edges:
      - `X_{t-1} -> X_t` (Markov transition)
      - `X_t -> A_{t}_{m}` (sensor emission at every machine)
    """
    if network_size < 1:
        raise ValueError("network_size must be positive")
    if steps < 0:
        raise ValueError("steps cannot be negative")
    dag: BayesNetDAG[str] = BayesNetDAG(["X_0"])
    for t in range(1, steps + 1):
        prev_state = f"X_{t - 1}"
        curr_state = f"X_{t}"
        dag.add_edge(prev_state, curr_state)
        for m in range(network_size):
            dag.add_edge(curr_state, f"A_{t}_{m}")
    return dag

