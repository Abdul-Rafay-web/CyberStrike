from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

from config import CONFIG, GameConfig
from core.network import Edge, Network, make_edge
from core.observation import AttackerObservation
from core.state import GameState
from search.heuristics import cost_to_goal


@dataclass(frozen=True, slots=True)
class RouteProblem:
    """Find the cheapest way for the attacker to reach the jewel."""

    adjacency: tuple[tuple[int, ...], ...]
    entry_cost: tuple[int, ...]
    start: int
    goal: int
    heuristic: tuple[int, ...]

    @property
    def size(self) -> int:
        return len(self.adjacency)

    def path_cost(self, path: tuple[int, ...]) -> int:
        """Turns needed to walk a path. The first machine is free: we are already there."""
        return sum(self.entry_cost[machine] for machine in path[1:])


def build_problem(
    network: Network,
    position: int,
    breach_target: int | None,
    breach_left: int,
    breached: Collection[int],
    patched: Collection[int],
    firewalls: Collection[Edge],
    config: GameConfig = CONFIG.game,
) -> RouteProblem:
    size = network.size
    entry_cost: list[int] = []
    floor_cost: list[int] = []
    for machine in range(size):
        if machine in breached:
            entry_cost.append(1)
            floor_cost.append(1)
        else:
            base = network.hardness[machine]
            bonus = config.patch_hardness_bonus if machine in patched else 0
            entry_cost.append(base + bonus)
            floor_cost.append(base)
    if breach_target is not None:
        entry_cost[breach_target] = breach_left
        floor_cost[breach_target] = min(floor_cost[breach_target], breach_left)

    open_adjacency = tuple(
        tuple(other for other in network.neighbours(m) if make_edge(m, other) not in firewalls)
        for m in range(size)
    )
    relaxed = cost_to_goal(network.adjacency, floor_cost, network.jewel)
    return RouteProblem(
        adjacency=open_adjacency,
        entry_cost=tuple(entry_cost),
        start=position,
        goal=network.jewel,
        heuristic=relaxed,
    )


def problem_from_state(state: GameState, config: GameConfig = CONFIG.game) -> RouteProblem:
    return build_problem(
        state.network,
        state.attacker_pos,
        state.breach_target,
        state.breach_left,
        state.breached,
        state.patched,
        state.firewalls,
        config,
    )


def problem_from_observation(
    observation: AttackerObservation, config: GameConfig = CONFIG.game
) -> RouteProblem:
    return build_problem(
        observation.network,
        observation.position,
        observation.breach_target,
        observation.breach_left,
        observation.breached,
        observation.patched,
        observation.firewalls,
        config,
    )