from __future__ import annotations

from dataclasses import dataclass

from config import CONFIG, GameConfig
from core.state import GameState, Side
from search.heuristics import UNREACHABLE, cost_to_goal
from search.problem import route_costs

MAX_EXITS = 5


@dataclass(frozen=True, slots=True)
class EvalWeights:
    """How much each worry costs, in score points. Phase 9 tunes these numbers."""

    patch: float = 20.0
    firewall: float = 40.0
    exits: float = 1.0
    honeypot: float = 10.0


DEFAULT_WEIGHTS = EvalWeights()


@dataclass(frozen=True, slots=True)
class RouteInfo:
    """What the attacker can say about its routes to the jewel in one position."""

    entry_cost: tuple[int, ...]
    adjacency: tuple[tuple[int, ...], ...]
    cost_left: tuple[int, ...]
    position: int
    goal: int

    @property
    def cost(self) -> int:
        return self.cost_left[self.position]

    @property
    def reachable(self) -> bool:
        return self.cost < UNREACHABLE

    def exits(self) -> int:
        """How many neighbouring machines still lead to the jewel."""
        return sum(1 for n in self.adjacency[self.position] if self.cost_left[n] < UNREACHABLE)

    def step_cost(self, machine: int) -> int:
        """Total cost of the cheapest route that starts by stepping onto this machine."""
        return self.entry_cost[machine] + self.cost_left[machine]

    def path(self) -> tuple[int, ...]:
        """The cheapest route, found by always stepping to the neighbour that keeps it cheapest."""
        route = [self.position]
        for _ in range(len(self.adjacency)):
            if route[-1] == self.goal:
                return tuple(route)
            here = route[-1]
            route.append(min(self.adjacency[here], key=lambda n: (self.step_cost(n), n)))
        raise RuntimeError("the cost table does not lead to the jewel")


def analyze_routes(state: GameState, game: GameConfig = CONFIG.game) -> RouteInfo:
    costs = route_costs(
        state.network,
        state.attacker_pos,
        state.breach_target,
        state.breach_left,
        state.breached,
        state.patched,
        state.firewalls,
        game,
    )
    left = cost_to_goal(costs.open_adjacency, costs.entry_cost, state.network.jewel)
    return RouteInfo(
        entry_cost=costs.entry_cost,
        adjacency=costs.open_adjacency,
        cost_left=left,
        position=state.attacker_pos,
        goal=state.network.jewel,
    )


def honeypot_risk(state: GameState, route: RouteInfo, game: GameConfig = CONFIG.game) -> float:
    """Expected turns lost to hidden honeypots on the cheapest route."""
    net = state.network
    hidden_left = game.stock.honeypots - len(state.triggered)
    suspects = {
        m
        for m in range(net.size)
        if m not in state.breached
        and m not in state.patched
        and m not in state.triggered
        and m not in (net.entry, net.jewel)
    }
    if hidden_left <= 0 or not suspects:
        return 0.0
    chance = min(1.0, hidden_left / len(suspects))
    exposed = sum(1 for machine in route.path()[1:] if machine in suspects)
    return chance * exposed * game.honeypot_delay_turns


def evaluate(
    state: GameState,
    route: RouteInfo,
    game: GameConfig = CONFIG.game,
    weights: EvalWeights = DEFAULT_WEIGHTS,
) -> float:
    """Estimated final score for the attacker if the game stopped being played out here.

    Start from the score the attacker gets if nobody interferes any more, then subtract
    a small price for each worry. A sure win therefore always beats an estimate.
    """
    wait = 0 if state.side is Side.ATTACKER else 1
    finish_turn = state.turn + wait + route.cost - 1
    if finish_turn > game.turn_limit:
        return -float(game.win_value - game.turn_penalty * game.turn_limit)
    projected = float(game.win_value - game.turn_penalty * finish_turn)
    exposed = sum(1 for machine in route.path()[1:] if machine not in state.breached)
    threat = weights.patch * min(state.stock.patches, exposed) + weights.firewall * min(
        state.stock.firewalls, exposed
    )
    exits_missing = MAX_EXITS - min(route.exits(), MAX_EXITS)
    return (
        projected
        - threat
        - weights.exits * exits_missing
        - weights.honeypot * honeypot_risk(state, route, game)
    )

