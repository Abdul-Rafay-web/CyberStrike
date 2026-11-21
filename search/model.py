from __future__ import annotations

from config import CONFIG, GameConfig, SearchConfig
from core.actions import Action, Advance, Firewall, Pass, Patch
from core.network import Network
from core.observation import AttackerObservation
from core.rules import attacker_actions
from core.state import GameState, Side, Stock
from search.evaluation import RouteInfo


def state_from_observation(
    observation: AttackerObservation, game: GameConfig = CONFIG.game
) -> GameState:
    """Rebuild the game as the attacker believes it is. Hidden honeypots stay out of the model."""
    patches_left = max(0, game.stock.patches - len(observation.patched))
    firewalls_left = max(0, game.stock.firewalls - len(observation.firewalls))
    return GameState(
        network=observation.network,
        attacker_pos=observation.position,
        breach_target=observation.breach_target,
        breach_left=observation.breach_left,
        breached=observation.breached,
        patched=observation.patched,
        firewalls=observation.firewalls,
        honeypots=frozenset(),
        triggered=observation.known_honeypots,
        stock=Stock(patches_left, firewalls_left, 0),
        turn=observation.turn,
        side=Side.ATTACKER,
    )


def _target(action: Action) -> int:
    if not isinstance(action, Advance):
        raise TypeError("attacker menus only contain Advance moves")
    return action.target


def ordered_attacker_moves(state: GameState, route: RouteInfo) -> list[Action]:
    """The attacker's legal moves, cheapest route first. Good order makes pruning work better."""
    moves = attacker_actions(state)
    if len(moves) > 1:
        moves.sort(key=lambda a: (route.step_cost(_target(a)), _target(a)))
    return moves


def _near_route(network: Network, path: tuple[int, ...], radius: int) -> set[int]:
    """Machines on the route, plus everything within a few links of it."""
    near = set(path)
    frontier = set(path)
    for _ in range(radius):
        frontier = {n for machine in frontier for n in network.neighbours(machine)} - near
        near |= frontier
    return near


def defender_replies(
    state: GameState,
    route: RouteInfo,
    game: GameConfig = CONFIG.game,
    search: SearchConfig = CONFIG.search,
) -> list[Action]:
    """The defender moves the attacker bothers to think about, most dangerous first.

    The defender may act anywhere, so we follow the attacker's cheapest route, not its
    position. Moves on that route come first, and only the best few are kept.
    """
    net = state.network
    options: list[tuple[int, int, Action]] = []
    if (state.stock.patches > 0 or state.stock.firewalls > 0) and route.reachable:
        path = route.path()
        near = _near_route(net, path, search.defender_reply_radius)
        on_path = set(path[1:])
        path_links = {(min(a, b), max(a, b)) for a, b in zip(path, path[1:], strict=False)}
        if state.stock.patches > 0:
            for m in sorted(near):
                if m not in state.breached and m not in state.patched:
                    options.append((-2 if m in on_path else 0, m, Patch(m)))
        if state.stock.firewalls > 0:
            for a, b in net.edges():
                if a not in near or b not in near or (a, b) in state.firewalls:
                    continue
                if a in state.breached and b in state.breached:
                    continue
                if (a, b) in path_links:
                    rank = -3
                elif net.jewel in (a, b):
                    rank = -1
                else:
                    rank = 0
                options.append((rank, 100 + a * 32 + b, Firewall(a, b)))
    options.sort(key=lambda item: (item[0], item[1]))
    limit = max(0, search.max_defender_replies - 1)
    replies = [action for _, _, action in options[:limit]]
    replies.append(Pass())
    return replies

