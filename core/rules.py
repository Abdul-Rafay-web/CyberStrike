from __future__ import annotations

from collections import deque
from dataclasses import replace
from enum import Enum
from typing import assert_never

from config import CONFIG, GameConfig
from core.actions import Action, Advance, Firewall, Honeypot, Pass, Patch
from core.network import make_edge
from core.state import GameState, Side


class IllegalActionError(ValueError):
    """Raised when someone tries a move the rules do not allow."""


class Outcome(Enum):
    ONGOING = "ongoing"
    ATTACKER_WINS = "attacker_wins"
    DEFENDER_WINS = "defender_wins"


def effective_hardness(
    state: GameState, machine: int, config: GameConfig = CONFIG.game
) -> int:
    base = state.network.hardness[machine]
    if machine in state.patched:
        return base + config.patch_hardness_bonus
    return base

def open_neighbours(state: GameState, machine: int) -> tuple[int, ...]:
    """Neighbours reachable without crossing a firewalled link."""
    return tuple(
        other
        for other in state.network.neighbours(machine)
        if make_edge(machine, other) not in state.firewalls
    )

def jewel_reachable(state: GameState) -> bool:
    jewel = state.network.jewel
    seen = {state.attacker_pos}
    queue = deque([state.attacker_pos])
    while queue:
        current = queue.popleft()
        if current == jewel:
            return True
        for nxt in open_neighbours(state, current):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False


def outcome(state: GameState, config: GameConfig = CONFIG.game) -> Outcome:
    if state.network.jewel in state.breached:
        return Outcome.ATTACKER_WINS
    if state.turn > config.turn_limit:
        return Outcome.DEFENDER_WINS
    if not jewel_reachable(state):
        return Outcome.DEFENDER_WINS
    return Outcome.ONGOING

def is_terminal(state: GameState, config: GameConfig = CONFIG.game) -> bool:
    return outcome(state, config) is not Outcome.ONGOING


def utility(state: GameState, config: GameConfig = CONFIG.game) -> int:
    """Score of a finished game from the attacker's point of view."""
    result = outcome(state, config)
    if result is Outcome.ONGOING:
        raise ValueError("the game is not over yet")
    turns_used = min(state.turn, config.turn_limit)
    value = config.win_value - config.turn_penalty * turns_used
    return value if result is Outcome.ATTACKER_WINS else -value


def _attacker_actions(state: GameState) -> list[Action]:
    if state.breach_target is not None:
        return [Advance(state.breach_target)]
    return [Advance(other) for other in open_neighbours(state, state.attacker_pos)]


def _defender_actions(state: GameState) -> list[Action]:
    net = state.network
    actions: list[Action] = []

    if state.stock.patches > 0:
        for machine in range(net.size):
            if (
                machine not in state.breached
                and machine not in state.patched
                and machine not in state.honeypots
            ):
                actions.append(Patch(machine))

    if state.stock.firewalls > 0:
        crossing = None
        if state.breach_target is not None:
            crossing = make_edge(state.attacker_pos, state.breach_target)
        for a, b in net.edges():
            if (a, b) not in state.firewalls and (a, b) != crossing:
                actions.append(Firewall(a, b))

    if state.stock.honeypots > 0:
        for machine in range(net.size):
            if (
                machine not in (net.entry, net.jewel)
                and machine not in state.breached
                and machine != state.breach_target
                and machine not in state.patched
                and machine not in state.honeypots
            ):
                actions.append(Honeypot(machine))

    actions.append(Pass())
    return actions


def legal_actions(state: GameState, config: GameConfig = CONFIG.game) -> list[Action]:
    if is_terminal(state, config):
        return []
    if state.side is Side.ATTACKER:
        return _attacker_actions(state)
    return _defender_actions(state)


def _finish_defender_turn(state: GameState) -> GameState:
    return replace(state, side=Side.ATTACKER, turn=state.turn + 1)


def _advance(state: GameState, target: int, config: GameConfig) -> GameState:
    triggered = state.triggered
    if state.breach_target == target:
        remaining = state.breach_left - 1
    else:
        total = 1 if target in state.breached else effective_hardness(state, target, config)
        if target in state.honeypots and target not in state.triggered:
            total += config.honeypot_delay_turns
            triggered = triggered | {target}
        remaining = total - 1

    if remaining <= 0:
        return replace(
            state,
            attacker_pos=target,
            breach_target=None,
            breach_left=0,
            breached=state.breached | {target},
            triggered=triggered,
            side=Side.DEFENDER,
        )
    return replace(
        state,
        breach_target=target,
        breach_left=remaining,
        triggered=triggered,
        side=Side.DEFENDER,
    )


def apply(
    state: GameState, action: Action, config: GameConfig = CONFIG.game
) -> GameState:
    """Return the new state after one move. The old state is never changed."""
    if action not in legal_actions(state, config):
        raise IllegalActionError(
            f"{action} is not legal for the {state.side.value} on turn {state.turn}"
        )
    match action:
        case Advance(target=target):
            return _advance(state, target, config)
        case Patch(machine=machine):
            extra = config.patch_hardness_bonus if state.breach_target == machine else 0
            updated = replace(
                state,
                patched=state.patched | {machine},
                stock=replace(state.stock, patches=state.stock.patches - 1),
                breach_left=state.breach_left + extra,
            )
            return _finish_defender_turn(updated)
        case Firewall(a=a, b=b):
            updated = replace(
                state,
                firewalls=state.firewalls | {(a, b)},
                stock=replace(state.stock, firewalls=state.stock.firewalls - 1),
            )
            return _finish_defender_turn(updated)
        case Honeypot(machine=machine):
            updated = replace(
                state,
                honeypots=state.honeypots | {machine},
                stock=replace(state.stock, honeypots=state.stock.honeypots - 1),
            )
            return _finish_defender_turn(updated)
        case Pass():
            return _finish_defender_turn(state)
        case _:
            assert_never(action)