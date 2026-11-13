from __future__ import annotations

from dataclasses import dataclass

from config import CONFIG, GameConfig
from core.actions import Action
from core.network import Edge, Network
from core.rules import legal_actions
from core.state import GameState, Stock


@dataclass(frozen=True, slots=True)
class AttackerObservation:
    """Everything the attacker is allowed to know."""

    network: Network
    position: int
    breach_target: int | None
    breach_left: int
    breached: frozenset[int]
    patched: frozenset[int]
    firewalls: frozenset[Edge]
    known_honeypots: frozenset[int]
    turn: int
    legal_actions: tuple[Action, ...]


@dataclass(frozen=True, slots=True)
class DefenderObservation:
    """Everything the defender is allowed to know. The attacker's position is not here."""

    network: Network
    patched: frozenset[int]
    firewalls: frozenset[Edge]
    honeypots: frozenset[int]
    stock: Stock
    turn: int
    alerts: tuple[bool, ...]
    intruder_known_at: int | None
    legal_actions: tuple[Action, ...]


def observe_attacker(state: GameState, config: GameConfig = CONFIG.game) -> AttackerObservation:
    return AttackerObservation(
        network=state.network,
        position=state.attacker_pos,
        breach_target=state.breach_target,
        breach_left=state.breach_left,
        breached=state.breached,
        patched=state.patched,
        firewalls=state.firewalls,
        known_honeypots=state.triggered,
        turn=state.turn,
        legal_actions=tuple(legal_actions(state, config)),
    )


def _intruder_known_at(state: GameState) -> int | None:
    """A triggered honeypot shows the defender where the intruder is."""
    if state.breach_target is not None and state.breach_target in state.triggered:
        return state.breach_target
    if state.attacker_pos in state.triggered:
        return state.attacker_pos
    return None


def observe_defender(
    state: GameState, alerts: tuple[bool, ...], config: GameConfig = CONFIG.game
) -> DefenderObservation:
    return DefenderObservation(
        network=state.network,
        patched=state.patched,
        firewalls=state.firewalls,
        honeypots=state.honeypots,
        stock=state.stock,
        turn=state.turn,
        alerts=alerts,
        intruder_known_at=_intruder_known_at(state),
        legal_actions=tuple(legal_actions(state, config)),
    )