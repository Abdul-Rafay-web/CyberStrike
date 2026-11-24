from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from core.actions import Action, Advance, Firewall, Honeypot, Pass, Patch
from core.network import Network, make_edge
from core.rules import Outcome


def _now() -> float:
    return time.time()


@dataclass(frozen=True, slots=True)
class GameStarted:
    seed: int
    size: int
    entry: int
    jewel: int
    hardness: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    version: int = 1
    timestamp: float = 0.0


@dataclass(frozen=True, slots=True)
class AttackerMoved:
    turn: int
    target: int
    duration_ms: float = 0.0
    version: int = 1
    timestamp: float = 0.0


@dataclass(frozen=True, slots=True)
class DefensePlaced:
    turn: int
    action_kind: str
    machine: int | None = None
    edge_a: int | None = None
    edge_b: int | None = None
    duration_ms: float = 0.0
    version: int = 1
    timestamp: float = 0.0


@dataclass(frozen=True, slots=True)
class AlertsRaised:
    turn: int
    alerts: tuple[bool, ...]
    version: int = 1
    timestamp: float = 0.0


@dataclass(frozen=True, slots=True)
class BeliefUpdated:
    turn: int
    probabilities: tuple[float, ...]
    version: int = 1
    timestamp: float = 0.0


@dataclass(frozen=True, slots=True)
class GameEnded:
    turn: int
    outcome: str
    utility: int
    version: int = 1
    timestamp: float = 0.0


GameEvent = GameStarted | AttackerMoved | DefensePlaced | AlertsRaised | BeliefUpdated | GameEnded


def make_game_started(seed: int, network: Network) -> GameStarted:
    return GameStarted(
        seed=seed,
        size=network.size,
        entry=network.entry,
        jewel=network.jewel,
        hardness=network.hardness,
        edges=tuple(network.edges()),
        timestamp=_now(),
    )


def make_defense_placed(turn: int, action: Action, duration_ms: float = 0.0) -> DefensePlaced:
    ts = _now()
    if isinstance(action, Patch):
        return DefensePlaced(
            turn=turn,
            action_kind="patch",
            machine=action.machine,
            duration_ms=duration_ms,
            timestamp=ts,
        )
    if isinstance(action, Firewall):
        return DefensePlaced(
            turn=turn,
            action_kind="firewall",
            edge_a=action.a,
            edge_b=action.b,
            duration_ms=duration_ms,
            timestamp=ts,
        )
    if isinstance(action, Honeypot):
        return DefensePlaced(
            turn=turn,
            action_kind="honeypot",
            machine=action.machine,
            duration_ms=duration_ms,
            timestamp=ts,
        )
    return DefensePlaced(turn=turn, action_kind="pass", duration_ms=duration_ms, timestamp=ts)


def defense_event_to_action(event: DefensePlaced) -> Action:
    if event.action_kind == "patch":
        assert event.machine is not None
        return Patch(event.machine)
    if event.action_kind == "firewall":
        assert event.edge_a is not None and event.edge_b is not None
        return Firewall(event.edge_a, event.edge_b)
    if event.action_kind == "honeypot":
        assert event.machine is not None
        return Honeypot(event.machine)
    if event.action_kind == "pass":
        return Pass()
    raise ValueError(f"unknown defense action_kind: {event.action_kind!r}")


def attacker_event_to_action(event: AttackerMoved) -> Advance:
    return Advance(event.target)


def event_to_dict(event: GameEvent) -> dict[str, Any]:
    if isinstance(event, GameStarted):
        return {
            "type": "GameStarted",
            "version": event.version,
            "timestamp": event.timestamp,
            "seed": event.seed,
            "size": event.size,
            "entry": event.entry,
            "jewel": event.jewel,
            "hardness": list(event.hardness),
            "edges": [[a, b] for a, b in event.edges],
        }
    if isinstance(event, AttackerMoved):
        return {
            "type": "AttackerMoved",
            "version": event.version,
            "timestamp": event.timestamp,
            "turn": event.turn,
            "target": event.target,
            "duration_ms": event.duration_ms,
        }
    if isinstance(event, DefensePlaced):
        return {
            "type": "DefensePlaced",
            "version": event.version,
            "timestamp": event.timestamp,
            "turn": event.turn,
            "action_kind": event.action_kind,
            "machine": event.machine,
            "edge_a": event.edge_a,
            "edge_b": event.edge_b,
            "duration_ms": event.duration_ms,
        }
    if isinstance(event, AlertsRaised):
        return {
            "type": "AlertsRaised",
            "version": event.version,
            "timestamp": event.timestamp,
            "turn": event.turn,
            "alerts": list(event.alerts),
        }
    if isinstance(event, BeliefUpdated):
        return {
            "type": "BeliefUpdated",
            "version": event.version,
            "timestamp": event.timestamp,
            "turn": event.turn,
            "probabilities": list(event.probabilities),
        }
    return {
        "type": "GameEnded",
        "version": event.version,
        "timestamp": event.timestamp,
        "turn": event.turn,
        "outcome": event.outcome,
        "utility": event.utility,
    }


def event_from_dict(data: dict[str, Any]) -> GameEvent:
    kind = str(data["type"])
    ver = int(data.get("version", 1))
    ts = float(data.get("timestamp", 0.0))
    if kind == "GameStarted":
        return GameStarted(
            seed=int(data["seed"]),
            size=int(data["size"]),
            entry=int(data["entry"]),
            jewel=int(data["jewel"]),
            hardness=tuple(int(h) for h in data["hardness"]),
            edges=tuple(make_edge(int(a), int(b)) for a, b in data["edges"]),
            version=ver,
            timestamp=ts,
        )
    if kind == "AttackerMoved":
        return AttackerMoved(
            turn=int(data["turn"]),
            target=int(data["target"]),
            duration_ms=float(data.get("duration_ms", 0.0)),
            version=ver,
            timestamp=ts,
        )
    if kind == "DefensePlaced":
        return DefensePlaced(
            turn=int(data["turn"]),
            action_kind=str(data["action_kind"]),
            machine=None if data.get("machine") is None else int(data["machine"]),
            edge_a=None if data.get("edge_a") is None else int(data["edge_a"]),
            edge_b=None if data.get("edge_b") is None else int(data["edge_b"]),
            duration_ms=float(data.get("duration_ms", 0.0)),
            version=ver,
            timestamp=ts,
        )
    if kind == "AlertsRaised":
        return AlertsRaised(
            turn=int(data["turn"]),
            alerts=tuple(bool(a) for a in data["alerts"]),
            version=ver,
            timestamp=ts,
        )
    if kind == "BeliefUpdated":
        return BeliefUpdated(
            turn=int(data["turn"]),
            probabilities=tuple(float(p) for p in data["probabilities"]),
            version=ver,
            timestamp=ts,
        )
    if kind == "GameEnded":
        return GameEnded(
            turn=int(data["turn"]),
            outcome=str(data["outcome"]),
            utility=int(data["utility"]),
            version=ver,
            timestamp=ts,
        )
    raise ValueError(f"unknown event type: {kind!r}")


def network_from_started(event: GameStarted) -> Network:
    return Network.from_edges(
        event.size,
        event.edges,
        event.hardness,
        event.entry,
        event.jewel,
    )


def make_game_ended(turn: int, result: Outcome, utility: int) -> GameEnded:
    return GameEnded(turn=turn, outcome=result.value, utility=utility, timestamp=_now())
