from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

from config import CONFIG, GameConfig
from core.actions import Action, Advance
from core.events import (
    AttackerMoved,
    DefensePlaced,
    GameEvent,
    GameStarted,
    attacker_event_to_action,
    defense_event_to_action,
    event_from_dict,
    event_to_dict,
    make_defense_placed,
    make_game_ended,
    make_game_started,
    network_from_started,
)
from core.network import Network
from core.rules import apply, outcome, utility
from core.state import GameState, Side, new_game


class EventStore:
    """Append-only JSON Lines event log (one JSON object per line)."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, event: GameEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event_to_dict(event)) + "\n")

    def append_many(self, events: Iterable[GameEvent]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event_to_dict(event)) + "\n")

    def read_all(self) -> list[GameEvent]:
        if not self.path.exists():
            return []
        events: list[GameEvent] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    events.append(event_from_dict(json.loads(stripped)))
        return events


def replay_events(events: Sequence[GameEvent], config: GameConfig = CONFIG.game) -> GameState:
    """Rebuild exact `GameState` by folding an event stream from `GameStarted`."""
    if not events or not isinstance(events[0], GameStarted):
        raise ValueError("event stream must begin with GameStarted")
    network = network_from_started(events[0])
    state = new_game(network, config)
    for event in events[1:]:
        if isinstance(event, AttackerMoved):
            state = apply(state, attacker_event_to_action(event), config)
        elif isinstance(event, DefensePlaced):
            state = apply(state, defense_event_to_action(event), config)
    return state


def build_events_from_steps(
    seed: int,
    network: Network,
    steps: Sequence[tuple[GameState, Action, GameState]],
    config: GameConfig = CONFIG.game,
) -> list[GameEvent]:
    """Convert recorded `(before, action, after)` game steps into a complete event stream."""
    events: list[GameEvent] = [make_game_started(seed, network)]
    final_state = new_game(network, config)
    for before, action, after in steps:
        if before.side is Side.ATTACKER and isinstance(action, Advance):
            events.append(
                AttackerMoved(turn=before.turn, target=action.target, timestamp=time.time())
            )
        elif before.side is Side.DEFENDER:
            events.append(make_defense_placed(before.turn, action))
        final_state = after
    res = outcome(final_state, config)
    events.append(
        make_game_ended(
            min(final_state.turn, config.turn_limit),
            res,
            utility(final_state, config),
        )
    )
    return events
