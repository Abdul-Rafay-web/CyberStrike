from pathlib import Path

import pytest

from agents.greedy_attacker import GreedyAttacker
from agents.simple_defenders import StaticDefender
from config import CONFIG
from core.actions import Action, Firewall, Honeypot, Pass, Patch
from core.events import (
    AlertsRaised,
    AttackerMoved,
    BeliefUpdated,
    GameEnded,
    GameEvent,
    event_from_dict,
    event_to_dict,
    make_defense_placed,
    make_game_started,
)
from core.eventstore import EventStore, build_events_from_steps, replay_events
from core.game import play_game
from core.generator import generate_network
from core.state import GameState
from utils.rng import STREAM_SENSORS, make_rng


def test_all_six_event_types_round_trip_through_json_dicts() -> None:
    net = generate_network(1)
    events: list[GameEvent] = [
        make_game_started(1, net),
        AttackerMoved(turn=1, target=net.neighbours(net.entry)[0], duration_ms=1.5),
        make_defense_placed(1, Patch(2), duration_ms=0.8),
        make_defense_placed(2, Firewall(*net.edges()[0]), duration_ms=0.4),
        make_defense_placed(3, Honeypot(3), duration_ms=0.3),
        make_defense_placed(4, Pass(), duration_ms=0.1),
        AlertsRaised(turn=1, alerts=(True,) + (False,) * (net.size - 1)),
        BeliefUpdated(turn=1, probabilities=(1.0,) + (0.0,) * (net.size - 1)),
        GameEnded(turn=8, outcome="attacker_wins", utility=920),
    ]
    for ev in events:
        assert event_from_dict(event_to_dict(ev)) == ev


def _record_game_steps(seed: int) -> tuple[list[tuple[GameState, Action, GameState]], GameState]:
    net = generate_network(seed)
    steps: list[tuple[GameState, Action, GameState]] = []

    def on_step(before: GameState, action: Action, after: GameState) -> None:
        steps.append((before, action, after))

    play_game(
        net,
        GreedyAttacker(),
        StaticDefender(),
        make_rng(seed, STREAM_SENSORS),
        CONFIG,
        on_step=on_step,
    )
    return steps, steps[-1][2]


def test_event_store_replays_finished_games_to_identical_final_state(tmp_path: Path) -> None:
    for seed in range(5):
        net = generate_network(seed)
        steps, final_state = _record_game_steps(seed)
        events = build_events_from_steps(seed, net, steps, CONFIG.game)
        store = EventStore(tmp_path / f"game_{seed}.jsonl")
        store.append_many(events)

        reloaded = store.read_all()
        assert len(reloaded) == len(events)
        replayed = replay_events(reloaded, CONFIG.game)
        assert replayed == final_state


def test_replay_rejects_missing_game_started() -> None:
    with pytest.raises(ValueError):
        replay_events([AttackerMoved(turn=1, target=1)])
