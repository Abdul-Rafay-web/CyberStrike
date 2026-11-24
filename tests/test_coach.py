from agents.coach import analyze_defender_game
from agents.greedy_attacker import GreedyAttacker
from agents.simple_defenders import PassiveDefender
from config import CONFIG
from core.actions import Action
from core.game import play_game
from core.generator import generate_network
from core.state import GameState
from utils.rng import STREAM_SENSORS, make_rng


def test_coach_produces_actionable_advice_for_passive_defender() -> None:
    recorded: list[tuple[GameState, Action]] = []

    def on_step(before: GameState, action: Action, after: GameState) -> None:
        del after
        recorded.append((before, action))

    play_game(
        generate_network(42),
        GreedyAttacker(),
        PassiveDefender(),
        make_rng(42, STREAM_SENSORS),
        CONFIG,
        on_step=on_step,
    )
    report = analyze_defender_game(recorded)
    assert report.decisions_reviewed >= 1
    assert len(report.advice) >= 1
    assert report.advice[0].turns_gained >= 1
    assert "COACH:" in report.headline


def test_coach_handles_empty_history_cleanly() -> None:
    report = analyze_defender_game([])
    assert report.decisions_reviewed == 0
    assert report.advice == ()
    assert "No Defender decisions" in report.headline

