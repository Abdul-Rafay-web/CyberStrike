from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import pytest

from agents.belief_csp_defender import BeliefCSPDefender
from agents.factory import build_ai_tutorial_caption, build_match_agents
from agents.minimax_attacker import MinimaxAttacker
from config import CONFIG
from core.actions import Pass
from core.game import play_game
from core.generator import generate_network
from core.state import Side
from ml.pipeline import load_sensor, train_sensor
from ml.sensor import ClassifierSensor
from ui.app import CyberStrikeApp
from ui.screens import SetupConfig
from utils.paths import DATA_DIR
from utils.rng import STREAM_AGENTS, STREAM_SENSORS, make_rng


@pytest.fixture(autouse=True)
def _pygame_init() -> None:
    if not pygame.get_init():
        pygame.init()


def test_all_four_modes_launch_cleanly_from_app() -> None:
    surf = pygame.Surface(CONFIG.theme.window_size)
    app = CyberStrikeApp(seed=42, start_screen="main_menu")

    # 1. Human vs AI (Defender facing Minimax + Naive Bayes sensor + Bayesian belief heatmap)
    app.start_operation(
        SetupConfig(mode="human_vs_ai", side=Side.DEFENDER, seed=42, difficulty="standard"),
        target_screen="defender_game",
    )
    app.render(surf)
    assert str(app.screen_id) == "defender_game"
    assert isinstance(app.ai_attacker, MinimaxAttacker)

    # 2. Human vs AI (Attacker facing Belief + CSP Defender)
    app.start_operation(
        SetupConfig(mode="human_vs_ai", side=Side.ATTACKER, seed=42, difficulty="standard"),
        target_screen="attacker_game",
    )
    app.render(surf)
    assert str(app.screen_id) == "attacker_game"
    assert isinstance(app.ai_defender, BeliefCSPDefender)

    # 3. AI vs AI Spectator watch mode
    app.start_operation(
        SetupConfig(mode="ai_vs_ai", side=Side.DEFENDER, seed=42, difficulty="easy"),
        target_screen="spectator_game",
    )
    app.step_spectator_once()
    app.render(surf)
    assert str(app.screen_id) == "spectator_game"
    assert app.plies >= 1

    # 4. Human vs Human hot-seat + Tutorial mode
    app.start_operation(
        SetupConfig(mode="human_vs_human", side=Side.DEFENDER, seed=42),
        target_screen="defender_game",
    )
    app.show_tutorial = True
    app.render(surf)
    assert app.show_tutorial is True


def test_tutorial_captions_mention_every_ai_technique() -> None:
    rng = make_rng(42, STREAM_AGENTS)
    bundle = build_match_agents("standard", rng)
    net = generate_network(42)
    codes = tuple(f"SVR-{i:02d}" for i in range(net.size))
    belief = tuple(1.0 / net.size for _ in range(net.size))
    alerts = (True,) + (False,) * (net.size - 1)

    t1_title, t1_body = build_ai_tutorial_caption(
        net, 1, Side.DEFENDER, belief, alerts, bundle.attacker, bundle.defender, codes
    )
    t2_title, t2_body = build_ai_tutorial_caption(
        net, 2, Side.ATTACKER, belief, alerts, bundle.attacker, bundle.defender, codes
    )
    t3_title, t3_body = build_ai_tutorial_caption(
        net, 2, Side.DEFENDER, belief, alerts, bundle.attacker, bundle.defender, codes
    )
    combined = f"{t1_title} {t1_body} {t2_title} {t2_body} {t3_title} {t3_body}".lower()
    for term in ("a*", "minimax", "alpha-beta", "naive bayes", "bayesian", "csp"):
        assert term in combined, f"Expected {term!r} in tutorial captions"


def test_replay_save_and_load_preserves_game_state(tmp_path: Path) -> None:
    app = CyberStrikeApp(seed=42, start_screen="defender_game")
    for _ in range(3):
        app.submit_player_action(Pass())
        app.flush_pending_ai_reply()

    replay_path = tmp_path / "saved_game.jsonl"
    app.save_replay(replay_path)
    assert replay_path.exists()
    original_state = app.state

    reloaded = CyberStrikeApp(seed=99, start_screen="main_menu")
    assert reloaded.load_replay(replay_path) is True
    assert reloaded.screen_id == "replay_viewer"
    assert reloaded.history[-1].state == original_state


def test_100_ai_vs_ai_games_finish_cleanly_without_crash_or_freeze() -> None:
    model_path = DATA_DIR / "sensor_model.json"
    trained = load_sensor(model_path) if model_path.exists() else train_sensor()
    sensor = ClassifierSensor.from_trained(trained)

    for seed in range(100):
        net = generate_network(seed)
        result = play_game(
            net,
            MinimaxAttacker(max_depth=2, time_budget=0.15),
            BeliefCSPDefender(rates=trained.rates),
            make_rng(seed, STREAM_SENSORS),
            CONFIG,
            sensor=sensor,
        )
        assert 1 <= result.turns <= CONFIG.game.turn_limit
        assert result.plies >= 1
