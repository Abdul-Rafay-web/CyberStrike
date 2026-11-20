from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import pytest

from config import CONFIG
from core.actions import Advance, Firewall, Honeypot, Pass, Patch
from core.chokepoints import chokepoints
from core.generator import generate_network
from core.observation import observe_attacker, observe_defender
from core.rules import is_terminal
from core.state import Side
from ui.app import CyberStrikeApp, ScreenId, run_window
from ui.layout import (
    MAP_PRESETS,
    build_canonical_13_network,
    compute_network_layout,
    is_canonical_13_network,
    project_node_to_screen,
)
from ui.panels import DrawerTab
from ui.screens import AtlasTab, SetupConfig, SpriteBgMode


@pytest.fixture(autouse=True)
def _pygame_session() -> None:
    if not pygame.get_init():
        pygame.init()


def test_canonical_13_network_is_valid_and_chokepoint_free() -> None:
    net = build_canonical_13_network()
    assert net.size == 13
    assert len(net.edges()) == 26
    assert net.is_connected()
    assert not chokepoints(net)
    assert is_canonical_13_network(net)


def test_all_nodes_and_rings_stay_within_battlefield_border() -> None:
    rect = pygame.Rect(12, 82, 920, 478)
    networks = [
        build_canonical_13_network(),
        *[generate_network(seed=100 + sz, size=sz, config=CONFIG) for sz in (10, 15, 20, 25)],
    ]
    for net in networks:
        metas = compute_network_layout(net)
        assert len(metas) == net.size
        pts: list[tuple[int, int]] = []
        for meta in metas:
            px, py = project_node_to_screen(meta.nx, meta.ny, rect)
            pts.append((px, py))
            # Outer ring radius max is 56px horizontal, 30px vertical around base
            assert rect.left + 32 <= px <= rect.right - 32
            assert rect.top + 36 <= py <= rect.bottom - 36

        # Ensure nodes do not collapse onto identical coordinates
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dx = pts[i][0] - pts[j][0]
                dy = pts[i][1] - pts[j][1]
                assert dx * dx + dy * dy >= 35 * 35


def test_no_fast_al2002_or_bse5a_strings_in_ui() -> None:
    ui_dir = Path(__file__).resolve().parent.parent / "ui"
    for preset in MAP_PRESETS:
        assert "fast-al2002" not in preset.name.lower()
        assert "bse-5a" not in preset.name.lower()

    for py_file in ui_dir.glob("*.py"):
        content = py_file.read_text(encoding="utf-8").lower()
        assert "fast-al2002" not in content, f"Unexpected legacy name in {py_file}"
        assert "fast_al2002" not in content, f"Unexpected legacy name in {py_file}"
        assert "bse-5a" not in content, f"Unexpected legacy name in {py_file}"


def test_ui_does_not_import_ai_backend_packages() -> None:
    ui_dir = Path(__file__).resolve().parent.parent / "ui"
    forbidden = ("search.", "prob.", "csp.", "ml.", "evo.", "db.", "runtime.")
    for py_file in ui_dir.glob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        for prefix in forbidden:
            assert f"from {prefix}" not in text
            assert f"import {prefix}" not in text


def test_app_renders_all_screens_and_modals_headless() -> None:
    surf = pygame.Surface(CONFIG.theme.window_size)
    app = CyberStrikeApp(seed=42, start_screen="main_menu")

    screens: tuple[ScreenId, ...] = (
        "main_menu",
        "game_setup",
        "defender_game",
        "attacker_game",
        "spectator_game",
        "replay_viewer",
        "map_library",
        "asset_atlas",
    )
    for scr in screens:
        app.screen_id = scr
        app.update(0.03)
        app.render(surf)

    tabs: tuple[AtlasTab, ...] = ("sprites", "ui_kit", "boards")
    bgs: tuple[SpriteBgMode, ...] = ("obsidian", "checker", "card")
    app.screen_id = "asset_atlas"
    for t in tabs:
        app.atlas_tab = t
        for bg in bgs:
            app.atlas_bg = bg
            app.render(surf)

    app.screen_id = "defender_game"
    app.show_tutorial = True
    app.show_drawer = True
    drawer_tabs: tuple[DrawerTab, ...] = ("peas", "sensors", "observation", "topology")
    for dt in drawer_tabs:
        app.drawer_tab = dt
        app.render(surf)

    app.is_paused = True
    app.render(surf)
    app.is_paused = False

    app.show_settings = True
    app.render(surf)
    app.show_settings = False

    app.game_mode = "human_vs_human"
    app.hotseat_curtain = Side.ATTACKER
    app.render(surf)


def test_defender_and_attacker_gameplay_and_picking() -> None:
    surf = pygame.Surface(CONFIG.theme.window_size)
    app = CyberStrikeApp(seed=42, start_screen="defender_game")
    app.render(surf)
    assert app.state.side is Side.DEFENDER

    # Test machine picking at projected coordinates of node 8 (SVR-12)
    meta_8 = app.metas[8]
    px, py = project_node_to_screen(
        meta_8.nx,
        meta_8.ny,
        app._battlefield_rect,
        pad_left=app.battlefield.PAD_LEFT,
        pad_right=app.battlefield.PAD_RIGHT,
        pad_top=app.battlefield.PAD_TOP,
        pad_bottom=app.battlefield.PAD_BOTTOM,
    )
    picked = app.battlefield.pick_machine(
        (px, py), app._battlefield_rect, app.metas, app.state.patched
    )
    assert picked == 8

    # Apply Defender actions (Patch, Firewall, Honeypot, Pass) and verify state transitions
    app.submit_player_action(Patch(8))
    app.flush_pending_ai_reply()
    assert 8 in app.state.patched
    assert app.state.stock.patches == CONFIG.game.stock.patches - 1

    edge_a, edge_b = app.state.network.edges()[0]
    app.submit_player_action(Firewall(edge_a, edge_b))
    app.flush_pending_ai_reply()
    assert (edge_a, edge_b) in app.state.firewalls

    if not is_terminal(app.state, CONFIG.game):
        app.submit_player_action(Honeypot(4))
        app.flush_pending_ai_reply()
        assert 4 in app.state.honeypots

    if not is_terminal(app.state, CONFIG.game):
        prev_turn = app.state.turn
        app.submit_player_action(Pass())
        app.flush_pending_ai_reply()
        assert app.state.turn >= prev_turn

    assert len(app.history) >= 3


def test_attacker_perspective_and_escape_pause() -> None:
    app = CyberStrikeApp(seed=42, start_screen="attacker_game")
    app.start_operation(
        SetupConfig(mode="human_vs_ai", side=Side.ATTACKER, seed=42),
        target_screen="attacker_game",
    )
    assert app.state.side is Side.ATTACKER
    app.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE}))
    assert app.is_paused is True
    app.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE}))
    assert app.is_paused is False


def test_illegal_action_is_caught_and_logged() -> None:
    app = CyberStrikeApp(seed=42, start_screen="defender_game")
    assert app.state.side is Side.DEFENDER

    # Attacker-only action during Defender turn must log [ILLEGAL MOVE] and not crash
    app.submit_player_action(Advance(0))
    assert app.logs
    assert "[ILLEGAL MOVE]" in app.logs[-1].text

    # Patching node 8 twice must reject the second attempt cleanly without spending extra stock
    app.submit_player_action(Patch(8))
    app.flush_pending_ai_reply()
    stock_after_first = app.state.stock.patches
    if not is_terminal(app.state, CONFIG.game):
        app.submit_player_action(Patch(8))
        assert app.state.stock.patches == stock_after_first
        assert "[ILLEGAL MOVE]" in app.logs[-1].text


def test_honeypots_stay_hidden_in_attacker_and_defender_observation_isolation() -> None:
    app = CyberStrikeApp(seed=42, start_screen="defender_game")
    app.submit_player_action(Honeypot(4))
    assert 4 in app.state.honeypots

    obs_atk = observe_attacker(app.state, CONFIG.game)
    atk_scene = app.battlefield._build_scene("attacker", True, None, obs_atk, None)
    assert atk_scene.honeypots == frozenset()
    assert 4 not in atk_scene.triggered

    obs_def = observe_defender(app.state, app.alerts, CONFIG.game)
    def_scene = app.battlefield._build_scene("defender", False, obs_def, None, None)
    assert def_scene.attacker_pos is None
    assert def_scene.can_see_intruder is False
    assert 4 in def_scene.honeypots


def test_twenty_machine_render_performance() -> None:
    screen = pygame.display.set_mode(CONFIG.theme.window_size)
    app = CyberStrikeApp(seed=42, start_screen="defender_game")
    app.vault.warm_up()
    app.start_operation(
        SetupConfig(
            mode="human_vs_ai",
            side=Side.DEFENDER,
            seed=42,
            map_size=20,
            sector_name="TEST_20_NODE_NET",
            use_canonical_13=False,
        ),
        target_screen="defender_game",
    )
    assert app.state.network.size == 20

    # Warm up frame-level caches before timing steady-state 20-node rendering
    for _ in range(3):
        app.update(0.016)
        app.render(screen)

    t0 = time.perf_counter()
    frames = 60
    for _ in range(frames):
        app.update(0.016)
        app.render(screen)
    avg_frame_sec = (time.perf_counter() - t0) / frames
    assert avg_frame_sec < 0.03


def test_run_window_headless_smoke() -> None:
    run_window(seed=42, start_screen="main_menu", max_frames=2)
