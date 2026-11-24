from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import pygame

from agents.belief_defender import BeliefSession, create_belief_session
from agents.random_agents import RandomAttacker, RandomDefender
from config import CONFIG
from core.actions import Action, Advance, Firewall, Honeypot, Pass, Patch
from core.generator import generate_network
from core.network import Network, make_edge
from core.observation import observe_attacker, observe_defender
from core.rules import IllegalActionError, Outcome, apply, is_terminal, outcome, utility
from core.sensors import sample_alerts
from core.state import GameState, Side, new_game
from ui.battlefield import BattlefieldView, ToolMode, ViewMode
from ui.draw import ButtonSpec
from ui.layout import (
    MAP_PRESETS,
    MapPreset,
    NodeVisualMeta,
    all_map_presets,
    build_canonical_13_network,
    compute_network_layout,
    load_network_from_json,
)
from ui.panels import (
    AuditEntry,
    DrawerTab,
    GameSettingsState,
    derive_attack_stage_from_breached,
    draw_attacker_command_panel,
    draw_audit_log_panel,
    draw_bottom_diagnostics_tray,
    draw_defender_command_panel,
    draw_hotseat_curtain_modal,
    draw_pause_modal,
    draw_replay_bar,
    draw_results_modal,
    draw_settings_modal,
    draw_side_panels_drawer,
    draw_spectator_bar,
    draw_top_hud_header,
    draw_tutorial_card,
)
from ui.screens import (
    AtlasTab,
    SetupConfig,
    SpriteBgMode,
    draw_asset_atlas_screen,
    draw_game_setup_screen,
    draw_main_menu_screen,
    draw_map_library_screen,
)
from ui.sound import TacticalSoundEngine
from ui.theme import AssetVault, FontBook, ThemePalette, get_palette
from utils.rng import STREAM_AGENTS, STREAM_SENSORS, make_rng

log = logging.getLogger(__name__)

ScreenId = Literal[
    "main_menu",
    "game_setup",
    "defender_game",
    "attacker_game",
    "spectator_game",
    "replay_viewer",
    "map_library",
    "asset_atlas",
]


@dataclass(frozen=True, slots=True)
class HistoryFrame:
    """Immutable snapshot stored for replay timeline scrubbing."""

    state: GameState
    alerts: tuple[bool, ...]
    logs: list[AuditEntry]
    belief: tuple[float, ...] = ()


class CyberStrikeApp:
    """Main Pygame UI controller wired strictly to Phase 1 core/ rules and agents/."""

    def __init__(
        self,
        seed: int = CONFIG.seeds.master_seed,
        start_screen: ScreenId = "main_menu",
    ) -> None:
        self.screen_id: ScreenId = start_screen
        self.prev_non_atlas_screen: ScreenId = "main_menu"
        self.running: bool = True

        self.settings = GameSettingsState(target_fps=CONFIG.theme.fps)
        self.palette: ThemePalette = get_palette(self.settings.theme)
        self.fonts = FontBook()
        self.vault = AssetVault()
        self.sfx = TacticalSoundEngine()
        self.sfx.configure(self.settings.sound_enabled, self.settings.master_volume)
        self.battlefield = BattlefieldView()

        self.setup_cfg = SetupConfig(seed=seed)
        self.selected_map_preset: MapPreset = MAP_PRESETS[0]

        # Preview cache for Setup / Map Library
        self._preview_cache_key: tuple[int, int, bool] | None = None
        self._preview_net: Network = build_canonical_13_network()
        self._preview_metas: tuple[NodeVisualMeta, ...] = compute_network_layout(self._preview_net)

        # UI modal / drawer states
        self.is_paused: bool = False
        self.show_settings: bool = False
        self.show_tutorial: bool = False
        self.show_drawer: bool = False
        self.show_quit_dialog: bool = False
        self.drawer_tab: DrawerTab = "peas"
        self.atlas_tab: AtlasTab = "sprites"
        self.atlas_bg: SpriteBgMode = "obsidian"

        # Active game runtime state
        self.game_mode: Literal["human_vs_ai", "ai_vs_ai", "human_vs_human"] = "human_vs_ai"
        self.player_side: Side = Side.DEFENDER
        self.active_seed: int = seed
        self.active_sector: str = "INTRA_NET_ALPHA_09"
        self.active_tool: ToolMode = None
        self.selected_node: int | None = None
        self.turn_seconds_left: float = 102.0
        self.plies: int = 0

        self.spectator_playing: bool = False
        self._spectator_timer: float = 0.0
        self._ai_reply_delay: float = 0.0
        self.replay_index: int = 0
        self.replay_playing: bool = False
        self._replay_timer: float = 0.0
        self.hotseat_curtain: Side | None = None

        self._active_buttons: list[ButtonSpec] = []
        self._hovered_btn_id: str | None = None
        self._battlefield_rect: pygame.Rect = pygame.Rect(12, 82, 920, 478)
        self._setup_slider_rect: pygame.Rect = pygame.Rect(0, 0, 0, 0)
        self._settings_vol_rect: pygame.Rect = pygame.Rect(0, 0, 0, 0)
        self._replay_track_rect: pygame.Rect = pygame.Rect(0, 0, 0, 0)
        self._dragging_slider: Literal["none", "setup_size", "settings_vol", "replay"] = "none"

        # Initialize default match state ready for quick launch
        self.start_operation(self.setup_cfg, target_screen=start_screen)

    def _build_network(
        self,
        size: int,
        seed: int,
        canonical: bool,
        evolved_map_path: str | None = None,
    ) -> Network:
        if evolved_map_path is not None:
            return load_network_from_json(evolved_map_path)
        if canonical and size == 13:
            return build_canonical_13_network()
        return generate_network(seed, size=size, config=CONFIG)

    def _get_preview(
        self, size: int, seed: int, canonical: bool
    ) -> tuple[Network, tuple[NodeVisualMeta, ...]]:
        key = (size, seed, canonical)
        if self._preview_cache_key != key:
            self._preview_net = self._build_network(size, seed, canonical)
            self._preview_metas = compute_network_layout(self._preview_net)
            self._preview_cache_key = key
        return self._preview_net, self._preview_metas

    def start_operation(self, cfg: SetupConfig, target_screen: ScreenId | None = None) -> None:
        """Initialize a fresh Phase 1 core.state.GameState for gameplay."""
        self.game_mode = cfg.mode
        self.player_side = cfg.side
        self.active_seed = cfg.seed
        self.active_sector = cfg.sector_name
        self.active_tool = None
        self.is_paused = False
        self.hotseat_curtain = None
        self.turn_seconds_left = 102.0
        self.plies = 0
        self._ai_reply_delay = 0.0
        self._spectator_timer = 0.0

        network = self._build_network(
            cfg.map_size, cfg.seed, cfg.use_canonical_13, cfg.evolved_map_path
        )
        self.metas = compute_network_layout(network)
        self.state = new_game(network, CONFIG.game)
        self.sensor_rng = make_rng(cfg.seed, STREAM_SENSORS)
        agent_rng = make_rng(cfg.seed, STREAM_AGENTS)
        self.ai_attacker = RandomAttacker(agent_rng)
        self.ai_defender = RandomDefender(agent_rng)

        self.selected_node = network.entry
        self.battlefield.reset_animations(self.state.attacker_pos)
        self.battlefield.peek_ref_mode = "none"

        self.logs: list[AuditEntry] = [
            AuditEntry(
                1,
                (
                    f"[T01] OPERATION STARTED ON {cfg.sector_name} "
                    f"(SEED #{cfg.seed}, {network.size} NODES)"
                ),
                "cyan",
            ),
            AuditEntry(
                1,
                f"[T01] INTRUSION DETECTED AT {self.metas[network.entry].code}",
                "info",
            ),
        ]
        self.alerts: tuple[bool, ...] = sample_alerts(self.state, self.sensor_rng, CONFIG.sensors)
        self.belief_session: BeliefSession = create_belief_session(network, CONFIG.sensors)
        self.belief: tuple[float, ...] = self.belief_session.probabilities
        self.history: list[HistoryFrame] = [
            HistoryFrame(self.state, self.alerts, list(self.logs), self.belief)
        ]
        self.replay_index = 0
        self.replay_playing = False

        # In Defender Human-vs-AI mode, core.new_game begins with Attacker to move;
        # run the opening Attacker ply so Defender immediately gets Turn 1 alerts.
        if (
            cfg.mode == "human_vs_ai"
            and cfg.side is Side.DEFENDER
            and self.state.side is Side.ATTACKER
        ):
            self._execute_ai_ply()

        if target_screen is not None:
            self.screen_id = target_screen
        elif cfg.mode == "ai_vs_ai":
            self.screen_id = "spectator_game"
            self.spectator_playing = True
        elif cfg.mode == "human_vs_human":
            self.screen_id = (
                "attacker_game" if self.state.side is Side.ATTACKER else "defender_game"
            )
            self.spectator_playing = False
        elif cfg.side is Side.ATTACKER:
            self.screen_id = "attacker_game"
            self.spectator_playing = False
        else:
            self.screen_id = "defender_game"
            self.spectator_playing = False

    def _log_action(self, before: GameState, action: Action, after: GameState) -> None:
        tag = f"[T{before.turn:02d}]"
        if isinstance(action, Advance):
            t_code = self.metas[action.target].code
            if len(after.triggered) > len(before.triggered):
                self.logs.append(
                    AuditEntry(
                        before.turn, f"{tag} HONEYPOT TRIGGERED AT {t_code}! (+2T DELAY)", "violet"
                    )
                )
            elif after.breach_target is not None:
                self.logs.append(
                    AuditEntry(
                        before.turn,
                        f"{tag} INTRUDER PROBING {t_code} ({after.breach_left}T REMAINING)",
                        "amber",
                    )
                )
            else:
                self.logs.append(
                    AuditEntry(before.turn, f"{tag} INTRUDER ADVANCED TO {t_code}", "red")
                )
        elif isinstance(action, Patch):
            m_code = self.metas[action.machine].code
            self.logs.append(
                AuditEntry(before.turn, f"{tag} DEFENDER PATCHED {m_code} (+2 HARDNESS)", "cyan")
            )
        elif isinstance(action, Firewall):
            a_code = self.metas[action.a].code
            b_code = self.metas[action.b].code
            self.logs.append(
                AuditEntry(
                    before.turn, f"{tag} FIREWALL DEPLOYED ON {a_code} <-> {b_code}", "amber"
                )
            )
        elif isinstance(action, Honeypot):
            m_code = self.metas[action.machine].code
            self.logs.append(
                AuditEntry(before.turn, f"{tag} COVERT HONEYPOT ARMED AT {m_code}", "violet")
            )
        elif isinstance(action, Pass):
            self.logs.append(AuditEntry(before.turn, f"{tag} DEFENDER PASSED TURN", "info"))

    def _apply_and_record(self, action: Action) -> bool:
        if is_terminal(self.state, CONFIG.game):
            return False
        before = self.state
        try:
            after = apply(before, action, CONFIG.game)
        except IllegalActionError as exc:
            self.logs.append(AuditEntry(before.turn, f"[ILLEGAL MOVE] {exc}", "red"))
            self.active_tool = None
            return False
        self.state = after
        self.plies += 1
        self.turn_seconds_left = 102.0

        # Spawn visual ripple & SFX
        if isinstance(action, Patch):
            self.sfx.defense_placed("patch")
            self.battlefield.spawn_ripple(
                self._battlefield_rect, self.metas, action.machine, self.palette.cyan
            )
        elif isinstance(action, Firewall):
            self.sfx.defense_placed("firewall")
            self.battlefield.spawn_edge_ripple(
                self._battlefield_rect,
                self.metas,
                make_edge(action.a, action.b),
                self.palette.amber,
            )
        elif isinstance(action, Honeypot):
            self.sfx.defense_placed("honeypot")
            self.battlefield.spawn_ripple(
                self._battlefield_rect, self.metas, action.machine, self.palette.violet
            )
        elif isinstance(action, Advance):
            if len(after.triggered) > len(before.triggered):
                self.sfx.honeypot_alarm()
                self.battlefield.spawn_ripple(
                    self._battlefield_rect,
                    self.metas,
                    action.target,
                    self.palette.violet,
                    max_rx=88,
                )
            else:
                self.sfx.breach_step()

        self._log_action(before, action, after)

        # Sample fresh noisy sensor alerts when entering Defender turn
        if after.side is Side.DEFENDER and not is_terminal(after, CONFIG.game):
            self.alerts = sample_alerts(after, self.sensor_rng, CONFIG.sensors)
            self.belief = self.belief_session.step(
                observe_defender(after, self.alerts, CONFIG.game)
            )
            alert_codes = [self.metas[i].code for i, flag in enumerate(self.alerts) if flag]
            if alert_codes:
                self.logs.append(
                    AuditEntry(
                        after.turn,
                        f"[T{after.turn:02d}] SENSOR ALERT ON {', '.join(alert_codes[:4])}",
                        "red",
                    )
                )

        res = outcome(after, CONFIG.game)
        if res is Outcome.DEFENDER_WINS:
            self.sfx.victory()
            self.spectator_playing = False
            self.logs.append(
                AuditEntry(after.turn, "OPERATION COMPLETE — DEFENDER VICTORY", "cyan")
            )
        elif res is Outcome.ATTACKER_WINS:
            self.sfx.defeat()
            self.spectator_playing = False
            self.logs.append(
                AuditEntry(after.turn, "CROWN JEWEL BREACHED — ATTACKER VICTORY", "red")
            )

        self.history.append(HistoryFrame(self.state, self.alerts, list(self.logs), self.belief))
        self.replay_index = len(self.history) - 1
        return True

    def _execute_ai_ply(self) -> None:
        if is_terminal(self.state, CONFIG.game):
            return
        if self.state.side is Side.ATTACKER:
            obs_a = observe_attacker(self.state, CONFIG.game)
            act = self.ai_attacker.choose_action(obs_a)
            self._apply_and_record(act)
        else:
            obs_d = observe_defender(self.state, self.alerts, CONFIG.game)
            act = self.ai_defender.choose_action(obs_d)
            self._apply_and_record(act)

    def submit_player_action(self, action: Action) -> None:
        """Apply a human player's action and trigger AI reply or hot-seat curtain."""
        if is_terminal(self.state, CONFIG.game):
            return
        if not self._apply_and_record(action):
            return
        self.active_tool = None
        if is_terminal(self.state, CONFIG.game):
            self.hotseat_curtain = None
            return

        if self.game_mode == "human_vs_ai":
            self._ai_reply_delay = 0.28
        elif self.game_mode == "human_vs_human":
            self.hotseat_curtain = self.state.side

    def flush_pending_ai_reply(self) -> None:
        """Immediately execute any pending AI reply (useful in tests and turn transitions)."""
        if self.game_mode == "human_vs_ai" and not is_terminal(self.state, CONFIG.game):
            expected_ai_side = self.player_side.other
            if self.state.side is expected_ai_side:
                self._ai_reply_delay = 0.0
                self._execute_ai_ply()

    def _is_battlefield_screen(self) -> bool:
        return self.screen_id in (
            "defender_game",
            "attacker_game",
            "spectator_game",
            "replay_viewer",
        )

    def _displayed_frame(self) -> HistoryFrame:
        if self.screen_id == "replay_viewer" and self.history:
            idx = max(0, min(len(self.history) - 1, self.replay_index))
            return self.history[idx]
        return (
            self.history[-1] if self.history else HistoryFrame(self.state, self.alerts, self.logs)
        )

    def _compute_win_reason(self, state: GameState) -> tuple[int, str]:
        res = outcome(state, CONFIG.game)
        score = abs(utility(state, CONFIG.game))
        if res is Outcome.ATTACKER_WINS:
            return score, "Attacker successfully breached Crown Jewel (DB-01)."
        if state.turn > CONFIG.game.turn_limit:
            return (
                score,
                f"Turn limit ({CONFIG.game.turn_limit} turns) held with Crown Jewel intact.",
            )
        return score, "All infiltration routes to Crown Jewel (DB-01) severed by firewalls."

    def update(self, dt: float) -> None:
        """Advance timers, AI cadence, replay playback, and battlefield animations."""
        self.palette = get_palette(self.settings.theme)
        self.sfx.configure(self.settings.sound_enabled, self.settings.master_volume)

        frame = self._displayed_frame()
        can_see_pos = self.screen_id == "attacker_game" or (
            self.screen_id in ("spectator_game", "replay_viewer")
            and self.settings.god_mode_spectator
        )
        visible_pos = frame.state.attacker_pos if can_see_pos else None
        self.battlefield.update(dt, visible_pos, self.metas)

        if self.is_paused or self.show_settings or self.hotseat_curtain is not None:
            return

        if self.screen_id in (
            "defender_game",
            "attacker_game",
            "spectator_game",
        ) and not is_terminal(self.state, CONFIG.game):
            self.turn_seconds_left = max(1.0, self.turn_seconds_left - dt)

        # AI reply after human move
        if self._ai_reply_delay > 0.0:
            self._ai_reply_delay -= dt
            if self._ai_reply_delay <= 0.0:
                self._ai_reply_delay = 0.0
                self.flush_pending_ai_reply()

        # Spectator auto-play loop
        if (
            self.screen_id == "spectator_game"
            and self.spectator_playing
            and not is_terminal(self.state, CONFIG.game)
        ):
            interval = max(0.22, 0.85 / max(0.5, self.settings.ai_speed_multiplier))
            self._spectator_timer += dt
            if self._spectator_timer >= interval:
                self._spectator_timer = 0.0
                self._execute_ai_ply()

        # Replay stream playback
        if self.screen_id == "replay_viewer" and self.replay_playing:
            if self.replay_index >= len(self.history) - 1:
                self.replay_playing = False
            else:
                self._replay_timer += dt
                if self._replay_timer >= 0.75:
                    self._replay_timer = 0.0
                    self.replay_index += 1

    def render(self, surface: pygame.Surface, fps_actual: float = 60.0) -> None:
        """Render active screen and any modal overlays onto surface."""
        size = surface.get_size()
        w, h = size
        self._active_buttons = []
        self._setup_slider_rect = pygame.Rect(0, 0, 0, 0)
        self._settings_vol_rect = pygame.Rect(0, 0, 0, 0)
        self._replay_track_rect = pygame.Rect(0, 0, 0, 0)

        if self.screen_id == "main_menu":
            self._active_buttons = draw_main_menu_screen(
                surface,
                size,
                self.palette,
                self.fonts,
                self.vault,
                self._hovered_btn_id,
                self.show_quit_dialog,
            )
        elif self.screen_id == "game_setup":
            net, metas = self._get_preview(
                self.setup_cfg.map_size, self.setup_cfg.seed, self.setup_cfg.use_canonical_13
            )
            btns, slider_rect = draw_game_setup_screen(
                surface,
                size,
                self.palette,
                self.fonts,
                self.vault,
                self.setup_cfg,
                net,
                metas,
                self._hovered_btn_id,
            )
            self._active_buttons = btns
            self._setup_slider_rect = slider_rect
        elif self.screen_id == "map_library":
            net, metas = self._get_preview(
                self.selected_map_preset.node_count,
                self.selected_map_preset.seed,
                self.selected_map_preset.is_canonical_13,
            )
            self._active_buttons = draw_map_library_screen(
                surface,
                size,
                self.palette,
                self.fonts,
                self.vault,
                self.selected_map_preset,
                net,
                metas,
                self._hovered_btn_id,
            )
        elif self.screen_id == "asset_atlas":
            self._active_buttons = draw_asset_atlas_screen(
                surface,
                size,
                self.palette,
                self.fonts,
                self.vault,
                self.atlas_tab,
                self.atlas_bg,
                self._hovered_btn_id,
            )
        else:
            self._render_battlefield_screen(surface, w, h, fps_actual)

        # Modals layered above active screen
        if self.is_paused:
            self._active_buttons = draw_pause_modal(
                surface, size, self.palette, self.fonts, self.vault, self._hovered_btn_id
            )
        elif self.show_settings:
            btns, vol_rect = draw_settings_modal(
                surface,
                size,
                self.palette,
                self.fonts,
                self.vault,
                self.settings,
                self._hovered_btn_id,
            )
            self._active_buttons = btns
            self._settings_vol_rect = vol_rect
        elif (
            self._is_battlefield_screen()
            and self.screen_id != "replay_viewer"
            and is_terminal(self.state, CONFIG.game)
        ):
            score, reason = self._compute_win_reason(self.state)
            self._active_buttons = draw_results_modal(
                surface,
                size,
                self.palette,
                self.fonts,
                self.vault,
                self.state,
                self.active_seed,
                self.plies,
                score,
                reason,
                self._hovered_btn_id,
            )
        elif (
            self.game_mode == "human_vs_human"
            and self.hotseat_curtain is not None
            and self._is_battlefield_screen()
        ):
            self._active_buttons = draw_hotseat_curtain_modal(
                surface,
                size,
                self.palette,
                self.fonts,
                self.vault,
                self.hotseat_curtain,
                self._hovered_btn_id,
            )

    def _render_battlefield_screen(
        self, surface: pygame.Surface, w: int, h: int, fps_actual: float
    ) -> None:
        surface.fill(self.palette.bg)
        frame = self._displayed_frame()
        state = frame.state

        view_mode: ViewMode = (
            "attacker"
            if self.screen_id == "attacker_game"
            else "spectator"
            if self.screen_id in ("spectator_game", "replay_viewer")
            else "defender"
        )
        hud_mode = "replay" if self.screen_id == "replay_viewer" else view_mode

        obs_def = (
            observe_defender(state, frame.alerts, CONFIG.game)
            if view_mode in ("defender", "spectator")
            else None
        )
        obs_atk = observe_attacker(state, CONFIG.game) if view_mode == "attacker" else None
        spectator_state = (
            state if (view_mode == "spectator" and self.settings.god_mode_spectator) else None
        )

        if obs_atk is not None:
            stage_label = derive_attack_stage_from_breached(
                obs_atk.network, obs_atk.breached, obs_atk.breach_target
            )
            stock_rem = 7
            breached_visible: int | None = len(obs_atk.breached)
            node_count = obs_atk.network.size
            net_ref = obs_atk.network
            patched_ref = obs_atk.patched
            fw_count = len(obs_atk.firewalls)
            hp_ref: frozenset[int] | None = None
            trig_count = len(obs_atk.known_honeypots)
            alerts_ref: tuple[bool, ...] | None = None
        elif spectator_state is not None:
            stage_label = derive_attack_stage_from_breached(
                spectator_state.network,
                spectator_state.breached,
                spectator_state.breach_target,
            )
            stock_rem = (
                spectator_state.stock.patches
                + spectator_state.stock.firewalls
                + spectator_state.stock.honeypots
            )
            breached_visible = len(spectator_state.breached)
            node_count = spectator_state.network.size
            net_ref = spectator_state.network
            patched_ref = spectator_state.patched
            fw_count = len(spectator_state.firewalls)
            hp_ref = spectator_state.honeypots
            trig_count = len(spectator_state.triggered)
            alerts_ref = frame.alerts
        else:
            assert obs_def is not None
            stage_label = (
                "INTRUDER CONFIRMED" if obs_def.intruder_known_at is not None else "ACTIVE DEFENSE"
            )
            stock_rem = obs_def.stock.patches + obs_def.stock.firewalls + obs_def.stock.honeypots
            breached_visible = None
            node_count = obs_def.network.size
            net_ref = obs_def.network
            patched_ref = obs_def.patched
            fw_count = len(obs_def.firewalls)
            hp_ref = obs_def.honeypots
            trig_count = 1 if obs_def.intruder_known_at is not None else 0
            alerts_ref = obs_def.alerts

        buttons = draw_top_hud_header(
            surface,
            w,
            self.palette,
            self.fonts,
            self.vault,
            state.turn,
            node_count,
            stock_rem,
            stage_label,
            self.active_sector,
            self.active_seed,
            hud_mode,
            int(self.turn_seconds_left),
            self.settings,
            self.show_drawer,
            self.show_tutorial,
            self._hovered_btn_id,
        )

        top_y = 82
        if self.screen_id == "spectator_game":
            s_rect = pygame.Rect(12, top_y, w - 24, 34)
            buttons.extend(
                draw_spectator_bar(
                    surface,
                    s_rect,
                    self.palette,
                    self.fonts,
                    self.vault,
                    self.spectator_playing,
                    self.settings.ai_speed_multiplier,
                    self.settings.god_mode_spectator,
                    self._hovered_btn_id,
                )
            )
            top_y += 40
        elif self.screen_id == "replay_viewer":
            r_rect = pygame.Rect(12, top_y, w - 24, 52)
            r_btns, track_rect = draw_replay_bar(
                surface,
                r_rect,
                self.palette,
                self.fonts,
                self.vault,
                self.replay_index,
                len(self.history),
                state.turn,
                self.history[-1].state.turn if self.history else state.turn,
                self.replay_playing,
                self._hovered_btn_id,
            )
            buttons.extend(r_btns)
            self._replay_track_rect = track_rect
            top_y += 58

        if self.show_tutorial:
            t_rect = pygame.Rect(12, top_y, w - 24, 54)
            buttons.extend(
                draw_tutorial_card(
                    surface,
                    t_rect,
                    self.palette,
                    self.fonts,
                    self.vault,
                    state.turn,
                    state.side,
                    stage_label,
                    self._hovered_btn_id,
                )
            )
            top_y += 60

        tray_h = 26 if self.settings.show_diagnostics_tray else 0
        bottom_y = h - tray_h - 8
        if self.settings.show_diagnostics_tray:
            draw_bottom_diagnostics_tray(
                surface,
                pygame.Rect(0, h - tray_h, w, tray_h),
                self.palette,
                self.fonts,
                net_ref,
                state.turn,
                state.side,
                breached_visible,
                self.active_seed,
                self.plies,
                fps_actual,
            )

        right_w = 304
        left_w = max(400, w - right_w - 34)
        log_h = 118
        bf_rect = pygame.Rect(12, top_y, left_w, max(240, bottom_y - top_y - log_h - 10))
        self._battlefield_rect = bf_rect

        log_rect = pygame.Rect(12, bf_rect.bottom + 10, left_w, bottom_y - bf_rect.bottom - 10)
        cmd_rect = pygame.Rect(bf_rect.right + 12, top_y, right_w, bottom_y - top_y)

        eff_tool: ToolMode = self.active_tool if self.screen_id == "defender_game" else None
        self.battlefield.draw(
            surface,
            bf_rect,
            self.metas,
            self.palette,
            self.fonts,
            self.vault,
            view_mode,
            self.settings.god_mode_spectator,
            eff_tool,
            self.selected_node,
            obs_def,
            obs_atk,
            spectator_state,
            self.active_seed,
            self.settings.show_heatmap,
            self.settings.show_breach_trace,
            self.settings.show_hardness_badges,
            belief=frame.belief,
        )

        draw_audit_log_panel(surface, log_rect, self.palette, self.fonts, frame.logs)

        if self.screen_id == "attacker_game":
            assert obs_atk is not None
            buttons.extend(
                draw_attacker_command_panel(
                    surface,
                    cmd_rect,
                    self.palette,
                    self.fonts,
                    self.vault,
                    self.metas,
                    obs_atk,
                    self._hovered_btn_id,
                )
            )
        else:
            assert obs_def is not None
            buttons.extend(
                draw_defender_command_panel(
                    surface,
                    cmd_rect,
                    self.palette,
                    self.fonts,
                    self.vault,
                    self.metas,
                    obs_def,
                    eff_tool,
                    self.selected_node,
                    self._hovered_btn_id,
                )
            )

        if self.show_drawer:
            drawer_rect = pygame.Rect(
                bf_rect.right - 340, bf_rect.top + 8, 332, bf_rect.height - 16
            )
            buttons.extend(
                draw_side_panels_drawer(
                    surface,
                    drawer_rect,
                    self.palette,
                    self.fonts,
                    self.vault,
                    net_ref,
                    patched_ref,
                    fw_count,
                    hp_ref,
                    trig_count,
                    state.turn,
                    self.metas,
                    alerts_ref,
                    self.drawer_tab,
                    self.selected_node,
                    self._hovered_btn_id,
                )
            )

        self._active_buttons = buttons

    def handle_event(self, event: pygame.event.Event) -> None:
        """Process one Pygame keyboard or mouse event."""
        if event.type == pygame.QUIT:
            self.running = False
            return

        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self._handle_escape()
            return

        if event.type == pygame.MOUSEMOTION:
            self._handle_mouse_move(event.pos)
            return

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._dragging_slider = "none"
            return

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._handle_mouse_down(event.pos)

    def _handle_escape(self) -> None:
        if self.show_settings:
            self.show_settings = False
            return
        if self.show_quit_dialog:
            self.show_quit_dialog = False
            return
        if self.show_drawer:
            self.show_drawer = False
            return
        if self.active_tool is not None:
            self.active_tool = None
            return
        if self.screen_id in ("defender_game", "attacker_game", "spectator_game"):
            self.is_paused = not self.is_paused
        elif self.screen_id == "replay_viewer":
            self.replay_playing = False
            self.screen_id = (
                "defender_game" if self.player_side is Side.DEFENDER else "attacker_game"
            )
        elif self.screen_id in ("game_setup", "map_library", "asset_atlas"):
            self.screen_id = (
                self.prev_non_atlas_screen if self.screen_id == "asset_atlas" else "main_menu"
            )
        elif self.screen_id == "main_menu":
            self.running = False

    def _handle_mouse_move(self, pos: tuple[int, int]) -> None:
        if self._dragging_slider == "setup_size" and self._setup_slider_rect.width > 0:
            self._update_setup_size_from_x(pos[0])
            return
        if self._dragging_slider == "settings_vol" and self._settings_vol_rect.width > 0:
            self._update_volume_from_x(pos[0])
            return
        if self._dragging_slider == "replay" and self._replay_track_rect.width > 0:
            self._update_replay_from_x(pos[0])
            return

        new_hover_btn: str | None = None
        for btn in reversed(self._active_buttons):
            if btn.rect.collidepoint(pos) and not btn.disabled and btn.force_state is None:
                new_hover_btn = btn.id
                break
        if new_hover_btn != self._hovered_btn_id and new_hover_btn is not None:
            self.sfx.hover()
        self._hovered_btn_id = new_hover_btn

        if (
            self._is_battlefield_screen()
            and not self.is_paused
            and not self.show_settings
            and self.hotseat_curtain is None
            and not self.show_drawer
            and self.battlefield.peek_ref_mode == "none"
        ):
            frame = self._displayed_frame()
            prev_node = self.battlefield.hovered_node
            self.battlefield.hovered_node = self.battlefield.pick_machine(
                pos, self._battlefield_rect, self.metas, frame.state.patched
            )
            if (
                self.battlefield.hovered_node is not None
                and self.battlefield.hovered_node != prev_node
            ):
                self.sfx.hover()

            if self.active_tool == "firewall" and self.screen_id == "defender_game":
                self.battlefield.hovered_edge = self.battlefield.pick_edge(
                    pos,
                    self._battlefield_rect,
                    self.metas,
                    frame.state.network,
                    frame.state.patched,
                )
            else:
                self.battlefield.hovered_edge = None
        else:
            self.battlefield.hovered_node = None
            self.battlefield.hovered_edge = None

    def _update_setup_size_from_x(self, x: int) -> None:
        r = self._setup_slider_rect
        t = max(0.0, min(1.0, (x - r.x) / max(1, r.width)))
        new_sz = round(
            CONFIG.game.min_machines + t * (CONFIG.game.max_machines - CONFIG.game.min_machines)
        )
        if new_sz != self.setup_cfg.map_size:
            self.setup_cfg.map_size = new_sz
            self.setup_cfg.use_canonical_13 = False
            self.setup_cfg.sector_name = f"GRID_SECTOR_{new_sz}"

    def _update_volume_from_x(self, x: int) -> None:
        r = self._settings_vol_rect
        t = max(0.0, min(1.0, (x - r.x) / max(1, r.width)))
        self.settings.master_volume = int(round(t * 100))

    def _update_replay_from_x(self, x: int) -> None:
        if not self.history:
            return
        r = self._replay_track_rect
        t = max(0.0, min(1.0, (x - r.x) / max(1, r.width)))
        self.replay_playing = False
        self.replay_index = int(round(t * (len(self.history) - 1)))

    def _handle_mouse_down(self, pos: tuple[int, int]) -> None:
        if self.screen_id == "game_setup" and self._setup_slider_rect.inflate(10, 14).collidepoint(
            pos
        ):
            self._dragging_slider = "setup_size"
            self._update_setup_size_from_x(pos[0])
            return
        if self.show_settings and self._settings_vol_rect.inflate(10, 14).collidepoint(pos):
            self._dragging_slider = "settings_vol"
            self._update_volume_from_x(pos[0])
            return
        if self.screen_id == "replay_viewer" and self._replay_track_rect.inflate(
            10, 14
        ).collidepoint(pos):
            self._dragging_slider = "replay"
            self._update_replay_from_x(pos[0])
            return

        for btn in reversed(self._active_buttons):
            if btn.rect.collidepoint(pos) and not btn.disabled and btn.force_state is None:
                self.sfx.click()
                self._dispatch_button(btn.id)
                return

        if (
            self._is_battlefield_screen()
            and not self.is_paused
            and not self.show_settings
            and self.hotseat_curtain is None
        ):
            if self.battlefield.ref_toggle_rect.collidepoint(pos):
                self.sfx.click()
                if self.battlefield.peek_ref_mode == "none":
                    self.battlefield.peek_ref_mode = (
                        "attacker" if self.screen_id == "attacker_game" else "defender"
                    )
                else:
                    self.battlefield.peek_ref_mode = "none"
                return

            if self.battlefield.peek_ref_mode != "none" or self.screen_id == "replay_viewer":
                return

            self.flush_pending_ai_reply()
            frame = self._displayed_frame()
            state = frame.state

            if self.screen_id == "defender_game" and self.active_tool == "firewall":
                edge = self.battlefield.pick_edge(
                    pos, self._battlefield_rect, self.metas, state.network, state.patched
                )
                if edge is not None:
                    obs_def = observe_defender(state, frame.alerts, CONFIG.game)
                    fw_act = Firewall(edge[0], edge[1])
                    if fw_act in obs_def.legal_actions:
                        self.submit_player_action(fw_act)
                        return

            picked = self.battlefield.pick_machine(
                pos, self._battlefield_rect, self.metas, state.patched
            )
            if picked is not None:
                self.selected_node = picked
                if self.screen_id == "defender_game" and self.active_tool in ("patch", "honeypot"):
                    obs_def = observe_defender(state, frame.alerts, CONFIG.game)
                    act: Action = Patch(picked) if self.active_tool == "patch" else Honeypot(picked)
                    if act in obs_def.legal_actions:
                        self.submit_player_action(act)
                        return
                elif self.screen_id == "attacker_game" and state.side is Side.ATTACKER:
                    obs = observe_attacker(state, CONFIG.game)
                    target_act = Advance(picked)
                    if target_act in obs.legal_actions:
                        self.submit_player_action(target_act)
                        return
                self.sfx.click()

    def _dispatch_button(self, bid: str) -> None:
        if bid == "menu_setup":
            self.screen_id = "game_setup"
        elif bid in ("menu_quick_def", "menu_quick_def_img"):
            self.show_tutorial = False
            self.start_operation(
                SetupConfig(mode="human_vs_ai", side=Side.DEFENDER, seed=self.setup_cfg.seed),
                target_screen="defender_game",
            )
        elif bid == "menu_spectator":
            self.show_tutorial = False
            self.start_operation(
                SetupConfig(mode="ai_vs_ai", side=Side.DEFENDER, seed=self.setup_cfg.seed),
                target_screen="spectator_game",
            )
        elif bid == "menu_tutorial":
            self.show_tutorial = True
            self.start_operation(
                SetupConfig(mode="human_vs_ai", side=Side.DEFENDER, seed=self.setup_cfg.seed),
                target_screen="defender_game",
            )
        elif bid == "menu_maps":
            self.screen_id = "map_library"
        elif bid == "menu_atlas" or bid.startswith("menu_sprite_"):
            self.prev_non_atlas_screen = self.screen_id
            self.screen_id = "asset_atlas"
        elif bid == "menu_settings":
            self.show_settings = True
        elif bid == "menu_quit":
            self.show_quit_dialog = True
        elif bid == "quit_cancel":
            self.show_quit_dialog = False
        elif bid == "quit_confirm":
            self.running = False

        elif bid == "setup_back":
            self.screen_id = "main_menu"
        elif bid.startswith("setup_mode_"):
            mode_val = bid.removeprefix("setup_mode_")
            if mode_val in ("human_vs_ai", "ai_vs_ai", "human_vs_human"):
                self.setup_cfg.mode = mode_val  # type: ignore[assignment]
        elif bid == "setup_side_defender":
            self.setup_cfg.side = Side.DEFENDER
        elif bid == "setup_side_attacker":
            self.setup_cfg.side = Side.ATTACKER
        elif bid == "setup_seed_dec":
            self.setup_cfg.seed = max(1, self.setup_cfg.seed - 1)
        elif bid == "setup_seed_inc":
            self.setup_cfg.seed += 1
        elif bid == "setup_seed_rand":
            self.setup_cfg.seed = (self.setup_cfg.seed * 73 + 137) % 9999 + 1
            self.setup_cfg.evolved_map_path = None
        elif bid.startswith("setup_preset_"):
            pid = bid.removeprefix("setup_preset_")
            for preset in all_map_presets():
                if preset.id == pid:
                    self.setup_cfg.seed = preset.seed
                    self.setup_cfg.map_size = preset.node_count
                    self.setup_cfg.sector_name = preset.sector_code
                    self.setup_cfg.use_canonical_13 = preset.is_canonical_13
                    self.setup_cfg.evolved_map_path = preset.evolved_map_path
                    break
        elif bid == "setup_start":
            self.show_tutorial = False
            self.start_operation(self.setup_cfg)

        elif bid == "maps_back":
            self.screen_id = "main_menu"
        elif bid.startswith("maps_select_"):
            pid = bid.removeprefix("maps_select_")
            for preset in all_map_presets():
                if preset.id == pid:
                    self.selected_map_preset = preset
                    break
        elif bid in ("maps_launch_def", "maps_launch_atk", "maps_launch_spec"):
            p = self.selected_map_preset
            mode: Literal["human_vs_ai", "ai_vs_ai"] = (
                "ai_vs_ai" if bid == "maps_launch_spec" else "human_vs_ai"
            )
            side = Side.ATTACKER if bid == "maps_launch_atk" else Side.DEFENDER
            target: ScreenId = (
                "spectator_game"
                if bid == "maps_launch_spec"
                else "attacker_game"
                if bid == "maps_launch_atk"
                else "defender_game"
            )
            self.start_operation(
                SetupConfig(
                    mode=mode,
                    side=side,
                    seed=p.seed,
                    map_size=p.node_count,
                    sector_name=p.sector_code,
                    use_canonical_13=p.is_canonical_13,
                    evolved_map_path=p.evolved_map_path,
                ),
                target_screen=target,
            )

        elif bid == "atlas_back":
            self.screen_id = self.prev_non_atlas_screen
        elif bid.startswith("atlas_tab_"):
            t = bid.removeprefix("atlas_tab_")
            if t in ("sprites", "ui_kit", "boards"):
                self.atlas_tab = t  # type: ignore[assignment]
        elif bid.startswith("atlas_bg_"):
            b = bid.removeprefix("atlas_bg_")
            if b in ("obsidian", "checker", "card"):
                self.atlas_bg = b  # type: ignore[assignment]

        elif bid == "hud_menu":
            self.spectator_playing = False
            self.replay_playing = False
            self.is_paused = False
            self.screen_id = "main_menu"
        elif bid == "hud_pause":
            self.is_paused = not self.is_paused
        elif bid == "hud_sound":
            self.settings.sound_enabled = not self.settings.sound_enabled
        elif bid == "hud_heatmap":
            self.settings.show_heatmap = not self.settings.show_heatmap
        elif bid == "hud_hardness":
            self.settings.show_hardness_badges = not self.settings.show_hardness_badges
        elif bid == "hud_drawer":
            self.show_drawer = not self.show_drawer
        elif bid == "hud_tutorial":
            self.show_tutorial = not self.show_tutorial
        elif bid == "hud_replay":
            if self.screen_id == "replay_viewer":
                self.replay_playing = False
                self.screen_id = (
                    "defender_game" if self.player_side is Side.DEFENDER else "attacker_game"
                )
            else:
                self.spectator_playing = False
                self.replay_index = max(0, len(self.history) - 1)
                self.screen_id = "replay_viewer"
        elif bid == "hud_settings":
            self.show_settings = True

        elif bid == "tool_patch":
            self.flush_pending_ai_reply()
            self.active_tool = None if self.active_tool == "patch" else "patch"
        elif bid == "tool_firewall":
            self.flush_pending_ai_reply()
            self.active_tool = None if self.active_tool == "firewall" else "firewall"
        elif bid == "tool_honeypot":
            self.flush_pending_ai_reply()
            self.active_tool = None if self.active_tool == "honeypot" else "honeypot"
        elif bid == "tool_pass":
            self.flush_pending_ai_reply()
            if self.state.side is Side.DEFENDER and not is_terminal(self.state, CONFIG.game):
                self.submit_player_action(Pass())

        elif bid.startswith("atk_move_"):
            self.flush_pending_ai_reply()
            target_idx = int(bid.removeprefix("atk_move_"))
            obs = observe_attacker(self.state, CONFIG.game)
            if Advance(target_idx) in obs.legal_actions:
                self.submit_player_action(Advance(target_idx))
        elif bid == "atk_continue":
            self.flush_pending_ai_reply()
            if self.state.side is Side.ATTACKER and not is_terminal(self.state, CONFIG.game):
                obs = observe_attacker(self.state, CONFIG.game)
                self.submit_player_action(self.ai_attacker.choose_action(obs))

        elif bid == "spec_play":
            self.spectator_playing = not self.spectator_playing
        elif bid == "spec_step":
            self._execute_ai_ply()
        elif bid.startswith("spec_speed_"):
            self.settings.ai_speed_multiplier = float(bid.removeprefix("spec_speed_"))
        elif bid == "spec_god":
            self.settings.god_mode_spectator = not self.settings.god_mode_spectator

        elif bid == "rep_prev":
            self.replay_playing = False
            self.replay_index = max(0, self.replay_index - 1)
        elif bid == "rep_next":
            self.replay_playing = False
            self.replay_index = min(max(0, len(self.history) - 1), self.replay_index + 1)
        elif bid == "rep_play":
            if self.replay_index >= len(self.history) - 1:
                self.replay_index = 0
            self.replay_playing = not self.replay_playing
        elif bid == "rep_exit":
            self.replay_playing = False
            self.screen_id = (
                "defender_game" if self.player_side is Side.DEFENDER else "attacker_game"
            )

        elif bid == "tut_close":
            self.show_tutorial = False
        elif bid == "tut_step":
            self.flush_pending_ai_reply()
            self._execute_ai_ply()

        elif bid == "drawer_close":
            self.show_drawer = False
        elif bid.startswith("drawer_tab_"):
            dt_val = bid.removeprefix("drawer_tab_")
            if dt_val in ("peas", "sensors", "observation", "topology"):
                self.drawer_tab = dt_val  # type: ignore[assignment]

        elif bid == "pause_resume":
            self.is_paused = False
        elif bid == "pause_restart":
            self.is_paused = False
            self.start_operation(
                SetupConfig(
                    mode=self.game_mode,
                    side=self.player_side,
                    seed=self.active_seed,
                    map_size=self.state.network.size,
                    sector_name=self.active_sector,
                    use_canonical_13=(
                        self.state.network.size == 13 and self.active_sector == "INTRA_NET_ALPHA_09"
                    ),
                ),
                target_screen=self.screen_id,
            )
        elif bid == "pause_replay":
            self.is_paused = False
            self.replay_index = max(0, len(self.history) - 1)
            self.screen_id = "replay_viewer"
        elif bid == "pause_settings":
            self.is_paused = False
            self.show_settings = True
        elif bid == "pause_menu":
            self.is_paused = False
            self.spectator_playing = False
            self.screen_id = "main_menu"

        elif bid == "res_menu":
            self.screen_id = "main_menu"
        elif bid == "res_replay":
            self.replay_index = 0
            self.screen_id = "replay_viewer"
        elif bid == "res_again":
            self.start_operation(
                SetupConfig(
                    mode=self.game_mode,
                    side=self.player_side,
                    seed=self.active_seed,
                    map_size=self.state.network.size,
                    sector_name=self.active_sector,
                    use_canonical_13=(
                        self.state.network.size == 13 and self.active_sector == "INTRA_NET_ALPHA_09"
                    ),
                ),
                target_screen=self.screen_id,
            )

        elif bid == "set_sound_toggle":
            self.settings.sound_enabled = not self.settings.sound_enabled
        elif bid.startswith("set_theme_"):
            tname = bid.removeprefix("set_theme_")
            if tname in ("hybrid", "playbook", "warroom"):
                self.settings.theme = tname  # type: ignore[assignment]
        elif bid.startswith("set_speed_"):
            self.settings.ai_speed_multiplier = float(bid.removeprefix("set_speed_"))
        elif bid.startswith("set_fps_"):
            self.settings.target_fps = int(bid.removeprefix("set_fps_"))
        elif bid == "set_ov_heatmap":
            self.settings.show_heatmap = not self.settings.show_heatmap
        elif bid == "set_ov_trace":
            self.settings.show_breach_trace = not self.settings.show_breach_trace
        elif bid == "set_ov_hardness":
            self.settings.show_hardness_badges = not self.settings.show_hardness_badges
        elif bid == "set_ov_tray":
            self.settings.show_diagnostics_tray = not self.settings.show_diagnostics_tray
        elif bid == "set_close":
            self.show_settings = False

        elif bid == "hotseat_reveal":
            if self.hotseat_curtain is Side.DEFENDER:
                self.screen_id = "defender_game"
            else:
                self.screen_id = "attacker_game"
            self.hotseat_curtain = None


def run_window(
    seed: int = CONFIG.seeds.master_seed,
    start_screen: ScreenId = "main_menu",
    max_frames: int | None = None,
) -> None:
    """Open the CyberStrike Pygame window and run the 60 FPS UI loop."""
    pygame.init()
    theme = CONFIG.theme
    screen = pygame.display.set_mode(theme.window_size, pygame.RESIZABLE)
    pygame.display.set_caption("CyberStrike")

    app = CyberStrikeApp(seed=seed, start_screen=start_screen)
    app.vault.warm_up()
    pygame.display.set_icon(app.vault.get_branding("window_icon_32"))
    clock = pygame.time.Clock()

    frames = 0
    while app.running:
        dt = min(0.05, clock.tick(app.settings.target_fps) / 1000.0)
        for event in pygame.event.get():
            app.handle_event(event)
        app.update(dt)
        app.render(screen, fps_actual=clock.get_fps() or float(app.settings.target_fps))
        pygame.display.flip()

        frames += 1
        if max_frames is not None and frames >= max_frames:
            break

    pygame.quit()
    log.info("window closed")
