from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pygame

from config import CONFIG
from core.actions import Advance
from core.chokepoints import chokepoints
from core.observation import AttackerObservation, DefenderObservation
from core.rules import Outcome, effective_hardness, outcome
from core.state import GameState, Side
from ui.draw import (
    ButtonSpec,
    draw_alpha_rect,
    draw_badge_pill,
    draw_cyber_logo,
    draw_stock_pip_bar,
    draw_tactical_button,
    draw_tactical_panel,
)
from ui.layout import NodeVisualMeta
from ui.theme import RGB, AssetVault, FontBook, ThemePalette, ThemePresetName, with_alpha

DrawerTab = Literal["peas", "sensors", "observation", "topology"]


@dataclass(frozen=True, slots=True)
class AuditEntry:
    """One formatted event line shown in the Message Log / Audit Trail."""

    turn: int
    text: str
    severity: Literal["info", "cyan", "amber", "red", "violet"] = "info"


@dataclass(slots=True)
class GameSettingsState:
    """Runtime UI settings configurable from the Settings modal."""

    sound_enabled: bool = True
    master_volume: int = 45
    ai_speed_multiplier: float = 1.0
    theme: ThemePresetName = "hybrid"
    target_fps: int = 60
    show_heatmap: bool = True
    show_breach_trace: bool = False
    show_hardness_badges: bool = False
    show_diagnostics_tray: bool = True
    god_mode_spectator: bool = True


def derive_attack_stage(state: GameState) -> str:
    """Compute human-readable intrusion stage strictly from GameState."""
    if state.network.jewel in state.breached:
        return "EXFILTRATION"
    if len(state.breached) >= 4:
        return "PRIVILEGE ESCALATION"
    if len(state.breached) >= 2 or state.breach_target is not None:
        return "LATERAL MOVEMENT"
    return "INITIAL ACCESS"


def _format_timer(seconds_left: int) -> str:
    m = seconds_left // 60
    s = seconds_left % 60
    return f"{m:02d}:{s:02d}s"


def draw_top_hud_header(
    surface: pygame.Surface,
    width: int,
    palette: ThemePalette,
    fonts: FontBook,
    vault: AssetVault,
    state: GameState,
    sector_name: str,
    seed: int,
    view_mode: str,
    turn_seconds_left: int,
    settings: GameSettingsState,
    show_drawer: bool,
    show_tutorial: bool,
    hovered_id: str | None,
) -> list[ButtonSpec]:
    """Render the realistic in-game tactical header and sub-bar, returning clickable buttons."""
    buttons: list[ButtonSpec] = []
    top_rect = pygame.Rect(0, 0, width, 44)
    draw_alpha_rect(surface, top_rect, with_alpha(palette.top_bar, 245))
    pygame.draw.line(surface, with_alpha(palette.cyan, 85), (0, 44), (width, 44), 1)

    menu_btn = ButtonSpec(
        id="hud_menu",
        rect=pygame.Rect(10, 8, 84, 28),
        label="MENU",
        variant="slate",
        icon_name="icon_home",
        font_size=11,
    )
    draw_tactical_button(
        surface, menu_btn, palette, fonts, vault, hovered=hovered_id == menu_btn.id
    )
    buttons.append(menu_btn)

    draw_cyber_logo(surface, (106, 10), palette, fonts, vault, compact=True)

    # Right side: SECTOR | TURN [04 / 25] | 01:42s LIMIT | AUDIO | PAUSE
    pause_btn = ButtonSpec(
        id="hud_pause",
        rect=pygame.Rect(width - 96, 8, 86, 28),
        label="PAUSE",
        variant="cyan",
        icon_name="icon_pause",
        font_size=11,
    )
    draw_tactical_button(
        surface, pause_btn, palette, fonts, vault, hovered=hovered_id == pause_btn.id
    )
    buttons.append(pause_btn)

    mute_btn = ButtonSpec(
        id="hud_sound",
        rect=pygame.Rect(pause_btn.rect.left - 66, 8, 60, 28),
        label="SFX" if settings.sound_enabled else "MUTE",
        variant="slate" if settings.sound_enabled else "red",
        font_size=11,
    )
    draw_tactical_button(
        surface, mute_btn, palette, fonts, vault, hovered=hovered_id == mute_btn.id
    )
    buttons.append(mute_btn)

    # Glowing Amber 01:42s LIMIT box
    limit_rect = pygame.Rect(mute_btn.rect.left - 146, 6, 134, 32)
    draw_alpha_rect(
        surface,
        limit_rect,
        (52, 34, 12, 240),
        border_rgba=with_alpha(palette.amber, 210),
        border_width=1,
    )
    timer_s = fonts.get("title", 15, bold=True).render(
        _format_timer(turn_seconds_left), True, (255, 184, 64)
    )
    lim_s = fonts.get("heading", 11, bold=True).render("LIMIT", True, palette.amber)
    total_w = timer_s.get_width() + 6 + lim_s.get_width()
    tx = limit_rect.centerx - total_w // 2
    surface.blit(timer_s, (tx, limit_rect.centery - timer_s.get_height() // 2))
    surface.blit(
        lim_s, (tx + timer_s.get_width() + 6, limit_rect.centery - lim_s.get_height() // 2 + 1)
    )

    # Turn counter
    turn_lbl = fonts.get("heading", 13, bold=True).render("TURN", True, (184, 208, 220))
    turn_val = fonts.get("title", 14, bold=True).render(
        f"[{state.turn:02d} / {CONFIG.game.turn_limit:02d}]", True, palette.text
    )
    turn_x = limit_rect.left - 18 - turn_val.get_width() - 6 - turn_lbl.get_width()
    surface.blit(turn_lbl, (turn_x, 22 - turn_lbl.get_height() // 2))
    surface.blit(turn_val, (turn_x + turn_lbl.get_width() + 6, 22 - turn_val.get_height() // 2))

    # Sector name
    sec_lbl = fonts.get("heading", 13, bold=True).render("SECTOR:", True, (184, 208, 220))
    sec_val = fonts.get("heading", 13, bold=True).render(sector_name, True, palette.text)
    sec_x = turn_x - 24 - sec_val.get_width() - 6 - sec_lbl.get_width()
    if sec_x > 475:
        surface.blit(sec_lbl, (sec_x, 22 - sec_lbl.get_height() // 2))
        surface.blit(sec_val, (sec_x + sec_lbl.get_width() + 6, 22 - sec_val.get_height() // 2))

    # Sub-header bar (y=45..74)
    sub_rect = pygame.Rect(0, 45, width, 30)
    draw_alpha_rect(surface, sub_rect, with_alpha(palette.sub_bar, 245))
    pygame.draw.line(surface, with_alpha(palette.cyan, 45), (0, 75), (width, 75), 1)

    rem_stock = state.stock.patches + state.stock.firewalls + state.stock.honeypots
    telem = (
        f"SEED: #{seed}   |   NODES: {state.network.size}   |   STOCK: {rem_stock}/7   |   "
        f"STAGE: {derive_attack_stage(state)}   |   VIEW: {view_mode.upper()}"
    )
    t_surf = fonts.get("mono", 11).render(telem, True, (156, 180, 194))
    surface.blit(t_surf, (14, 60 - t_surf.get_height() // 2))

    # Sub-bar right toggle pills
    rx = width - 10
    sub_defs: list[
        tuple[str, str, str | None, bool, Literal["cyan", "amber", "violet", "red", "slate"]]
    ] = [
        ("hud_settings", "SETTINGS", "icon_settings", False, "slate"),
        ("hud_replay", "REPLAY", "icon_replay", view_mode == "replay", "amber"),
        ("hud_tutorial", "TUTORIAL", None, show_tutorial, "cyan"),
        ("hud_drawer", "INFO DRAWER", None, show_drawer, "violet"),
        ("hud_hardness", "HARDNESS", None, settings.show_hardness_badges, "red"),
    ]
    if view_mode != "attacker":
        sub_defs.append(("hud_heatmap", "HEATMAP", "icon_heatmap", settings.show_heatmap, "amber"))

    for bid, blabel, bicon, barmed, bvar in sub_defs:
        bw = len(blabel) * 7 + (34 if bicon else 18)
        brect = pygame.Rect(rx - bw, 48, bw, 23)
        spec = ButtonSpec(
            id=bid,
            rect=brect,
            label=blabel,
            variant=bvar,
            icon_name=bicon,
            armed=barmed,
            font_size=10,
        )
        draw_tactical_button(surface, spec, palette, fonts, vault, hovered=hovered_id == bid)
        buttons.append(spec)
        rx = brect.left - 6

    return buttons


def draw_defender_command_panel(
    surface: pygame.Surface,
    rect: pygame.Rect,
    palette: ThemePalette,
    fonts: FontBook,
    vault: AssetVault,
    state: GameState,
    metas: tuple[NodeVisualMeta, ...],
    observation: DefenderObservation,
    active_tool: str | None,
    selected_node: int | None,
    hovered_id: str | None,
) -> list[ButtonSpec]:
    """Render the Defender Command right sidebar matching the War-Room layout."""
    inner = draw_tactical_panel(
        surface,
        rect,
        palette,
        fonts,
        title="DEFENDER COMMAND",
        sub_header="TACTICAL STOCK",
        accent="cyan",
        sub_accent="amber",
    )
    buttons: list[ButtonSpec] = []
    y = inner.top + 4
    w = inner.width

    y += (
        draw_stock_pip_bar(
            surface,
            (inner.left, y),
            w,
            "PATCHES",
            state.stock.patches,
            CONFIG.game.stock.patches,
            palette.cyan,
            palette,
            fonts,
        )
        + 12
    )
    y += (
        draw_stock_pip_bar(
            surface,
            (inner.left, y),
            w,
            "FIREWALLS",
            state.stock.firewalls,
            CONFIG.game.stock.firewalls,
            palette.amber,
            palette,
            fonts,
        )
        + 12
    )
    y += (
        draw_stock_pip_bar(
            surface,
            (inner.left, y),
            w,
            "HONEYPOTS",
            state.stock.honeypots,
            CONFIG.game.stock.honeypots,
            palette.violet,
            palette,
            fonts,
        )
        + 10
    )

    # Sensor Status & Selected Node Box
    pygame.draw.line(surface, with_alpha(palette.cyan, 45), (inner.left, y), (inner.right, y), 1)
    y += 8
    alert_ids = [metas[i].code for i, flag in enumerate(observation.alerts) if flag]
    hdr_l = fonts.get("mono", 10).render("SENSOR TELEMETRY", True, palette.muted)
    hdr_r = fonts.get("mono", 10).render(
        f"{len(alert_ids)} ALERTS" if alert_ids else "QUIET",
        True,
        palette.red if alert_ids else palette.cyan,
    )
    surface.blit(hdr_l, (inner.left, y))
    surface.blit(hdr_r, (inner.right - hdr_r.get_width(), y))
    y += 16

    box_rect = pygame.Rect(inner.left, y, w, 46)
    draw_alpha_rect(
        surface,
        box_rect,
        (16, 28, 38, 215),
        border_rgba=with_alpha(palette.red if alert_ids else palette.cyan, 90),
    )
    if observation.intruder_known_at is not None:
        known_code = metas[observation.intruder_known_at].code
        s_txt = f"HONEYPOT HIT: INTRUDER AT {known_code}"
        s_col = palette.violet
    elif alert_ids:
        shown = ", ".join(alert_ids[:4])
        extra = f" +{len(alert_ids) - 4}" if len(alert_ids) > 4 else ""
        s_txt = f"ALERTS ON: {shown}{extra}"
        s_col = palette.red
    else:
        s_txt = "NO ANOMALOUS PACKETS DETECTED"
        s_col = palette.cyan

    if selected_node is not None and 0 <= selected_node < len(metas):
        sel_hrd = effective_hardness(state, selected_node, CONFIG.game)
        sel_txt = f"SELECTED: {metas[selected_node].code} (HRD {sel_hrd})"
    else:
        sel_txt = "CLICK A NODE OR ARMED TOOL"
    surface.blit(fonts.get("mono", 10).render(s_txt, True, s_col), (box_rect.x + 8, box_rect.y + 6))
    surface.blit(
        fonts.get("mono", 10).render(sel_txt, True, palette.muted),
        (box_rect.x + 8, box_rect.y + 24),
    )
    y = box_rect.bottom + 10

    # Operation Panel divider
    pygame.draw.line(
        surface, with_alpha(palette.cyan, 65), (inner.left - 8, y), (inner.right + 8, y), 1
    )
    y += 8
    op_title = fonts.get("heading", 13, bold=True).render("DEFENDER COMMAND", True, palette.text)
    op_sub = fonts.get("mono", 10).render("[The operation panel]", True, palette.muted)
    surface.blit(op_title, (inner.left, y))
    surface.blit(op_sub, (inner.left, y + 17))
    y += 38

    game_over = outcome(state, CONFIG.game) is not Outcome.ONGOING
    not_turn = state.side is not Side.DEFENDER or game_over
    btn_h = 36
    gap = 8

    tool_defs: list[tuple[str, str, str, str, bool, bool]] = [
        (
            "tool_patch",
            "CANCEL PATCH" if active_tool == "patch" else "PLACE PATCH",
            "icon_patch",
            "+2 HRD",
            active_tool == "patch",
            not_turn or state.stock.patches <= 0,
        ),
        (
            "tool_firewall",
            "CANCEL FIREWALL" if active_tool == "firewall" else "DEPLOY FIREWALL",
            "icon_firewall",
            "CUT LINK",
            active_tool == "firewall",
            not_turn or state.stock.firewalls <= 0,
        ),
        (
            "tool_honeypot",
            "CANCEL HONEYPOT" if active_tool == "honeypot" else "SET HONEYPOT",
            "icon_honeypot",
            "TRAP 2T",
            active_tool == "honeypot",
            not_turn or state.stock.honeypots <= 0,
        ),
    ]

    for bid, blabel, bicon, btag, barmed, bdisabled in tool_defs:
        spec = ButtonSpec(
            id=bid,
            rect=pygame.Rect(inner.left, y, w, btn_h),
            label=blabel,
            variant="cyan",
            icon_name=bicon,
            right_tag=btag,
            armed=barmed,
            disabled=bdisabled,
            font_size=12,
        )
        draw_tactical_button(surface, spec, palette, fonts, vault, hovered=hovered_id == bid)
        buttons.append(spec)
        y += btn_h + gap

    pass_y = max(y + 4, inner.bottom - 44)
    pass_btn = ButtonSpec(
        id="tool_pass",
        rect=pygame.Rect(inner.left, pass_y, w, 40),
        label="PASS TURN",
        variant="amber",
        icon_name="icon_pass",
        disabled=not_turn,
        font_size=14,
    )
    draw_tactical_button(
        surface, pass_btn, palette, fonts, vault, hovered=hovered_id == "tool_pass"
    )
    buttons.append(pass_btn)

    return buttons


def draw_attacker_command_panel(
    surface: pygame.Surface,
    rect: pygame.Rect,
    palette: ThemePalette,
    fonts: FontBook,
    vault: AssetVault,
    state: GameState,
    metas: tuple[NodeVisualMeta, ...],
    observation: AttackerObservation,
    hovered_id: str | None,
) -> list[ButtonSpec]:
    """Render the Attacker Command right sidebar driven directly by AttackerObservation."""
    stage = derive_attack_stage(state)
    inner = draw_tactical_panel(
        surface,
        rect,
        palette,
        fonts,
        title="ATTACKER COMMAND",
        sub_header=f"STAGE // {stage}",
        accent="red",
        sub_accent="red",
    )
    buttons: list[ButtonSpec] = []
    y = inner.top + 4
    w = inner.width

    cur_idx = observation.position
    cur_meta = metas[cur_idx]
    target_idx = observation.breach_target
    active_idx = target_idx if target_idx is not None else cur_idx
    eff_h = effective_hardness(state, active_idx, CONFIG.game)

    # Active vector & breach progress card
    card = pygame.Rect(inner.left, y, w, 78)
    draw_alpha_rect(surface, card, (28, 11, 19, 225), border_rgba=with_alpha(palette.red, 125))
    surface.blit(
        fonts.get("heading", 12, bold=True).render(
            f"ACTIVE VECTOR: {cur_meta.code}", True, (255, 107, 142)
        ),
        (card.x + 10, card.y + 8),
    )
    hrd_s = fonts.get("mono", 11).render(f"HRD {eff_h}T", True, palette.amber)
    surface.blit(hrd_s, (card.right - 10 - hrd_s.get_width(), card.y + 9))

    if target_idx is not None:
        t_code = metas[target_idx].code
        status_txt = f"BREACHING {t_code} ({observation.breach_left}T LEFT)"
        completed = max(1, eff_h - observation.breach_left)
        pct = min(1.0, completed / max(1, eff_h))
    else:
        status_txt = "NODE OWNED — SELECT TARGET"
        pct = 1.0

    surface.blit(
        fonts.get("mono", 10).render(status_txt, True, palette.text), (card.x + 10, card.y + 30)
    )
    bar = pygame.Rect(card.x + 10, card.y + 50, card.width - 20, 10)
    draw_alpha_rect(surface, bar, (18, 7, 13, 230), border_rgba=with_alpha(palette.red, 95))
    if pct > 0.0:
        draw_alpha_rect(
            surface,
            pygame.Rect(bar.x + 1, bar.y + 1, int((bar.width - 2) * pct), bar.height - 2),
            with_alpha(palette.red, 220),
        )
    y = card.bottom + 12

    legal_targets: list[int] = [
        act.target for act in observation.legal_actions if isinstance(act, Advance)
    ]
    surface.blit(
        fonts.get("heading", 12, bold=True).render(
            f"LEGAL LATERAL MOVES ({len(legal_targets)})", True, palette.text
        ),
        (inner.left, y),
    )
    y += 22

    game_over = outcome(state, CONFIG.game) is not Outcome.ONGOING
    not_turn = state.side is not Side.ATTACKER or game_over
    max_list_bottom = inner.bottom - 52

    if observation.breach_target is not None:
        info = pygame.Rect(inner.left, y, w, 66)
        draw_alpha_rect(
            surface, info, (23, 18, 36, 230), border_rgba=with_alpha(palette.amber, 125)
        )
        t_code = metas[observation.breach_target].code
        surface.blit(
            fonts.get("heading", 12, bold=True).render(
                f"BREACH IN PROGRESS: {t_code}", True, palette.amber
            ),
            (info.x + 10, info.y + 10),
        )
        surface.blit(
            fonts.get("body", 13).render(
                "Advance turn to finish breaching this shell.", True, palette.text
            ),
            (info.x + 10, info.y + 34),
        )
    else:
        for nb_idx in legal_targets:
            if y + 32 > max_list_bottom:
                break
            nb_meta = metas[nb_idx]
            nb_h = effective_hardness(state, nb_idx, CONFIG.game)
            bid = f"atk_move_{nb_idx}"
            spec = ButtonSpec(
                id=bid,
                rect=pygame.Rect(inner.left, y, w, 30),
                label=f"{nb_meta.code}" + (" [JEWEL]" if nb_idx == state.network.jewel else ""),
                variant="red" if nb_idx == state.network.jewel else "slate",
                right_tag=f"HRD:{nb_h}",
                disabled=not_turn,
                font_size=12,
                align_left=True,
            )
            draw_tactical_button(surface, spec, palette, fonts, vault, hovered=hovered_id == bid)
            buttons.append(spec)
            y += 34

    auto_lbl = (
        "CONTINUE BREACH CYCLE" if observation.breach_target is not None else "EXECUTE RANDOM MOVE"
    )
    auto_btn = ButtonSpec(
        id="atk_continue",
        rect=pygame.Rect(inner.left, inner.bottom - 44, w, 40),
        label=auto_lbl,
        variant="amber",
        disabled=not_turn or not legal_targets,
        font_size=13,
    )
    draw_tactical_button(
        surface, auto_btn, palette, fonts, vault, hovered=hovered_id == "atk_continue"
    )
    buttons.append(auto_btn)
    return buttons


def draw_audit_log_panel(
    surface: pygame.Surface,
    rect: pygame.Rect,
    palette: ThemePalette,
    fonts: FontBook,
    logs: list[AuditEntry],
) -> None:
    """Render the bottom-left Message Log / Audit Trail console."""
    inner = draw_tactical_panel(
        surface,
        rect,
        palette,
        fonts,
        title="MESSAGE LOG / AUDIT TRAIL",
        accent="cyan",
    )
    sev_map: dict[str, RGB] = {
        "info": (198, 228, 216),
        "cyan": (107, 242, 220),
        "amber": (255, 192, 77),
        "red": (255, 117, 151),
        "violet": (216, 180, 254),
    }
    mono = fonts.get("mono", 11)
    max_lines = max(1, inner.height // 17)
    recent = logs[-max_lines:]
    y = inner.top
    for entry in recent:
        col = sev_map.get(entry.severity, palette.text)
        txt = entry.text
        if len(txt) > 96:
            txt = txt[:93] + "..."
        surface.blit(mono.render(txt, True, col), (inner.left + 2, y))
        y += 17


def draw_bottom_diagnostics_tray(
    surface: pygame.Surface,
    rect: pygame.Rect,
    palette: ThemePalette,
    fonts: FontBook,
    state: GameState,
    seed: int,
    plies: int,
    fps_actual: float,
) -> None:
    """Render the bottom diagnostics bar strictly from live core/ metrics."""
    draw_alpha_rect(surface, rect, (9, 15, 23, 245))
    pygame.draw.line(surface, with_alpha(palette.cyan, 75), rect.topleft, (rect.right, rect.top), 1)
    choke_cnt = len(chokepoints(state.network))
    text = (
        f"DIAGNOSTICS  |  SEED #{seed}  |  TURN {state.turn}/{CONFIG.game.turn_limit}  |  "
        f"PLIES: {plies}  |  BREACHED: {len(state.breached)}/{state.network.size}  |  "
        f"CHOKEPOINTS: {choke_cnt}  |  SIDE: {state.side.value.upper()}  |  "
        f"FPS: {fps_actual:.0f}"
    )
    surf = fonts.get("mono", 11).render(text, True, (155, 176, 193))
    surface.blit(surf, (rect.left + 14, rect.centery - surf.get_height() // 2))


def draw_spectator_bar(
    surface: pygame.Surface,
    rect: pygame.Rect,
    palette: ThemePalette,
    fonts: FontBook,
    vault: AssetVault,
    is_playing: bool,
    speed: float,
    god_mode: bool,
    hovered_id: str | None,
) -> list[ButtonSpec]:
    """Render the AI vs AI Spectator playback bar."""
    draw_alpha_rect(
        surface,
        rect,
        (17, 21, 43, 240),
        border_rgba=with_alpha(palette.violet, 140),
    )
    buttons: list[ButtonSpec] = []
    draw_badge_pill(
        surface,
        (rect.left + 12, rect.top + 6),
        "SPECTATOR: AI VS AI (RANDOM AGENTS)",
        (235, 217, 255),
        with_alpha(palette.violet, 60),
        with_alpha(palette.violet, 200),
        fonts.get("heading", 11, bold=True),
        centered=False,
    )

    x = rect.right - 10
    god_btn = ButtonSpec(
        id="spec_god",
        rect=pygame.Rect(x - 148, rect.top + 5, 138, 24),
        label="OMNISCIENT: ON" if god_mode else "FOG OF WAR",
        variant="red" if god_mode else "slate",
        armed=god_mode,
        font_size=10,
    )
    draw_tactical_button(surface, god_btn, palette, fonts, vault, hovered=hovered_id == god_btn.id)
    buttons.append(god_btn)
    x = god_btn.rect.left - 8

    for sp in (4.0, 2.0, 1.0, 0.5):
        lbl = f"{sp:g}x"
        bid = f"spec_speed_{sp}"
        sp_btn = ButtonSpec(
            id=bid,
            rect=pygame.Rect(x - 42, rect.top + 5, 42, 24),
            label=lbl,
            variant="amber",
            armed=abs(speed - sp) < 0.01,
            font_size=10,
        )
        draw_tactical_button(surface, sp_btn, palette, fonts, vault, hovered=hovered_id == bid)
        buttons.append(sp_btn)
        x = sp_btn.rect.left - 4

    step_btn = ButtonSpec(
        id="spec_step",
        rect=pygame.Rect(x - 106, rect.top + 5, 98, 24),
        label="STEP TURN",
        variant="slate",
        icon_name="icon_step",
        disabled=is_playing,
        font_size=10,
    )
    draw_tactical_button(
        surface, step_btn, palette, fonts, vault, hovered=hovered_id == "spec_step"
    )
    buttons.append(step_btn)
    x = step_btn.rect.left - 6

    play_btn = ButtonSpec(
        id="spec_play",
        rect=pygame.Rect(x - 118, rect.top + 5, 118, 24),
        label="PAUSE AI" if is_playing else "AUTO-PLAY AI",
        variant="amber" if is_playing else "cyan",
        icon_name="icon_pause" if is_playing else "icon_play",
        font_size=10,
    )
    draw_tactical_button(
        surface, play_btn, palette, fonts, vault, hovered=hovered_id == "spec_play"
    )
    buttons.append(play_btn)
    return buttons


def draw_replay_bar(
    surface: pygame.Surface,
    rect: pygame.Rect,
    palette: ThemePalette,
    fonts: FontBook,
    vault: AssetVault,
    replay_index: int,
    history_len: int,
    current_turn: int,
    max_turn: int,
    is_playing: bool,
    hovered_id: str | None,
) -> tuple[list[ButtonSpec], pygame.Rect]:
    """Render the Replay Timeline Scrubber bar and return (buttons, track_rect)."""
    draw_alpha_rect(
        surface,
        rect,
        (15, 23, 38, 242),
        border_rgba=with_alpha(palette.amber, 145),
    )
    buttons: list[ButtonSpec] = []

    replay_badge = (
        f"REPLAY STREAM  [ STEP {replay_index + 1}/{max(1, history_len)} "
        f"// T{current_turn:02d}/T{max_turn:02d} ]"
    )
    draw_badge_pill(
        surface,
        (rect.left + 12, rect.top + 6),
        replay_badge,
        palette.amber,
        with_alpha(palette.amber, 45),
        with_alpha(palette.amber, 195),
        fonts.get("heading", 11, bold=True),
        centered=False,
    )

    exit_btn = ButtonSpec(
        id="rep_exit",
        rect=pygame.Rect(rect.right - 134, rect.top + 5, 124, 24),
        label="EXIT REPLAY",
        variant="cyan",
        font_size=10,
    )
    next_btn = ButtonSpec(
        id="rep_next",
        rect=pygame.Rect(exit_btn.rect.left - 44, rect.top + 5, 38, 24),
        label=">",
        variant="slate",
        disabled=replay_index >= history_len - 1,
        font_size=12,
    )
    play_btn = ButtonSpec(
        id="rep_play",
        rect=pygame.Rect(next_btn.rect.left - 102, rect.top + 5, 96, 24),
        label="PAUSE" if is_playing else "PLAY",
        variant="amber",
        icon_name="icon_pause" if is_playing else "icon_play",
        font_size=10,
    )
    prev_btn = ButtonSpec(
        id="rep_prev",
        rect=pygame.Rect(play_btn.rect.left - 44, rect.top + 5, 38, 24),
        label="<",
        variant="slate",
        disabled=replay_index <= 0,
        font_size=12,
    )

    for btn in (exit_btn, next_btn, play_btn, prev_btn):
        draw_tactical_button(surface, btn, palette, fonts, vault, hovered=hovered_id == btn.id)
        buttons.append(btn)

    track_rect = pygame.Rect(rect.left + 16, rect.bottom - 14, rect.width - 32, 8)
    draw_alpha_rect(
        surface, track_rect, (27, 38, 56, 230), border_rgba=with_alpha(palette.amber, 90)
    )
    ratio = replay_index / max(1, history_len - 1)
    fill_w = int(track_rect.width * ratio)
    if fill_w > 0:
        draw_alpha_rect(
            surface,
            pygame.Rect(track_rect.x, track_rect.y, fill_w, track_rect.height),
            with_alpha(palette.amber, 195),
        )
    knob_x = track_rect.x + fill_w
    pygame.draw.circle(surface, palette.amber, (knob_x, track_rect.centery), 6)
    pygame.draw.circle(surface, palette.text, (knob_x, track_rect.centery), 3)
    return buttons, track_rect


def draw_tutorial_card(
    surface: pygame.Surface,
    rect: pygame.Rect,
    palette: ThemePalette,
    fonts: FontBook,
    vault: AssetVault,
    state: GameState,
    hovered_id: str | None,
) -> list[ButtonSpec]:
    """Render the Explainable Rules Tutorial banner describing current turn state."""
    draw_alpha_rect(
        surface,
        rect,
        (19, 34, 46, 240),
        border_rgba=with_alpha(palette.cyan, 160),
    )
    buttons: list[ButtonSpec] = []
    stage = derive_attack_stage(state)
    draw_badge_pill(
        surface,
        (rect.left + 12, rect.top + 6),
        f"TUTORIAL COACH // {stage}",
        palette.cyan,
        with_alpha(palette.cyan, 50),
        with_alpha(palette.cyan, 200),
        fonts.get("heading", 10, bold=True),
        centered=False,
    )

    if state.side is Side.DEFENDER:
        hint = (
            f"Defender Turn {state.turn}: Watch noisy sensor alerts and protect Crown Jewel paths "
            "using Patch (+2 hardness), Firewall (cuts link), or hidden Honeypot (+2T delay)."
        )
    else:
        hint = (
            f"Attacker Turn {state.turn}: Advance to an open neighbor. Multi-turn hardness holds "
            "the intruder in place until breach_left reaches 0."
        )
    surface.blit(
        fonts.get("body", 14).render(hint, True, palette.text), (rect.left + 14, rect.top + 30)
    )

    close_btn = ButtonSpec(
        id="tut_close",
        rect=pygame.Rect(rect.right - 36, rect.top + 8, 26, 26),
        label="X",
        variant="slate",
        font_size=11,
    )
    step_btn = ButtonSpec(
        id="tut_step",
        rect=pygame.Rect(close_btn.rect.left - 148, rect.top + 8, 140, 26),
        label="STEP DEMO TURN",
        variant="cyan",
        font_size=10,
    )
    for btn in (close_btn, step_btn):
        draw_tactical_button(surface, btn, palette, fonts, vault, hovered=hovered_id == btn.id)
        buttons.append(btn)
    return buttons


def draw_side_panels_drawer(
    surface: pygame.Surface,
    rect: pygame.Rect,
    palette: ThemePalette,
    fonts: FontBook,
    vault: AssetVault,
    state: GameState,
    metas: tuple[NodeVisualMeta, ...],
    alerts: tuple[bool, ...],
    active_tab: DrawerTab,
    selected_node: int | None,
    hovered_id: str | None,
) -> list[ButtonSpec]:
    """Render the slide-over Info Drawer displaying Phase 1 PEAS, Sensors & Topology."""
    inner = draw_tactical_panel(
        surface,
        rect,
        palette,
        fonts,
        title="SYSTEM INSPECTOR // PHASE 1–2",
        accent="violet",
        fill_alpha=246,
    )
    buttons: list[ButtonSpec] = []

    close_btn = ButtonSpec(
        id="drawer_close",
        rect=pygame.Rect(rect.right - 32, rect.top + 4, 24, 22),
        label="X",
        variant="slate",
        font_size=11,
    )
    draw_tactical_button(
        surface, close_btn, palette, fonts, vault, hovered=hovered_id == "drawer_close"
    )
    buttons.append(close_btn)

    tabs: list[tuple[DrawerTab, str]] = [
        ("peas", "PEAS"),
        ("sensors", "SENSORS"),
        ("observation", "OBSERVE"),
        ("topology", "GRAPH"),
    ]
    tab_w = inner.width // len(tabs)
    for idx, (tid, tlabel) in enumerate(tabs):
        trect = pygame.Rect(inner.left + idx * tab_w, inner.top, tab_w - 3, 26)
        spec = ButtonSpec(
            id=f"drawer_tab_{tid}",
            rect=trect,
            label=tlabel,
            variant="violet",
            armed=active_tab == tid,
            font_size=10,
        )
        draw_tactical_button(surface, spec, palette, fonts, vault, hovered=hovered_id == spec.id)
        buttons.append(spec)

    y = inner.top + 36
    mono = fonts.get("mono", 11)
    body = fonts.get("body", 14)
    focus = selected_node if selected_node is not None and 0 <= selected_node < len(metas) else 0
    focus_meta = metas[focus]

    if active_tab == "peas":
        lines = [
            ("ENVIRONMENT PROPERTIES", palette.violet),
            ("  Partially Observable, Stochastic Sensors", palette.text),
            ("  Discrete, Turn-Based, Zero-Sum", palette.text),
            ("", palette.text),
            ("ATTACKER AGENT (PEAS)", palette.red),
            ("  P: Reach DB-01 in minimal turns", palette.muted),
            ("  E: Network graph + hardness 1..5", palette.muted),
            ("  A: Advance(target_machine)", palette.muted),
            ("  S: Topology, patches, firewalls", palette.muted),
            ("", palette.text),
            ("DEFENDER AGENT (PEAS)", palette.cyan),
            ("  P: Hold DB-01 through Turn 25", palette.muted),
            ("  E: Attacker position strictly hidden", palette.muted),
            ("  A: Patch, Firewall, Honeypot, Pass", palette.muted),
            ("  S: Noisy alerts + triggered traps", palette.muted),
        ]
    elif active_tab == "sensors":
        active_alerts = [metas[i].code for i, a in enumerate(alerts) if a]
        lines = [
            (f"DETECTION RATE: {int(CONFIG.sensors.detection_rate * 100)}%", palette.cyan),
            (f"FALSE ALARM RATE: {int(CONFIG.sensors.false_alarm_rate * 100)}%", palette.amber),
            (f"ACTIVE ALERTS ({len(active_alerts)}):", palette.red),
            (f"  {', '.join(active_alerts[:6]) or 'NONE'}", palette.text),
            ("", palette.text),
            (f"SELECTED HOST: {focus_meta.code}", palette.violet),
            (f"  SENSOR STATE: {'ALERT' if alerts[focus] else 'NORMAL'}", palette.text),
            (f"  BASE HARDNESS: {state.network.hardness[focus]}T", palette.muted),
            (f"  EFF HARDNESS: {effective_hardness(state, focus, CONFIG.game)}T", palette.cyan),
            (f"  HONEYPOT TRAP: {'ARMED' if focus in state.honeypots else 'NONE'}", palette.muted),
        ]
    elif active_tab == "observation":
        lines = [
            ("DEFENDER OBSERVATION BOUNDARY", palette.cyan),
            (f"  Turn: {state.turn} / {CONFIG.game.turn_limit}", palette.text),
            (f"  Patched Nodes: {len(state.patched)}", palette.text),
            (f"  Firewalled Links: {len(state.firewalls)}", palette.text),
            (f"  Honeypots Armed: {len(state.honeypots)}", palette.text),
            (f"  Triggered Traps: {len(state.triggered)}", palette.violet),
            ("  True Attacker Pos: [CONCEALED]", palette.amber),
        ]
    else:
        chokes = chokepoints(state.network)
        nbs = [metas[nb].code for nb in state.network.neighbours(focus)]
        lines = [
            (f"NETWORK GRAPH ({state.network.size} NODES)", palette.cyan),
            (f"  Total Edges: {len(state.network.edges())}", palette.text),
            (f"  Connected: {state.network.is_connected()}", palette.green),
            (f"  Articulation Chokepoints: {len(chokes)}", palette.amber),
            ("", palette.text),
            (f"NEIGHBORS OF {focus_meta.code}:", palette.violet),
            (f"  {', '.join(nbs)}", palette.text),
        ]

    for text, col in lines:
        if y + 18 > inner.bottom:
            break
        if text:
            f = (
                fonts.get("heading", 11, bold=True)
                if text.isupper() and not text.startswith("  ")
                else body
            )
            if text.startswith("  "):
                f = mono
            surface.blit(f.render(text, True, col), (inner.left + 4, y))
        y += 20

    return buttons


def draw_pause_modal(
    surface: pygame.Surface,
    size: tuple[int, int],
    palette: ThemePalette,
    fonts: FontBook,
    vault: AssetVault,
    hovered_id: str | None,
) -> list[ButtonSpec]:
    """Render the Pause Menu overlay modal."""
    w, h = size
    draw_alpha_rect(surface, pygame.Rect(0, 0, w, h), (6, 6, 13, 200))
    modal = pygame.Rect(w // 2 - 190, h // 2 - 175, 380, 350)
    inner = draw_tactical_panel(
        surface,
        modal,
        palette,
        fonts,
        title="OPERATION PAUSED",
        sub_header="TACTICAL HOLD — STATE FROZEN",
        accent="cyan",
        sub_accent="amber",
        fill_alpha=248,
    )
    buttons: list[ButtonSpec] = []
    defs: list[tuple[str, str, Literal["amber", "cyan", "violet", "slate", "red"], str | None]] = [
        ("pause_resume", "RESUME OPERATION", "amber", "icon_play"),
        ("pause_restart", "RESTART CURRENT SEED", "cyan", "icon_replay"),
        ("pause_replay", "OPEN REPLAY TIMELINE", "violet", "icon_step"),
        ("pause_settings", "SYSTEM SETTINGS", "slate", "icon_settings"),
        ("pause_menu", "QUIT TO MAIN MENU", "red", "icon_home"),
    ]
    y = inner.top + 8
    for bid, blabel, bvar, bicon in defs:
        spec = ButtonSpec(
            id=bid,
            rect=pygame.Rect(inner.left + 6, y, inner.width - 12, 42),
            label=blabel,
            variant=bvar,
            icon_name=bicon,
            font_size=13,
        )
        draw_tactical_button(surface, spec, palette, fonts, vault, hovered=hovered_id == bid)
        buttons.append(spec)
        y += 52
    return buttons


def draw_results_modal(
    surface: pygame.Surface,
    size: tuple[int, int],
    palette: ThemePalette,
    fonts: FontBook,
    vault: AssetVault,
    state: GameState,
    seed: int,
    plies: int,
    score: int,
    win_reason: str,
    hovered_id: str | None,
) -> list[ButtonSpec]:
    """Render the Post-Operation Results modal when a game terminates."""
    w, h = size
    draw_alpha_rect(surface, pygame.Rect(0, 0, w, h), (6, 6, 13, 215))
    modal = pygame.Rect(w // 2 - 320, h // 2 - 210, 640, 420)
    is_def = outcome(state, CONFIG.game) is Outcome.DEFENDER_WINS
    accent: Literal["cyan", "red"] = "cyan" if is_def else "red"
    inner = draw_tactical_panel(
        surface,
        modal,
        palette,
        fonts,
        title="POST-OPERATION DEBRIEF",
        sub_header=(
            "DEFENDER VICTORY — CROWN JEWEL SECURED"
            if is_def
            else "BREACH COMPLETE — CROWN JEWEL COMPROMISED"
        ),
        accent=accent,
        sub_accent=accent,
        fill_alpha=248,
    )

    banner = pygame.Rect(inner.left, inner.top + 6, inner.width, 82)
    bcol = palette.cyan if is_def else palette.red
    draw_alpha_rect(surface, banner, with_alpha(bcol, 32), border_rgba=with_alpha(bcol, 165))
    surface.blit(
        fonts.get("title", 20, bold=True).render(
            "DEFENDER VICTORY" if is_def else "ATTACKER VICTORY", True, palette.text
        ),
        (banner.x + 16, banner.y + 12),
    )
    surface.blit(
        fonts.get("body", 14).render(win_reason, True, (215, 222, 238)),
        (banner.x + 16, banner.y + 46),
    )
    sc_lbl = fonts.get("heading", 10, bold=True).render("TACTICAL SCORE", True, palette.muted)
    sc_val = fonts.get("title", 22, bold=True).render(str(score), True, palette.amber)
    surface.blit(sc_lbl, (banner.right - 16 - sc_lbl.get_width(), banner.y + 12))
    surface.blit(sc_val, (banner.right - 16 - sc_val.get_width(), banner.y + 32))

    # 6 real stats tiles
    def_spent = (
        (CONFIG.game.stock.patches - state.stock.patches)
        + (CONFIG.game.stock.firewalls - state.stock.firewalls)
        + (CONFIG.game.stock.honeypots - state.stock.honeypots)
    )
    stats = [
        (
            "TURNS USED",
            f"{min(state.turn, CONFIG.game.turn_limit)} / {CONFIG.game.turn_limit}",
            palette.text,
        ),
        ("MACHINES OWNED", f"{len(state.breached)} / {state.network.size}", palette.red),
        ("DEFENSES SPENT", f"{def_spent} / 7 UNITS", palette.cyan),
        ("HONEYPOTS HIT", f"{len(state.triggered)}", palette.violet),
        ("TOTAL PLIES", f"{plies}", palette.amber),
        ("MAP SEED", f"#{seed}", palette.text),
    ]
    gy = banner.bottom + 16
    tile_w = (inner.width - 20) // 3
    for i, (sk, sv, scol) in enumerate(stats):
        col_i = i % 3
        row_i = i // 3
        trect = pygame.Rect(inner.left + col_i * (tile_w + 10), gy + row_i * 64, tile_w, 54)
        draw_alpha_rect(
            surface, trect, (10, 16, 27, 235), border_rgba=with_alpha(palette.muted, 65)
        )
        surface.blit(
            fonts.get("mono", 10).render(sk, True, palette.muted), (trect.x + 10, trect.y + 8)
        )
        surface.blit(
            fonts.get("heading", 15, bold=True).render(sv, True, scol), (trect.x + 10, trect.y + 26)
        )

    buttons: list[ButtonSpec] = []
    by = inner.bottom - 44
    b_defs: list[tuple[str, str, Literal["slate", "cyan", "amber"], str | None]] = [
        ("res_menu", "MAIN MENU", "slate", "icon_home"),
        ("res_replay", "SCRUB REPLAY", "cyan", "icon_replay"),
        ("res_again", "PLAY AGAIN", "amber", "icon_play"),
    ]
    bw = (inner.width - 24) // 3
    for idx, (bid, blabel, bvar, bicon) in enumerate(b_defs):
        spec = ButtonSpec(
            id=bid,
            rect=pygame.Rect(inner.left + idx * (bw + 12), by, bw, 40),
            label=blabel,
            variant=bvar,
            icon_name=bicon,
            font_size=13,
        )
        draw_tactical_button(surface, spec, palette, fonts, vault, hovered=hovered_id == bid)
        buttons.append(spec)
    return buttons


def draw_settings_modal(
    surface: pygame.Surface,
    size: tuple[int, int],
    palette: ThemePalette,
    fonts: FontBook,
    vault: AssetVault,
    settings: GameSettingsState,
    hovered_id: str | None,
) -> tuple[list[ButtonSpec], pygame.Rect]:
    """Render the System & Engine Settings modal and return (buttons, volume_slider_rect)."""
    w, h = size
    draw_alpha_rect(surface, pygame.Rect(0, 0, w, h), (6, 6, 13, 210))
    modal = pygame.Rect(w // 2 - 270, h // 2 - 220, 540, 440)
    inner = draw_tactical_panel(
        surface,
        modal,
        palette,
        fonts,
        title="SYSTEM & ENGINE SETTINGS",
        sub_header="AUDIO SYNTHESIZER, THEME PRESET, CADENCE & OVERLAYS",
        accent="cyan",
        sub_accent="cyan",
        fill_alpha=248,
    )
    buttons: list[ButtonSpec] = []
    y = inner.top + 8

    # Audio toggle + Master Gain slider
    surface.blit(
        fonts.get("heading", 12, bold=True).render("TACTICAL SFX SYNTHESIZER", True, palette.text),
        (inner.left, y),
    )
    sfx_btn = ButtonSpec(
        id="set_sound_toggle",
        rect=pygame.Rect(inner.right - 104, y - 4, 104, 26),
        label="ENABLED" if settings.sound_enabled else "MUTED",
        variant="cyan" if settings.sound_enabled else "red",
        armed=settings.sound_enabled,
        font_size=11,
    )
    draw_tactical_button(surface, sfx_btn, palette, fonts, vault, hovered=hovered_id == sfx_btn.id)
    buttons.append(sfx_btn)
    y += 32

    surface.blit(fonts.get("mono", 11).render("MASTER GAIN", True, palette.muted), (inner.left, y))
    vol_slider = pygame.Rect(inner.left + 115, y + 4, inner.width - 175, 8)
    draw_alpha_rect(
        surface, vol_slider, (22, 34, 48, 230), border_rgba=with_alpha(palette.cyan, 90)
    )
    fill_w = int(vol_slider.width * (settings.master_volume / 100.0))
    if fill_w > 0:
        draw_alpha_rect(
            surface,
            pygame.Rect(vol_slider.x, vol_slider.y, fill_w, vol_slider.height),
            with_alpha(palette.cyan, 210),
        )
    pygame.draw.circle(surface, palette.cyan, (vol_slider.x + fill_w, vol_slider.centery), 6)
    vol_lbl = fonts.get("mono", 11).render(f"{settings.master_volume}%", True, palette.cyan)
    surface.blit(vol_lbl, (inner.right - vol_lbl.get_width(), y))
    y += 34

    # Theme presets
    surface.blit(
        fonts.get("heading", 12, bold=True).render(
            "COLOR PALETTE PROFILE (CONFIG.PY)", True, palette.text
        ),
        (inner.left, y),
    )
    y += 22
    tw = (inner.width - 16) // 3
    themes: tuple[tuple[ThemePresetName, str], ...] = (
        ("hybrid", "NEON HYBRID"),
        ("playbook", "PLAYBOOK"),
        ("warroom", "WAR-ROOM"),
    )
    for idx, (tid, tlabel) in enumerate(themes):
        spec = ButtonSpec(
            id=f"set_theme_{tid}",
            rect=pygame.Rect(inner.left + idx * (tw + 8), y, tw, 30),
            label=tlabel,
            variant="cyan",
            armed=settings.theme == tid,
            font_size=11,
        )
        draw_tactical_button(surface, spec, palette, fonts, vault, hovered=hovered_id == spec.id)
        buttons.append(spec)
    y += 42

    # AI speed & FPS cap
    half_w = (inner.width - 14) // 2
    surface.blit(
        fonts.get("heading", 11, bold=True).render("AI TURN SPEED", True, palette.text),
        (inner.left, y),
    )
    surface.blit(
        fonts.get("heading", 11, bold=True).render("TARGET FRAME CAP", True, palette.text),
        (inner.left + half_w + 14, y),
    )
    y += 20
    sp_w = (half_w - 12) // 4
    for idx, sp in enumerate((0.5, 1.0, 2.0, 4.0)):
        spec = ButtonSpec(
            id=f"set_speed_{sp}",
            rect=pygame.Rect(inner.left + idx * (sp_w + 4), y, sp_w, 28),
            label=f"{sp:g}x",
            variant="amber",
            armed=abs(settings.ai_speed_multiplier - sp) < 0.01,
            font_size=11,
        )
        draw_tactical_button(surface, spec, palette, fonts, vault, hovered=hovered_id == spec.id)
        buttons.append(spec)

    fps_w = (half_w - 8) // 3
    for idx, fps in enumerate((30, 60, 120)):
        spec = ButtonSpec(
            id=f"set_fps_{fps}",
            rect=pygame.Rect(inner.left + half_w + 14 + idx * (fps_w + 4), y, fps_w, 28),
            label=str(fps),
            variant="cyan",
            armed=settings.target_fps == fps,
            font_size=11,
        )
        draw_tactical_button(surface, spec, palette, fonts, vault, hovered=hovered_id == spec.id)
        buttons.append(spec)
    y += 42

    # Overlay toggles
    surface.blit(
        fonts.get("heading", 12, bold=True).render("BATTLEFIELD OVERLAYS", True, palette.text),
        (inner.left, y),
    )
    y += 22
    ov_defs = (
        ("set_ov_heatmap", "SENSOR OVERLAY", settings.show_heatmap),
        ("set_ov_trace", "BREACH TRAIL", settings.show_breach_trace),
        ("set_ov_hardness", "HARDNESS BADGES", settings.show_hardness_badges),
        ("set_ov_tray", "DIAGNOSTICS TRAY", settings.show_diagnostics_tray),
    )
    for idx, (bid, blabel, active) in enumerate(ov_defs):
        col_i = idx % 2
        row_i = idx // 2
        spec = ButtonSpec(
            id=bid,
            rect=pygame.Rect(inner.left + col_i * (half_w + 14), y + row_i * 34, half_w, 28),
            label=blabel,
            variant="cyan",
            armed=active,
            right_tag="ON" if active else "OFF",
            font_size=10,
            align_left=True,
        )
        draw_tactical_button(surface, spec, palette, fonts, vault, hovered=hovered_id == bid)
        buttons.append(spec)

    close_btn = ButtonSpec(
        id="set_close",
        rect=pygame.Rect(inner.right - 166, inner.bottom - 40, 166, 36),
        label="SAVE & APPLY",
        variant="amber",
        icon_name="icon_settings",
        font_size=12,
    )
    draw_tactical_button(
        surface, close_btn, palette, fonts, vault, hovered=hovered_id == "set_close"
    )
    buttons.append(close_btn)
    return buttons, vol_slider


def draw_hotseat_curtain_modal(
    surface: pygame.Surface,
    size: tuple[int, int],
    palette: ThemePalette,
    fonts: FontBook,
    vault: AssetVault,
    next_side: Side,
    hovered_id: str | None,
) -> list[ButtonSpec]:
    """Render the partial-observability privacy curtain between Human vs Human turns."""
    w, h = size
    draw_alpha_rect(surface, pygame.Rect(0, 0, w, h), (6, 6, 13, 245))
    modal = pygame.Rect(w // 2 - 230, h // 2 - 125, 460, 250)
    is_def = next_side is Side.DEFENDER
    accent: Literal["cyan", "red"] = "cyan" if is_def else "red"
    inner = draw_tactical_panel(
        surface,
        modal,
        palette,
        fonts,
        title="HOT-SEAT PRIVACY CURTAIN",
        sub_header=f"PASS TERMINAL TO {next_side.value.upper()} COMMANDER",
        accent=accent,
        sub_accent=accent,
        fill_alpha=248,
    )
    msg = (
        "Attacker infiltration step committed. Intruder location remains hidden."
        if is_def
        else "Defender move locked in secretly. Armed honeypots remain concealed."
    )
    surface.blit(
        fonts.get("body", 16).render(msg, True, palette.text), (inner.left + 8, inner.top + 36)
    )

    reveal_btn = ButtonSpec(
        id="hotseat_reveal",
        rect=pygame.Rect(inner.left + 12, inner.bottom - 48, inner.width - 24, 42),
        label=f"REVEAL {next_side.value.upper()} HUD",
        variant="amber",
        font_size=14,
    )
    draw_tactical_button(
        surface, reveal_btn, palette, fonts, vault, hovered=hovered_id == "hotseat_reveal"
    )
    return [reveal_btn]
