from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Literal

import pygame

from config import CONFIG
from core.chokepoints import chokepoints
from core.network import Network
from core.state import Side
from ui.draw import (
    ButtonSpec,
    ButtonVariant,
    ButtonVisualState,
    draw_alpha_rect,
    draw_badge_pill,
    draw_corner_brackets,
    draw_cyber_logo,
    draw_glow_line,
    draw_iso_ellipse_ring,
    draw_stock_pip_bar,
    draw_tactical_button,
    draw_tactical_panel,
)
from ui.layout import (
    MAP_PRESETS,
    MapPreset,
    NodeVisualMeta,
    all_map_presets,
    build_canonical_13_network,
    compute_network_layout,
    project_node_to_screen,
)
from ui.theme import RGB, AssetVault, FontBook, ThemePalette, with_alpha

AtlasTab = Literal["sprites", "ui_kit", "boards"]
SpriteBgMode = Literal["obsidian", "checker", "card"]


@dataclass(slots=True)
class SetupConfig:
    """User-selected mission configuration in Operation Setup."""

    mode: Literal["human_vs_ai", "ai_vs_ai", "human_vs_human"] = "human_vs_ai"
    side: Side = Side.DEFENDER
    seed: int = CONFIG.seeds.master_seed
    map_size: int = 13
    sector_name: str = "INTRA_NET_ALPHA_09"
    use_canonical_13: bool = True
    evolved_map_path: str | None = None


def _draw_network_preview(
    surface: pygame.Surface,
    rect: pygame.Rect,
    network: Network,
    metas: tuple[NodeVisualMeta, ...],
    palette: ThemePalette,
    fonts: FontBook,
    line_rgb: RGB,
) -> None:
    """Render a clean graph preview strictly contained inside rect."""
    draw_alpha_rect(surface, rect, (7, 11, 20, 235), border_rgba=with_alpha(line_rgb, 95))
    prev_clip = surface.get_clip()
    surface.set_clip(rect)
    try:
        pts: dict[int, tuple[int, int]] = {}
        for meta in metas:
            pts[meta.index] = project_node_to_screen(
                meta.nx,
                meta.ny,
                rect,
                pad_left=44,
                pad_right=44,
                pad_top=30,
                pad_bottom=34,
            )

        for a, b in network.edges():
            pygame.draw.line(surface, with_alpha(line_rgb, 110), pts[a], pts[b], 2)

        mono = fonts.get("mono", 10)
        for meta in metas:
            px, py = pts[meta.index]
            if meta.role == "entry":
                stroke = palette.green
            elif meta.role == "crown_jewel":
                stroke = palette.amber
            elif meta.role == "gateway":
                stroke = palette.violet
            else:
                stroke = palette.cyan

            diamond = [(px, py - 9), (px + 15, py), (px, py + 9), (px - 15, py)]
            pygame.draw.polygon(surface, (18, 18, 42), diamond)
            pygame.draw.polygon(surface, stroke, diamond, 2)
            lbl = mono.render(meta.code, True, palette.text)
            lx = px - lbl.get_width() // 2
            ly = py + 11
            draw_alpha_rect(
                surface,
                pygame.Rect(lx - 3, ly - 1, lbl.get_width() + 6, lbl.get_height() + 2),
                (7, 11, 20, 210),
            )
            surface.blit(lbl, (lx, ly))
    finally:
        surface.set_clip(prev_clip)


def _draw_menu_diorama(
    surface: pygame.Surface,
    rect: pygame.Rect,
    palette: ThemePalette,
    fonts: FontBook,
    vault: AssetVault,
) -> None:
    """Render a live miniature isometric topology diorama using cropped 2x sprites."""
    prev_clip = surface.get_clip()
    surface.set_clip(rect)
    try:
        bg = vault.get_cover("backgrounds/warroom_console_4k.jpg", rect.size).copy()
        bg.set_alpha(200)
        surface.blit(bg, rect.topleft)
        surface.blit(vault.get_tiled_iso_grid(rect.size, alpha=130), rect.topleft)

        net = build_canonical_13_network()
        metas = compute_network_layout(net)
        pts = {
            m.index: project_node_to_screen(
                m.nx, m.ny, rect, pad_left=46, pad_right=46, pad_top=30, pad_bottom=72
            )
            for m in metas
        }

        for a, b in net.edges():
            p1 = (pts[a][0], pts[a][1] + 8)
            p2 = (pts[b][0], pts[b][1] + 8)
            draw_glow_line(surface, p1, p2, (132, 175, 190), width=1, core_alpha=115, glow_alpha=0)

        for meta in sorted(metas, key=lambda m: pts[m.index][1]):
            px, py = pts[meta.index]
            bx, by = px, py + 9
            if meta.index == net.jewel:
                skey = "crown_jewel"
                sz = 46
                draw_iso_ellipse_ring(surface, (bx, by), 26, 13, palette.amber, alpha=190, width=1)
            elif meta.index == 8:
                skey = "patched"
                sz = 42
                draw_iso_ellipse_ring(surface, (bx, by), 21, 10, palette.cyan, alpha=170, width=1)
            elif meta.index == net.entry:
                skey = "entry_point"
                sz = 40
                draw_iso_ellipse_ring(surface, (bx, by), 20, 10, palette.green, alpha=150, width=1)
            elif meta.index == 4:
                skey = "selected"
                sz = 40
                draw_iso_ellipse_ring(surface, (bx, by), 21, 10, palette.red, alpha=190, width=1)
            else:
                skey = "normal"
                sz = 38
                draw_iso_ellipse_ring(surface, (bx, by), 18, 9, (90, 118, 132), alpha=95, width=1)

            sprite = vault.get_machine(skey, sz)
            surface.blit(sprite, (px - sz // 2, py - sz // 2))
            lbl = fonts.get("mono", 9).render(meta.code, True, palette.text)
            surface.blit(lbl, (px - lbl.get_width() // 2, by + 10))
    finally:
        surface.set_clip(prev_clip)
    pygame.draw.rect(surface, with_alpha(palette.cyan, 115), rect, 1)


def draw_main_menu_screen(
    surface: pygame.Surface,
    size: tuple[int, int],
    palette: ThemePalette,
    fonts: FontBook,
    vault: AssetVault,
    hovered_id: str | None,
    show_quit_dialog: bool,
) -> list[ButtonSpec]:
    """Render the Main Menu screen with full branding, navigation, and sprite showcase."""
    w, h = size
    surface.fill(palette.bg)

    backdrop = vault.get_cover("backgrounds/warroom_console_4k.jpg", size).copy()
    backdrop.set_alpha(90)
    surface.blit(backdrop, (0, 0))
    surface.blit(vault.get_tiled_iso_grid(size, alpha=130), (0, 0))
    draw_alpha_rect(surface, pygame.Rect(0, 0, w, h), (6, 6, 13, 135))
    draw_corner_brackets(
        surface, pygame.Rect(16, 16, w - 32, h - 32), palette.cyan, size=18, thickness=2
    )

    buttons: list[ButtonSpec] = []

    # Left column: Hero Logo + Description + Navigation Panel
    left_x = 64
    draw_cyber_logo(surface, (left_x, 54), palette, fonts, vault, compact=False)

    desc_lines = (
        "Two adversarial forces clash across a corporate network.",
        "Command defense controls under partial observability or",
        "infiltrate toward the Crown Jewel server (DB-01).",
    )
    dy = 176
    body_font = fonts.get("body", 15)
    for line in desc_lines:
        surface.blit(body_font.render(line, True, (190, 196, 224)), (left_x, dy))
        dy += 21

    nav_rect = pygame.Rect(left_x, 256, 460, 356)
    inner = draw_tactical_panel(
        surface,
        nav_rect,
        palette,
        fonts,
        title="COMMAND NAVIGATION",
        accent="cyan",
        fill_alpha=236,
    )

    ny = inner.top + 4
    nw = inner.width
    nav_defs: list[tuple[str, str, Literal["amber", "cyan", "violet"], str, str, int]] = [
        ("menu_setup", "PLAY OPERATION", "amber", "icon_play", "SETUP ->", 44),
        ("menu_quick_def", "DEFENDER BATTLEFIELD", "cyan", "icon_patch", "QUICK PLAY", 38),
        ("menu_spectator", "WATCH AI VS AI", "violet", "icon_speed", "SPECTATOR", 38),
        ("menu_tutorial", "TUTORIAL MODE", "cyan", "icon_warning", "RULES COACH", 38),
    ]
    for bid, blabel, bvar, bicon, btag, bh in nav_defs:
        spec = ButtonSpec(
            id=bid,
            rect=pygame.Rect(inner.left, ny, nw, bh),
            label=blabel,
            variant=bvar,
            icon_name=bicon,
            right_tag=btag,
            font_size=14 if bh >= 42 else 12,
            align_left=True,
        )
        draw_tactical_button(surface, spec, palette, fonts, vault, hovered=hovered_id == bid)
        buttons.append(spec)
        ny += bh + 10

    half_w = (nw - 10) // 2
    row1: tuple[tuple[str, str, ButtonVariant, str], ...] = (
        ("menu_maps", "MAP LIBRARY", "slate", "icon_heatmap"),
        ("menu_atlas", "4K ASSET ATLAS", "slate", "icon_astar"),
    )
    for idx, (r_id, r_label, r_var, r_icon) in enumerate(row1):
        spec = ButtonSpec(
            id=r_id,
            rect=pygame.Rect(inner.left + idx * (half_w + 10), ny, half_w, 34),
            label=r_label,
            variant=r_var,
            icon_name=r_icon,
            font_size=11,
        )
        draw_tactical_button(surface, spec, palette, fonts, vault, hovered=hovered_id == r_id)
        buttons.append(spec)
    ny += 42

    row2: tuple[tuple[str, str, ButtonVariant, str], ...] = (
        ("menu_settings", "SETTINGS", "slate", "icon_settings"),
        ("menu_quit", "QUIT", "red", "icon_back"),
    )
    for idx, (r_id, r_label, r_var, r_icon) in enumerate(row2):
        spec = ButtonSpec(
            id=r_id,
            rect=pygame.Rect(inner.left + idx * (half_w + 10), ny, half_w, 34),
            label=r_label,
            variant=r_var,
            icon_name=r_icon,
            font_size=11,
        )
        draw_tactical_button(surface, spec, palette, fonts, vault, hovered=hovered_id == r_id)
        buttons.append(spec)

    # Bottom tech stack tags
    bx = left_x
    for tag, col in (
        ("PYTHON 3 / CORE", palette.cyan),
        ("PYGAME 1280x720", palette.violet),
        ("NUMPY / RNG", palette.amber),
        (f"SEED #{CONFIG.seeds.master_seed}", palette.muted),
    ):
        pill = draw_badge_pill(
            surface,
            (bx, 636),
            tag,
            col,
            (10, 14, 24, 210),
            with_alpha(col, 130),
            fonts.get("mono", 11),
            centered=False,
        )
        bx = pill.right + 10

    # Right column: Tactical Preview + 4 Machine Sprites Showcase
    right_rect = pygame.Rect(560, 74, w - 624, 540)
    r_inner = draw_tactical_panel(
        surface,
        right_rect,
        palette,
        fonts,
        title="TACTICAL BATTLEFIELD PREVIEW // INTRA_NET_ALPHA_09",
        sub_header="CROPPED 4K ISOMETRIC SPRITES & WAR-ROOM CONSOLE",
        accent="amber",
        sub_accent="cyan",
    )

    preview_img_rect = pygame.Rect(r_inner.left, r_inner.top + 4, r_inner.width, 286)
    _draw_menu_diorama(surface, preview_img_rect, palette, fonts, vault)

    preview_btn = ButtonSpec(
        id="menu_quick_def_img",
        rect=pygame.Rect(preview_img_rect.left + 12, preview_img_rect.bottom - 40, 260, 28),
        label="ENTER DEFENDER CONSOLE",
        variant="cyan",
        icon_name="icon_play",
        font_size=11,
    )
    draw_tactical_button(
        surface, preview_btn, palette, fonts, vault, hovered=hovered_id == preview_btn.id
    )
    buttons.append(preview_btn)

    fps_font = fonts.get("mono", 10)
    fps_w = fps_font.size("60 FPS READY")[0] + 14
    draw_badge_pill(
        surface,
        (preview_img_rect.right - fps_w - 12, preview_img_rect.bottom - 36),
        "60 FPS READY",
        palette.amber,
        (6, 6, 13, 230),
        with_alpha(palette.amber, 165),
        fps_font,
        centered=False,
    )

    # 4 Core Machine Sprites Showcase row
    card_y = preview_img_rect.bottom + 16
    card_w = (r_inner.width - 30) // 4
    card_h = r_inner.bottom - card_y - 4
    showcase = (
        ("entry_point", "ENTRY POINT", palette.green),
        ("patched", "PATCHED", palette.cyan),
        ("honeypot", "HONEYPOT", palette.violet),
        ("crown_jewel", "CROWN JEWEL", palette.amber),
    )
    for idx, (skey, slabel, scol) in enumerate(showcase):
        crect = pygame.Rect(r_inner.left + idx * (card_w + 10), card_y, card_w, card_h)
        is_hov = hovered_id == f"menu_sprite_{skey}"
        draw_alpha_rect(
            surface,
            crect,
            (9, 14, 24, 235),
            border_rgba=with_alpha(palette.cyan if is_hov else scol, 195 if is_hov else 85),
        )
        sprite = vault.get_machine(skey, 84)
        surface.blit(sprite, (crect.centerx - 42, crect.top + 12))
        ls = fonts.get("heading", 11, bold=True).render(slabel, True, scol)
        surface.blit(ls, (crect.centerx - ls.get_width() // 2, crect.bottom - 24))
        buttons.append(ButtonSpec(id=f"menu_sprite_{skey}", rect=crect, label=""))

    if show_quit_dialog:
        draw_alpha_rect(surface, pygame.Rect(0, 0, w, h), (6, 6, 13, 200))
        q_rect = pygame.Rect(w // 2 - 210, h // 2 - 105, 420, 210)
        q_in = draw_tactical_panel(
            surface,
            q_rect,
            palette,
            fonts,
            title="TERMINATE CYBERSTRIKE",
            accent="red",
            fill_alpha=248,
        )
        surface.blit(
            fonts.get("body", 16).render(
                "Exit the CyberStrike desktop window cleanly?", True, palette.text
            ),
            (q_in.left + 12, q_in.top + 24),
        )
        cancel_btn = ButtonSpec(
            id="quit_cancel",
            rect=pygame.Rect(q_in.left + 12, q_in.bottom - 44, 175, 36),
            label="CANCEL",
            variant="slate",
            font_size=12,
        )
        confirm_btn = ButtonSpec(
            id="quit_confirm",
            rect=pygame.Rect(q_in.right - 187, q_in.bottom - 44, 175, 36),
            label="EXIT GAME",
            variant="red",
            font_size=12,
        )
        draw_tactical_button(
            surface, cancel_btn, palette, fonts, vault, hovered=hovered_id == "quit_cancel"
        )
        draw_tactical_button(
            surface, confirm_btn, palette, fonts, vault, hovered=hovered_id == "quit_confirm"
        )
        buttons.extend([cancel_btn, confirm_btn])

    return buttons


def draw_game_setup_screen(
    surface: pygame.Surface,
    size: tuple[int, int],
    palette: ThemePalette,
    fonts: FontBook,
    vault: AssetVault,
    cfg: SetupConfig,
    preview_network: Network,
    preview_metas: tuple[NodeVisualMeta, ...],
    hovered_id: str | None,
) -> tuple[list[ButtonSpec], pygame.Rect]:
    """Render Operation Setup screen and return (buttons, map_size_slider_rect)."""
    w, h = size
    surface.fill(palette.bg)
    surface.blit(vault.get_tiled_iso_grid(size, alpha=110), (0, 0))
    buttons: list[ButtonSpec] = []

    surface.blit(
        fonts.get("title", 22, bold=True).render("02 // OPERATION SETUP", True, palette.text),
        (52, 28),
    )
    surface.blit(
        fonts.get("heading", 12, bold=True).render(
            "CONFIGURE MODE, PERSPECTIVE, DETERMINISTIC SEED & TOPOLOGY SCALE", True, palette.cyan
        ),
        (52, 58),
    )
    back_btn = ButtonSpec(
        id="setup_back",
        rect=pygame.Rect(w - 180, 32, 128, 34),
        label="MAIN MENU",
        variant="slate",
        icon_name="icon_back",
        font_size=11,
    )
    draw_tactical_button(
        surface, back_btn, palette, fonts, vault, hovered=hovered_id == "setup_back"
    )
    buttons.append(back_btn)

    left_rect = pygame.Rect(52, 92, 690, h - 128)
    inner = draw_tactical_panel(
        surface,
        left_rect,
        palette,
        fonts,
        title="MISSION PARAMETERS",
        sub_header="MODE, FACTION, SEED & TOPOLOGY SIZE",
        accent="cyan",
        sub_accent="cyan",
    )

    y = inner.top + 8
    # 1. Mode
    surface.blit(
        fonts.get("heading", 12, bold=True).render("1. ENGAGEMENT MODE", True, palette.muted),
        (inner.left, y),
    )
    y += 22
    mw = (inner.width - 16) // 3
    for idx, (mid, mlbl) in enumerate(
        (
            ("human_vs_ai", "HUMAN VS AI"),
            ("ai_vs_ai", "AI VS AI"),
            ("human_vs_human", "HUMAN VS HUMAN"),
        )
    ):
        spec = ButtonSpec(
            id=f"setup_mode_{mid}",
            rect=pygame.Rect(inner.left + idx * (mw + 8), y, mw, 36),
            label=mlbl,
            variant="cyan",
            armed=cfg.mode == mid,
            font_size=12,
        )
        draw_tactical_button(surface, spec, palette, fonts, vault, hovered=hovered_id == spec.id)
        buttons.append(spec)
    y += 52

    # 2. Perspective
    surface.blit(
        fonts.get("heading", 12, bold=True).render("2. COMMAND PERSPECTIVE", True, palette.muted),
        (inner.left, y),
    )
    y += 22
    sw = (inner.width - 12) // 2
    for idx, (sid, slbl, svar) in enumerate(
        (
            ("defender", "DEFENDER VIEW (PARTIAL OBS)", "cyan"),
            ("attacker", "ATTACKER VIEW (INFILTRATION)", "red"),
        )
    ):
        is_armed = (cfg.side is Side.DEFENDER and sid == "defender") or (
            cfg.side is Side.ATTACKER and sid == "attacker"
        )
        spec = ButtonSpec(
            id=f"setup_side_{sid}",
            rect=pygame.Rect(inner.left + idx * (sw + 12), y, sw, 42),
            label=slbl,
            variant=svar,  # type: ignore[arg-type]
            armed=is_armed,
            font_size=12,
        )
        draw_tactical_button(surface, spec, palette, fonts, vault, hovered=hovered_id == spec.id)
        buttons.append(spec)
    y += 58

    # 3. Seed & 4. Map Size
    half_w = (inner.width - 16) // 2
    surface.blit(
        fonts.get("heading", 12, bold=True).render(
            "3. DETERMINISTIC MAP SEED", True, palette.muted
        ),
        (inner.left, y),
    )
    surface.blit(
        fonts.get("heading", 12, bold=True).render(
            f"4. MAP SIZE: {cfg.map_size} NODES", True, palette.amber
        ),
        (inner.left + half_w + 16, y),
    )
    y += 22

    seed_box = pygame.Rect(inner.left, y, half_w - 170, 34)
    draw_alpha_rect(surface, seed_box, (9, 14, 24, 240), border_rgba=with_alpha(palette.cyan, 130))
    s_txt = fonts.get("mono", 15).render(f"#{cfg.seed}", True, palette.cyan)
    surface.blit(
        s_txt,
        (seed_box.centerx - s_txt.get_width() // 2, seed_box.centery - s_txt.get_height() // 2),
    )

    sx = seed_box.right + 6
    for bid, blbl in (
        ("setup_seed_dec", "-1"),
        ("setup_seed_inc", "+1"),
        ("setup_seed_rand", "DICE"),
    ):
        bw = 44 if bid != "setup_seed_rand" else 68
        spec = ButtonSpec(
            id=bid,
            rect=pygame.Rect(sx, y, bw, 34),
            label=blbl,
            variant="amber" if bid == "setup_seed_rand" else "slate",
            font_size=11,
        )
        draw_tactical_button(surface, spec, palette, fonts, vault, hovered=hovered_id == bid)
        buttons.append(spec)
        sx += bw + 4

    slider_x = inner.left + half_w + 16
    slider_rect = pygame.Rect(slider_x, y + 12, half_w, 10)
    draw_alpha_rect(
        surface, slider_rect, (20, 34, 46, 235), border_rgba=with_alpha(palette.cyan, 90)
    )
    ratio = (cfg.map_size - CONFIG.game.min_machines) / max(
        1, CONFIG.game.max_machines - CONFIG.game.min_machines
    )
    fill_w = int(slider_rect.width * ratio)
    if fill_w > 0:
        draw_alpha_rect(
            surface,
            pygame.Rect(slider_rect.x, slider_rect.y, fill_w, slider_rect.height),
            with_alpha(palette.cyan, 210),
        )
    pygame.draw.circle(surface, palette.amber, (slider_rect.x + fill_w, slider_rect.centery), 7)
    surface.blit(fonts.get("mono", 10).render("10 (MIN)", True, palette.muted), (slider_x, y + 26))
    surface.blit(
        fonts.get("mono", 10).render("15 (DEF)", True, palette.muted),
        (slider_x + half_w // 2 - 24, y + 26),
    )
    surface.blit(
        fonts.get("mono", 10).render("25 (MAX)", True, palette.muted),
        (slider_x + half_w - 48, y + 26),
    )
    y += 58

    # 5. Preset quick selectors
    surface.blit(
        fonts.get("heading", 12, bold=True).render(
            "5. CURATED TOPOLOGY PRESETS", True, palette.muted
        ),
        (inner.left, y),
    )
    y += 22
    pw = (inner.width - 12) // 2
    for idx, preset in enumerate(MAP_PRESETS):
        col_i = idx % 2
        row_i = idx // 2
        bid = f"setup_preset_{preset.id}"
        spec = ButtonSpec(
            id=bid,
            rect=pygame.Rect(inner.left + col_i * (pw + 12), y + row_i * 42, pw, 36),
            label=preset.name,
            variant="violet",
            armed=cfg.sector_name == preset.sector_code,
            right_tag=f"{preset.node_count}N",
            font_size=11,
            align_left=True,
        )
        draw_tactical_button(surface, spec, palette, fonts, vault, hovered=hovered_id == bid)
        buttons.append(spec)

    start_btn = ButtonSpec(
        id="setup_start",
        rect=pygame.Rect(inner.right - 220, inner.bottom - 48, 220, 44),
        label="START OPERATION",
        variant="amber",
        icon_name="icon_play",
        font_size=14,
    )
    draw_tactical_button(
        surface, start_btn, palette, fonts, vault, hovered=hovered_id == "setup_start"
    )
    buttons.append(start_btn)

    # Right column: Live Seeded Graph Preview
    right_rect = pygame.Rect(762, 92, w - 814, h - 128)
    sub_hdr = (
        f"SEED #{cfg.seed} // {preview_network.size} MACHINES "
        f"// {len(preview_network.edges())} LINKS"
    )
    r_inner = draw_tactical_panel(
        surface,
        right_rect,
        palette,
        fonts,
        title=f"LIVE TOPOLOGY // {cfg.sector_name}",
        sub_header=sub_hdr,
        accent="violet",
        sub_accent="violet",
    )

    graph_rect = pygame.Rect(r_inner.left, r_inner.top + 4, r_inner.width, 320)
    _draw_network_preview(
        surface, graph_rect, preview_network, preview_metas, palette, fonts, palette.cyan
    )

    choke_cnt = len(chokepoints(preview_network))
    info_y = graph_rect.bottom + 18
    for k, v, col in (
        (
            "CONNECTIVITY:",
            "CONNECTED" if preview_network.is_connected() else "DISCONNECTED",
            palette.green,
        ),
        ("CHOKEPOINTS:", f"{choke_cnt} ARTICULATION NODES", palette.amber),
        ("DEFENDER STOCK:", "3 PATCH // 2 FW // 2 HONEYPOT", palette.cyan),
        ("TURN LIMIT:", f"{CONFIG.game.turn_limit} TURNS", palette.text),
    ):
        surface.blit(fonts.get("mono", 11).render(k, True, palette.muted), (r_inner.left, info_y))
        vs = fonts.get("mono", 11).render(v, True, col)
        surface.blit(vs, (r_inner.right - vs.get_width(), info_y))
        info_y += 28

    return buttons, slider_rect


def draw_map_library_screen(
    surface: pygame.Surface,
    size: tuple[int, int],
    palette: ThemePalette,
    fonts: FontBook,
    vault: AssetVault,
    selected_preset: MapPreset,
    preview_network: Network,
    preview_metas: tuple[NodeVisualMeta, ...],
    hovered_id: str | None,
) -> list[ButtonSpec]:
    """Render the Network Map Library screen."""
    w, h = size
    surface.fill(palette.bg)
    surface.blit(vault.get_tiled_iso_grid(size, alpha=110), (0, 0))
    buttons: list[ButtonSpec] = []

    surface.blit(
        fonts.get("title", 22, bold=True).render("NETWORK MAP LIBRARY", True, palette.text),
        (52, 28),
    )
    surface.blit(
        fonts.get("heading", 12, bold=True).render(
            "CURATED & PROCEDURAL ENTERPRISE TOPOLOGIES FROM CORE.GENERATOR", True, palette.violet
        ),
        (52, 58),
    )
    back_btn = ButtonSpec(
        id="maps_back",
        rect=pygame.Rect(w - 180, 32, 128, 34),
        label="BACK TO MENU",
        variant="slate",
        icon_name="icon_back",
        font_size=11,
    )
    draw_tactical_button(
        surface, back_btn, palette, fonts, vault, hovered=hovered_id == "maps_back"
    )
    buttons.append(back_btn)

    # Left list of preset cards
    ly = 96
    lw = 445
    for preset in all_map_presets():
        crect = pygame.Rect(52, ly, lw, 118)
        active = preset.id == selected_preset.id
        bid = f"maps_select_{preset.id}"
        is_hov = hovered_id == bid
        draw_alpha_rect(
            surface,
            crect,
            with_alpha(palette.violet, 40) if active else (12, 19, 34, 232),
            border_rgba=with_alpha(
                palette.violet if active or is_hov else palette.cyan, 210 if active else 75
            ),
        )
        surface.blit(
            fonts.get("heading", 14, bold=True).render(preset.name, True, palette.text),
            (crect.x + 14, crect.y + 12),
        )
        badge_font = fonts.get("mono", 10)
        badge_w = badge_font.size(preset.badge)[0] + 14
        draw_badge_pill(
            surface,
            (crect.right - badge_w - 12, crect.y + 12),
            preset.badge,
            palette.cyan,
            with_alpha(palette.cyan, 35),
            with_alpha(palette.cyan, 120),
            badge_font,
            centered=False,
        )
        wrapped = textwrap.wrap(preset.description, width=54)[:2]
        for line_i, wline in enumerate(wrapped):
            surface.blit(
                fonts.get("body", 14).render(wline, True, (165, 174, 208)),
                (crect.x + 14, crect.y + 42 + line_i * 19),
            )
        surface.blit(
            fonts.get("mono", 11).render(
                f"NODES: {preset.node_count}   |   SEED: #{preset.seed}", True, palette.amber
            ),
            (crect.x + 14, crect.bottom - 24),
        )
        buttons.append(ButtonSpec(id=bid, rect=crect, label=""))
        ly += 130

    # Right selected blueprint panel
    right_rect = pygame.Rect(522, 96, w - 574, h - 132)
    r_inner = draw_tactical_panel(
        surface,
        right_rect,
        palette,
        fonts,
        title=f"BLUEPRINT // {selected_preset.sector_code}",
        sub_header=f"SEED #{selected_preset.seed} // {preview_network.size} MACHINES",
        accent="cyan",
        sub_accent="violet",
    )
    graph_rect = pygame.Rect(r_inner.left, r_inner.top + 4, r_inner.width, 320)
    _draw_network_preview(
        surface, graph_rect, preview_network, preview_metas, palette, fonts, palette.violet
    )

    mean_h = sum(preview_network.hardness) / max(1, preview_network.size)
    choke_cnt = len(chokepoints(preview_network))
    stats = (
        ("MACHINES", str(preview_network.size), palette.cyan),
        ("LINKS", str(len(preview_network.edges())), palette.text),
        ("CHOKEPOINTS", f"{choke_cnt} NODES", palette.violet),
        ("MEAN HARDNESS", f"{mean_h:.1f} TURNS", palette.amber),
    )
    sy = graph_rect.bottom + 14
    sw = (r_inner.width - 30) // 4
    for idx, (sk, sv, scol) in enumerate(stats):
        srect = pygame.Rect(r_inner.left + idx * (sw + 10), sy, sw, 56)
        draw_alpha_rect(surface, srect, (11, 18, 30, 235), border_rgba=with_alpha(palette.cyan, 65))
        surface.blit(
            fonts.get("mono", 10).render(sk, True, palette.muted), (srect.x + 8, srect.y + 8)
        )
        surface.blit(
            fonts.get("heading", 14, bold=True).render(sv, True, scol), (srect.x + 8, srect.y + 28)
        )

    by = r_inner.bottom - 44
    bw = (r_inner.width - 20) // 3
    for idx, (bid, blbl, bvar) in enumerate(
        (
            ("maps_launch_spec", "WATCH AI VS AI", "violet"),
            ("maps_launch_atk", "PLAY AS ATTACKER", "red"),
            ("maps_launch_def", "DEFEND SECTOR", "amber"),
        )
    ):
        spec = ButtonSpec(
            id=bid,
            rect=pygame.Rect(r_inner.left + idx * (bw + 10), by, bw, 40),
            label=blbl,
            variant=bvar,  # type: ignore[arg-type]
            icon_name="icon_play" if bid == "maps_launch_def" else None,
            font_size=12,
        )
        draw_tactical_button(surface, spec, palette, fonts, vault, hovered=hovered_id == bid)
        buttons.append(spec)

    return buttons


def draw_asset_atlas_screen(
    surface: pygame.Surface,
    size: tuple[int, int],
    palette: ThemePalette,
    fonts: FontBook,
    vault: AssetVault,
    tab: AtlasTab,
    bg_mode: SpriteBgMode,
    hovered_id: str | None,
) -> list[ButtonSpec]:
    """Render the 4K Asset Atlas & UI Design System screen."""
    w, h = size
    surface.fill(palette.bg)
    surface.blit(vault.get_tiled_iso_grid(size, alpha=105), (0, 0))
    buttons: list[ButtonSpec] = []

    surface.blit(
        fonts.get("title", 20, bold=True).render(
            "CYBERSTRIKE // 4K ASSET ATLAS", True, palette.text
        ),
        (44, 24),
    )
    surface.blit(
        fonts.get("heading", 11, bold=True).render(
            "CROPPED RGBA SPRITES, 64x64 ICONS, EFFECTS, UI KIT STATES & CONFIG.PY TOKENS",
            True,
            palette.cyan,
        ),
        (44, 52),
    )

    back_btn = ButtonSpec(
        id="atlas_back",
        rect=pygame.Rect(w - 150, 26, 106, 32),
        label="BACK",
        variant="slate",
        icon_name="icon_back",
        font_size=11,
    )
    draw_tactical_button(
        surface, back_btn, palette, fonts, vault, hovered=hovered_id == "atlas_back"
    )
    buttons.append(back_btn)

    # Tabs row
    tx = 44
    for tid, tlbl in (
        ("sprites", "01 MACHINES"),
        ("ui_kit", "02 ICONS & UI KIT"),
        ("boards", "03 PALETTE & 4K BOARDS"),
    ):
        spec = ButtonSpec(
            id=f"atlas_tab_{tid}",
            rect=pygame.Rect(tx, 78, 190, 30),
            label=tlbl,
            variant="cyan",
            armed=tab == tid,
            font_size=11,
        )
        draw_tactical_button(surface, spec, palette, fonts, vault, hovered=hovered_id == spec.id)
        buttons.append(spec)
        tx += 200

    if tab == "sprites":
        # Background mode toggles
        bx = w - 418
        for mid, mlbl in (("obsidian", "OBSIDIAN"), ("checker", "CHECKER"), ("card", "ATLAS CARD")):
            spec = ButtonSpec(
                id=f"atlas_bg_{mid}",
                rect=pygame.Rect(bx, 78, 118, 30),
                label=mlbl,
                variant="amber",
                armed=bg_mode == mid,
                font_size=10,
            )
            draw_tactical_button(
                surface, spec, palette, fonts, vault, hovered=hovered_id == spec.id
            )
            buttons.append(spec)
            bx += 124

        specs = (
            ("normal", "NORMAL NODE", "card_normal.jpg", palette.cyan),
            ("breached", "BREACHED", "card_breached.jpg", palette.red),
            ("under_breach", "UNDER BREACH", "card_under_breach.jpg", palette.amber),
            ("patched", "PATCHED (+2 HRD)", "card_patched.jpg", palette.cyan),
            ("honeypot", "HONEYPOT", "card_honeypot.jpg", palette.violet),
            ("crown_jewel", "CROWN JEWEL", "card_crown_jewel.jpg", palette.amber),
            ("entry_point", "ENTRY POINT", "card_entry_point.jpg", palette.green),
            ("selected", "HOVER / SELECTED", "card_selected.jpg", palette.cyan),
        )
        cw = (w - 118) // 4
        ch = (h - 150) // 2
        for idx, (skey, stitle, card_file, col) in enumerate(specs):
            col_i = idx % 4
            row_i = idx // 4
            crect = pygame.Rect(44 + col_i * (cw + 10), 120 + row_i * (ch + 12), cw, ch)
            draw_alpha_rect(surface, crect, (10, 16, 28, 235), border_rgba=with_alpha(col, 115))
            img_area = pygame.Rect(crect.x + 8, crect.y + 8, crect.width - 16, crect.height - 54)
            if bg_mode == "checker":
                for cy in range(img_area.top, img_area.bottom, 16):
                    for cx in range(img_area.left, img_area.right, 16):
                        c = (23, 32, 48) if ((cx // 16 + cy // 16) % 2 == 0) else (14, 20, 32)
                        pygame.draw.rect(
                            surface,
                            c,
                            pygame.Rect(
                                cx, cy, min(16, img_area.right - cx), min(16, img_area.bottom - cy)
                            ),
                        )
            else:
                pygame.draw.rect(surface, (6, 6, 13), img_area)

            if bg_mode == "card":
                c_img = vault.get_contain(
                    f"atlas/{card_file}", (img_area.width - 8, img_area.height - 8)
                )
                surface.blit(
                    c_img,
                    (
                        img_area.centerx - c_img.get_width() // 2,
                        img_area.centery - c_img.get_height() // 2,
                    ),
                )
            else:
                s_sz = min(img_area.width, img_area.height) - 16
                sp_img = vault.get_machine(skey, max(64, s_sz))
                surface.blit(
                    sp_img,
                    (
                        img_area.centerx - sp_img.get_width() // 2,
                        img_area.centery - sp_img.get_height() // 2,
                    ),
                )

            surface.blit(
                fonts.get("heading", 12, bold=True).render(stitle, True, palette.text),
                (crect.x + 12, crect.bottom - 40),
            )
            surface.blit(
                fonts.get("mono", 10).render(f"machine_{skey}.png", True, col),
                (crect.x + 12, crect.bottom - 22),
            )

    elif tab == "ui_kit":
        # Left: 18 icons + 5 effects
        icons_panel = pygame.Rect(44, 122, 690, h - 154)
        i_in = draw_tactical_panel(
            surface,
            icons_panel,
            palette,
            fonts,
            title="64x64 ICONS & TINTED EFFECTS",
            accent="cyan",
        )
        col_n = 6
        tile_w = (i_in.width - (col_n - 1) * 8) // col_n
        for idx, ic in enumerate(vault.ICON_NAMES):
            ci = idx % col_n
            ri = idx // col_n
            trect = pygame.Rect(i_in.left + ci * (tile_w + 8), i_in.top + 4 + ri * 76, tile_w, 68)
            draw_alpha_rect(
                surface, trect, (8, 13, 22, 230), border_rgba=with_alpha(palette.cyan, 60)
            )
            surface.blit(vault.get_icon(ic, size=32), (trect.centerx - 16, trect.y + 8))
            short = ic.replace("icon_", "").replace("link_", "")
            ls = fonts.get("mono", 9).render(short[:12], True, palette.muted)
            surface.blit(ls, (trect.centerx - ls.get_width() // 2, trect.bottom - 18))

        ey = i_in.top + 240
        surface.blit(
            fonts.get("heading", 12, bold=True).render("EFFECT PRIMITIVES", True, palette.violet),
            (i_in.left, ey),
        )
        ew = (i_in.width - 4 * 8) // 5
        for idx, ef in enumerate(vault.EFFECT_NAMES):
            erect = pygame.Rect(i_in.left + idx * (ew + 8), ey + 24, ew, 110)
            draw_alpha_rect(
                surface, erect, (8, 13, 22, 230), border_rgba=with_alpha(palette.violet, 85)
            )
            esurf = vault.get_effect(ef, (64, 64), tint=palette.cyan, alpha=230)
            surface.blit(esurf, (erect.centerx - 32, erect.y + 12))
            el = fonts.get("mono", 9).render(ef.replace("effect_", ""), True, (216, 180, 254))
            surface.blit(el, (erect.centerx - el.get_width() // 2, erect.bottom - 20))

        # Right: 4-state buttons & Stock bar demo
        kit_panel = pygame.Rect(754, 122, w - 798, h - 154)
        k_in = draw_tactical_panel(
            surface,
            kit_panel,
            palette,
            fonts,
            title="4-STATE BUTTONS & STOCK WIDGETS",
            accent="amber",
        )
        ky = k_in.top + 8
        demo_states: tuple[ButtonVisualState, ...] = ("normal", "hover", "pressed", "disabled")
        for state_name in demo_states:
            spec = ButtonSpec(
                id=f"demo_{state_name}",
                rect=pygame.Rect(k_in.left, ky, k_in.width, 38),
                label=f"STATE: {state_name.upper()}",
                variant="cyan",
                icon_name="icon_patch",
                force_state=state_name,
                font_size=12,
            )
            draw_tactical_button(surface, spec, palette, fonts, vault)
            ky += 48

        ky += 16
        draw_stock_pip_bar(
            surface, (k_in.left, ky), k_in.width, "PATCHES", 2, 3, palette.cyan, palette, fonts
        )
        ky += 48
        draw_stock_pip_bar(
            surface, (k_in.left, ky), k_in.width, "FIREWALLS", 1, 2, palette.amber, palette, fonts
        )
        ky += 48
        draw_stock_pip_bar(
            surface, (k_in.left, ky), k_in.width, "HONEYPOTS", 2, 2, palette.violet, palette, fonts
        )

    else:
        # 03: Config.py palette swatches + 4K Reference Boards
        pal_rect = pygame.Rect(44, 122, w - 88, 130)
        p_in = draw_tactical_panel(
            surface, pal_rect, palette, fonts, title="CONFIG.PY COLOR TOKENS", accent="cyan"
        )
        swatches = (
            ("BG", CONFIG.theme.background),
            ("PANEL", CONFIG.theme.panel),
            ("VIOLET", CONFIG.theme.violet),
            ("CYAN", CONFIG.theme.cyan),
            ("ORANGE", CONFIG.theme.orange),
            ("RED", CONFIG.theme.red),
            ("TEXT", CONFIG.theme.text),
            ("MUTED", CONFIG.theme.muted),
        )
        sw = (p_in.width - 7 * 10) // 8
        for idx, (sname, srgb) in enumerate(swatches):
            srect = pygame.Rect(p_in.left + idx * (sw + 10), p_in.top + 4, sw, 68)
            draw_alpha_rect(surface, srect, (9, 14, 24, 235), border_rgba=(255, 255, 255, 45))
            pygame.draw.rect(
                surface, srgb, pygame.Rect(srect.x + 6, srect.y + 6, srect.width - 12, 24)
            )
            hex_s = f"#{srgb[0]:02X}{srgb[1]:02X}{srgb[2]:02X}"
            surface.blit(
                fonts.get("mono", 10).render(hex_s, True, palette.text), (srect.x + 6, srect.y + 34)
            )
            surface.blit(
                fonts.get("heading", 10, bold=True).render(sname, True, palette.muted),
                (srect.x + 6, srect.y + 50),
            )

        boards_rect = pygame.Rect(44, 266, w - 88, h - 298)
        b_in = draw_tactical_panel(
            surface, boards_rect, palette, fonts, title="4K REFERENCE BOARDS", accent="amber"
        )
        bw = (b_in.width - 24) // 3
        for idx, (fname, title_txt) in enumerate(
            (
                ("atlas/defender_ref_4k.jpg", "DEFENDER CONSOLE (2752x1536)"),
                ("atlas/attacker_ref_4k.jpg", "SECTOR VARIANT (2752x1536)"),
                ("atlas/asset_atlas_4k.jpg", "SPRITE ATLAS (2400x1792)"),
            )
        ):
            rrect = pygame.Rect(b_in.left + idx * (bw + 12), b_in.top + 8, bw, b_in.height - 16)
            draw_alpha_rect(
                surface, rrect, (8, 13, 22, 235), border_rgba=with_alpha(palette.cyan, 85)
            )
            thumb = vault.get_cover(fname, (rrect.width - 12, rrect.height - 46))
            surface.blit(thumb, (rrect.x + 6, rrect.y + 6))
            surface.blit(
                fonts.get("mono", 10).render(title_txt, True, palette.text),
                (rrect.x + 10, rrect.bottom - 26),
            )

    return buttons
