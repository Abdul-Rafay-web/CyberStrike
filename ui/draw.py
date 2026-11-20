from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import pygame

from ui.theme import RGB, RGBA, AssetVault, FontBook, ThemePalette, lerp_color, with_alpha

AccentKind = Literal["cyan", "amber", "violet", "red", "slate"]
ButtonVariant = Literal["cyan", "amber", "red", "violet", "slate"]
ButtonVisualState = Literal["normal", "hover", "pressed", "disabled"]


@dataclass(slots=True)
class ButtonSpec:
    """Interactive tactical button specification and state."""

    id: str
    rect: pygame.Rect
    label: str
    variant: ButtonVariant = "cyan"
    icon_name: str | None = None
    right_tag: str = ""
    armed: bool = False
    disabled: bool = False
    force_state: ButtonVisualState | None = None
    font_size: int = 13
    align_left: bool = False


def accent_rgb(palette: ThemePalette, accent: AccentKind) -> RGB:
    if accent == "amber":
        return palette.amber
    if accent == "violet":
        return palette.violet
    if accent == "red":
        return palette.red
    if accent == "slate":
        return palette.muted
    return palette.cyan


def draw_alpha_rect(
    surface: pygame.Surface,
    rect: pygame.Rect,
    fill_rgba: RGBA,
    border_rgba: RGBA | None = None,
    border_width: int = 1,
    radius: int = 0,
) -> None:
    """Draw a translucent filled rectangle with optional translucent border."""
    if rect.width <= 0 or rect.height <= 0:
        return
    overlay = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    pygame.draw.rect(
        overlay,
        fill_rgba,
        pygame.Rect(0, 0, rect.width, rect.height),
        border_radius=radius,
    )
    if border_rgba is not None and border_width > 0:
        pygame.draw.rect(
            overlay,
            border_rgba,
            pygame.Rect(0, 0, rect.width, rect.height),
            width=border_width,
            border_radius=radius,
        )
    surface.blit(overlay, rect.topleft)


def draw_scanlines(surface: pygame.Surface, rect: pygame.Rect, alpha: int = 8) -> None:
    """Draw subtle horizontal scanlines clipped inside rect."""
    if rect.width <= 0 or rect.height <= 0 or alpha <= 0:
        return
    stripe = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    line_col = (255, 255, 255, alpha)
    for y in range(0, rect.height, 4):
        pygame.draw.line(stripe, line_col, (0, y), (rect.width - 1, y), 1)
    surface.blit(stripe, rect.topleft)


def draw_corner_brackets(
    surface: pygame.Surface,
    rect: pygame.Rect,
    color: RGB,
    size: int = 10,
    thickness: int = 2,
) -> None:
    """Draw sharp military corner brackets around rect."""
    x0, y0 = rect.left, rect.top
    x1, y1 = rect.right - 1, rect.bottom - 1
    pygame.draw.lines(
        surface, color, False, [(x0, y0 + size), (x0, y0), (x0 + size, y0)], thickness
    )
    pygame.draw.lines(
        surface, color, False, [(x1 - size, y0), (x1, y0), (x1, y0 + size)], thickness
    )
    pygame.draw.lines(
        surface, color, False, [(x0, y1 - size), (x0, y1), (x0 + size, y1)], thickness
    )
    pygame.draw.lines(
        surface, color, False, [(x1 - size, y1), (x1, y1), (x1, y1 - size)], thickness
    )


def draw_tactical_panel(
    surface: pygame.Surface,
    rect: pygame.Rect,
    palette: ThemePalette,
    fonts: FontBook,
    title: str = "",
    sub_header: str = "",
    accent: AccentKind = "cyan",
    sub_accent: AccentKind = "amber",
    right_badge: str = "",
    fill_alpha: int = 232,
) -> pygame.Rect:
    """Render an angled tactical panel and return its usable interior rectangle."""
    col = accent_rgb(palette, accent)
    draw_alpha_rect(
        surface,
        rect,
        with_alpha(palette.panel, fill_alpha),
        border_rgba=with_alpha(col, 95),
        border_width=1,
    )
    draw_scanlines(surface, rect, alpha=7)
    draw_corner_brackets(surface, rect, col, size=9, thickness=2)

    cursor_y = rect.top
    if title:
        header_rect = pygame.Rect(rect.left + 1, cursor_y + 1, max(1, rect.width - 2), 30)
        draw_alpha_rect(
            surface,
            header_rect,
            with_alpha(palette.panel_header, 235),
            border_rgba=None,
        )
        pygame.draw.line(
            surface,
            with_alpha(col, 80),
            (rect.left + 1, header_rect.bottom),
            (rect.right - 2, header_rect.bottom),
            1,
        )
        t_surf = fonts.get("heading", 13, bold=True).render(title.upper(), True, palette.text)
        surface.blit(t_surf, (rect.left + 12, header_rect.centery - t_surf.get_height() // 2))
        if right_badge:
            b_surf = fonts.get("mono", 11).render(right_badge, True, col)
            surface.blit(
                b_surf,
                (
                    rect.right - 12 - b_surf.get_width(),
                    header_rect.centery - b_surf.get_height() // 2,
                ),
            )
        cursor_y = header_rect.bottom + 1

    if sub_header:
        sub_col = accent_rgb(palette, sub_accent)
        sub_rect = pygame.Rect(rect.left + 1, cursor_y, max(1, rect.width - 2), 24)
        grad = pygame.Surface((sub_rect.width, sub_rect.height), pygame.SRCALPHA)
        for px in range(sub_rect.width):
            ratio = 1.0 - (px / max(1, sub_rect.width))
            a = int(58 * (ratio**1.35))
            pygame.draw.line(grad, with_alpha(sub_col, a), (px, 0), (px, sub_rect.height - 1))
        surface.blit(grad, sub_rect.topleft)
        pygame.draw.line(
            surface,
            with_alpha(sub_col, 95),
            (rect.left + 1, sub_rect.bottom),
            (rect.right - 2, sub_rect.bottom),
            1,
        )
        s_surf = fonts.get("heading", 11, bold=True).render(sub_header.upper(), True, sub_col)
        surface.blit(s_surf, (rect.left + 12, sub_rect.centery - s_surf.get_height() // 2))
        cursor_y = sub_rect.bottom + 1

    return pygame.Rect(
        rect.left + 10,
        cursor_y + 8,
        max(1, rect.width - 20),
        max(1, rect.bottom - cursor_y - 16),
    )


def draw_tactical_button(
    surface: pygame.Surface,
    spec: ButtonSpec,
    palette: ThemePalette,
    fonts: FontBook,
    vault: AssetVault | None = None,
    hovered: bool = False,
) -> pygame.Rect:
    """Draw a 4-state tactical button with corner notches, icon, and optional tag."""
    rect = spec.rect
    if spec.force_state is not None:
        state: ButtonVisualState = spec.force_state
    elif spec.disabled:
        state = "disabled"
    elif spec.armed:
        state = "pressed"
    elif hovered:
        state = "hover"
    else:
        state = "normal"

    base_col = accent_rgb(palette, spec.variant)
    if state == "disabled":
        bg_rgba: RGBA = (14, 18, 28, 175)
        border_rgba: RGBA = with_alpha(palette.muted, 55)
        text_col: RGB = lerp_color(palette.muted, palette.bg, 0.45)
    elif state == "pressed":
        bg_rgba = with_alpha(base_col, 68)
        border_rgba = with_alpha(base_col, 245)
        text_col = (255, 255, 255) if spec.variant != "amber" else (255, 224, 163)
    elif state == "hover":
        bg_rgba = with_alpha(lerp_color(palette.panel, base_col, 0.25), 238)
        border_rgba = with_alpha(base_col, 235)
        text_col = (255, 255, 255)
    else:
        bg_rgba = with_alpha(lerp_color(palette.panel, base_col, 0.12), 225)
        border_rgba = with_alpha(base_col, 130)
        text_col = (255, 196, 86) if spec.variant == "amber" else palette.text

    draw_alpha_rect(surface, rect, bg_rgba, border_rgba=border_rgba, border_width=1)

    if state in ("hover", "pressed"):
        glow = pygame.Rect(rect.x - 1, rect.y - 1, rect.width + 2, rect.height + 2)
        draw_alpha_rect(
            surface, glow, (0, 0, 0, 0), border_rgba=with_alpha(base_col, 90), border_width=1
        )

    # Corner micro-notches matching TacticalButton
    notch_col = with_alpha(text_col if state != "disabled" else palette.muted, 160)
    pygame.draw.lines(
        surface,
        notch_col,
        False,
        [
            (rect.left + 3, rect.top + 8),
            (rect.left + 3, rect.top + 3),
            (rect.left + 8, rect.top + 3),
        ],
        1,
    )
    pygame.draw.lines(
        surface,
        notch_col,
        False,
        [
            (rect.right - 9, rect.bottom - 4),
            (rect.right - 4, rect.bottom - 4),
            (rect.right - 4, rect.bottom - 9),
        ],
        1,
    )

    label_font = fonts.get("heading", spec.font_size, bold=True)
    label_surf = label_font.render(spec.label.upper(), True, text_col)
    icon_surf: pygame.Surface | None = None
    if spec.icon_name and vault is not None:
        icon_tint: RGB | None = text_col if state == "disabled" else None
        icon_surf = vault.get_icon(spec.icon_name, size=16, tint=icon_tint)

    tag_surf = (
        fonts.get("mono", max(10, spec.font_size - 2)).render(spec.right_tag, True, base_col)
        if spec.right_tag
        else None
    )

    gap = 8
    pad_x = 12
    content_w = label_surf.get_width() + (icon_surf.get_width() + gap if icon_surf else 0)

    if spec.align_left or tag_surf is not None:
        x = rect.left + pad_x
    else:
        x = rect.centerx - content_w // 2

    if icon_surf is not None:
        surface.blit(icon_surf, (x, rect.centery - icon_surf.get_height() // 2))
        x += icon_surf.get_width() + gap

    surface.blit(label_surf, (x, rect.centery - label_surf.get_height() // 2))

    if tag_surf is not None:
        surface.blit(
            tag_surf,
            (rect.right - pad_x - tag_surf.get_width(), rect.centery - tag_surf.get_height() // 2),
        )
    return rect


def draw_stock_pip_bar(
    surface: pygame.Surface,
    pos: tuple[int, int],
    width: int,
    label: str,
    current: int,
    max_count: int,
    color: RGB,
    palette: ThemePalette,
    fonts: FontBook,
) -> int:
    """Draw a Defender stock pip row + tactical slider rail bar. Returns total height."""
    x, y = pos
    mono = fonts.get("mono", 12)
    lbl_surf = mono.render(f"{label.upper()}:", True, palette.text)
    surface.blit(lbl_surf, (x, y))

    pip_x = x + lbl_surf.get_width() + 8
    bracket_l = mono.render("[", True, palette.muted)
    surface.blit(bracket_l, (pip_x, y))
    pip_x += bracket_l.get_width() + 3

    pip_size = 9
    pip_y = y + (lbl_surf.get_height() - pip_size) // 2
    for idx in range(max_count):
        pip_rect = pygame.Rect(pip_x, pip_y, pip_size, pip_size)
        if idx < current:
            pygame.draw.rect(surface, color, pip_rect)
            pygame.draw.rect(surface, (235, 245, 255), pip_rect, 1)
        else:
            pygame.draw.rect(surface, with_alpha(palette.muted, 130), pip_rect, 1)
        pip_x += pip_size + 4

    bracket_r = mono.render("]", True, palette.muted)
    surface.blit(bracket_r, (pip_x, y))

    ratio_surf = mono.render(f"[{current}/{max_count}]", True, palette.muted)
    surface.blit(ratio_surf, (x + width - ratio_surf.get_width(), y))

    bar_y = y + lbl_surf.get_height() + 5
    bar_rect = pygame.Rect(x, bar_y, width, 7)
    draw_alpha_rect(
        surface,
        bar_rect,
        (22, 35, 43, 220),
        border_rgba=with_alpha(palette.cyan, 55),
        border_width=1,
    )
    pct = current / max(1, max_count)
    fill_w = int(round((width - 2) * pct))
    if fill_w > 0:
        fill_rect = pygame.Rect(x + 1, bar_y + 1, fill_w, 5)
        draw_alpha_rect(surface, fill_rect, with_alpha(color, 190))

    notch_x = max(x + 1, min(x + width - 4, x + fill_w - 1))
    pygame.draw.rect(surface, (230, 244, 248), pygame.Rect(notch_x, bar_y - 3, 3, 13))
    return lbl_surf.get_height() + 18


def draw_cyber_logo(
    surface: pygame.Surface,
    pos: tuple[int, int],
    palette: ThemePalette,
    fonts: FontBook,
    vault: AssetVault,
    compact: bool = False,
) -> pygame.Rect:
    """Draw the CyberStrike brand mark in compact header or full hero mode."""
    x, y = pos
    if compact:
        icon = vault.get_branding("window_icon_256", size=24)
        surface.blit(icon, (x, y))
        title_s = fonts.get("title", 14, bold=True).render("CYBERSTRIKE", True, palette.text)
        tx = x + 32
        ty = y + (24 - title_s.get_height()) // 2
        surface.blit(title_s, (tx, ty))
        sub_s = fonts.get("heading", 13, bold=True).render(
            "// INTEL DEFENSE OPERATION", True, (155, 183, 196)
        )
        surface.blit(sub_s, (tx + title_s.get_width() + 10, ty + 1))
        return pygame.Rect(x, y, 32 + title_s.get_width() + 10 + sub_s.get_width(), 24)

    # Full Hero Main Menu branding
    bar_rect = pygame.Rect(x, y + 4, 5, 58)
    for dy in range(bar_rect.height):
        t = dy / max(1, bar_rect.height - 1)
        col = (
            lerp_color(palette.violet, palette.cyan, t * 2)
            if t < 0.5
            else lerp_color(palette.cyan, palette.amber, (t - 0.5) * 2)
        )
        pygame.draw.line(
            surface, col, (bar_rect.left, bar_rect.top + dy), (bar_rect.right, bar_rect.top + dy)
        )

    emblem = vault.get_branding("window_icon_256", size=58)
    surface.blit(emblem, (x + 16, y + 4))

    title_surf = fonts.get("title", 34, bold=True).render("CYBERSTRIKE", True, palette.text)
    surface.blit(title_surf, (x + 86, y + 2))
    sub_surf = fonts.get("heading", 14, bold=True).render(
        "NETWORK INFILTRATION GAME // ADVERSARIAL AI UI", True, palette.muted
    )
    surface.blit(sub_surf, (x + 88, y + 44))

    # Slanted title tabs underneath
    tab_y = y + 74
    tab_x = x
    for text, col in (
        ("AUTONOMOUS SYSTEM TRAFFIC", palette.violet),
        ("PARTIAL OBSERVABILITY CONSOLE", palette.cyan),
    ):
        ts = fonts.get("heading", 11, bold=True).render(text, True, col)
        w = ts.get_width() + 28
        h = 22
        pts = [(tab_x, tab_y), (tab_x + w, tab_y), (tab_x + w - 12, tab_y + h), (tab_x, tab_y + h)]
        poly = pygame.Surface((w + 2, h + 2), pygame.SRCALPHA)
        rel_pts = [(px - tab_x, py - tab_y) for px, py in pts]
        pygame.draw.polygon(poly, with_alpha(col, 42), rel_pts)
        pygame.draw.polygon(poly, with_alpha(col, 165), rel_pts, 1)
        surface.blit(poly, (tab_x, tab_y))
        surface.blit(ts, (tab_x + 10, tab_y + (h - ts.get_height()) // 2))
        tab_x += w + 10

    return pygame.Rect(x, y, max(440, title_surf.get_width() + 90), 102)


def draw_iso_ellipse_ring(
    surface: pygame.Surface,
    center: tuple[int, int],
    rx: int,
    ry: int,
    color: RGB,
    alpha: int = 190,
    width: int = 1,
    dashed: bool = False,
    phase: float = 0.0,
) -> None:
    """Draw a smooth anti-aliased isometric ellipse ring centered strictly on center."""
    if rx <= 1 or ry <= 1 or alpha <= 0:
        return
    pad = width + 3
    w, h = rx * 2 + pad * 2, ry * 2 + pad * 2
    ring_surf = pygame.Surface((w, h), pygame.SRCALPHA)
    cx, cy = w // 2, h // 2
    rgba = with_alpha(color, alpha)

    if not dashed:
        pygame.draw.ellipse(
            ring_surf,
            rgba,
            pygame.Rect(cx - rx, cy - ry, rx * 2, ry * 2),
            width=width,
        )
    else:
        segments = 16
        step = (2.0 * math.pi) / segments
        for i in range(segments):
            if i % 2 == 1:
                continue
            a0 = i * step + phase
            a1 = a0 + step * 0.65
            pts: list[tuple[float, float]] = []
            for k in range(6):
                ang = a0 + (a1 - a0) * (k / 5.0)
                pts.append((cx + rx * math.cos(ang), cy + ry * math.sin(ang)))
            pygame.draw.lines(ring_surf, rgba, False, pts, width)

    surface.blit(ring_surf, (center[0] - cx, center[1] - cy))


def draw_target_reticle_ring(
    surface: pygame.Surface,
    center: tuple[int, int],
    rx: int,
    ry: int,
    color: RGB,
    alpha: int = 230,
    phase: float = 0.0,
) -> None:
    """Draw a tight concentric reticle with rotating ticks locked to machine center."""
    draw_iso_ellipse_ring(
        surface, center, rx, ry, color, alpha=alpha, width=2, dashed=True, phase=phase
    )
    draw_iso_ellipse_ring(
        surface,
        center,
        max(4, rx - 6),
        max(3, ry - 4),
        color,
        alpha=max(40, alpha // 2),
        width=1,
        dashed=False,
    )
    # 4 crisp cardinal tick marks anchored to the exact ellipse perimeter
    tick_len = 6
    cx, cy = center
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        px = cx + dx * rx
        py = cy + dy * ry
        pygame.draw.line(
            surface,
            color,
            (px, py),
            (px + dx * tick_len, py + dy * tick_len),
            2,
        )


def draw_glow_line(
    surface: pygame.Surface,
    start: tuple[int, int],
    end: tuple[int, int],
    color: RGB,
    width: int = 2,
    core_alpha: int = 210,
    glow_alpha: int = 75,
) -> None:
    """Draw a clean glowing neon link line on a tight bounding alpha surface."""
    x0 = min(start[0], end[0]) - 8
    y0 = min(start[1], end[1]) - 8
    x1 = max(start[0], end[0]) + 8
    y1 = max(start[1], end[1]) + 8
    w, h = max(2, x1 - x0), max(2, y1 - y0)
    buf = pygame.Surface((w, h), pygame.SRCALPHA)
    p1 = (start[0] - x0, start[1] - y0)
    p2 = (end[0] - x0, end[1] - y0)
    if glow_alpha > 0:
        pygame.draw.line(buf, with_alpha(color, glow_alpha // 2), p1, p2, width + 6)
        pygame.draw.line(buf, with_alpha(color, glow_alpha), p1, p2, width + 3)
    pygame.draw.line(buf, with_alpha(color, core_alpha), p1, p2, width)
    surface.blit(buf, (x0, y0))


def draw_animated_dash_line(
    surface: pygame.Surface,
    start: tuple[int, int],
    end: tuple[int, int],
    color: RGB,
    width: int = 2,
    dash_len: float = 9.0,
    gap_len: float = 12.0,
    phase: float = 0.0,
) -> None:
    """Draw animated marching dashes from start to end."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length < 2.0:
        return
    ux, uy = dx / length, dy / length
    period = dash_len + gap_len
    offset = (phase * 48.0) % period - period
    dist = offset
    while dist < length:
        s_d = max(0.0, dist)
        e_d = min(length, dist + dash_len)
        if e_d > s_d:
            p1 = (int(round(start[0] + ux * s_d)), int(round(start[1] + uy * s_d)))
            p2 = (int(round(start[0] + ux * e_d)), int(round(start[1] + uy * e_d)))
            pygame.draw.line(surface, color, p1, p2, width)
        dist += period


def draw_firewall_dot_matrix(
    surface: pygame.Surface,
    center: tuple[int, int],
    vault: AssetVault,
    fonts: FontBook,
    palette: ThemePalette,
) -> None:
    """Draw the iconic 4x13 amber dotted firewall matrix barrier at a link midpoint."""
    mx, my = center
    backdrop = pygame.Rect(mx - 54, my - 16, 108, 28)
    draw_alpha_rect(surface, backdrop, (255, 138, 18, 28))

    for r_idx, ry in enumerate((-10, -4, 2, 8)):
        bright = r_idx == 3
        col = (255, 181, 59) if bright else (224, 123, 24)
        rad = 2 if bright else 1
        for c_idx in range(13):
            cx = mx - 48 + c_idx * 8
            pygame.draw.circle(surface, col, (cx, my + ry), rad)

    barrier_icon = vault.get_icon("link_firewall_barrier", size=18)
    surface.blit(barrier_icon, (mx - 9, my - 22))

    lbl = fonts.get("heading", 10, bold=True).render("FIREWALLED", True, palette.amber)
    surface.blit(lbl, (mx - lbl.get_width() // 2, my + 14))


def draw_badge_pill(
    surface: pygame.Surface,
    anchor: tuple[int, int],
    text: str,
    fg: RGB,
    bg_rgba: RGBA,
    border_rgba: RGBA | None,
    font: pygame.font.Font,
    centered: bool = True,
    pad_x: int = 6,
    pad_y: int = 2,
) -> pygame.Rect:
    """Render a compact tactical badge pill and return its drawn rect."""
    ts = font.render(text, True, fg)
    w = ts.get_width() + pad_x * 2
    h = ts.get_height() + pad_y * 2
    x = anchor[0] - w // 2 if centered else anchor[0]
    y = anchor[1] - h // 2 if centered else anchor[1]
    rect = pygame.Rect(x, y, w, h)
    draw_alpha_rect(surface, rect, bg_rgba, border_rgba=border_rgba, border_width=1)
    surface.blit(ts, (rect.x + pad_x, rect.y + pad_y))
    return rect
