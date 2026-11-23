from __future__ import annotations

import math

import pygame

from ui.draw import draw_alpha_rect, draw_badge_pill
from ui.layout import NodeVisualMeta
from ui.theme import RGB, FontBook, ThemePalette, with_alpha

_DISC_CACHE: dict[tuple[int, int, RGB, int], pygame.Surface] = {}
_MAX_DISC_CACHE = 256


def _belief_color(prob: float, palette: ThemePalette) -> RGB:
    """Map belief probability in [0, 1] to CyberStrike's neon gradient."""
    if prob >= 0.45:
        return palette.red
    if prob >= 0.22:
        return palette.amber
    if prob >= 0.10:
        return palette.violet
    return palette.cyan


def _get_glow_disc(rx: int, ry: int, color: RGB, peak_alpha: int) -> pygame.Surface:
    """Pre-render and cache a soft multi-layered isometric glowing disc."""
    key = (rx, ry, color, peak_alpha)
    cached = _DISC_CACHE.get(key)
    if cached is not None:
        return cached

    if len(_DISC_CACHE) >= _MAX_DISC_CACHE:
        _DISC_CACHE.clear()

    pad = 4
    w = rx * 2 + pad * 2
    h = ry * 2 + pad * 2
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    layers = 5
    for layer in range(layers, 0, -1):
        frac = layer / layers
        cur_rx = max(4, int(rx * frac))
        cur_ry = max(3, int(ry * frac))
        alpha = max(6, int(peak_alpha * (1.0 - frac * 0.72) / (layers * 0.42)))
        ellipse_rect = pygame.Rect(
            w // 2 - cur_rx,
            h // 2 - cur_ry,
            cur_rx * 2,
            cur_ry * 2,
        )
        pygame.draw.ellipse(surf, with_alpha(color, min(220, alpha)), ellipse_rect)

    if pygame.display.get_surface() is not None:
        surf = surf.convert_alpha()
    _DISC_CACHE[key] = surf
    return surf


def draw_belief_heatmap(
    surface: pygame.Surface,
    metas: tuple[NodeVisualMeta, ...],
    base_pts: dict[int, tuple[int, int]],
    belief: tuple[float, ...],
    palette: ThemePalette,
    fonts: FontBook,
    time_s: float,
) -> None:
    """Draw soft pre-rendered glowing discs sized by belief probability around each machine."""
    if len(belief) != len(metas):
        return
    pulse = 0.5 + 0.5 * math.sin(time_s * 2.8)
    for meta in metas:
        prob = belief[meta.index]
        if prob < 0.025:
            continue
        bx, by = base_pts[meta.index]
        # Quantize radius and alpha buckets so cached discs stay bounded and fast (60+ FPS)
        scale = math.sqrt(min(1.0, prob))
        rx = int(round((26 + 44 * scale + 3 * pulse * scale) / 4.0)) * 4
        ry = max(14, int(round(rx * 0.52 / 2.0)) * 2)
        alpha = int(round((45 + 145 * scale) / 10.0)) * 10
        color = _belief_color(prob, palette)

        disc = _get_glow_disc(rx, ry, color, alpha)
        surface.blit(disc, (bx - disc.get_width() // 2, by - disc.get_height() // 2))

        if prob >= 0.12:
            pct = f"{round(prob * 100):d}%"
            draw_badge_pill(
                surface,
                (bx, by - 46),
                pct,
                color,
                (9, 13, 24, 215),
                with_alpha(color, 165),
                fonts.get("mono", 10, bold=True),
            )


def draw_belief_top3_card(
    surface: pygame.Surface,
    battlefield_rect: pygame.Rect,
    metas: tuple[NodeVisualMeta, ...],
    belief: tuple[float, ...],
    palette: ThemePalette,
    fonts: FontBook,
) -> None:
    """Draw a compact Top-3 suspected machines HUD overlay inside the battlefield."""
    if len(belief) != len(metas) or not belief:
        return

    ranked = sorted(range(len(belief)), key=lambda i: (-belief[i], i))[:3]
    card_w = 196
    card_h = 84
    card = pygame.Rect(
        battlefield_rect.left + 14,
        battlefield_rect.bottom - card_h - 12,
        card_w,
        card_h,
    )
    draw_alpha_rect(
        surface,
        card,
        (7, 11, 20, 215),
        border_rgba=with_alpha(palette.violet, 145),
        border_width=1,
    )

    title = fonts.get("heading", 10, bold=True).render(
        "BAYESIAN BELIEF // TOP 3", True, palette.cyan
    )
    surface.blit(title, (card.left + 10, card.top + 7))

    row_font = fonts.get("mono", 11, bold=True)
    y = card.top + 24
    bar_x = card.left + 68
    bar_w = 78
    for idx in ranked:
        prob = belief[idx]
        col = _belief_color(prob, palette)
        code = metas[idx].code
        code_surf = row_font.render(code, True, palette.text)
        surface.blit(code_surf, (card.left + 10, y))

        track = pygame.Rect(bar_x, y + 4, bar_w, 6)
        pygame.draw.rect(surface, (20, 28, 46), track)
        fill_w = max(2, int(bar_w * min(1.0, prob)))
        pygame.draw.rect(surface, col, pygame.Rect(bar_x, y + 4, fill_w, 6))

        pct_surf = row_font.render(f"{prob * 100:4.0f}%", True, col)
        surface.blit(pct_surf, (bar_x + bar_w + 6, y))
        y += 18
