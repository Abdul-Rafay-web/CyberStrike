from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import pygame

from config import CONFIG
from core.network import Edge, make_edge
from core.rules import effective_hardness
from core.state import GameState
from ui.draw import (
    draw_alpha_rect,
    draw_animated_dash_line,
    draw_badge_pill,
    draw_firewall_dot_matrix,
    draw_glow_line,
    draw_iso_ellipse_ring,
    draw_target_reticle_ring,
)
from ui.layout import NodeVisualMeta, project_node_to_screen
from ui.theme import RGB, AssetVault, FontBook, ThemePalette, with_alpha

ViewMode = Literal["defender", "attacker", "spectator"]
ToolMode = Literal["patch", "firewall", "honeypot"] | None


@dataclass(slots=True)
class RippleEffect:
    """Expanding isometric ring spawned on defense deployment or honeypot trigger."""

    cx: int
    cy: int
    color: RGB
    age: float = 0.0
    duration: float = 0.55
    max_rx: int = 72
    max_ry: int = 38


def _point_to_segment_distance(
    px: float,
    py: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    dx = x2 - x1
    dy = y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq <= 1e-6:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def _ease_out_cubic(t: float) -> float:
    u = 1.0 - max(0.0, min(1.0, t))
    return 1.0 - u * u * u


class BattlefieldView:
    """Renders the isometric network map with unified node/ring anchoring and smooth animations."""

    PAD_LEFT = 88
    PAD_RIGHT = 88
    PAD_TOP = 92
    PAD_BOTTOM = 82

    def __init__(self) -> None:
        self.time_s: float = 0.0
        self.hovered_node: int | None = None
        self.hovered_edge: Edge | None = None
        self.peek_ref_mode: Literal["none", "defender", "attacker"] = "none"
        self.ref_toggle_rect: pygame.Rect = pygame.Rect(0, 0, 0, 0)

        self._hover_lift: dict[int, float] = {}
        self._ripples: list[RippleEffect] = []

        self._last_attacker_pos: int | None = None
        self._move_from_idx: int | None = None
        self._move_to_idx: int | None = None
        self._move_progress: float = 1.0
        self._move_duration: float = 0.36

        self._screen_pts: dict[int, tuple[int, int]] = {}
        self._base_pts: dict[int, tuple[int, int]] = {}
        self._sprite_sizes: dict[int, int] = {}

    def reset_animations(self, attacker_pos: int | None = None) -> None:
        self._hover_lift.clear()
        self._ripples.clear()
        self._last_attacker_pos = attacker_pos
        self._move_from_idx = None
        self._move_to_idx = None
        self._move_progress = 1.0

    def spawn_ripple(
        self,
        rect: pygame.Rect,
        metas: tuple[NodeVisualMeta, ...],
        machine_idx: int,
        color: RGB,
        max_rx: int = 72,
    ) -> None:
        self._sync_geometry(rect, metas, None)
        base = self._base_pts.get(machine_idx)
        if base is not None:
            self._ripples.append(
                RippleEffect(cx=base[0], cy=base[1], color=color, max_rx=max_rx, max_ry=max_rx // 2)
            )

    def spawn_edge_ripple(
        self,
        rect: pygame.Rect,
        metas: tuple[NodeVisualMeta, ...],
        edge: Edge,
        color: RGB,
    ) -> None:
        self._sync_geometry(rect, metas, None)
        p1 = self._base_pts.get(edge[0])
        p2 = self._base_pts.get(edge[1])
        if p1 is not None and p2 is not None:
            mx = (p1[0] + p2[0]) // 2
            my = (p1[1] + p2[1]) // 2
            self._ripples.append(RippleEffect(cx=mx, cy=my, color=color, max_rx=68, max_ry=34))

    def update(self, dt: float, state: GameState, metas: tuple[NodeVisualMeta, ...]) -> None:
        """Advance hover lifts, ripple expansions, and eased attacker transitions."""
        self.time_s += dt

        if self._last_attacker_pos is None:
            self._last_attacker_pos = state.attacker_pos
        elif state.attacker_pos != self._last_attacker_pos:
            self._move_from_idx = self._last_attacker_pos
            self._move_to_idx = state.attacker_pos
            self._move_progress = 0.0
            self._last_attacker_pos = state.attacker_pos

        if self._move_progress < 1.0:
            self._move_progress = min(1.0, self._move_progress + dt / self._move_duration)

        lerp_rate = min(1.0, dt * 14.0)
        for meta in metas:
            target_lift = 5.0 if self.hovered_node == meta.index else 0.0
            cur = self._hover_lift.get(meta.index, 0.0)
            self._hover_lift[meta.index] = cur + (target_lift - cur) * lerp_rate

        alive: list[RippleEffect] = []
        for rp in self._ripples:
            rp.age += dt
            if rp.age < rp.duration:
                alive.append(rp)
        self._ripples = alive

    def _base_sprite_size(self, node_count: int) -> int:
        if node_count <= 14:
            return 76
        if node_count <= 19:
            return 68
        return 58

    def _sync_geometry(
        self,
        rect: pygame.Rect,
        metas: tuple[NodeVisualMeta, ...],
        state: GameState | None,
    ) -> None:
        base_sz = self._base_sprite_size(len(metas))
        self._screen_pts.clear()
        self._base_pts.clear()
        self._sprite_sizes.clear()

        for meta in metas:
            px, py = project_node_to_screen(
                meta.nx,
                meta.ny,
                rect,
                pad_left=self.PAD_LEFT,
                pad_right=self.PAD_RIGHT,
                pad_top=self.PAD_TOP,
                pad_bottom=self.PAD_BOTTOM,
            )
            sz = base_sz
            if meta.role == "crown_jewel":
                sz = base_sz + 14
            elif state is not None and meta.index in state.patched:
                sz = base_sz + 6

            self._screen_pts[meta.index] = (px, py)
            self._base_pts[meta.index] = (px, py + int(round(sz * 0.24)))
            self._sprite_sizes[meta.index] = sz

    def is_node_legal_for_tool(
        self,
        state: GameState,
        machine_idx: int,
        active_tool: ToolMode,
    ) -> bool:
        if active_tool == "patch":
            return (
                state.stock.patches > 0
                and machine_idx not in state.patched
                and (machine_idx not in state.honeypots)
            )
        if active_tool == "honeypot":
            net = state.network
            return (
                state.stock.honeypots > 0
                and machine_idx not in (net.entry, net.jewel)
                and machine_idx not in state.patched
                and machine_idx not in state.honeypots
            )
        return False

    def is_edge_legal_for_firewall(
        self,
        state: GameState,
        edge: Edge,
        active_tool: ToolMode,
    ) -> bool:
        return (
            active_tool == "firewall" and state.stock.firewalls > 0 and edge not in state.firewalls
        )

    def pick_machine(
        self,
        pos: tuple[int, int],
        rect: pygame.Rect,
        metas: tuple[NodeVisualMeta, ...],
        state: GameState,
    ) -> int | None:
        """Return the machine index under pos, or None."""
        if not rect.collidepoint(pos):
            return None
        self._sync_geometry(rect, metas, state)
        best_idx: int | None = None
        best_dist = float("inf")
        # Iterate back-to-front in reverse so foreground sprites win ties.
        for meta in reversed(sorted(metas, key=lambda m: self._screen_pts[m.index][1])):
            px, py = self._screen_pts[meta.index]
            sz = self._sprite_sizes[meta.index]
            dx = abs(pos[0] - px)
            dy = abs(pos[1] - py)
            if dx <= sz * 0.48 and dy <= sz * 0.54:
                dist = math.hypot(dx, dy)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = meta.index
        return best_idx

    def pick_edge(
        self,
        pos: tuple[int, int],
        rect: pygame.Rect,
        metas: tuple[NodeVisualMeta, ...],
        state: GameState,
    ) -> Edge | None:
        """Return the network Edge under pos (or midpoint pill), or None."""
        if not rect.collidepoint(pos):
            return None
        self._sync_geometry(rect, metas, state)
        best_edge: Edge | None = None
        best_dist = 20.0
        for a, b in state.network.edges():
            p1 = self._base_pts[a]
            p2 = self._base_pts[b]
            mx, my = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
            pill_rect = pygame.Rect(int(mx) - 36, int(my) - 12, 72, 24)
            if pill_rect.collidepoint(pos):
                return make_edge(a, b)
            d = _point_to_segment_distance(pos[0], pos[1], p1[0], p1[1], p2[0], p2[1])
            if d < best_dist:
                best_dist = d
                best_edge = make_edge(a, b)
        return best_edge

    def _resolve_sprite_key(
        self,
        state: GameState,
        meta: NodeVisualMeta,
        can_see_intruder: bool,
        highlighted: bool,
    ) -> str:
        idx = meta.index
        if idx == state.network.jewel:
            return "crown_jewel"
        if idx in state.triggered:
            return "honeypot"
        if can_see_intruder and state.breach_target == idx:
            return "under_breach"
        if can_see_intruder and (
            idx == state.attacker_pos or (idx in state.breached and idx != state.network.entry)
        ):
            return "breached"
        if idx in state.patched:
            return "patched"
        if idx == state.network.entry:
            return "entry_point"
        if highlighted:
            return "selected"
        return "normal"

    def draw(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        state: GameState,
        metas: tuple[NodeVisualMeta, ...],
        palette: ThemePalette,
        fonts: FontBook,
        vault: AssetVault,
        view_mode: ViewMode,
        god_mode: bool,
        active_tool: ToolMode,
        selected_node: int | None,
        alerts: tuple[bool, ...],
        intruder_known_at: int | None,
        legal_attacker_targets: set[int],
        seed: int,
        show_heatmap: bool,
        show_breach_trace: bool,
        show_hardness_badges: bool,
    ) -> None:
        """Render the complete battlefield inside rect with strict border clipping."""
        self._sync_geometry(rect, metas, state)
        prev_clip = surface.get_clip()
        surface.set_clip(rect)

        try:
            # 1. Background war-room console surface + tiled 2:1 isometric grid + vignette
            surface.fill(palette.bg, rect)
            bg_img = vault.get_cover("backgrounds/warroom_console_4k.jpg", rect.size)
            bg_copy = bg_img.copy()
            bg_copy.set_alpha(195)
            surface.blit(bg_copy, rect.topleft)
            grid_tile = vault.get_tiled_iso_grid(rect.size, alpha=115)
            surface.blit(grid_tile, rect.topleft)

            self._draw_vignette(surface, rect)

            # 2. Optional 4K reference comparison overlay
            if self.peek_ref_mode != "none":
                self._draw_reference_overlay(surface, rect, palette, fonts, vault)
                self._draw_top_chrome(surface, rect, state, seed, active_tool, palette, fonts)
                return

            can_see_intruder = view_mode == "attacker" or (view_mode == "spectator" and god_mode)
            effective_heatmap = view_mode != "attacker" and show_heatmap

            # 3. Alert thermal halo & Crown Jewel golden base aura (centered strictly on base_pts!)
            if effective_heatmap:
                self._draw_heatmap_rings(
                    surface,
                    state,
                    metas,
                    alerts,
                    intruder_known_at,
                    palette,
                    fonts,
                    vault,
                )

            # 4. Isometric base pedestal wireframe rings (centered strictly on base_pts)
            self._draw_pedestal_rings(surface, state, metas, alerts, view_mode, palette)

            # 5. Network links + Firewalled 4x13 matrices + Breach trace + Firewall hit pills
            self._draw_links(
                surface,
                state,
                palette,
                fonts,
                vault,
                can_see_intruder,
                active_tool,
                show_breach_trace,
            )

            # 6. Expanding ripple effects (defense placement / honeypot trigger)
            self._draw_ripples(surface, vault)

            # 7. Eased intruder vector pulse moving from previous node to new node
            if (
                can_see_intruder
                and self._move_progress < 1.0
                and (self._move_from_idx is not None and self._move_to_idx is not None)
            ):
                self._draw_eased_intruder_transit(surface, palette)

            # 8. Machines back-to-front (painter's order by y)
            ordered_metas = sorted(metas, key=lambda m: self._screen_pts[m.index][1])
            for meta in ordered_metas:
                self._draw_machine_node(
                    surface,
                    rect,
                    state,
                    meta,
                    palette,
                    fonts,
                    vault,
                    view_mode,
                    can_see_intruder,
                    active_tool,
                    selected_node,
                    alerts,
                    intruder_known_at,
                    legal_attacker_targets,
                    show_hardness_badges,
                )

            # 9. Clamped hover micro-tooltip drawn on top inside rect
            if self.hovered_node is not None and 0 <= self.hovered_node < len(metas):
                self._draw_hover_tooltip(
                    surface,
                    rect,
                    state,
                    metas[self.hovered_node],
                    palette,
                    fonts,
                    view_mode,
                    alerts,
                )

            # 10. Top header labels + Ref Compare button
            self._draw_top_chrome(surface, rect, state, seed, active_tool, palette, fonts)

            # Border frame
            pygame.draw.rect(surface, with_alpha(palette.cyan, 80), rect, 1)
        finally:
            surface.set_clip(prev_clip)

    def _draw_vignette(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        edge_fade = pygame.Surface(rect.size, pygame.SRCALPHA)
        w, h = rect.size
        pygame.draw.rect(edge_fade, (4, 4, 10, 120), pygame.Rect(0, 0, w, 28))
        pygame.draw.rect(edge_fade, (4, 4, 10, 130), pygame.Rect(0, h - 28, w, 28))
        pygame.draw.rect(edge_fade, (4, 4, 10, 120), pygame.Rect(0, 0, 26, h))
        pygame.draw.rect(edge_fade, (4, 4, 10, 120), pygame.Rect(w - 26, 0, 26, h))
        surface.blit(edge_fade, rect.topleft)

    def _draw_heatmap_rings(
        self,
        surface: pygame.Surface,
        state: GameState,
        metas: tuple[NodeVisualMeta, ...],
        alerts: tuple[bool, ...],
        intruder_known_at: int | None,
        palette: ThemePalette,
        fonts: FontBook,
        vault: AssetVault,
    ) -> None:
        pulse = 0.5 + 0.5 * math.sin(self.time_s * 3.0)
        for meta in metas:
            idx = meta.index
            bx, by = self._base_pts[idx]
            is_known = intruder_known_at == idx
            is_alert = idx < len(alerts) and alerts[idx]

            if is_known or is_alert:
                col = palette.red if is_alert or is_known else palette.amber
                blob = vault.get_effect(
                    "effect_heat_blob",
                    (176, 104),
                    tint=col,
                    alpha=int(120 + 45 * pulse) if is_known else int(90 + 35 * pulse),
                )
                surface.blit(blob, (bx - blob.get_width() // 2, by - blob.get_height() // 2))
                draw_iso_ellipse_ring(
                    surface,
                    (bx, by),
                    56 + int(4 * pulse),
                    29 + int(2 * pulse),
                    col,
                    alpha=145,
                    width=1,
                )
                draw_iso_ellipse_ring(surface, (bx, by), 72, 37, col, alpha=85, width=1)

        # Crown Jewel golden concentric halo strictly centered on jewel base
        j_bx, j_by = self._base_pts[state.network.jewel]
        jewel_blob = vault.get_effect("effect_heat_blob", (184, 108), tint=palette.amber, alpha=105)
        surface.blit(
            jewel_blob, (j_bx - jewel_blob.get_width() // 2, j_by - jewel_blob.get_height() // 2)
        )
        draw_iso_ellipse_ring(surface, (j_bx, j_by), 50, 25, palette.amber, alpha=205, width=2)
        draw_iso_ellipse_ring(surface, (j_bx, j_by), 68, 35, palette.amber, alpha=110, width=1)

        lbl = fonts.get("heading", 10, bold=True).render(
            "SENSOR THERMAL OVERLAY", True, (217, 163, 78)
        )
        surface.blit(lbl, (j_bx - lbl.get_width() // 2, j_by + 68))

    def _draw_pedestal_rings(
        self,
        surface: pygame.Surface,
        state: GameState,
        metas: tuple[NodeVisualMeta, ...],
        alerts: tuple[bool, ...],
        view_mode: ViewMode,
        palette: ThemePalette,
    ) -> None:
        for meta in metas:
            idx = meta.index
            bx, by = self._base_pts[idx]
            is_alert = idx < len(alerts) and alerts[idx] and view_mode != "attacker"
            if is_alert:
                col = palette.red
                alpha_outer, alpha_inner = 165, 210
            elif idx in state.patched:
                col = palette.cyan
                alpha_outer, alpha_inner = 155, 200
            elif idx == state.network.jewel or meta.role == "gateway":
                col = palette.amber
                alpha_outer, alpha_inner = 115, 160
            else:
                col = (90, 118, 132)
                alpha_outer, alpha_inner = 85, 60

            rx_out = 44 if idx == state.network.jewel else 32
            ry_out = 22 if idx == state.network.jewel else 16
            rx_in = 33 if idx == state.network.jewel else 23
            ry_in = 16 if idx == state.network.jewel else 11
            draw_iso_ellipse_ring(
                surface, (bx, by), rx_out, ry_out, col, alpha=alpha_outer, width=1
            )
            draw_iso_ellipse_ring(surface, (bx, by), rx_in, ry_in, col, alpha=alpha_inner, width=1)

    def _draw_links(
        self,
        surface: pygame.Surface,
        state: GameState,
        palette: ThemePalette,
        fonts: FontBook,
        vault: AssetVault,
        can_see_intruder: bool,
        active_tool: ToolMode,
        show_breach_trace: bool,
    ) -> None:
        for a, b in state.network.edges():
            edge = make_edge(a, b)
            p1 = self._base_pts[a]
            p2 = self._base_pts[b]
            mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
            is_firewalled = edge in state.firewalls
            is_hovered = self.hovered_edge == edge
            can_fw = self.is_edge_legal_for_firewall(state, edge, active_tool)

            is_active_breach = (
                can_see_intruder
                and state.breach_target is not None
                and make_edge(state.attacker_pos, state.breach_target) == edge
            )
            is_breached_trail = (
                (can_see_intruder or show_breach_trace)
                and not is_firewalled
                and a in state.breached
                and b in state.breached
            )

            if is_firewalled:
                draw_glow_line(surface, p1, p2, palette.amber, width=1, core_alpha=85, glow_alpha=0)
                draw_firewall_dot_matrix(surface, (mx, my), vault, fonts, palette)
            elif is_active_breach:
                draw_glow_line(
                    surface, p1, p2, palette.red, width=3, core_alpha=235, glow_alpha=110
                )
                draw_animated_dash_line(
                    surface,
                    p1,
                    p2,
                    palette.amber,
                    width=2,
                    phase=self.time_s,
                )
            elif is_breached_trail and can_see_intruder:
                draw_glow_line(surface, p1, p2, palette.red, width=2, core_alpha=175, glow_alpha=65)
            elif can_fw and is_hovered:
                draw_glow_line(
                    surface, p1, p2, palette.amber, width=3, core_alpha=240, glow_alpha=100
                )
            else:
                draw_glow_line(
                    surface,
                    p1,
                    p2,
                    (132, 175, 190),
                    width=1,
                    core_alpha=125,
                    glow_alpha=0,
                )

            if can_fw:
                pill_bg = (255, 164, 27, 235) if is_hovered else (14, 25, 34, 230)
                pill_fg = palette.bg if is_hovered else palette.amber
                draw_badge_pill(
                    surface,
                    (mx, my),
                    "+ FIREWALL",
                    pill_fg,
                    pill_bg,
                    with_alpha(palette.amber, 240),
                    fonts.get("heading", 10, bold=True),
                    centered=True,
                )

    def _draw_ripples(self, surface: pygame.Surface, vault: AssetVault) -> None:
        for rp in self._ripples:
            t = rp.age / max(0.01, rp.duration)
            eased = _ease_out_cubic(t)
            rx = max(14, int(24 + (rp.max_rx - 24) * eased))
            ry = max(7, int(12 + (rp.max_ry - 12) * eased))
            alpha = int(210 * (1.0 - t))
            eff = vault.get_effect(
                "effect_shield_ripple", (rx * 2, ry * 2), tint=rp.color, alpha=alpha
            )
            surface.blit(eff, (rp.cx - rx, rp.cy - ry))
            draw_iso_ellipse_ring(surface, (rp.cx, rp.cy), rx, ry, rp.color, alpha=alpha, width=2)

    def _draw_eased_intruder_transit(self, surface: pygame.Surface, palette: ThemePalette) -> None:
        assert self._move_from_idx is not None and self._move_to_idx is not None
        p0 = self._base_pts[self._move_from_idx]
        p1 = self._base_pts[self._move_to_idx]
        t = _ease_out_cubic(self._move_progress)
        cx = int(round(p0[0] + (p1[0] - p0[0]) * t))
        cy = int(round(p0[1] + (p1[1] - p0[1]) * t))
        draw_iso_ellipse_ring(surface, (cx, cy), 18, 9, palette.red, alpha=230, width=2)
        pygame.draw.circle(surface, palette.amber, (cx, cy), 5)

    def _draw_machine_node(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        state: GameState,
        meta: NodeVisualMeta,
        palette: ThemePalette,
        fonts: FontBook,
        vault: AssetVault,
        view_mode: ViewMode,
        can_see_intruder: bool,
        active_tool: ToolMode,
        selected_node: int | None,
        alerts: tuple[bool, ...],
        intruder_known_at: int | None,
        legal_attacker_targets: set[int],
        show_hardness_badges: bool,
    ) -> None:
        idx = meta.index
        px, py = self._screen_pts[idx]
        bx, by = self._base_pts[idx]
        sz = self._sprite_sizes[idx]
        lift = int(round(self._hover_lift.get(idx, 0.0)))

        is_selected = selected_node == idx
        is_hovered = self.hovered_node == idx
        defender_legal = view_mode == "defender" and self.is_node_legal_for_tool(
            state, idx, active_tool
        )
        attacker_legal = view_mode == "attacker" and idx in legal_attacker_targets
        is_alert = view_mode != "attacker" and idx < len(alerts) and alerts[idx]
        is_known_intruder = view_mode != "attacker" and intruder_known_at == idx

        # Concentric targeting / selection rings locked strictly around machine base (bx, by)
        pulse_phase = self.time_s * 1.8
        if defender_legal or attacker_legal:
            ring_col = (
                palette.amber if is_hovered else (palette.red if attacker_legal else palette.cyan)
            )
            draw_target_reticle_ring(
                surface,
                (bx, by),
                rx=int(sz * 0.50),
                ry=int(sz * 0.26),
                color=ring_col,
                alpha=235 if is_hovered else 175,
                phase=pulse_phase,
            )
        elif is_selected:
            draw_target_reticle_ring(
                surface,
                (bx, by),
                rx=int(sz * 0.46),
                ry=int(sz * 0.24),
                color=palette.cyan,
                alpha=215,
                phase=pulse_phase * 0.5,
            )
        elif is_hovered:
            draw_iso_ellipse_ring(
                surface,
                (bx, by),
                rx=int(sz * 0.44),
                ry=int(sz * 0.23),
                color=palette.cyan,
                alpha=170,
                width=2,
            )

        # Glow disc under alerted / Crown Jewel / active intruder nodes
        if is_alert or is_known_intruder:
            disc = vault.get_effect(
                "effect_glow_disc", (sz + 18, (sz + 18) // 2), tint=palette.red, alpha=175
            )
            surface.blit(disc, (bx - disc.get_width() // 2, by - disc.get_height() // 2))
        elif idx == state.network.jewel:
            disc = vault.get_effect(
                "effect_glow_disc", (sz + 22, (sz + 22) // 2), tint=palette.amber, alpha=155
            )
            surface.blit(disc, (bx - disc.get_width() // 2, by - disc.get_height() // 2))

        # Sprite image centered at (px, py - lift)
        sprite_key = self._resolve_sprite_key(
            state,
            meta,
            can_see_intruder,
            highlighted=is_selected or is_hovered or defender_legal or attacker_legal,
        )
        sprite = vault.get_machine(sprite_key, sz)
        surface.blit(sprite, (px - sz // 2, py - sz // 2 - lift))

        # Optional hardness badge on top-right corner of sprite
        if show_hardness_badges or view_mode == "attacker":
            eff_h = effective_hardness(state, idx, CONFIG.game)
            badge_col = palette.cyan if idx in state.patched else palette.amber
            draw_badge_pill(
                surface,
                (px + sz // 2 - 4, py - sz // 2 + 6 - lift),
                f"H={eff_h}",
                badge_col,
                (6, 6, 13, 225),
                with_alpha(badge_col, 170),
                fonts.get("mono", 10),
                centered=True,
                pad_x=4,
                pad_y=1,
            )

        # Fixed-offset top callout banner (above sprite, never shifts rings!)
        banner_y = py - sz // 2 - 8 - lift
        if is_alert or is_known_intruder:
            banner_txt = "! INTRUDER CONFIRMED" if is_known_intruder else "! ANOMALY ALERT"
            draw_badge_pill(
                surface,
                (px, banner_y),
                banner_txt,
                (255, 195, 208),
                (38, 10, 18, 240),
                with_alpha(palette.red, 225),
                fonts.get("heading", 10, bold=True),
                centered=True,
                pad_x=7,
                pad_y=2,
            )
            banner_y -= 18
        elif can_see_intruder and idx == state.attacker_pos:
            status_txt = (
                f"BREACHING [{state.breach_left}T]"
                if state.breach_target is not None
                else "INTRUDER VECTOR"
            )
            draw_badge_pill(
                surface,
                (px, banner_y),
                status_txt,
                (255, 215, 225),
                (55, 14, 26, 235),
                with_alpha(palette.red, 225),
                fonts.get("mono", 10),
                centered=True,
                pad_x=6,
                pad_y=2,
            )
            banner_y -= 18

        if view_mode != "attacker" and idx in state.honeypots and idx not in state.triggered:
            draw_badge_pill(
                surface,
                (px, banner_y),
                "TRAP ARMED",
                (233, 213, 255),
                (36, 16, 62, 235),
                with_alpha(palette.violet, 210),
                fonts.get("mono", 9),
                centered=True,
                pad_x=5,
                pad_y=1,
            )

        # Fixed-offset bottom labels directly below the isometric pedestal
        label_y = by + int(sz * 0.24) + 5
        lbl_surf = fonts.get("heading", 12, bold=True).render(meta.label, True, palette.text)
        surface.blit(lbl_surf, (px - lbl_surf.get_width() // 2, label_y))

        sub_text = meta.sub_label
        sub_fg: RGB = palette.muted
        sub_bg: tuple[int, int, int, int] | None = None
        sub_border: tuple[int, int, int, int] | None = None

        if idx in state.triggered:
            sub_text = "HONEYPOT HIT"
            sub_fg = (233, 213, 255)
            sub_bg = (50, 20, 80, 230)
            sub_border = with_alpha(palette.violet, 230)
        elif idx in state.patched:
            sub_text = "PATCHED"
            sub_fg = palette.cyan
            sub_bg = (18, 62, 66, 230)
            sub_border = with_alpha(palette.cyan, 160)
        elif meta.role == "entry":
            sub_fg = (140, 233, 154)
            sub_bg = (37, 68, 43, 220)
            sub_border = (81, 207, 102, 130)
        elif meta.role in ("crown_jewel", "gateway"):
            sub_fg = palette.amber

        if sub_text:
            sub_y = label_y + lbl_surf.get_height() + 6
            if sub_bg is not None:
                draw_badge_pill(
                    surface,
                    (px, sub_y),
                    sub_text,
                    sub_fg,
                    sub_bg,
                    sub_border,
                    fonts.get("heading", 10, bold=True),
                    centered=True,
                    pad_x=6,
                    pad_y=1,
                )
            else:
                ss = fonts.get("heading", 10, bold=True).render(sub_text, True, sub_fg)
                surface.blit(ss, (px - ss.get_width() // 2, sub_y - ss.get_height() // 2))

    def _draw_hover_tooltip(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        state: GameState,
        meta: NodeVisualMeta,
        palette: ThemePalette,
        fonts: FontBook,
        view_mode: ViewMode,
        alerts: tuple[bool, ...],
    ) -> None:
        idx = meta.index
        px, py = self._screen_pts[idx]
        eff_h = effective_hardness(state, idx, CONFIG.game)

        lines: list[tuple[str, RGB]] = [
            (f"{meta.code}  //  CVSS {meta.cvss:.1f}", palette.cyan),
            (
                f"HARDNESS: {eff_h}T {'(+2 PATCH)' if idx in state.patched else ''}",
                palette.text,
            ),
            (f"ROLE: {meta.role.replace('_', ' ').upper()}", palette.muted),
        ]
        if view_mode != "attacker":
            alert_txt = "ACTIVE" if (idx < len(alerts) and alerts[idx]) else "QUIET"
            alert_col = palette.red if alert_txt == "ACTIVE" else palette.cyan
            lines.append((f"SENSOR ALERT: {alert_txt}", alert_col))
        else:
            owned_txt = "OWNED" if idx in state.breached else "UNBREACHED"
            lines.append((f"STATUS: {owned_txt}", palette.amber))

        w = 192
        h = 16 + len(lines) * 18
        tx = px + 52
        if tx + w > rect.right - 12:
            tx = px - 52 - w
        ty = py - h // 2
        tx = max(rect.left + 12, min(rect.right - w - 12, tx))
        ty = max(rect.top + 38, min(rect.bottom - h - 12, ty))

        tip_rect = pygame.Rect(tx, ty, w, h)
        draw_alpha_rect(
            surface,
            tip_rect,
            (9, 14, 24, 244),
            border_rgba=with_alpha(palette.cyan, 180),
            border_width=1,
        )
        y = ty + 8
        mono = fonts.get("mono", 11)
        for text, col in lines:
            surface.blit(mono.render(text, True, col), (tx + 10, y))
            y += 18

    def _draw_top_chrome(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        state: GameState,
        seed: int,
        active_tool: ToolMode,
        palette: ThemePalette,
        fonts: FontBook,
    ) -> None:
        nx = rect.left + 16
        ny = rect.top + 10
        title_s = fonts.get("heading", 17, bold=True).render("NETWORK Map", True, palette.text)
        surface.blit(title_s, (nx, ny))

        meta_txt = (
            f"{state.network.size} NODES // {len(state.network.edges())} LINKS // SEED #{seed}"
        )
        pill = draw_badge_pill(
            surface,
            (nx + title_s.get_width() + 12, ny + 2),
            meta_txt,
            palette.cyan,
            (14, 30, 38, 215),
            with_alpha(palette.cyan, 85),
            fonts.get("mono", 11),
            centered=False,
        )

        if active_tool is not None:
            target_kind = "LINK" if active_tool == "firewall" else "MACHINE"
            draw_badge_pill(
                surface,
                (pill.right + 10, ny + 2),
                f"TARGETING ARMED: CLICK LEGAL {target_kind}",
                palette.amber,
                (45, 28, 8, 230),
                with_alpha(palette.amber, 230),
                fonts.get("heading", 11, bold=True),
                centered=False,
            )

        ref_label = "HIDE 4K REF" if self.peek_ref_mode != "none" else "4K REF COMPARE"
        ref_surf = fonts.get("mono", 11).render(
            ref_label, True, palette.cyan if self.peek_ref_mode == "none" else palette.amber
        )
        btn_w = ref_surf.get_width() + 20
        btn_h = 24
        self.ref_toggle_rect = pygame.Rect(rect.right - btn_w - 14, rect.top + 10, btn_w, btn_h)
        is_active = self.peek_ref_mode != "none"
        draw_alpha_rect(
            surface,
            self.ref_toggle_rect,
            with_alpha(palette.amber, 55) if is_active else (12, 22, 32, 225),
            border_rgba=with_alpha(palette.amber if is_active else palette.cyan, 160),
            border_width=1,
        )
        surface.blit(
            ref_surf,
            (
                self.ref_toggle_rect.centerx - ref_surf.get_width() // 2,
                self.ref_toggle_rect.centery - ref_surf.get_height() // 2,
            ),
        )

    def _draw_reference_overlay(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        palette: ThemePalette,
        fonts: FontBook,
        vault: AssetVault,
    ) -> None:
        draw_alpha_rect(surface, rect, (6, 6, 13, 240))
        rel = (
            "atlas/attacker_ref_4k.jpg"
            if self.peek_ref_mode == "attacker"
            else "atlas/defender_ref_4k.jpg"
        )
        max_w = max(100, rect.width - 36)
        max_h = max(100, rect.height - 76)
        img = vault.get_contain(rel, (max_w, max_h))
        ix = rect.centerx - img.get_width() // 2
        iy = rect.centery - img.get_height() // 2 + 8
        surface.blit(img, (ix, iy))
        pygame.draw.rect(
            surface,
            with_alpha(palette.cyan, 135),
            pygame.Rect(ix, iy, img.get_width(), img.get_height()),
            1,
        )
        draw_badge_pill(
            surface,
            (rect.centerx, rect.bottom - 20),
            f"REFERENCE BOARD ({self.peek_ref_mode.upper()}) — CLICK 4K REF COMPARE TO RETURN",
            palette.amber,
            (6, 6, 13, 235),
            with_alpha(palette.amber, 180),
            fonts.get("heading", 11, bold=True),
            centered=True,
        )
