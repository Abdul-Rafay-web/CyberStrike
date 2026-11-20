from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pygame

from config import CONFIG

RGB = tuple[int, int, int]
RGBA = tuple[int, int, int, int]
ThemePresetName = Literal["hybrid", "playbook", "warroom"]
FontRole = Literal["title", "heading", "body", "mono"]

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
IMAGES_DIR = ASSETS_DIR / "images"
FONTS_DIR = ASSETS_DIR / "fonts"


@dataclass(frozen=True, slots=True)
class ThemePalette:
    """Color tokens synchronized with config.py and War-Room UI presets."""

    name: ThemePresetName
    label: str
    bg: RGB
    panel: RGB
    panel_alt: RGB
    panel_header: RGB
    top_bar: RGB
    sub_bar: RGB
    cyan: RGB
    amber: RGB
    violet: RGB
    red: RGB
    green: RGB
    text: RGB
    muted: RGB
    link_idle: RGBA
    grid_tint: RGB


THEME_PRESETS: dict[ThemePresetName, ThemePalette] = {
    "hybrid": ThemePalette(
        name="hybrid",
        label="NEON HYBRID",
        bg=CONFIG.theme.background,
        panel=(13, 21, 32),
        panel_alt=CONFIG.theme.panel,
        panel_header=(23, 43, 52),
        top_bar=(12, 25, 33),
        sub_bar=(8, 17, 24),
        cyan=CONFIG.theme.cyan,
        amber=CONFIG.theme.orange,
        violet=CONFIG.theme.violet,
        red=CONFIG.theme.red,
        green=(81, 207, 102),
        text=CONFIG.theme.text,
        muted=CONFIG.theme.muted,
        link_idle=(132, 175, 190, 140),
        grid_tint=(34, 232, 200),
    ),
    "playbook": ThemePalette(
        name="playbook",
        label="PLAYBOOK #12122A",
        bg=CONFIG.theme.background,
        panel=CONFIG.theme.panel,
        panel_alt=(16, 16, 38),
        panel_header=(24, 24, 56),
        top_bar=(16, 16, 40),
        sub_bar=(10, 10, 28),
        cyan=CONFIG.theme.cyan,
        amber=CONFIG.theme.orange,
        violet=CONFIG.theme.violet,
        red=CONFIG.theme.red,
        green=(81, 207, 102),
        text=CONFIG.theme.text,
        muted=CONFIG.theme.muted,
        link_idle=(140, 155, 210, 145),
        grid_tint=(168, 85, 247),
    ),
    "warroom": ThemePalette(
        name="warroom",
        label="WAR-ROOM SLATE",
        bg=(7, 11, 16),
        panel=(15, 25, 34),
        panel_alt=(13, 22, 30),
        panel_header=(26, 48, 56),
        top_bar=(12, 22, 28),
        sub_bar=(8, 15, 20),
        cyan=CONFIG.theme.cyan,
        amber=CONFIG.theme.orange,
        violet=CONFIG.theme.violet,
        red=CONFIG.theme.red,
        green=(81, 207, 102),
        text=CONFIG.theme.text,
        muted=(150, 175, 190),
        link_idle=(125, 180, 195, 145),
        grid_tint=(34, 232, 200),
    ),
}


def get_palette(preset: ThemePresetName = "hybrid") -> ThemePalette:
    return THEME_PRESETS.get(preset, THEME_PRESETS["hybrid"])


def with_alpha(color: RGB, alpha: int) -> RGBA:
    return (color[0], color[1], color[2], max(0, min(255, alpha)))


def lerp_color(a: RGB, b: RGB, t: float) -> RGB:
    clamped = max(0.0, min(1.0, t))
    return (
        int(a[0] + (b[0] - a[0]) * clamped),
        int(a[1] + (b[1] - a[1]) * clamped),
        int(a[2] + (b[2] - a[2]) * clamped),
    )


class FontBook:
    """Cached font lookup with bundled TTF support and clean system fallbacks."""

    _ROLE_FILES: dict[FontRole, tuple[str, ...]] = {
        "title": ("Michroma-Regular.ttf", "ChakraPetch-Bold.ttf"),
        "heading": ("ChakraPetch-SemiBold.ttf", "ChakraPetch-Bold.ttf"),
        "body": ("Rajdhani-Medium.ttf", "Rajdhani-Regular.ttf"),
        "mono": ("ShareTechMono-Regular.ttf",),
    }

    _SYS_FALLBACKS: dict[FontRole, tuple[str, ...]] = {
        "title": ("michroma", "freesansbold", "nimbussans", "liberationsans", "notosans"),
        "heading": ("chakrapetch", "nimbussans", "liberationsans", "notosans", "adwaitasans"),
        "body": ("rajdhani", "liberationsans", "notosans", "adwaitasans", "freesans"),
        "mono": ("sharetechmono", "liberationmono", "notosansmono", "adwaitamono", "nimbusmonops"),
    }

    def __init__(self) -> None:
        if not pygame.font.get_init():
            pygame.font.init()
        self._cache: dict[tuple[FontRole, int, bool], pygame.font.Font] = {}
        self._resolved_paths: dict[tuple[FontRole, bool], str | None] = {}

    def _resolve_path(self, role: FontRole, bold: bool) -> str | None:
        key = (role, bold)
        if key in self._resolved_paths:
            return self._resolved_paths[key]

        if FONTS_DIR.exists():
            for fname in self._ROLE_FILES[role]:
                candidate = FONTS_DIR / fname
                if candidate.exists():
                    self._resolved_paths[key] = str(candidate)
                    return str(candidate)

        for sys_name in self._SYS_FALLBACKS[role]:
            matched = pygame.font.match_font(sys_name, bold=bold)
            if matched:
                self._resolved_paths[key] = matched
                return matched

        self._resolved_paths[key] = None
        return None

    def get(self, role: FontRole, size: int, bold: bool = False) -> pygame.font.Font:
        key = (role, size, bold)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        path = self._resolve_path(role, bold)
        font = (
            pygame.font.Font(path, size) if path else pygame.font.SysFont("sans", size, bold=bold)
        )
        if path and bold:
            font.set_bold(True)
        self._cache[key] = font
        return font


class AssetVault:
    """Loads, scales, tints, and caches all extracted 4K/2x UI surfaces."""

    MACHINE_FILES: dict[str, tuple[str, str]] = {
        "normal": ("machine_normal_2x.png", "machine_normal.png"),
        "breached": ("machine_breached_2x.png", "machine_breached.png"),
        "under_breach": ("machine_under_breach_2x.png", "machine_under_breach.png"),
        "patched": ("machine_patched_2x.png", "machine_patched.png"),
        "honeypot": ("machine_honeypot_2x.png", "machine_honeypot.png"),
        "crown_jewel": ("machine_crown_jewel_2x.png", "machine_crown_jewel.png"),
        "entry_point": ("machine_entry_point_2x.png", "machine_entry_point.png"),
        "selected": ("machine_selected_2x.png", "machine_selected.png"),
    }

    ICON_NAMES: tuple[str, ...] = (
        "icon_patch",
        "icon_firewall",
        "icon_honeypot",
        "icon_pass",
        "icon_alert_bell",
        "icon_warning",
        "icon_play",
        "icon_pause",
        "icon_step",
        "icon_speed",
        "icon_settings",
        "icon_back",
        "icon_replay",
        "icon_home",
        "icon_heatmap",
        "icon_astar",
        "link_firewall_barrier",
        "link_pruned_x",
    )

    EFFECT_NAMES: tuple[str, ...] = (
        "effect_glow_disc",
        "effect_pulse_ring",
        "effect_heat_blob",
        "effect_shield_ripple",
        "effect_spark",
    )

    def __init__(self) -> None:
        self._raw: dict[Path, pygame.Surface] = {}
        self._scaled: dict[tuple[str, int, int], pygame.Surface] = {}
        self._tinted: dict[tuple[str, int, int, RGB, int], pygame.Surface] = {}
        self._cover_cache: dict[tuple[str, int, int], pygame.Surface] = {}
        self._contain_cache: dict[tuple[str, int, int], pygame.Surface] = {}
        self._grid_tiles: dict[tuple[int, int, int], pygame.Surface] = {}

    def _load_raw(self, path: Path) -> pygame.Surface:
        cached = self._raw.get(path)
        if cached is not None:
            return cached
        if not path.exists():
            fallback = pygame.Surface((64, 64), pygame.SRCALPHA)
            fallback.fill((34, 232, 200, 40))
            self._raw[path] = fallback
            return fallback
        surf = pygame.image.load(str(path))
        if pygame.display.get_surface() is not None:
            surf = surf.convert_alpha() if path.suffix.lower() == ".png" else surf.convert()
        self._raw[path] = surf
        return surf

    def get_machine(self, state_key: str, size: int, high_res: bool = False) -> pygame.Surface:
        pair = self.MACHINE_FILES.get(state_key, self.MACHINE_FILES["normal"])
        filename = pair[1] if high_res else pair[0]
        cache_key = (f"machines/{filename}", size, size)
        cached = self._scaled.get(cache_key)
        if cached is not None:
            return cached
        raw = self._load_raw(IMAGES_DIR / "machines" / filename)
        scaled = pygame.transform.smoothscale(raw, (size, size))
        self._scaled[cache_key] = scaled
        return scaled

    def get_icon(self, name: str, size: int = 16, tint: RGB | None = None) -> pygame.Surface:
        fname = name if name.endswith(".png") else f"{name}.png"
        if tint is None:
            cache_key = (f"icons/{fname}", size, size)
            cached = self._scaled.get(cache_key)
            if cached is not None:
                return cached
            raw = self._load_raw(IMAGES_DIR / "icons" / fname)
            scaled = pygame.transform.smoothscale(raw, (size, size))
            self._scaled[cache_key] = scaled
            return scaled

        tkey = (f"icons/{fname}", size, size, tint, 255)
        tcached = self._tinted.get(tkey)
        if tcached is not None:
            return tcached
        base = self.get_icon(name, size=size).copy()
        base.fill((tint[0], tint[1], tint[2], 255), special_flags=pygame.BLEND_RGBA_MULT)
        self._tinted[tkey] = base
        return base

    def get_effect(
        self,
        name: str,
        size: tuple[int, int],
        tint: RGB | None = None,
        alpha: int = 255,
    ) -> pygame.Surface:
        fname = name if name.endswith(".png") else f"{name}.png"
        tint_rgb = tint if tint is not None else (255, 255, 255)
        clamped_alpha = max(0, min(255, alpha))
        tkey = (f"effects/{fname}", size[0], size[1], tint_rgb, clamped_alpha)
        tcached = self._tinted.get(tkey)
        if tcached is not None:
            return tcached

        base_key = (f"effects/{fname}", size[0], size[1])
        scaled = self._scaled.get(base_key)
        if scaled is None:
            raw = self._load_raw(IMAGES_DIR / "effects" / fname)
            scaled = pygame.transform.smoothscale(raw, size)
            self._scaled[base_key] = scaled

        tinted = scaled.copy()
        if tint is not None or clamped_alpha < 255:
            tinted.fill(
                (tint_rgb[0], tint_rgb[1], tint_rgb[2], clamped_alpha),
                special_flags=pygame.BLEND_RGBA_MULT,
            )
        self._tinted[tkey] = tinted
        return tinted

    def get_branding(self, name: str, size: int | None = None) -> pygame.Surface:
        fname = name if name.endswith(".png") else f"{name}.png"
        raw = self._load_raw(IMAGES_DIR / "branding" / fname)
        if size is None:
            return raw
        cache_key = (f"branding/{fname}", size, size)
        cached = self._scaled.get(cache_key)
        if cached is not None:
            return cached
        scaled = pygame.transform.smoothscale(raw, (size, size))
        self._scaled[cache_key] = scaled
        return scaled

    def get_background_raw(self, filename: str) -> pygame.Surface:
        return self._load_raw(IMAGES_DIR / "backgrounds" / filename)

    def get_atlas_raw(self, filename: str) -> pygame.Surface:
        return self._load_raw(IMAGES_DIR / "atlas" / filename)

    def get_cover(self, rel_path: str, target_size: tuple[int, int]) -> pygame.Surface:
        """Scale and center-crop an image to fill target_size without distortion."""
        key = (rel_path, target_size[0], target_size[1])
        cached = self._cover_cache.get(key)
        if cached is not None:
            return cached

        raw = self._load_raw(IMAGES_DIR / rel_path)
        rw, rh = raw.get_size()
        tw, th = max(1, target_size[0]), max(1, target_size[1])
        scale = max(tw / max(1, rw), th / max(1, rh))
        sw, sh = max(1, int(rw * scale)), max(1, int(rh * scale))
        scaled = pygame.transform.smoothscale(raw, (sw, sh))
        crop_x = max(0, (sw - tw) // 2)
        crop_y = max(0, (sh - th) // 2)
        sub = scaled.subsurface(pygame.Rect(crop_x, crop_y, min(tw, sw), min(th, sh))).copy()
        self._cover_cache[key] = sub
        return sub

    def get_contain(self, rel_path: str, max_size: tuple[int, int]) -> pygame.Surface:
        """Scale an image proportionally to fit within max_size."""
        key = (rel_path, max_size[0], max_size[1])
        cached = self._contain_cache.get(key)
        if cached is not None:
            return cached

        raw = self._load_raw(IMAGES_DIR / rel_path)
        rw, rh = raw.get_size()
        mw, mh = max(1, max_size[0]), max(1, max_size[1])
        scale = min(mw / max(1, rw), mh / max(1, rh))
        tw, th = max(1, int(rw * scale)), max(1, int(rh * scale))
        scaled = pygame.transform.smoothscale(raw, (tw, th))
        self._contain_cache[key] = scaled
        return scaled

    def get_tiled_iso_grid(self, size: tuple[int, int], alpha: int = 110) -> pygame.Surface:
        """Build a tiled 2:1 isometric grid surface matching target dimensions."""
        w, h = max(1, size[0]), max(1, size[1])
        key = (w, h, alpha)
        cached = self._grid_tiles.get(key)
        if cached is not None:
            return cached

        tile = self._load_raw(IMAGES_DIR / "backgrounds" / "bg_isometric_grid.png")
        tw, th = tile.get_size()
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        for y in range(0, h, max(1, th)):
            for x in range(0, w, max(1, tw)):
                surf.blit(tile, (x, y))
        if alpha < 255:
            surf.set_alpha(alpha)
        self._grid_tiles[key] = surf
        return surf
