from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StockConfig:
    patches: int = 3
    firewalls: int = 2
    honeypots: int = 2


@dataclass(frozen=True)
class GameConfig:
    turn_limit: int = 25
    default_machines: int = 15
    min_machines: int = 10
    max_machines: int = 25
    min_hardness: int = 1
    max_hardness: int = 3
    patch_hardness_bonus: int = 2
    honeypot_delay_turns: int = 2
    stock: StockConfig = field(default_factory=StockConfig)
    win_value: int = 1000
    turn_penalty: int = 10


@dataclass(frozen=True)
class SearchConfig:
    max_depth_plies: int = 6
    time_budget_seconds: float = 2.0
    defender_reply_radius: int = 2


@dataclass(frozen=True)
class SeedConfig:
    master_seed: int = 42
    dev_seed_start: int = 0
    dev_seed_count: int = 100
    test_seed_start: int = 1000
    test_seed_count: int = 300


@dataclass(frozen=True)
class Theme:
    window_size: tuple[int, int] = (1280, 720)
    fps: int = 60
    background: tuple[int, int, int] = (6, 6, 13)
    panel: tuple[int, int, int] = (18, 18, 42)
    violet: tuple[int, int, int] = (168, 85, 247)
    cyan: tuple[int, int, int] = (34, 232, 200)
    orange: tuple[int, int, int] = (255, 164, 27)
    red: tuple[int, int, int] = (255, 45, 98)
    text: tuple[int, int, int] = (233, 233, 245)
    muted: tuple[int, int, int] = (141, 141, 180)


@dataclass(frozen=True)
class AppConfig:
    game: GameConfig = field(default_factory=GameConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    seeds: SeedConfig = field(default_factory=SeedConfig)
    theme: Theme = field(default_factory=Theme)


CONFIG = AppConfig()
