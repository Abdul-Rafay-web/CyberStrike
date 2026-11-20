from __future__ import annotations

import contextlib
from typing import Literal

import numpy as np
import pygame

WaveKind = Literal["sine", "triangle", "sawtooth", "square"]
DefenseKind = Literal["patch", "firewall", "honeypot"]


class TacticalSoundEngine:
    """Synthesizes tactical WebAudio-style sound effects using NumPy and Pygame."""

    def __init__(self) -> None:
        self._enabled: bool = True
        self._volume: float = 0.45
        self._mixer_ready: bool = False
        self._sample_rate: int = 22050
        self._channels: int = 2
        self._cache: dict[tuple[object, ...], pygame.mixer.Sound] = {}

    def _ensure_mixer(self) -> bool:
        if not self._enabled:
            return False
        init_info = pygame.mixer.get_init()
        if init_info is None:
            try:
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                init_info = pygame.mixer.get_init()
            except pygame.error:
                self._mixer_ready = False
                return False
        if init_info is None:
            self._mixer_ready = False
            return False
        self._sample_rate = int(init_info[0])
        self._channels = int(init_info[2])
        self._mixer_ready = True
        return True

    def configure(self, enabled: bool, volume_percent: int) -> None:
        self._enabled = enabled
        self._volume = max(0.0, min(1.0, volume_percent / 100.0))

    def _synthesize_tone(
        self,
        freq: float,
        duration: float,
        kind: WaveKind = "sine",
        gain_scale: float = 0.16,
        slide_to: float | None = None,
    ) -> pygame.mixer.Sound | None:
        if not self._ensure_mixer():
            return None
        key = (freq, duration, kind, gain_scale, slide_to, self._sample_rate, self._channels)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        n_samples = max(64, int(self._sample_rate * duration))
        t = np.linspace(0.0, duration, n_samples, endpoint=False)
        if slide_to is not None and slide_to > 0.0 and freq > 0.0:
            # Exponential frequency sweep integrated for smooth phase continuity.
            ratio = slide_to / freq
            if abs(ratio - 1.0) > 1e-4:
                ln_r = float(np.log(ratio))
                phase = 2.0 * np.pi * freq * duration * (np.power(ratio, t / duration) - 1.0) / ln_r
            else:
                phase = 2.0 * np.pi * freq * t
        else:
            phase = 2.0 * np.pi * freq * t

        if kind == "sine":
            wave = np.sin(phase)
        elif kind == "triangle":
            wave = (2.0 / np.pi) * np.arcsin(np.sin(phase))
        elif kind == "sawtooth":
            cycles = phase / (2.0 * np.pi)
            wave = 2.0 * (cycles - np.floor(cycles + 0.5))
        else:
            wave = np.where(np.sin(phase) >= 0.0, 1.0, -1.0)

        # Crisp attack / exponential decay envelope to prevent clicks.
        attack_n = max(1, min(n_samples // 4, int(self._sample_rate * 0.008)))
        env = np.exp(-3.5 * (t / max(duration, 1e-4)))
        env[:attack_n] *= np.linspace(0.0, 1.0, attack_n)

        samples = (wave * env * gain_scale * 32767.0).clip(-32767, 32767).astype(np.int16)
        buf = np.column_stack([samples] * self._channels) if self._channels >= 2 else samples

        try:
            sound = pygame.sndarray.make_sound(buf)
        except (pygame.error, ValueError):
            return None
        self._cache[key] = sound
        return sound

    def _play(
        self,
        freq: float,
        duration: float,
        kind: WaveKind = "sine",
        gain_scale: float = 0.15,
        slide_to: float | None = None,
    ) -> None:
        if not self._enabled or self._volume <= 0.0:
            return
        snd = self._synthesize_tone(freq, duration, kind, gain_scale, slide_to)
        if snd is not None:
            snd.set_volume(self._volume)
            with contextlib.suppress(pygame.error):
                snd.play()

    def hover(self) -> None:
        self._play(1320.0, 0.028, "sine", 0.05, 1680.0)

    def click(self) -> None:
        self._play(680.0, 0.055, "triangle", 0.12, 980.0)

    def defense_placed(self, kind: DefenseKind) -> None:
        if kind == "patch":
            self._play(520.0, 0.14, "triangle", 0.14, 960.0)
        elif kind == "firewall":
            self._play(240.0, 0.16, "sawtooth", 0.11, 180.0)
        else:
            self._play(620.0, 0.15, "sine", 0.13, 1120.0)

    def honeypot_alarm(self) -> None:
        self._play(880.0, 0.18, "sawtooth", 0.16, 620.0)

    def breach_step(self) -> None:
        self._play(190.0, 0.14, "sawtooth", 0.12, 130.0)

    def victory(self) -> None:
        self._play(523.25, 0.25, "triangle", 0.15, 1046.5)

    def defeat(self) -> None:
        self._play(440.0, 0.28, "sawtooth", 0.14, 220.0)
