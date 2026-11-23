from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml.dataset import FEATURES, Dataset

Weights = tuple[float, ...]


@dataclass(frozen=True, slots=True)
class Profile:
    """One kind of traffic. Each list gives the chance of each value of that feature."""

    name: str
    attack: bool
    share: float
    protocol: Weights
    port_group: Weights
    bytes_group: Weights
    failed_logins: Weights
    hour_group: Weights
    connections: Weights

    def tables(self) -> tuple[Weights, ...]:
        return (
            self.protocol,
            self.port_group,
            self.bytes_group,
            self.failed_logins,
            self.hour_group,
            self.connections,
        )


PROFILES: tuple[Profile, ...] = (
    Profile(
        name="web browsing",
        attack=False,
        share=0.45,
        protocol=(0.90, 0.08, 0.02),
        port_group=(0.80, 0.00, 0.00, 0.05, 0.10, 0.05),
        bytes_group=(0.05, 0.30, 0.40, 0.20, 0.05),
        failed_logins=(0.97, 0.03, 0.00, 0.00),
        hour_group=(0.05, 0.30, 0.40, 0.25),
        connections=(0.50, 0.42, 0.07, 0.01),
    ),
    Profile(
        name="database and file work",
        attack=False,
        share=0.20,
        protocol=(0.95, 0.04, 0.01),
        port_group=(0.05, 0.05, 0.60, 0.00, 0.10, 0.20),
        bytes_group=(0.02, 0.18, 0.40, 0.30, 0.10),
        failed_logins=(0.97, 0.03, 0.00, 0.00),
        hour_group=(0.05, 0.35, 0.45, 0.15),
        connections=(0.30, 0.50, 0.19, 0.01),
    ),
    Profile(
        name="admin sessions",
        attack=False,
        share=0.12,
        protocol=(1.00, 0.00, 0.00),
        port_group=(0.00, 0.90, 0.00, 0.00, 0.05, 0.05),
        bytes_group=(0.10, 0.60, 0.25, 0.05, 0.00),
        failed_logins=(0.78, 0.17, 0.04, 0.01),
        hour_group=(0.03, 0.42, 0.45, 0.10),
        connections=(0.50, 0.40, 0.10, 0.00),
    ),
    Profile(
        name="mail and chat",
        attack=False,
        share=0.13,
        protocol=(0.60, 0.40, 0.00),
        port_group=(0.10, 0.00, 0.00, 0.55, 0.30, 0.05),
        bytes_group=(0.20, 0.60, 0.17, 0.03, 0.00),
        failed_logins=(0.99, 0.01, 0.00, 0.00),
        hour_group=(0.05, 0.35, 0.35, 0.25),
        connections=(0.55, 0.38, 0.07, 0.00),
    ),
    Profile(
        name="backup jobs",
        attack=False,
        share=0.10,
        protocol=(1.00, 0.00, 0.00),
        port_group=(0.00, 0.00, 0.35, 0.00, 0.15, 0.50),
        bytes_group=(0.00, 0.05, 0.15, 0.40, 0.40),
        failed_logins=(1.00, 0.00, 0.00, 0.00),
        hour_group=(0.70, 0.10, 0.05, 0.15),
        connections=(0.80, 0.18, 0.02, 0.00),
    ),
    Profile(
        name="port scan",
        attack=True,
        share=0.32,
        protocol=(0.70, 0.10, 0.20),
        port_group=(0.10, 0.10, 0.05, 0.05, 0.30, 0.40),
        bytes_group=(0.80, 0.18, 0.02, 0.00, 0.00),
        failed_logins=(0.85, 0.12, 0.03, 0.00),
        hour_group=(0.35, 0.20, 0.20, 0.25),
        connections=(0.02, 0.10, 0.38, 0.50),
    ),
    Profile(
        name="password guessing",
        attack=True,
        share=0.32,
        protocol=(1.00, 0.00, 0.00),
        port_group=(0.08, 0.84, 0.05, 0.03, 0.00, 0.00),
        bytes_group=(0.25, 0.60, 0.13, 0.02, 0.00),
        failed_logins=(0.05, 0.12, 0.38, 0.45),
        hour_group=(0.40, 0.10, 0.20, 0.30),
        connections=(0.05, 0.25, 0.60, 0.10),
    ),
    Profile(
        name="data theft",
        attack=True,
        share=0.26,
        protocol=(0.95, 0.05, 0.00),
        port_group=(0.20, 0.00, 0.00, 0.00, 0.50, 0.30),
        bytes_group=(0.00, 0.00, 0.08, 0.32, 0.60),
        failed_logins=(0.85, 0.12, 0.03, 0.00),
        hour_group=(0.60, 0.05, 0.10, 0.25),
        connections=(0.55, 0.38, 0.07, 0.00),
    ),
    Profile(
        name="quiet beacon",
        attack=True,
        share=0.10,
        protocol=(0.60, 0.40, 0.00),
        port_group=(0.30, 0.00, 0.00, 0.00, 0.70, 0.00),
        bytes_group=(0.45, 0.45, 0.10, 0.00, 0.00),
        failed_logins=(0.97, 0.03, 0.00, 0.00),
        hour_group=(0.25, 0.25, 0.25, 0.25),
        connections=(0.50, 0.45, 0.05, 0.00),
    ),
)


class TrafficModel:
    """Draws made up log records. Mixes of profiles, with some overlap between the classes.

    `stealth` is the share of attack records that look like normal traffic.
    `odd_normal` is the share of normal records that look like attacks.
    """

    def __init__(
        self,
        profiles: tuple[Profile, ...] = PROFILES,
        stealth: float = 0.05,
        odd_normal: float = 0.02,
    ) -> None:
        self.profiles = profiles
        widest = max(len(f.values) for f in FEATURES)
        self._cumulative = np.ones((len(profiles), len(FEATURES), widest))
        for p, profile in enumerate(profiles):
            for j, table in enumerate(profile.tables()):
                if abs(sum(table) - 1.0) > 1e-9 or len(table) != len(FEATURES[j].values):
                    raise ValueError(f"bad table for {profile.name} feature {FEATURES[j].name}")
                self._cumulative[p, j, : len(table)] = np.cumsum(table)
        self._class_weights = np.zeros((2, len(profiles)))
        for p, profile in enumerate(profiles):
            same, other = (1 - stealth, stealth) if profile.attack else (1 - odd_normal, odd_normal)
            for label in (0, 1):
                is_own_class = profile.attack == bool(label)
                self._class_weights[label, p] = profile.share * (same if is_own_class else other)
        self._class_weights /= self._class_weights.sum(axis=1, keepdims=True)

    def sample(self, label: int, count: int, rng: np.random.Generator) -> np.ndarray:
        """`count` records of one class, as a whole number matrix."""
        chosen = rng.choice(len(self.profiles), size=count, p=self._class_weights[label])
        records = np.zeros((count, len(FEATURES)), dtype=np.int64)
        for j in range(len(FEATURES)):
            draws = rng.random(count)
            records[:, j] = (draws[:, None] >= self._cumulative[chosen, j]).sum(axis=1)
        return records


DEFAULT_TRAFFIC = TrafficModel()


def generate_dataset(
    rows: int,
    attack_share: float,
    rng: np.random.Generator,
    traffic: TrafficModel = DEFAULT_TRAFFIC,
) -> Dataset:
    """A shuffled labelled dataset with about `attack_share` of the rows being attacks."""
    if not 0.0 < attack_share < 1.0:
        raise ValueError("attack_share must be between 0 and 1")
    attacks = int(round(rows * attack_share))
    normal_rows = rows - attacks
    features = np.vstack([traffic.sample(0, normal_rows, rng), traffic.sample(1, attacks, rng)])
    labels = np.concatenate(
        [np.zeros(normal_rows, dtype=np.int64), np.ones(attacks, dtype=np.int64)]
    )
    order = rng.permutation(rows)
    return Dataset(features[order], labels[order])

