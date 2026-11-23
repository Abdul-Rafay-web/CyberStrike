from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LABEL_NAMES = ("normal", "attack")


@dataclass(frozen=True, slots=True)
class Feature:
    """One column of the log: its name and the fixed list of values it can take."""

    name: str
    values: tuple[str, ...]

    def encode(self, value: str) -> int:
        try:
            return self.values.index(value)
        except ValueError:
            raise ValueError(f"{value!r} is not a value of {self.name}") from None


FEATURES: tuple[Feature, ...] = (
    Feature("protocol", ("tcp", "udp", "icmp")),
    Feature("port_group", ("web", "remote", "database", "mail", "high", "other")),
    Feature("bytes_group", ("tiny", "small", "medium", "large", "huge")),
    Feature("failed_logins", ("none", "few", "several", "many")),
    Feature("hour_group", ("night", "morning", "afternoon", "evening")),
    Feature("connections", ("low", "normal", "busy", "flood")),
)


@dataclass(frozen=True, slots=True)
class Dataset:
    """Log records stored as whole numbers: one row per record, one column per feature."""

    features: np.ndarray
    labels: np.ndarray
    schema: tuple[Feature, ...] = FEATURES

    def __post_init__(self) -> None:
        if self.features.ndim != 2 or self.features.shape[1] != len(self.schema):
            raise ValueError("features must have one column per schema feature")
        if len(self.labels) != len(self.features):
            raise ValueError("there must be one label per row")

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def sizes(self) -> tuple[int, ...]:
        return tuple(len(feature.values) for feature in self.schema)

    def take(self, rows: np.ndarray) -> Dataset:
        return Dataset(self.features[rows], self.labels[rows], self.schema)


def write_csv(dataset: Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([feature.name for feature in dataset.schema] + ["label"])
        for row, label in zip(dataset.features, dataset.labels, strict=True):
            names = [f.values[int(v)] for f, v in zip(dataset.schema, row, strict=True)]
            writer.writerow(names + [LABEL_NAMES[int(label)]])


def read_csv(path: Path, schema: Sequence[Feature] = FEATURES) -> Dataset:
    rows: list[list[int]] = []
    labels: list[int] = []
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        expected = [feature.name for feature in schema] + ["label"]
        if header != expected:
            raise ValueError(f"unexpected columns {header}, expected {expected}")
        for line in reader:
            rows.append([f.encode(value) for f, value in zip(schema, line[:-1], strict=True)])
            labels.append(LABEL_NAMES.index(line[-1]))
    features = np.array(rows, dtype=np.int64).reshape(len(rows), len(schema))
    return Dataset(features, np.array(labels, dtype=np.int64), tuple(schema))

