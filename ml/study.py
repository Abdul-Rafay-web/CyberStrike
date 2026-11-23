from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml.crossval import best_threshold
from ml.dataset import Dataset
from ml.metrics import confusion
from ml.naive_bayes import NaiveBayes
from utils.rng import STREAM_DATA, make_rng


@dataclass(frozen=True, slots=True)
class SmoothingRow:
    train_rows: int
    f1_without: float
    f1_with: float
    ruled_out_without: float


def smoothing_curve(
    pool: Dataset, test: Dataset, sizes: tuple[int, ...], repeats: int, seed: int
) -> list[SmoothingRow]:
    """Train on tiny random samples. Compare smoothing off and on, averaged over many samples."""
    rows: list[SmoothingRow] = []
    for size in sizes:
        f1_off: list[float] = []
        f1_on: list[float] = []
        ruled_out: list[float] = []
        for repeat in range(repeats):
            rng = make_rng(seed + repeat, STREAM_DATA)
            sample = pool.take(rng.choice(len(pool), size=size, replace=False))
            if len(np.unique(sample.labels)) < 2:
                continue
            for smoothing, bucket in ((0.0, f1_off), (1.0, f1_on)):
                model = NaiveBayes(smoothing).fit(sample.features, sample.labels, pool.sizes)
                bucket.append(confusion(test.labels, model.predict(test.features)).f1)
                if smoothing == 0.0:
                    ruled_out.append(model.impossible_rows(test.features) / len(test))
        rows.append(
            SmoothingRow(
                size,
                float(np.mean(f1_off)),
                float(np.mean(f1_on)),
                float(np.mean(ruled_out)),
            )
        )
    return rows


@dataclass(frozen=True, slots=True)
class FeatureStep:
    features: tuple[str, ...]
    train_f1: float
    validation_f1: float


def _fit_on(train: Dataset, columns: list[int], smoothing: float) -> NaiveBayes:
    sizes = tuple(train.sizes[c] for c in columns)
    return NaiveBayes(smoothing).fit(train.features[:, columns], train.labels, sizes)


def forward_selection(train: Dataset, validation: Dataset, smoothing: float) -> list[FeatureStep]:
    """Add the most helpful feature one at a time. Shows where more features stop paying off."""
    chosen: list[int] = []
    steps: list[FeatureStep] = []
    remaining = list(range(len(train.schema)))
    while remaining:
        best: tuple[float, int, float] | None = None
        for column in remaining:
            columns = [*chosen, column]
            model = _fit_on(train, columns, smoothing)
            probability = model.predict_proba(validation.features[:, columns])[:, 1]
            threshold = best_threshold(validation.labels, probability)
            score = confusion(
                validation.labels, (probability >= threshold).astype(np.int64)
            ).f1
            if best is None or score > best[0]:
                best = (score, column, threshold)
        assert best is not None
        score, column, threshold = best
        chosen.append(column)
        remaining.remove(column)
        model = _fit_on(train, chosen, smoothing)
        train_probability = model.predict_proba(train.features[:, chosen])[:, 1]
        train_f1 = confusion(train.labels, (train_probability >= threshold).astype(np.int64)).f1
        steps.append(
            FeatureStep(tuple(train.schema[c].name for c in chosen), train_f1, score)
        )
    return steps

