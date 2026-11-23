from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _check_ranges(features: np.ndarray, sizes: Sequence[int]) -> None:
    """Every value of feature `column` must be a whole number from 0 up to its size minus 1."""
    if not len(features):
        return
    for column, size in enumerate(sizes):
        if features[:, column].min() < 0 or features[:, column].max() >= size:
            raise ValueError(f"feature {column} has a value outside 0 to {size - 1}")


class NaiveBayes:
    """Naive Bayes for features that are whole number categories.

    Score of a class = log of its share of the data, plus for every feature the log of how
    often that value showed up in that class. Smoothing adds a small count to every value
    so a value never seen in training is rare, not impossible.
    """

    def __init__(self, smoothing: float = 1.0) -> None:
        if smoothing < 0:
            raise ValueError("smoothing cannot be negative")
        self.smoothing = smoothing
        self._sizes: tuple[int, ...] = ()
        self._class_counts = np.zeros(0)
        self._counts: list[np.ndarray] = []
        self._log_prior = np.zeros(0)
        self._log_likelihood: list[np.ndarray] = []

    @property
    def is_fitted(self) -> bool:
        return bool(self._sizes)

    @property
    def n_classes(self) -> int:
        return len(self._class_counts)

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sizes: Sequence[int],
        n_classes: int = 2,
    ) -> NaiveBayes:
        if features.ndim != 2 or features.shape[1] != len(sizes):
            raise ValueError("features needs one column per entry of sizes")
        if len(labels) != len(features):
            raise ValueError("there must be one label per row")
        if len(labels) and (labels.min() < 0 or labels.max() >= n_classes):
            raise ValueError("labels must be whole numbers from 0 to n_classes minus 1")
        _check_ranges(features, sizes)
        class_counts = np.bincount(labels, minlength=n_classes).astype(float)
        if (class_counts == 0).any():
            raise ValueError("every class needs at least one training row")

        counts = []
        for column, size in enumerate(sizes):
            combined = labels * size + features[:, column]
            table = np.bincount(combined, minlength=n_classes * size).reshape(n_classes, size)
            counts.append(table.astype(float))

        self._sizes = tuple(sizes)
        self._class_counts = class_counts
        self._counts = counts
        self._refresh_tables()
        return self

    def _refresh_tables(self) -> None:
        alpha = self.smoothing
        total = self._class_counts.sum()
        with np.errstate(divide="ignore"):
            self._log_prior = np.log(
                (self._class_counts + alpha) / (total + alpha * self.n_classes)
            )
            self._log_likelihood = [
                np.log((table + alpha) / (self._class_counts[:, None] + alpha * size))
                for table, size in zip(self._counts, self._sizes, strict=True)
            ]

    def _check(self, features: np.ndarray) -> None:
        if not self.is_fitted:
            raise RuntimeError("fit the model before using it")
        if features.ndim != 2 or features.shape[1] != len(self._sizes):
            raise ValueError("features has the wrong number of columns")
        _check_ranges(features, self._sizes)

    def log_scores(self, features: np.ndarray) -> np.ndarray:
        """For every row and class: log prior plus the sum of the log likelihoods."""
        self._check(features)
        scores = np.tile(self._log_prior, (len(features), 1))
        for column, table in enumerate(self._log_likelihood):
            scores += table[:, features[:, column]].T
        return scores

    def impossible_rows(self, features: np.ndarray) -> int:
        """Rows that every class rules out. This only happens when smoothing is off."""
        return int(np.isneginf(self.log_scores(features)).all(axis=1).sum())

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        scores = self.log_scores(features)
        top = scores.max(axis=1, keepdims=True)
        ruled_out = np.isneginf(top).ravel()
        safe_top = np.where(np.isneginf(top), 0.0, top)
        weights = np.exp(scores - safe_top)
        totals = weights.sum(axis=1, keepdims=True)
        proba = np.where(totals > 0, weights / np.where(totals > 0, totals, 1.0), 0.0)
        if ruled_out.any():
            prior = np.exp(self._log_prior)
            proba[ruled_out] = prior / prior.sum()
        return proba

    def predict(self, features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Two classes: 1 (attack) when its probability reaches the threshold."""
        if self.n_classes != 2:
            return np.argmax(self.log_scores(features), axis=1)
        return (self.predict_proba(features)[:, 1] >= threshold).astype(np.int64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "smoothing": self.smoothing,
            "sizes": list(self._sizes),
            "class_counts": self._class_counts.tolist(),
            "counts": [table.tolist() for table in self._counts],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NaiveBayes:
        model = cls(float(data["smoothing"]))
        model._sizes = tuple(int(s) for s in data["sizes"])
        model._class_counts = np.array(data["class_counts"], dtype=float)
        model._counts = [np.array(table, dtype=float) for table in data["counts"]]
        model._refresh_tables()
        return model

