from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from config import CONFIG  # noqa: E402
from ml.crossval import CrossValResult, Spread  # noqa: E402
from ml.metrics import ConfusionMatrix  # noqa: E402
from ml.study import FeatureStep, SmoothingRow  # noqa: E402

THEME = CONFIG.theme


def _unit(color: tuple[int, int, int]) -> tuple[float, float, float]:
    return (color[0] / 255, color[1] / 255, color[2] / 255)


BACKGROUND = _unit(THEME.background)
PANEL = _unit(THEME.panel)
TEXT = _unit(THEME.text)
MUTED = _unit(THEME.muted)
VIOLET = _unit(THEME.violet)
CYAN = _unit(THEME.cyan)
ORANGE = _unit(THEME.orange)
RED = _unit(THEME.red)


def _style(ax: Axes, title: str) -> None:
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=TEXT, fontsize=13, pad=12, loc="left")
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.grid(color=MUTED, alpha=0.15)
    ax.set_axisbelow(True)


def _figure(width: float, height: float) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(width, height), facecolor=BACKGROUND)
    return fig, ax


def _save(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_confusion(matrix: ConfusionMatrix, path: Path) -> None:
    fig, ax = _figure(5.6, 4.8)
    grid = np.array([[matrix.tn, matrix.fp], [matrix.fn, matrix.tp]])
    shares = grid / grid.sum(axis=1, keepdims=True)
    ax.imshow(shares, cmap="magma", vmin=0, vmax=1)
    labels = [["true negative", "false alarm"], ["miss", "true positive"]]
    for row in range(2):
        for col in range(2):
            ink = BACKGROUND if shares[row, col] > 0.5 else TEXT
            ax.text(
                col,
                row,
                f"{labels[row][col]}\n{grid[row, col]}\n{shares[row, col]:.0%}",
                ha="center",
                va="center",
                color=ink,
                fontsize=11,
            )
    ax.set_xticks([0, 1], ["no alert", "alert"])
    ax.set_yticks([0, 1], ["normal", "attack"])
    ax.set_xlabel("what the sensor said")
    ax.set_ylabel("what really happened")
    _style(ax, "Confusion matrix on the test set")
    ax.grid(False)
    _save(fig, path)


def plot_kfold(results: dict[str, CrossValResult], path: Path) -> None:
    fig, ax = _figure(8.6, 4.8)
    names = list(results)
    positions = np.arange(len(names))
    width = 0.36
    series: tuple[
        tuple[float, str, tuple[float, float, float], Callable[[CrossValResult], Spread]], ...
    ] = (
        (-width / 2, "accuracy", CYAN, lambda r: r.accuracy),
        (width / 2, "F1 score", VIOLET, lambda r: r.f1),
    )
    for offset, label, color, pick in series:
        means = [pick(results[name]).mean for name in names]
        spread = [pick(results[name]).std for name in names]
        ax.bar(
            positions + offset,
            means,
            width,
            yerr=spread,
            color=color,
            label=label,
            error_kw={"ecolor": TEXT, "capsize": 3},
        )
    ax.set_xticks(positions, names, rotation=12)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score (mean of the folds, bars show spread)")
    ax.legend(facecolor=PANEL, edgecolor=MUTED, labelcolor=TEXT)
    _style(ax, "5 fold cross validation, smoothing on and off")
    _save(fig, path)


def plot_smoothing_curve(rows: list[SmoothingRow], path: Path) -> None:
    fig, ax = _figure(7.6, 4.8)
    sizes = [row.train_rows for row in rows]
    ax.plot(sizes, [r.f1_with for r in rows], color=CYAN, marker="o", label="smoothing on")
    ax.plot(sizes, [r.f1_without for r in rows], color=RED, marker="o", label="smoothing off")
    ax.set_xscale("log")
    ax.set_xlabel("training rows")
    ax.set_ylabel("F1 score on the test set")
    ax.legend(facecolor=PANEL, edgecolor=MUTED, labelcolor=TEXT)
    _style(ax, "Smoothing matters most when data is scarce")
    _save(fig, path)


def plot_feature_steps(steps: list[FeatureStep], path: Path) -> None:
    fig, ax = _figure(7.6, 4.8)
    counts = list(range(1, len(steps) + 1))
    ax.plot(counts, [s.train_f1 for s in steps], color=ORANGE, marker="o", label="training F1")
    validation = [s.validation_f1 for s in steps]
    ax.plot(counts, validation, color=VIOLET, marker="o", label="validation F1")
    for count, step in zip(counts, steps, strict=True):
        ax.annotate(
            step.features[-1],
            (count, step.validation_f1),
            textcoords="offset points",
            xytext=(0, -16),
            ha="center",
            color=MUTED,
            fontsize=8,
        )
    ax.set_xlabel("number of features used")
    ax.set_ylabel("F1 score")
    ax.legend(facecolor=PANEL, edgecolor=MUTED, labelcolor=TEXT)
    _style(ax, "Adding features: where the gains stop")
    _save(fig, path)
