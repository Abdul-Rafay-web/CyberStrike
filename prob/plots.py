from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from config import CONFIG  # noqa: E402

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


@dataclass(frozen=True, slots=True)
class TurnBeliefPoint:
    """Average tracking quality across games at one turn number."""

    turn: int
    samples: int
    belief_true_prob: float
    uniform_true_prob: float
    last_alert_true_prob: float
    top1_hit_rate: float
    top3_hit_rate: float
    mean_entropy: float


@dataclass(frozen=True, slots=True)
class TrackerSummaryRow:
    """Aggregate comparison across methods over all observed Defender turns."""

    name: str
    mean_true_prob: float
    top1_hit_rate: float
    top3_hit_rate: float


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


def plot_belief_by_turn(points: list[TurnBeliefPoint], path: Path) -> None:
    """Playbook Chart #5: probability on the true machine per turn, belief tracker vs baselines."""
    fig, ax = _figure(8.2, 4.8)
    turns = [p.turn for p in points]
    ax.plot(
        turns,
        [p.belief_true_prob for p in points],
        color=CYAN,
        marker="o",
        linewidth=2.2,
        label="Bayesian belief tracker",
    )
    ax.plot(
        turns,
        [p.last_alert_true_prob for p in points],
        color=ORANGE,
        marker="s",
        linewidth=1.6,
        linestyle="-.",
        label="Last-alert guess",
    )
    ax.plot(
        turns,
        [p.uniform_true_prob for p in points],
        color=RED,
        linestyle="--",
        linewidth=1.5,
        label="Uniform baseline (1 / N)",
    )
    ax.set_xlabel("turn")
    ax.set_ylabel("mean probability assigned to true machine")
    ax.set_ylim(0.0, 1.0)
    if turns:
        ax.set_xticks(turns)
    ax.legend(facecolor=PANEL, edgecolor=MUTED, labelcolor=TEXT)
    _style(ax, "Probability on the true machine per turn: belief tracker vs baselines")
    _save(fig, path)


def plot_tracker_comparison(rows: list[TrackerSummaryRow], path: Path) -> None:
    """Grouped bar chart comparing true-machine probability, Top-1 hit rate, and Top-3 hit rate."""
    fig, ax = _figure(8.4, 4.8)
    names = [row.name for row in rows]
    positions = np.arange(len(names))
    width = 0.25

    ax.bar(
        positions - width,
        [r.mean_true_prob for r in rows],
        width,
        color=CYAN,
        label="P(true machine)",
    )
    ax.bar(
        positions,
        [r.top1_hit_rate for r in rows],
        width,
        color=VIOLET,
        label="Top-1 hit rate",
    )
    ax.bar(
        positions + width,
        [r.top3_hit_rate for r in rows],
        width,
        color=ORANGE,
        label="Top-3 hit rate",
    )
    ax.set_xticks(positions, names)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("score")
    ax.legend(facecolor=PANEL, edgecolor=MUTED, labelcolor=TEXT)
    _style(ax, "Bayesian belief tracker against baselines (Naive Bayes noisy sensor)")
    _save(fig, path)

