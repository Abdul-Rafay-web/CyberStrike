from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from config import CONFIG  # noqa: E402
from evo.genetic import GenerationRecord  # noqa: E402

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
    ax.set_title(title, color=TEXT, fontsize=12, pad=10, loc="left")
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.grid(color=MUTED, alpha=0.15)
    ax.set_axisbelow(True)


def _save(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_ga_fitness_and_balance(
    history: tuple[GenerationRecord, ...],
    random_win_rates: list[float],
    evolved_win_rates: list[float],
    path: Path,
) -> None:
    """Playbook Chart #6: GA fitness per generation + evolved vs random map balance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.6), facecolor=BACKGROUND)

    gens = [h.generation for h in history]
    ax1.plot(
        gens, [h.best_fitness for h in history], color=CYAN, linewidth=2.2, label="Best fitness"
    )
    ax1.plot(
        gens, [h.mean_fitness for h in history], color=VIOLET, linewidth=2.0, label="Mean fitness"
    )
    ax1.set_xlabel("generation")
    ax1.set_ylabel("fitness score")
    ax1.legend(facecolor=PANEL, edgecolor=MUTED, labelcolor=TEXT)
    _style(ax1, "GA fitness over 40 generations")

    labels = ["Random maps", "Evolved maps"]
    means = [float(np.mean(random_win_rates)), float(np.mean(evolved_win_rates))]
    stds = [float(np.std(random_win_rates)), float(np.std(evolved_win_rates))]
    positions = np.arange(len(labels))
    ax2.bar(
        positions,
        means,
        width=0.45,
        yerr=stds,
        color=[ORANGE, CYAN],
        error_kw={"ecolor": TEXT, "capsize": 4},
    )
    ax2.axhline(0.5, color=RED, linestyle="--", linewidth=1.4, label="50% balanced target")
    ax2.set_xticks(positions, labels)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_ylabel("simulated attacker win rate")
    ax2.legend(facecolor=PANEL, edgecolor=MUTED, labelcolor=TEXT)
    _style(ax2, "Balance: evolved vs random maps")

    _save(fig, path)


def plot_regression_fit(
    true_turns: tuple[float, ...],
    pred_turns: tuple[float, ...],
    r2: float,
    path: Path,
) -> None:
    """Scatter plot of predicted vs simulated turns to breach on held-out test maps."""
    fig, ax = plt.subplots(figsize=(6.4, 5.0), facecolor=BACKGROUND)
    ax.scatter(true_turns, pred_turns, color=CYAN, alpha=0.75, s=28, label="Held-out test maps")
    low = min(min(true_turns), min(pred_turns))
    high = max(max(true_turns), max(pred_turns))
    ax.plot(
        [low, high], [low, high], color=ORANGE, linestyle="--", linewidth=1.5, label="Ideal fit"
    )
    ax.set_xlabel("simulated mean turns")
    ax.set_ylabel("linear regression predicted turns")
    ax.legend(facecolor=PANEL, edgecolor=MUTED, labelcolor=TEXT)
    _style(ax, f"Linear regression surrogate on held-out maps (R^2 = {r2:.3f})")
    _save(fig, path)
