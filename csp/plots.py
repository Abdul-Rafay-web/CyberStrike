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
class CSPBenchRow:
    """Mean solver metrics over many instances at one map size and one solver setting."""

    map_size: int
    solver_name: str
    mean_assignments: float
    mean_backtracks: float
    mean_ms: float
    solve_rate: float


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


def plot_csp_backtracks(rows: list[CSPBenchRow], path: Path) -> None:
    """Playbook Chart #3: CSP backtracks versus map size for four solver settings."""
    fig, ax = _figure(8.4, 4.8)
    sizes = sorted({r.map_size for r in rows})
    solver_styles = (
        ("plain_bt", "Plain backtracking", RED, "o"),
        ("fc", "Forward checking", ORANGE, "s"),
        ("mrv_lcv_fc", "MRV + LCV + FC", VIOLET, "^"),
        ("mac_mrv_lcv", "MAC (AC-3) + MRV + LCV", CYAN, "D"),
    )
    by_key = {(r.map_size, r.solver_name): r for r in rows}
    for key, label, color, marker in solver_styles:
        vals = [by_key[(sz, key)].mean_backtracks for sz in sizes if (sz, key) in by_key]
        xs = [sz for sz in sizes if (sz, key) in by_key]
        if vals:
            ax.plot(xs, vals, color=color, marker=marker, linewidth=2.2, label=label)

    ax.set_xticks(sizes)
    ax.set_xlabel("map size (machines)")
    ax.set_ylabel("mean backtracks per instance")
    if any(r.mean_backtracks > 50 for r in rows):
        ax.set_yscale("symlog", linthresh=1.0)
    ax.legend(facecolor=PANEL, edgecolor=MUTED, labelcolor=TEXT)
    _style(ax, "CSP backtracks vs map size across four solver settings")
    _save(fig, path)


def plot_csp_defender_matchup(
    matchup_rows: list[tuple[str, float, float]], path: Path
) -> None:
    """Bar chart comparing defender win rates (BeliefCSPDefender vs StaticDefender)."""
    fig, ax = _figure(7.8, 4.6)
    labels = [row[0] for row in matchup_rows]
    static_rates = [row[1] for row in matchup_rows]
    csp_rates = [row[2] for row in matchup_rows]
    positions = np.arange(len(labels))
    width = 0.34

    ax.bar(positions - width / 2, static_rates, width, color=ORANGE, label="Static Defender")
    ax.bar(positions + width / 2, csp_rates, width, color=CYAN, label="Belief + CSP Defender")
    ax.set_xticks(positions, labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("defender win rate")
    ax.legend(facecolor=PANEL, edgecolor=MUTED, labelcolor=TEXT)
    _style(ax, "Defender win rate over 300 maps: Belief + CSP vs Static")
    _save(fig, path)

