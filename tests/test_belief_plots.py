from pathlib import Path

from prob.plots import (
    TrackerSummaryRow,
    TurnBeliefPoint,
    plot_belief_by_turn,
    plot_tracker_comparison,
)

PNG = b"\x89PNG"


def _is_png(path: Path) -> bool:
    return path.exists() and path.read_bytes()[:4] == PNG and path.stat().st_size > 2000


def test_plot_belief_by_turn_creates_png(tmp_path: Path) -> None:
    points = [
        TurnBeliefPoint(1, 50, 0.42, 0.067, 0.25, 0.55, 0.85, 2.1),
        TurnBeliefPoint(2, 50, 0.58, 0.067, 0.28, 0.68, 0.91, 1.6),
        TurnBeliefPoint(3, 50, 0.64, 0.067, 0.27, 0.72, 0.93, 1.4),
    ]
    target = tmp_path / "belief_by_turn.png"
    plot_belief_by_turn(points, target)
    assert _is_png(target)


def test_plot_tracker_comparison_creates_png(tmp_path: Path) -> None:
    rows = [
        TrackerSummaryRow("Bayesian belief", 0.56, 0.66, 0.88),
        TrackerSummaryRow("Last-alert guess", 0.29, 0.34, 0.61),
        TrackerSummaryRow("Uniform (1/N)", 0.067, 0.067, 0.20),
    ]
    target = tmp_path / "nested" / "belief_comparison.png"
    plot_tracker_comparison(rows, target)
    assert _is_png(target)

