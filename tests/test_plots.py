from pathlib import Path

from ml.crossval import cross_validate
from ml.metrics import ConfusionMatrix
from ml.plots import plot_confusion, plot_feature_steps, plot_kfold, plot_smoothing_curve
from ml.study import FeatureStep, SmoothingRow
from ml.synthetic_logs import generate_dataset
from utils.rng import STREAM_DATA, make_rng

PNG = b"\x89PNG"


def _is_png(path: Path) -> bool:
    return path.exists() and path.read_bytes()[:4] == PNG and path.stat().st_size > 2000


def test_confusion_chart(tmp_path: Path) -> None:
    plot_confusion(ConfusionMatrix(290, 191, 1409, 110), tmp_path / "c.png")
    assert _is_png(tmp_path / "c.png")


def test_cross_validation_chart(tmp_path: Path) -> None:
    data = generate_dataset(300, 0.2, make_rng(1, STREAM_DATA))
    results = {
        "smoothing on": cross_validate(data, 5, 1.0, make_rng(2, STREAM_DATA)),
        "smoothing off": cross_validate(data, 5, 0.0, make_rng(2, STREAM_DATA)),
    }
    plot_kfold(results, tmp_path / "k.png")
    assert _is_png(tmp_path / "k.png")


def test_smoothing_chart(tmp_path: Path) -> None:
    rows = [SmoothingRow(10, 0.1, 0.4, 0.5), SmoothingRow(100, 0.5, 0.58, 0.02)]
    plot_smoothing_curve(rows, tmp_path / "s.png")
    assert _is_png(tmp_path / "s.png")


def test_feature_chart_and_missing_folders_are_created(tmp_path: Path) -> None:
    steps = [FeatureStep(("a",), 0.5, 0.5), FeatureStep(("a", "b"), 0.6, 0.58)]
    target = tmp_path / "deep" / "folder" / "f.png"
    plot_feature_steps(steps, target)
    assert _is_png(target)

