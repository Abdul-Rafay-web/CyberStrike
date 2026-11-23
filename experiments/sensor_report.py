from __future__ import annotations

import numpy as np

from config import CONFIG
from ml.crossval import (
    CrossValResult,
    cross_validate,
    leave_one_out,
    train_validation_test_split,
)
from ml.dataset import Dataset, write_csv
from ml.metrics import ConfusionMatrix, majority_baseline
from ml.pipeline import save_sensor, train_sensor
from ml.plots import plot_confusion, plot_feature_steps, plot_kfold, plot_smoothing_curve
from ml.study import FeatureStep, SmoothingRow, forward_selection, smoothing_curve
from ml.synthetic_logs import generate_dataset
from utils.paths import DATA_DIR, RESULTS_DIR
from utils.rng import STREAM_DATA, make_rng

SMALL_SAMPLE = 100
HEADER = f"{'':<30}{'accuracy (spread)':>18}{'precision':>10}{'recall':>8}{'F1':>8}{'false':>8}"


def _fold_line(name: str, result: CrossValResult) -> str:
    accuracy = f"{result.accuracy.mean:.3f} ({result.accuracy.std:.3f})"
    return (
        f"{name:<30}{accuracy:>18}{result.precision.mean:>10.3f}{result.recall.mean:>8.3f}"
        f"{result.f1.mean:>8.3f}{result.false_alarm_rate.mean:>8.3f}"
    )


def _matrix_line(name: str, matrix: ConfusionMatrix) -> str:
    return (
        f"{name:<30}{matrix.accuracy:>18.3f}{matrix.precision:>10.3f}"
        f"{matrix.recall:>8.3f}{matrix.f1:>8.3f}{matrix.false_alarm_rate:>8.3f}"
    )


def _cross_validation(development: Dataset, seed: int) -> dict[str, CrossValResult]:
    folds = CONFIG.ml.folds
    print(f"{folds} fold cross validation on the development rows ({len(development)})")
    print(HEADER)
    small = development.take(np.arange(SMALL_SAMPLE))
    cases = (
        ("all rows, smoothing on", development, CONFIG.ml.smoothing),
        ("all rows, smoothing off", development, 0.0),
        (f"{SMALL_SAMPLE} rows, smoothing on", small, CONFIG.ml.smoothing),
        (f"{SMALL_SAMPLE} rows, smoothing off", small, 0.0),
    )
    results: dict[str, CrossValResult] = {}
    for name, data, smoothing in cases:
        results[name] = cross_validate(data, folds, smoothing, make_rng(seed, STREAM_DATA))
        print(_fold_line(name, results[name]))
    labels = development.labels
    print(_matrix_line("always say normal", majority_baseline(labels, labels)))
    return results


def _leave_one_out(development: Dataset) -> None:
    rows = CONFIG.ml.loocv_rows
    subset = development.take(np.arange(rows))
    print(f"\nleave one out on {rows} rows")
    print(_matrix_line("smoothing on", leave_one_out(subset, CONFIG.ml.smoothing)))
    print(_matrix_line("smoothing off", leave_one_out(subset, 0.0)))


def _smoothing(train: Dataset, test: Dataset, seed: int) -> list[SmoothingRow]:
    print("\nsmoothing with scarce data (average of 60 random samples)")
    print(f"{'training rows':>14}{'F1 off':>9}{'F1 on':>9}{'rows ruled out (off)':>24}")
    rows = smoothing_curve(train, test, (10, 20, 40, 80, 160, 640), 60, seed)
    for row in rows:
        print(
            f"{row.train_rows:>14}{row.f1_without:>9.3f}{row.f1_with:>9.3f}"
            f"{row.ruled_out_without:>24.1%}"
        )
    return rows


def _features(train: Dataset, validation: Dataset) -> list[FeatureStep]:
    print("\nforward feature selection (training F1 against validation F1)")
    steps = forward_selection(train, validation, CONFIG.ml.smoothing)
    for number, step in enumerate(steps, start=1):
        added = f"+{step.features[-1]}"
        print(f"{number} features: {added:<16}{step.train_f1:>8.3f}{step.validation_f1:>8.3f}")
    return steps


def main() -> None:
    ml = CONFIG.ml
    seed = CONFIG.seeds.master_seed
    rng = make_rng(seed, STREAM_DATA)
    data = generate_dataset(ml.rows, ml.attack_share, rng)
    write_csv(data, DATA_DIR / "synthetic_logs.csv")
    print(f"wrote {len(data)} records to data/synthetic_logs.csv")

    rows = train_validation_test_split(data.labels, ml.train_share, ml.validation_share, rng)
    train, validation, test = (data.take(r) for r in rows)
    development = data.take(np.concatenate([rows[0], rows[1]]))
    print(f"train {len(train)}, validation {len(validation)}, test {len(test)}\n")

    results = _cross_validation(development, seed)
    _leave_one_out(development)
    curve = _smoothing(train, test, seed)
    steps = _features(train, validation)

    sensor = train_sensor(ml, seed)
    print(f"\nthreshold chosen on validation data: {sensor.threshold:.2f}")
    print(_matrix_line("final model on the test set", sensor.test))
    print(
        f"\nsensor rates to use in Phase 6: detection_rate={sensor.rates.detection_rate:.3f}, "
        f"false_alarm_rate={sensor.rates.false_alarm_rate:.3f}"
    )

    save_sensor(sensor, DATA_DIR / "sensor_model.json")
    plot_confusion(sensor.test, RESULTS_DIR / "confusion_matrix.png")
    plot_kfold(results, RESULTS_DIR / "kfold_smoothing.png")
    plot_smoothing_curve(curve, RESULTS_DIR / "smoothing_curve.png")
    plot_feature_steps(steps, RESULTS_DIR / "feature_steps.png")
    print("\nsaved data/sensor_model.json and four charts in results/")


if __name__ == "__main__":
    main()

