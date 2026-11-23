from pathlib import Path

import numpy as np

from config import CONFIG
from ml.crossval import train_validation_test_split
from ml.dataset import FEATURES, Dataset
from ml.metrics import majority_baseline
from ml.pipeline import load_sensor, save_sensor, train_sensor
from ml.study import forward_selection, smoothing_curve
from ml.synthetic_logs import generate_dataset
from utils.rng import STREAM_DATA, make_rng


def test_the_trained_sensor_beats_always_saying_normal() -> None:
    sensor = train_sensor()
    assert 0.0 < sensor.threshold < 1.0
    assert sensor.test.recall > 0.6
    assert sensor.test.false_alarm_rate < 0.2
    baseline = majority_baseline(np.zeros(10, dtype=np.int64), np.array([0] * 8 + [1] * 2))
    assert sensor.test.accuracy > baseline.accuracy
    assert sensor.test.f1 > 0.5


def test_training_is_repeatable_and_fast_to_rerun() -> None:
    first = train_sensor()
    second = train_sensor()
    assert first.test == second.test
    assert first.threshold == second.threshold


def test_rates_are_the_measured_recall_and_false_alarm_rate() -> None:
    sensor = train_sensor()
    assert sensor.rates.detection_rate == sensor.test.recall
    assert sensor.rates.false_alarm_rate == sensor.test.false_alarm_rate


def test_a_saved_sensor_loads_back_the_same(tmp_path: Path) -> None:
    sensor = train_sensor()
    path = tmp_path / "sensor.json"
    save_sensor(sensor, path)
    loaded = load_sensor(path)
    sample = generate_dataset(200, 0.2, make_rng(9, STREAM_DATA))
    assert np.allclose(
        sensor.model.predict_proba(sample.features), loaded.model.predict_proba(sample.features)
    )
    assert loaded.threshold == sensor.threshold
    assert loaded.test == sensor.test


def _split() -> tuple[Dataset, Dataset, Dataset, Dataset]:
    rng = make_rng(CONFIG.seeds.master_seed, STREAM_DATA)
    data = generate_dataset(CONFIG.ml.rows, CONFIG.ml.attack_share, rng)
    train, validation, test = train_validation_test_split(data.labels, 0.6, 0.2, rng)
    return data, data.take(train), data.take(validation), data.take(test)


def test_smoothing_rescues_tiny_training_sets() -> None:
    _, train, _, test = _split()
    rows = smoothing_curve(train, test, (10, 640), repeats=20, seed=1)
    tiny, large = rows
    assert tiny.f1_with > tiny.f1_without + 0.1
    assert tiny.ruled_out_without > 0.1
    assert abs(large.f1_with - large.f1_without) < 0.03
    assert large.ruled_out_without < 0.01


def test_forward_selection_adds_one_feature_at_a_time() -> None:
    _, train, validation, _ = _split()
    steps = forward_selection(train, validation, 1.0)
    assert len(steps) == len(FEATURES)
    assert [len(step.features) for step in steps] == list(range(1, len(FEATURES) + 1))
    assert steps[-1].validation_f1 > steps[0].validation_f1 + 0.1
    last_gain = steps[-1].validation_f1 - steps[-2].validation_f1
    assert last_gain < 0.02

