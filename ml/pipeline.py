from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import CONFIG, MLConfig, SensorConfig
from ml.crossval import best_threshold, train_validation_test_split
from ml.metrics import ConfusionMatrix, confusion
from ml.naive_bayes import NaiveBayes
from ml.synthetic_logs import DEFAULT_TRAFFIC, TrafficModel, generate_dataset
from utils.rng import STREAM_DATA, make_rng


@dataclass(frozen=True, slots=True)
class TrainedSensor:
    """A trained classifier plus everything we measured about it."""

    model: NaiveBayes
    threshold: float
    validation: ConfusionMatrix
    test: ConfusionMatrix

    @property
    def rates(self) -> SensorConfig:
        """How often the sensor alerts on an attacker, and on harmless traffic."""
        return SensorConfig(
            detection_rate=self.test.recall, false_alarm_rate=self.test.false_alarm_rate
        )


def train_sensor(
    config: MLConfig = CONFIG.ml,
    seed: int = CONFIG.seeds.master_seed,
    traffic: TrafficModel = DEFAULT_TRAFFIC,
) -> TrainedSensor:
    """Make data, split it, train on one part, tune the threshold on another, test on the last."""
    rng = make_rng(seed, STREAM_DATA)
    data = generate_dataset(config.rows, config.attack_share, rng, traffic)
    train_rows, validation_rows, test_rows = train_validation_test_split(
        data.labels, config.train_share, config.validation_share, rng
    )
    train = data.take(train_rows)
    validation = data.take(validation_rows)
    test = data.take(test_rows)

    model = NaiveBayes(config.smoothing).fit(train.features, train.labels, data.sizes)
    validation_probability = model.predict_proba(validation.features)[:, 1]
    threshold = best_threshold(validation.labels, validation_probability)
    on_validation = confusion(
        validation.labels, (validation_probability >= threshold).astype(np.int64)
    )
    on_test = confusion(test.labels, model.predict(test.features, threshold))
    return TrainedSensor(model, threshold, on_validation, on_test)


def save_sensor(sensor: TrainedSensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": sensor.model.to_dict(),
        "threshold": sensor.threshold,
        "validation": _matrix_to_dict(sensor.validation),
        "test": _matrix_to_dict(sensor.test),
        "rates": {
            "detection_rate": sensor.rates.detection_rate,
            "false_alarm_rate": sensor.rates.false_alarm_rate,
        },
    }
    path.write_text(json.dumps(payload, indent=2))


def load_sensor(path: Path) -> TrainedSensor:
    payload = json.loads(path.read_text())
    return TrainedSensor(
        model=NaiveBayes.from_dict(payload["model"]),
        threshold=float(payload["threshold"]),
        validation=ConfusionMatrix(**payload["validation"]),
        test=ConfusionMatrix(**payload["test"]),
    )


def _matrix_to_dict(matrix: ConfusionMatrix) -> dict[str, int]:
    return {"tp": matrix.tp, "fp": matrix.fp, "tn": matrix.tn, "fn": matrix.fn}

