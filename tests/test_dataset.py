from pathlib import Path

import numpy as np
import pytest

from ml.dataset import FEATURES, Dataset, Feature, read_csv, write_csv
from ml.synthetic_logs import generate_dataset
from utils.rng import STREAM_DATA, make_rng


def test_csv_round_trip_keeps_every_record(tmp_path: Path) -> None:
    data = generate_dataset(300, 0.25, make_rng(1, STREAM_DATA))
    path = tmp_path / "logs.csv"
    write_csv(data, path)
    again = read_csv(path)
    assert (again.features == data.features).all()
    assert (again.labels == data.labels).all()


def test_the_csv_is_readable_text(tmp_path: Path) -> None:
    data = generate_dataset(50, 0.2, make_rng(1, STREAM_DATA))
    path = tmp_path / "logs.csv"
    write_csv(data, path)
    lines = path.read_text().splitlines()
    assert lines[0] == "protocol,port_group,bytes_group,failed_logins,hour_group,connections,label"
    assert len(lines) == 51
    assert lines[1].split(",")[-1] in ("normal", "attack")


def test_a_wrong_header_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("a,b,c\n1,2,3\n")
    with pytest.raises(ValueError):
        read_csv(path)


def test_an_unknown_value_is_rejected() -> None:
    with pytest.raises(ValueError):
        FEATURES[0].encode("carrier pigeon")
    assert Feature("x", ("a", "b")).encode("b") == 1


def test_dataset_checks_its_shape() -> None:
    with pytest.raises(ValueError):
        Dataset(np.zeros((5, 2), dtype=np.int64), np.zeros(5, dtype=np.int64))
    with pytest.raises(ValueError):
        Dataset(np.zeros((5, 6), dtype=np.int64), np.zeros(4, dtype=np.int64))


def test_take_and_sizes() -> None:
    data = generate_dataset(100, 0.2, make_rng(1, STREAM_DATA))
    part = data.take(np.array([3, 5, 7]))
    assert len(part) == 3
    assert (part.features[0] == data.features[3]).all()
    assert data.sizes == (3, 6, 5, 4, 4, 4)

