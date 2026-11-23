import numpy as np
import pytest

from ml.dataset import FEATURES
from ml.metrics import confusion
from ml.naive_bayes import NaiveBayes
from ml.synthetic_logs import PROFILES, Profile, TrafficModel, generate_dataset
from utils.rng import STREAM_DATA, make_rng


def test_every_profile_table_is_a_valid_distribution() -> None:
    TrafficModel()  # building the model checks every table
    for profile in PROFILES:
        for table, feature in zip(profile.tables(), FEATURES, strict=True):
            assert len(table) == len(feature.values)
            assert sum(table) == pytest.approx(1.0)


def test_a_broken_table_is_rejected() -> None:
    good = PROFILES[0]
    broken = Profile(
        "broken",
        False,
        1.0,
        (0.5, 0.2, 0.1),
        good.port_group,
        good.bytes_group,
        good.failed_logins,
        good.hour_group,
        good.connections,
    )
    with pytest.raises(ValueError):
        TrafficModel((broken,))


def test_samples_have_the_right_shape_and_value_ranges() -> None:
    records = TrafficModel().sample(1, 500, make_rng(1, STREAM_DATA))
    assert records.shape == (500, len(FEATURES))
    for column, feature in enumerate(FEATURES):
        assert records[:, column].min() >= 0
        assert records[:, column].max() < len(feature.values)


def test_the_same_seed_gives_the_same_records() -> None:
    first = TrafficModel().sample(0, 200, make_rng(7, STREAM_DATA))
    second = TrafficModel().sample(0, 200, make_rng(7, STREAM_DATA))
    other = TrafficModel().sample(0, 200, make_rng(8, STREAM_DATA))
    assert (first == second).all()
    assert not (first == other).all()


def test_attacks_really_do_look_different() -> None:
    rng = make_rng(2, STREAM_DATA)
    normal = TrafficModel().sample(0, 4000, rng)
    attack = TrafficModel().sample(1, 4000, rng)
    failed = FEATURES.index(next(f for f in FEATURES if f.name == "failed_logins"))
    assert attack[:, failed].mean() > normal[:, failed].mean() + 0.2


def test_generate_dataset_has_the_requested_mix_and_is_shuffled() -> None:
    data = generate_dataset(1000, 0.2, make_rng(3, STREAM_DATA))
    assert len(data) == 1000
    assert data.labels.sum() == 200
    assert not (np.diff(data.labels) >= 0).all()


def test_attack_share_must_be_a_real_share() -> None:
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            generate_dataset(100, bad, make_rng(1, STREAM_DATA))


def test_less_overlap_makes_the_problem_easier() -> None:
    def accuracy(model: TrafficModel) -> float:
        rng = make_rng(4, STREAM_DATA)
        train = generate_dataset(4000, 0.2, rng, model)
        test = generate_dataset(4000, 0.2, rng, model)
        nb = NaiveBayes(1.0).fit(train.features, train.labels, train.sizes)
        return confusion(test.labels, nb.predict(test.features)).accuracy

    assert accuracy(TrafficModel(stealth=0.0, odd_normal=0.0)) > accuracy(TrafficModel())

