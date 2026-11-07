from utils.rng import STREAM_AGENTS, STREAM_MAP, make_rng


def test_same_seed_gives_same_numbers() -> None:
    first = make_rng(42, STREAM_MAP).integers(0, 1000, size=5)
    second = make_rng(42, STREAM_MAP).integers(0, 1000, size=5)
    assert first.tolist() == second.tolist()


def test_streams_are_independent() -> None:
    a = make_rng(42, STREAM_MAP).integers(0, 1000, size=5)
    b = make_rng(42, STREAM_AGENTS).integers(0, 1000, size=5)
    assert a.tolist() != b.tolist()


def test_different_seeds_differ() -> None:
    a = make_rng(1).integers(0, 1000, size=5)
    b = make_rng(2).integers(0, 1000, size=5)
    assert a.tolist() != b.tolist()
