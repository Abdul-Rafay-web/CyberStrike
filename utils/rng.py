from __future__ import annotations

import numpy as np

STREAM_MAP = 0
STREAM_SENSORS = 1
STREAM_AGENTS = 2


def make_rng(seed: int, stream: int = STREAM_MAP) -> np.random.Generator:
    """Return a random generator for one seed and one named stream."""
    sequence = np.random.SeedSequence([seed, stream])
    return np.random.default_rng(sequence)
