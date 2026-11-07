from dataclasses import FrozenInstanceError

import pytest

from config import CONFIG


def test_defaults_match_the_playbook() -> None:
    assert CONFIG.game.turn_limit == 25
    assert CONFIG.game.stock.patches == 3
    assert CONFIG.seeds.master_seed == 42


def test_config_cannot_be_changed() -> None:
    with pytest.raises(FrozenInstanceError):
        setattr(CONFIG.game, "turn_limit", 5)  # noqa: B010
 
