from dataclasses import FrozenInstanceError, replace

import pytest

from core.state import Side, new_game
from tests.factories import diamond_network


def test_new_game_starts_correctly() -> None:
    state = new_game(diamond_network())
    assert state.attacker_pos == 0
    assert state.breached == frozenset({0})
    assert state.turn == 1
    assert state.side is Side.ATTACKER
    assert (state.stock.patches, state.stock.firewalls, state.stock.honeypots) == (3, 2, 2)


def test_state_cannot_be_edited() -> None:
    state = new_game(diamond_network())
    with pytest.raises(FrozenInstanceError):
        state.turn = 2  # type: ignore[misc]


def test_replace_makes_a_new_state_and_keeps_the_old_one() -> None:
    old = new_game(diamond_network())
    new = replace(old, turn=2)
    assert new.turn == 2
    assert old.turn == 1
    assert new.network is old.network


def test_other_side() -> None:
    assert Side.ATTACKER.other is Side.DEFENDER
    assert Side.DEFENDER.other is Side.ATTACKER