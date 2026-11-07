import pytest

from core.actions import Advance, Firewall, Patch


def test_equal_actions_are_equal_and_hashable() -> None:
    assert Advance(3) == Advance(3)
    assert len({Patch(1), Patch(1), Patch(2)}) == 2


def test_firewall_needs_smaller_number_first() -> None:
    with pytest.raises(ValueError):
        Firewall(5, 2)
    assert Firewall(2, 5).b == 5