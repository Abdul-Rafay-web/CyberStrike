from dataclasses import replace

import pytest

from core.actions import Advance, Firewall, Honeypot, Pass, Patch
from core.rules import (
    IllegalActionError,
    Outcome,
    apply,
    legal_actions,
    outcome,
    utility,
)
from core.state import Side, Stock, new_game
from tests.factories import chain_network, diamond_network


def test_attacker_options_at_the_start() -> None:
    assert legal_actions(new_game(chain_network())) == [Advance(1)]
    assert legal_actions(new_game(diamond_network())) == [Advance(1), Advance(2)]


def test_easy_machine_is_breached_in_one_turn() -> None:
    state = apply(new_game(diamond_network()), Advance(1))
    assert state.attacker_pos == 1
    assert state.breach_target is None
    assert state.breached == frozenset({0, 1})
    assert state.side is Side.DEFENDER
    assert state.turn == 1


def test_hard_machine_takes_several_turns() -> None:
    s1 = apply(new_game(chain_network()), Advance(1))
    assert s1.breach_target == 1
    assert s1.breach_left == 1
    assert s1.attacker_pos == 0

    s2 = apply(s1, Pass())
    assert s2.side is Side.ATTACKER
    assert s2.turn == 2
    assert legal_actions(s2) == [Advance(1)]

    s3 = apply(s2, Advance(1))
    assert s3.attacker_pos == 1
    assert s3.breach_target is None


def test_patch_during_a_breach_adds_turns() -> None:
    s1 = apply(new_game(chain_network()), Advance(1))
    s2 = apply(s1, Patch(1))
    assert s2.breach_left == 3
    assert 1 in s2.patched
    assert s2.stock.patches == 2


def test_patch_rules() -> None:
    defender = replace(new_game(diamond_network()), side=Side.DEFENDER)
    actions = legal_actions(defender)
    assert Patch(0) in actions
    assert Patch(1) in actions
    after = replace(apply(defender, Patch(1)), side=Side.DEFENDER)
    assert Patch(1) not in legal_actions(after)
def test_firewall_on_the_crossing_link_interrupts_the_breach() -> None:
    mid_breach = apply(new_game(diamond_network()), Advance(2))
    assert mid_breach.breach_target == 2
    assert Firewall(0, 2) in legal_actions(mid_breach)

    after = apply(mid_breach, Firewall(0, 2))
    assert after.breach_target is None
    assert after.breach_left == 0
    assert after.attacker_pos == 0
    assert legal_actions(after) == [Advance(1)]


def test_defender_menu_does_not_depend_on_hidden_facts() -> None:
    quiet = replace(new_game(diamond_network()), side=Side.DEFENDER)
    deep = replace(quiet, attacker_pos=1, breached=frozenset({0, 1}))
    mid_breach = apply(new_game(diamond_network()), Advance(2))
    assert legal_actions(quiet) == legal_actions(deep) == legal_actions(mid_breach)

def test_honeypot_placement_rules() -> None:
    defender = replace(new_game(diamond_network()), side=Side.DEFENDER)
    actions = legal_actions(defender)
    assert Honeypot(0) not in actions
    assert Honeypot(3) not in actions
    assert Honeypot(1) in actions


def test_honeypot_delays_and_triggers() -> None:
    state = replace(new_game(diamond_network()), honeypots=frozenset({2}))
    after = apply(state, Advance(2))
    assert 2 in after.triggered
    assert after.breach_target == 2
    assert after.breach_left == 3


def test_breaching_the_jewel_wins() -> None:
    state = replace(
        new_game(diamond_network()), attacker_pos=1, breached=frozenset({0, 1})
    )
    after = apply(state, Advance(3))
    assert outcome(after) is Outcome.ATTACKER_WINS
    assert utility(after) == 990
    assert legal_actions(after) == []


def test_running_out_of_turns_means_defender_wins() -> None:
    state = replace(new_game(chain_network()), turn=26)
    assert outcome(state) is Outcome.DEFENDER_WINS
    assert utility(state) == -750


def test_cutting_off_the_jewel_means_defender_wins() -> None:
    defender = replace(new_game(chain_network()), side=Side.DEFENDER)
    after = apply(defender, Firewall(0, 1))
    assert outcome(after) is Outcome.DEFENDER_WINS
    assert legal_actions(after) == []
    assert utility(after) == -980


def test_utility_needs_a_finished_game() -> None:
    with pytest.raises(ValueError):
        utility(new_game(chain_network()))


def test_wrong_side_move_is_rejected() -> None:
    with pytest.raises(IllegalActionError):
        apply(new_game(chain_network()), Patch(1))


def test_apply_never_changes_the_old_state() -> None:
    start = new_game(diamond_network())
    apply(start, Advance(1))
    assert start.attacker_pos == 0
    assert start.breached == frozenset({0})


def test_no_stock_leaves_only_pass() -> None:
    defender = replace(
        new_game(diamond_network()), side=Side.DEFENDER, stock=Stock(0, 0, 0)
    )
    assert legal_actions(defender) == [Pass()]


def test_moving_back_onto_an_owned_machine_costs_one_turn() -> None:
    state = replace(
        new_game(chain_network()), attacker_pos=1, breached=frozenset({0, 1})
    )
    assert legal_actions(state) == [Advance(0), Advance(2)]
    after = apply(state, Advance(0))
    assert after.attacker_pos == 0
    assert after.breach_target is None