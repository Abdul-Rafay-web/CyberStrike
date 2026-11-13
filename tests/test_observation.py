from dataclasses import fields, replace

from core.actions import Advance
from core.observation import DefenderObservation, observe_attacker, observe_defender
from core.rules import apply
from core.state import Side, new_game
from tests.factories import chain_network, diamond_network

NO_ALERTS = (False, False, False, False)


def test_defender_observation_has_no_attacker_fields() -> None:
    names = {f.name for f in fields(DefenderObservation)}
    hidden = {"attacker_pos", "position", "breached", "breach_target", "breach_left"}
    assert names.isdisjoint(hidden)


def test_attacker_cannot_see_untriggered_honeypots() -> None:
    state = replace(new_game(diamond_network()), honeypots=frozenset({2}))
    seen = observe_attacker(state)
    assert seen.known_honeypots == frozenset()
    assert not hasattr(seen, "honeypots")


def test_attacker_learns_a_honeypot_once_it_triggers() -> None:
    state = replace(new_game(diamond_network()), honeypots=frozenset({2}))
    after = apply(state, Advance(2))
    assert observe_attacker(after).known_honeypots == frozenset({2})


def test_defender_sees_the_intruder_only_through_a_triggered_honeypot() -> None:
    state = replace(new_game(diamond_network()), side=Side.DEFENDER)
    assert observe_defender(state, NO_ALERTS).intruder_known_at is None

    trapped = replace(new_game(diamond_network()), honeypots=frozenset({2}))
    after = apply(trapped, Advance(2))
    assert observe_defender(after, NO_ALERTS).intruder_known_at == 2


def test_observation_carries_the_legal_menu() -> None:
    state = new_game(chain_network())
    assert observe_attacker(state).legal_actions == (Advance(1),)