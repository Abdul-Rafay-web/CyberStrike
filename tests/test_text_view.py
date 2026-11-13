from dataclasses import replace

from core.actions import Advance, Firewall, Honeypot, Pass, Patch
from core.rules import apply
from core.state import new_game
from tests.factories import chain_network, diamond_network
from ui.text_view import StepPrinter, describe_action, render_state


def test_every_action_has_a_sentence() -> None:
    assert describe_action(Advance(3)) == "advances on machine 3"
    assert describe_action(Patch(2)) == "patches machine 2"
    assert describe_action(Firewall(1, 4)) == "firewalls the link 1 to 4"
    assert describe_action(Honeypot(5)) == "sets a honeypot on machine 5"
    assert describe_action(Pass()) == "passes"


def test_step_printer_writes_one_line_per_move() -> None:
    lines: list[str] = []
    printer = StepPrinter(lines.append)
    before = new_game(chain_network())
    action = Advance(1)
    printer(before, action, apply(before, action))
    assert lines == ["turn  1 | Attacker | advances on machine 1 | breach on 1, 1 more turns"]


def test_step_printer_flags_a_triggered_honeypot() -> None:
    lines: list[str] = []
    before = replace(new_game(diamond_network()), honeypots=frozenset({2}))
    after = apply(before, Advance(2))
    StepPrinter(lines.append)(before, Advance(2), after)
    assert "HONEYPOT TRIGGERED" in lines[0]


def test_render_state_mentions_the_key_facts() -> None:
    text = render_state(new_game(chain_network()))
    assert "turn 1" in text
    assert "attacker on machine 0" in text
    assert "3 patches" in text