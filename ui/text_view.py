from __future__ import annotations

from collections.abc import Callable
from typing import assert_never

from core.actions import Action, Advance, Firewall, Honeypot, Pass, Patch
from core.state import GameState, Side


def describe_action(action: Action) -> str:
    match action:
        case Advance(target=target):
            return f"advances on machine {target}"
        case Patch(machine=machine):
            return f"patches machine {machine}"
        case Firewall(a=a, b=b):
            return f"firewalls the link {a} to {b}"
        case Honeypot(machine=machine):
            return f"sets a honeypot on machine {machine}"
        case Pass():
            return "passes"
        case _:
            assert_never(action)


def render_state(state: GameState) -> str:
    """A short multi line summary of the whole game. It shows hidden facts too."""
    lines = [
        f"turn {state.turn}, {state.side.value} to move",
        f"attacker on machine {state.attacker_pos}, owns {sorted(state.breached)}",
    ]
    if state.breach_target is not None:
        lines.append(f"breaching machine {state.breach_target}, {state.breach_left} more turns")
    lines.append(f"patched {sorted(state.patched)}, firewalls {sorted(state.firewalls)}")
    lines.append(f"honeypots {sorted(state.honeypots)}, triggered {sorted(state.triggered)}")
    s = state.stock
    lines.append(f"stock: {s.patches} patches, {s.firewalls} firewalls, {s.honeypots} honeypots")
    return "\n".join(lines)


class StepPrinter:
    """Prints one line per move. A spectator view, so it shows hidden facts too."""

    def __init__(self, write: Callable[[str], None] = print) -> None:
        self._write = write

    def __call__(self, before: GameState, action: Action, after: GameState) -> None:
        who = before.side.value.capitalize()
        line = f"turn {before.turn:>2} | {who:<8} | {describe_action(action)}"
        notes: list[str] = []
        if before.side is Side.ATTACKER:
            if after.breach_target is not None:
                notes.append(
                    f"breach on {after.breach_target}, {after.breach_left} more turns"
                )
            elif after.attacker_pos != before.attacker_pos:
                notes.append(f"now owns machine {after.attacker_pos}")
        if len(after.triggered) > len(before.triggered):
            notes.append("HONEYPOT TRIGGERED")
        if notes:
            line += " | " + "; ".join(notes)
        self._write(line)