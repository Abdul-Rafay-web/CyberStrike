from __future__ import annotations

from core.actions import Action, Firewall, Honeypot, Pass, Patch
from core.network import Edge
from core.observation import DefenderObservation
from search.heuristics import UNREACHABLE
from search.problem import build_problem
from search.uninformed import uniform_cost


class PassiveDefender:
    """Does nothing every turn. A useful yardstick for how fast an attacker can be."""

    name = "passive defender"

    def choose_action(self, observation: DefenderObservation) -> Action:
        return Pass()


class StaticDefender:
    """Follows a fixed shopping list worked out from the map alone. It never reacts."""

    name = "static defender"

    def choose_action(self, observation: DefenderObservation) -> Action:
        wanted = self._shopping_list(observation)
        menu = set(observation.legal_actions)
        for action in wanted:
            if action in menu:
                return action
        return Pass()

    @staticmethod
    def _shopping_list(observation: DefenderObservation) -> list[Action]:
        net = observation.network
        busy = sorted(
            (m for m in range(net.size) if m != net.entry),
            key=lambda m: (-len(net.neighbours(m)), m),
        )
        middle = [m for m in busy if m != net.jewel]
        links = sorted(
            (e for e in net.edges() if net.entry not in e and net.jewel not in e),
            key=lambda e: (-(len(net.neighbours(e[0])) + len(net.neighbours(e[1]))), e),
        )
        plan: list[Action] = [Patch(m) for m in busy]
        plan += [Honeypot(m) for m in middle]
        plan += [Firewall(a, b) for a, b in links]
        return plan


class ReactiveDefender:
    """Reads the alerts to find the intruder, then picks the move that delays it most.

    With perfect sensors this is a strong yardstick. With noisy sensors it often guesses wrong.
    """

    name = "reactive defender"

    def choose_action(self, observation: DefenderObservation) -> Action:
        position = self._guess_position(observation)
        if position is None:
            return Pass()
        base = self._route_cost(observation, position, observation.patched, observation.firewalls)
        best: Action = Pass()
        best_gain = 0
        for action in observation.legal_actions:
            patched = observation.patched
            firewalls = observation.firewalls
            if isinstance(action, Patch):
                patched = patched | {action.machine}
            elif isinstance(action, Firewall):
                firewalls = firewalls | {(action.a, action.b)}
            else:
                continue
            after = self._route_cost(observation, position, patched, firewalls)
            gain = after - base
            if gain > best_gain:
                best, best_gain = action, gain
        return best

    @staticmethod
    def _guess_position(observation: DefenderObservation) -> int | None:
        if observation.intruder_known_at is not None:
            return observation.intruder_known_at
        flagged = [m for m, alert in enumerate(observation.alerts) if alert]
        return flagged[0] if len(flagged) == 1 else None

    @staticmethod
    def _route_cost(
        observation: DefenderObservation,
        position: int,
        patched: frozenset[int],
        firewalls: frozenset[Edge],
    ) -> int:
        problem = build_problem(
            observation.network, position, None, 0, frozenset({position}), patched, firewalls
        )
        result = uniform_cost(problem)
        return UNREACHABLE if result.cost is None else result.cost
