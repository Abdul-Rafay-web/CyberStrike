from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from agents.greedy_attacker import GreedyAttacker
from agents.simple_defenders import StaticDefender
from config import CONFIG, AppConfig
from core.actions import Action, Firewall, Honeypot, Pass, Patch
from core.network import make_edge
from core.observation import observe_attacker, observe_defender
from core.rules import Outcome, apply, is_terminal, legal_actions, outcome
from core.sensors import sample_alerts
from core.state import GameState, Side
from search.problem import problem_from_state
from search.uninformed import uniform_cost
from utils.rng import STREAM_SENSORS, make_rng


@dataclass(frozen=True, slots=True)
class CoachAdvice:
    """One concrete better Defender move found by local search replay."""

    turn: int
    played: Action
    suggested: Action
    turns_gained: int
    summary: str


@dataclass(frozen=True, slots=True)
class CoachReport:
    """Post-game Defender coaching report sorted by largest turns gained."""

    decisions_reviewed: int
    advice: tuple[CoachAdvice, ...]
    headline: str


def _simulate_remaining_turns(
    start_state: GameState,
    config: AppConfig = CONFIG,
    seed: int = 42,
) -> tuple[int, Outcome]:
    """Roll out a game from `start_state` with GreedyAttacker vs StaticDefender."""
    game = config.game
    state = start_state
    attacker = GreedyAttacker()
    defender = StaticDefender()
    rng = make_rng(seed, STREAM_SENSORS)
    plies = 0
    max_plies = 2 * game.turn_limit + 2

    while not is_terminal(state, game) and plies < max_plies:
        if state.side is Side.ATTACKER:
            action = attacker.choose_action(observe_attacker(state, game))
        else:
            alerts = sample_alerts(state, rng, config.sensors)
            action = defender.choose_action(observe_defender(state, alerts, game))
        state = apply(state, action, game)
        plies += 1

    res = outcome(state, game)
    turns_survived = (
        game.turn_limit + 5 if res is Outcome.DEFENDER_WINS else min(state.turn, game.turn_limit)
    )
    return turns_survived, res


def _nearby_defender_alternatives(
    state: GameState, played: Action, config: AppConfig = CONFIG
) -> list[Action]:
    """Select candidate legal alternatives on or near the attacker's shortest route."""
    legal = legal_actions(state, config.game)
    route_res = uniform_cost(problem_from_state(state, config.game))
    path = route_res.path if route_res.path else (state.attacker_pos, state.network.jewel)
    path_nodes = set(path[1:])
    path_edges = {make_edge(a, b) for a, b in zip(path, path[1:], strict=False)}

    candidates: list[Action] = []
    for act in legal:
        if act == played:
            continue
        is_relevant = (
            (
                isinstance(act, Firewall)
                and ((act.a, act.b) in path_edges or state.network.jewel in (act.a, act.b))
            )
            or (
                isinstance(act, Patch)
                and (act.machine in path_nodes or act.machine == state.network.jewel)
            )
            or (isinstance(act, Honeypot) and act.machine in path_nodes)
            or isinstance(act, Pass)
        )
        if is_relevant:
            candidates.append(act)
    return candidates[:14]


def _format_action(action: Action) -> str:
    if isinstance(action, Patch):
        return f"Patch(M{action.machine})"
    if isinstance(action, Firewall):
        return f"Firewall(M{action.a}-M{action.b})"
    if isinstance(action, Honeypot):
        return f"Honeypot(M{action.machine})"
    return "Pass()"


def analyze_defender_game(
    history: Sequence[tuple[GameState, Action]],
    *,
    config: AppConfig = CONFIG,
    max_tips: int = 3,
) -> CoachReport:
    """Replay every Defender move in `history` and search nearby alternatives for extra turns."""
    findings: list[CoachAdvice] = []
    reviewed = 0

    for before, played in history:
        if before.side is not Side.DEFENDER or is_terminal(before, config.game):
            continue
        reviewed += 1
        actual_after = apply(before, played, config.game)
        actual_turns, _ = _simulate_remaining_turns(actual_after, config, seed=before.turn)

        best_alt: Action | None = None
        best_gain = 0
        for alt in _nearby_defender_alternatives(before, played, config):
            alt_after = apply(before, alt, config.game)
            alt_turns, _ = _simulate_remaining_turns(alt_after, config, seed=before.turn)
            gain = alt_turns - actual_turns
            if gain > best_gain:
                best_gain = gain
                best_alt = alt

        if best_alt is not None and best_gain > 0:
            summary = (
                f"Turn {before.turn}: {_format_action(best_alt)} instead of "
                f"{_format_action(played)} gains +{best_gain} turn(s)."
            )
            findings.append(
                CoachAdvice(
                    turn=before.turn,
                    played=played,
                    suggested=best_alt,
                    turns_gained=best_gain,
                    summary=summary,
                )
            )

    findings.sort(key=lambda item: (-item.turns_gained, item.turn))
    top_advice = tuple(findings[:max_tips])
    if top_advice:
        headline = f"COACH: {top_advice[0].summary}"
    elif reviewed == 0:
        headline = "COACH: No Defender decisions found in game history."
    else:
        headline = "COACH: Solid defense — no single-move alternative gained extra turns."

    return CoachReport(
        decisions_reviewed=reviewed,
        advice=top_advice,
        headline=headline,
    )
