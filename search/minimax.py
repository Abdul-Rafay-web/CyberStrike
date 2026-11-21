from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from config import CONFIG, GameConfig, SearchConfig
from core.actions import Action
from core.rules import Outcome, apply_unchecked, final_score, outcome
from core.state import GameState, Side
from search.evaluation import EvalWeights, analyze_routes, evaluate
from search.model import defender_replies, ordered_attacker_moves

CHECK_EVERY = 256


class SearchTimeout(Exception):
    """Raised inside a search when the time budget runs out."""


@dataclass(frozen=True, slots=True)
class SearchReport:
    action: Action
    value: float
    depth: int
    nodes: int
    evaluations: int
    cutoffs: int
    seconds: float


@dataclass
class SearchContext:
    game: GameConfig = CONFIG.game
    search: SearchConfig = CONFIG.search
    weights: EvalWeights = field(default_factory=EvalWeights)
    pruning: bool = True
    margin: float = 0.0
    deadline: float | None = None
    nodes: int = 0
    evaluations: int = 0
    cutoffs: int = 0
    trace: list[str] | None = None

    def tick(self) -> None:
        self.nodes += 1
        if (
            self.deadline is not None
            and self.nodes % CHECK_EVERY == 0
            and time.perf_counter() > self.deadline
        ):
            raise SearchTimeout


def _value(
    state: GameState, depth: int, alpha: float, beta: float, ctx: SearchContext, root: int
) -> float:
    ctx.tick()
    result = outcome(state, ctx.game)
    if result is not Outcome.ONGOING:
        return float(final_score(state, result, ctx.game))
    route = analyze_routes(state, ctx.game)
    if depth == 0:
        ctx.evaluations += 1
        return evaluate(state, route, ctx.game, ctx.weights)

    maximizing = state.side is Side.ATTACKER
    if maximizing:
        moves = ordered_attacker_moves(state, route)
    else:
        moves = defender_replies(state, route, ctx.game, ctx.search)
    best = -math.inf if maximizing else math.inf
    for index, action in enumerate(moves):
        child = apply_unchecked(state, action, ctx.game)
        value = _value(child, depth - 1, alpha, beta, ctx, root)
        if ctx.trace is not None:
            who = "attacker" if maximizing else "defender"
            ctx.trace.append(f"{'  ' * (root - depth)}{who} {action} -> {value:.0f}")
        if maximizing:
            best = max(best, value)
            alpha = max(alpha, best)
        else:
            best = min(best, value)
            beta = min(beta, best)
        if ctx.pruning and alpha >= beta:
            ctx.cutoffs += 1
            if ctx.trace is not None and index < len(moves) - 1:
                ctx.trace.append(f"{'  ' * (root - depth)}  pruned {len(moves) - index - 1} more")
            break
    return best


def search_root(
    state: GameState, depth: int, ctx: SearchContext, first: Action | None = None
) -> tuple[Action, float]:
    """Pick the attacker's move for a fixed depth.

    Normally that is the move with the best value. With a margin above zero, the attacker
    prefers the cheapest route whenever its value is within `margin` points of the best.
    """
    route = analyze_routes(state, ctx.game)
    canonical = ordered_attacker_moves(state, route)
    order = list(canonical)
    if first is not None and first in order:
        order.remove(first)
        order.insert(0, first)

    values: dict[Action, float] = {}
    best_move = order[0]
    best = -math.inf
    for action in order:
        child = apply_unchecked(state, action, ctx.game)
        alpha = best - ctx.margin if ctx.pruning else -math.inf
        value = _value(child, depth - 1, alpha, math.inf, ctx, depth)
        values[action] = value
        if ctx.trace is not None:
            ctx.trace.append(f"attacker {action} -> {value:.0f}")
        if value > best:
            best, best_move = value, action

    chosen = best_move
    if ctx.margin > 0:
        for action in canonical:
            if action == best_move:
                break
            if values[action] > best - ctx.margin:
                chosen = action
                break
    return chosen, best


def search_to_depth(
    state: GameState,
    depth: int,
    pruning: bool = True,
    game: GameConfig = CONFIG.game,
    search: SearchConfig = CONFIG.search,
    weights: EvalWeights | None = None,
    trace: list[str] | None = None,
    margin: float = 0.0,
) -> SearchReport:
    """Search a fixed number of plies with no time limit. Used for tests and comparisons."""
    started = time.perf_counter()
    ctx = SearchContext(
        game=game,
        search=search,
        weights=weights or EvalWeights(),
        pruning=pruning,
        margin=margin,
        trace=trace,
    )
    action, value = search_root(state, depth, ctx)
    return SearchReport(
        action, value, depth, ctx.nodes, ctx.evaluations, ctx.cutoffs, time.perf_counter() - started
    )


def iterative_search(
    state: GameState,
    max_depth: int,
    time_budget: float,
    pruning: bool = True,
    game: GameConfig = CONFIG.game,
    search: SearchConfig = CONFIG.search,
    weights: EvalWeights | None = None,
) -> SearchReport:
    """Search depth 1, then 2, then 3 and so on until time or depth runs out.

    If every move loses against perfect play, race along the cheapest route anyway:
    a real defender is not perfect, and stalling only wastes turns.
    """
    started = time.perf_counter()
    ctx = SearchContext(
        game=game,
        search=search,
        weights=weights or EvalWeights(),
        pruning=pruning,
        margin=search.commit_margin,
        deadline=started + time_budget,
    )
    route = analyze_routes(state, game)
    fallback = ordered_attacker_moves(state, route)[0]
    best_action, best_value, reached = fallback, 0.0, 0
    for depth in range(1, max_depth + 1):
        try:
            action, value = search_root(state, depth, ctx, first=best_action)
        except SearchTimeout:
            break
        best_action, best_value, reached = action, value, depth
    if best_value < 0:
        best_action = fallback
    return SearchReport(
        best_action,
        best_value,
        reached,
        ctx.nodes,
        ctx.evaluations,
        ctx.cutoffs,
        time.perf_counter() - started,
    )

