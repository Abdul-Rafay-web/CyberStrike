from __future__ import annotations

from dataclasses import replace

from config import CONFIG, AppConfig
from core.actions import Pass
from core.generator import generate_network
from core.rules import apply, is_terminal, legal_actions
from core.state import GameState, Side, new_game
from utils.rng import STREAM_DEFENDER, make_rng


def defended_state(seed: int, size: int | None = None, config: AppConfig = CONFIG) -> GameState:
    """A new game after a random defender placed its stock, never cutting the jewel off."""
    game = config.game
    network = generate_network(seed, size, config)
    rng = make_rng(seed, STREAM_DEFENDER)
    state = replace(new_game(network, game), side=Side.DEFENDER)
    tries = game.stock.patches + game.stock.firewalls + game.stock.honeypots
    for _ in range(tries):
        options = [a for a in legal_actions(state, game) if not isinstance(a, Pass)]
        if not options:
            break
        action = options[int(rng.integers(0, len(options)))]
        placed = apply(state, action, game)
        if is_terminal(placed, game):
            continue
        state = replace(placed, side=Side.DEFENDER, turn=1)
    return replace(state, side=Side.ATTACKER, turn=1)