from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from config import CONFIG, GameConfig
from core.network import Edge, Network


class Side(Enum):
    ATTACKER="attacker"
    DEFENDER="defender"

    @property
    def other(self)->Side:
        return Side.DEFENDER if self is Side.ATTACKER else Side.ATTACKER
    
@dataclass(frozen=True,slots=True)
class Stock:
    patches:int
    firewalls:int
    honeypots:int 

@dataclass(frozen=True, slots=True)
class GameState:
    """One frozen snapshot of the whole game."""

    network: Network
    attacker_pos: int
    breach_target: int | None
    breach_left: int
    breached: frozenset[int]
    patched: frozenset[int]
    firewalls: frozenset[Edge]
    honeypots: frozenset[int]
    triggered: frozenset[int]
    stock: Stock
    turn: int
    side: Side

def new_game(network: Network, config: GameConfig = CONFIG.game) -> GameState:
    return GameState(
        network=network,
        attacker_pos=network.entry,
        breach_target=None,
        breach_left=0,
        breached=frozenset({network.entry}),
        patched=frozenset(),
        firewalls=frozenset(),
        honeypots=frozenset(),
        triggered=frozenset(),
        stock=Stock(config.stock.patches, config.stock.firewalls, config.stock.honeypots),
        turn=1,
        side=Side.ATTACKER,
    )