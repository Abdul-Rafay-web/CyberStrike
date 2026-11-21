from dataclasses import replace

from config import CONFIG
from core.actions import Advance, Firewall, Pass, Patch
from core.observation import observe_attacker
from core.rules import legal_actions
from core.state import Side, Stock, new_game
from experiments.scenarios import defended_state
from search.evaluation import analyze_routes
from search.model import defender_replies, ordered_attacker_moves, state_from_observation
from tests.factories import diamond_network


def test_the_attacker_model_hides_honeypots_and_counts_visible_defenses() -> None:
    state = defended_state(3)
    rebuilt = state_from_observation(observe_attacker(state))
    assert rebuilt.honeypots == frozenset()
    assert rebuilt.patched == state.patched
    assert rebuilt.firewalls == state.firewalls
    assert rebuilt.stock.patches == 3 - len(state.patched)
    assert rebuilt.stock.firewalls == 2 - len(state.firewalls)
    assert rebuilt.stock.honeypots == 0


def test_attacker_moves_start_with_the_cheapest_route() -> None:
    state = new_game(diamond_network())
    assert ordered_attacker_moves(state, analyze_routes(state)) == [Advance(1), Advance(2)]
    harder = replace(state, patched=frozenset({1}))
    assert ordered_attacker_moves(harder, analyze_routes(harder)) == [Advance(2), Advance(1)]


def test_defender_replies_are_legal_few_and_end_with_pass() -> None:
    for seed in range(40):
        state = replace(defended_state(seed), side=Side.DEFENDER)
        replies = defender_replies(state, analyze_routes(state))
        assert replies[-1] == Pass()
        assert len(replies) <= CONFIG.search.max_defender_replies
        menu = legal_actions(state)
        assert all(reply in menu for reply in replies)


def test_the_most_dangerous_replies_come_first() -> None:
    state = replace(new_game(diamond_network()), side=Side.DEFENDER)
    route = analyze_routes(state)
    replies = defender_replies(state, route)
    assert replies[0] in (Firewall(0, 1), Firewall(1, 3))
    assert Patch(1) in replies


def test_with_no_stock_the_defender_can_only_pass() -> None:
    state = replace(new_game(diamond_network()), side=Side.DEFENDER, stock=Stock(0, 0, 0))
    assert defender_replies(state, analyze_routes(state)) == [Pass()]

