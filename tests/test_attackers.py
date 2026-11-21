from agents.minimax_attacker import MinimaxAttacker
from agents.simple_defenders import PassiveDefender
from core.actions import Advance, Pass
from core.game import play_game
from core.generator import generate_network
from core.observation import observe_attacker
from core.rules import Outcome, apply
from core.state import new_game
from experiments.scenarios import defended_state
from tests.factories import chain_network
from utils.rng import STREAM_SENSORS, make_rng


def test_minimax_attacker_returns_a_legal_move_and_keeps_a_report() -> None:
    agent = MinimaxAttacker(max_depth=3, time_budget=1.0)
    for seed in range(20):
        observation = observe_attacker(defended_state(seed))
        assert agent.choose_action(observation) in observation.legal_actions
    assert agent.last_report is not None


def test_pruning_does_not_change_the_attackers_choice() -> None:
    for seed in range(20):
        observation = observe_attacker(defended_state(seed))
        fast = MinimaxAttacker(max_depth=3, time_budget=10.0, pruning=True)
        slow = MinimaxAttacker(max_depth=3, time_budget=10.0, pruning=False)
        assert fast.choose_action(observation) == slow.choose_action(observation)


def test_a_forced_move_is_returned_without_searching() -> None:
    mid_breach = apply(apply(new_game(chain_network()), Advance(1)), Pass())
    agent = MinimaxAttacker(max_depth=3, time_budget=1.0)
    assert agent.choose_action(observe_attacker(mid_breach)) == Advance(1)
    assert agent.last_report is None


def test_minimax_beats_a_passive_defender_on_every_map() -> None:
    for seed in range(15):
        result = play_game(
            generate_network(seed),
            MinimaxAttacker(max_depth=3, time_budget=1.0),
            PassiveDefender(),
            make_rng(seed, STREAM_SENSORS),
        )
        assert result.outcome is Outcome.ATTACKER_WINS

