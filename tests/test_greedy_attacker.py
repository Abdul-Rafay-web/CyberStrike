from agents.greedy_attacker import GreedyAttacker
from agents.random_agents import RandomAttacker
from agents.simple_defenders import PassiveDefender
from core.game import play_game
from core.generator import generate_network
from core.observation import observe_attacker
from core.rules import Outcome
from core.state import new_game
from experiments.scenarios import defended_state
from search.astar import astar
from search.problem import problem_from_state
from utils.rng import STREAM_AGENTS, STREAM_SENSORS, make_rng


def test_greedy_attacker_takes_the_first_step_of_the_astar_route() -> None:
    state = new_game(generate_network(2))
    route = astar(problem_from_state(state)).path
    action = GreedyAttacker().choose_action(observe_attacker(state))
    assert action.target == route[1]  # type: ignore[union-attr]


def test_against_a_passive_defender_turns_equal_the_astar_cost() -> None:
    for seed in range(60):
        network = generate_network(seed)
        expected = astar(problem_from_state(new_game(network))).cost
        result = play_game(
            network, GreedyAttacker(), PassiveDefender(), make_rng(seed, STREAM_SENSORS)
        )
        assert result.outcome is Outcome.ATTACKER_WINS
        assert result.turns == expected


def test_greedy_beats_a_random_attacker_by_a_wide_margin() -> None:
    greedy_turns = 0
    random_turns = 0
    for seed in range(60):
        network = generate_network(seed)
        sensors = make_rng(seed, STREAM_SENSORS)
        greedy_turns += play_game(network, GreedyAttacker(), PassiveDefender(), sensors).turns
        random_turns += play_game(
            network, RandomAttacker(make_rng(seed, STREAM_AGENTS)), PassiveDefender(), sensors
        ).turns
    assert greedy_turns * 2 < random_turns


def test_greedy_attacker_always_returns_a_legal_move() -> None:
    for seed in range(60):
        observation = observe_attacker(defended_state(seed))
        assert GreedyAttacker().choose_action(observation) in observation.legal_actions