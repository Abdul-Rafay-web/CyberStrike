from agents.random_agents import RandomAttacker, RandomDefender
from config import CONFIG
from core.actions import Action
from core.game import GameResult, play_game
from core.generator import generate_network
from core.rules import Outcome
from core.state import GameState
from utils.rng import STREAM_AGENTS, STREAM_DEFENDER, STREAM_SENSORS, make_rng

Step = tuple[GameState, Action, GameState]


def _play(seed: int, steps: list[Step] | None = None) -> GameResult:
    def record(before: GameState, action: Action, after: GameState) -> None:
        if steps is not None:
            steps.append((before, action, after))

    return play_game(
        generate_network(seed),
        RandomAttacker(make_rng(seed, STREAM_AGENTS)),
        RandomDefender(make_rng(seed, STREAM_DEFENDER)),
        make_rng(seed, STREAM_SENSORS),
        on_step=record,
    )


def test_a_game_finishes_with_a_consistent_result() -> None:
    result = _play(3)
    assert result.outcome in (Outcome.ATTACKER_WINS, Outcome.DEFENDER_WINS)
    assert 1 <= result.turns <= CONFIG.game.turn_limit
    assert 1 <= result.plies <= 2 * CONFIG.game.turn_limit + 1
    assert (result.utility > 0) == (result.outcome is Outcome.ATTACKER_WINS)


def test_the_step_callback_sees_every_move() -> None:
    steps: list[Step] = []
    result = _play(3, steps)
    assert len(steps) == result.plies
    for before, _, after in steps:
        assert before.network is after.network


def test_the_same_seed_gives_the_same_game() -> None:
    assert _play(11) == _play(11)


def test_five_hundred_random_games_never_crash() -> None:
    for seed in range(500):
        assert _play(seed).outcome is not Outcome.ONGOING