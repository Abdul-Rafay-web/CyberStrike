from experiments.random_games import run_random_games


def test_summary_adds_up() -> None:
    summary = run_random_games(40)
    assert summary.games == 40
    assert summary.attacker_wins + summary.defender_wins == 40
    assert 1.0 <= summary.mean_turns <= 25.0
    assert "40 games" in summary.report()


def test_empty_run_is_safe() -> None:
    summary = run_random_games(0)
    assert summary.games == 0
    assert summary.report().startswith("0 games")