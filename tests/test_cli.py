import pytest

from main import main


def test_tournament_runs_without_a_window(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["tournament", "--games", "3"]) == 0
    assert "3 games" in capsys.readouterr().out