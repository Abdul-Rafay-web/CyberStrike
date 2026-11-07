import pytest

from main import build_parser, main


def test_play_command_reads_seed() -> None:
    args = build_parser().parse_args(["play", "--seed", "7"])
    assert args.command == "play"
    assert args.seed == 7


def test_missing_command_exits() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_tournament_runs_without_a_window() -> None:
    assert main(["tournament", "--games", "3"]) == 0