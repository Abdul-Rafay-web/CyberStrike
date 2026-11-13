from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from config import CONFIG
from utils.logging_setup import setup_logging

log = logging.getLogger("Cyberstrike")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cyberstrike", description="CyberStrike network infiltration game"
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    parser.add_argument("--json-logs", action="store_true", help="write logs as JSON lines")

    sub = parser.add_subparsers(dest="command", required=True)

    play = sub.add_parser("play", help="open the game window")
    play.add_argument("--seed", type=int, default=CONFIG.seeds.master_seed)

    tournament = sub.add_parser("tournament", help="run headless games")
    tournament.add_argument("--games", type=int, default=100)
    tournament.add_argument("--size", type=int, default=None, help="number of machines")

    replay = sub.add_parser("replay", help="replay a saved game")
    replay.add_argument("path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level, args.json_logs)
    log.info("starting", extra={"data": {"command": args.command}})

    if args.command == "play":
        log.info("play requested", extra={"data": {"seed": args.seed}})
        from ui.app import run_window  # loaded only here, so headless commands skip pygame

        run_window()
    elif args.command == "tournament":
        from experiments.random_games import run_random_games

        summary = run_random_games(args.games, args.size)
        log.info("games finished", extra={"data": {"games": summary.games}})
        print(summary.report())
    elif args.command == "replay":
        log.info("replay is not built yet")
    return 0


if __name__ == "__main__":
    sys.exit(main())