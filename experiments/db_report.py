from __future__ import annotations

import time
from pathlib import Path

from agents.greedy_attacker import GreedyAttacker
from agents.simple_defenders import StaticDefender
from config import CONFIG
from core.actions import Action
from core.eventstore import EventStore, build_events_from_steps, replay_events
from core.game import play_game
from core.generator import generate_network
from core.rules import Outcome
from core.state import GameState
from db.db import CyberStrikeDB, GameRecordInput
from utils.paths import RESULTS_DIR
from utils.rng import STREAM_SENSORS, make_rng


def verify_event_replay(jsonl_path: Path) -> bool:
    """Record a full game to JSONL, reload it, and verify identical final `GameState`."""
    if jsonl_path.exists():
        jsonl_path.unlink()
    net = generate_network(42)
    steps: list[tuple[GameState, Action, GameState]] = []

    def on_step(before: GameState, action: Action, after: GameState) -> None:
        steps.append((before, action, after))

    play_game(
        net, GreedyAttacker(), StaticDefender(), make_rng(42, STREAM_SENSORS), CONFIG, on_step
    )
    events = build_events_from_steps(42, net, steps, CONFIG.game)
    store = EventStore(jsonl_path)
    store.append_many(events)

    reloaded = store.read_all()
    replayed_state = replay_events(reloaded, CONFIG.game)
    return replayed_state == steps[-1][2]


def run_index_plan_benchmark(db_path: Path) -> str:
    """Compare EXPLAIN QUERY PLAN and timing on 100,000 move rows before vs after indexes."""
    if db_path.exists():
        db_path.unlink()
    lines: list[str] = []
    query = "SELECT ply, action, duration_ms FROM moves WHERE game_id = ?"

    with CyberStrikeDB(db_path, apply_all_migrations=False) as db:
        db.apply_migrations(max_version=1)
        exp_id = db.get_or_create_experiment("phase10_benchmark", "10k games + 100k moves")
        m_greedy = db.get_or_create_matchup("greedy_astar", "static_defender", "nb_sensor")
        m_minimax = db.get_or_create_matchup("minimax_ab", "belief_csp_defender", "nb_sensor")
        map_id = db.get_or_create_map(seed=42, size=15, entry=0, jewel=14, edge_count=22)

        # Insert 10,000 games in one batch transaction (enforcing foreign keys)
        t0 = time.perf_counter()
        game_rows = [
            GameRecordInput(
                experiment_id=exp_id,
                matchup_id=m_greedy if i % 2 == 0 else m_minimax,
                map_id=map_id,
                seed=i,
                outcome=Outcome.ATTACKER_WINS.value
                if (i % 5 != 0)
                else Outcome.DEFENDER_WINS.value,
                turns=8 + (i % 12),
                plies=16 + (i % 20),
                utility=920 - 10 * (i % 12) if (i % 5 != 0) else -750,
            )
            for i in range(10_000)
        ]
        db.insert_games_batch(game_rows)
        t_games = time.perf_counter() - t0
        lines.append(f"Inserted {db.game_count()} games in 1 transaction ({t_games:.3f}s)")

        # Insert 100,000 move rows (10 moves per game)
        moves_batch = [
            (
                (i // 10) + 1,
                i % 10,
                (i % 10) // 2 + 1,
                "attacker" if i % 2 == 0 else "defender",
                f"Advance({i % 15})" if i % 2 == 0 else f"Patch({i % 15})",
                round((i % 97) * 0.45, 2),
            )
            for i in range(100_000)
        ]
        db.insert_moves_batch(moves_batch)

        plan_before = db.explain_query_plan(query, (5000,))
        t1 = time.perf_counter()
        for gid in range(4900, 5000):
            db.conn.execute(query, (gid,)).fetchall()
        ms_before = (time.perf_counter() - t1) * 1000.0

        # Apply migration 002 (indexes)
        db.apply_migrations()
        plan_after = db.explain_query_plan(query, (5000,))
        t2 = time.perf_counter()
        for gid in range(4900, 5000):
            db.conn.execute(query, (gid,)).fetchall()
        ms_after = (time.perf_counter() - t2) * 1000.0

        stats = db.win_rate_by_matchup()
        rolling = db.rolling_win_rate(m_greedy, window=10)
        slowest = db.slowest_moves(limit=5)

        lines.append(
            f"BEFORE index (100,000 moves): {plan_before} | 100 queries = {ms_before:.2f} ms"
        )
        lines.append(
            f"AFTER  index (100,000 moves): {plan_after} | 100 queries = {ms_after:.2f} ms"
        )
        lines.append("\nGROUP BY win rates:")
        for row in stats:
            lines.append(
                f"  {row.attacker_name} vs {row.defender_name}: "
                f"games={row.games}, win_rate={row.attacker_win_rate:.1%}, "
                f"mean_turns={row.mean_turns:.2f}"
            )
        lines.append(f"Rolling window sample (last 3): {rolling[-3:]}")
        lines.append(f"Slowest move top-1: {slowest[0]}")

    return "\n".join(lines)


def main() -> None:
    replay_ok = verify_event_replay(RESULTS_DIR / "sample_game.jsonl")
    print(f"EventStore replay exact state match: {replay_ok}")
    report = run_index_plan_benchmark(RESULTS_DIR / "cyberstrike_phase10.db")
    print(report)
    out_path = RESULTS_DIR / "explain_query_plan.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report + "\n", encoding="utf-8")
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
