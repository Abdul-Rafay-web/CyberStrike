import sqlite3
from pathlib import Path

import pytest

from db.db import CyberStrikeDB, GameRecordInput


def test_migrations_are_idempotent_and_enforce_foreign_keys_and_checks(tmp_path: Path) -> None:
    db_file = tmp_path / "test.db"
    with CyberStrikeDB(db_file) as db:
        assert db.current_version() == 2
        assert db.apply_migrations() == []

        # Foreign key violation: inserting a move for non-existent game_id=999 must fail
        with pytest.raises(sqlite3.IntegrityError):
            db.insert_moves_batch([(999, 0, 1, "attacker", "Advance(1)", 1.0)])

        # CHECK constraint violation: probability outside [0, 1] must fail
        m_id = db.get_or_create_matchup("a", "d")
        map_id = db.get_or_create_map(1, 10, 0, 9, 15)
        db.insert_games_batch([GameRecordInput(m_id, map_id, 1, "attacker_wins", 7, 13, 930)])
        with pytest.raises(sqlite3.IntegrityError):
            db.insert_beliefs_batch([(1, 1, 0, 1.5)])


def test_batch_insert_10000_games_and_sql_analytics_match_python(tmp_path: Path) -> None:
    db_file = tmp_path / "scale.db"
    with CyberStrikeDB(db_file) as db:
        m1 = db.get_or_create_matchup("greedy", "static", "nb")
        m2 = db.get_or_create_matchup("minimax", "csp", "nb")
        map_id = db.get_or_create_map(42, 15, 0, 14, 22)

        records: list[GameRecordInput] = []
        py_m1_wins = 0
        py_m2_wins = 0
        for i in range(10_000):
            if i % 2 == 0:
                won = i % 4 == 0
                py_m1_wins += int(won)
                records.append(
                    GameRecordInput(
                        m1,
                        map_id,
                        i,
                        "attacker_wins" if won else "defender_wins",
                        10,
                        20,
                        900 if won else -750,
                    )
                )
            else:
                won = i % 5 == 0
                py_m2_wins += int(won)
                records.append(
                    GameRecordInput(
                        m2,
                        map_id,
                        i,
                        "attacker_wins" if won else "defender_wins",
                        14,
                        28,
                        860 if won else -750,
                    )
                )

        db.insert_games_batch(records)
        assert db.game_count() == 10_000

        stats = {r.attacker_name: r for r in db.win_rate_by_matchup()}
        assert stats["greedy"].games == 5000
        assert stats["greedy"].attacker_win_rate == pytest.approx(py_m1_wins / 5000)
        assert stats["minimax"].attacker_win_rate == pytest.approx(py_m2_wins / 5000)

        rolling = db.rolling_win_rate(m1, window=10)
        assert len(rolling) == 5000
        assert all(0.0 <= val <= 1.0 for _, val in rolling)

        db.insert_moves_batch(
            [
                (1, ply, ply + 1, "attacker", f"Advance({ply})", float(ply * 3.5))
                for ply in range(15)
            ]
        )
        slowest = db.slowest_moves(limit=10)
        assert len(slowest) == 10
        assert slowest[0].duration_ms == pytest.approx(14 * 3.5)


def test_explain_query_plan_shows_table_scan_before_and_index_search_after(tmp_path: Path) -> None:
    db_file = tmp_path / "plan.db"
    query = "SELECT ply, action FROM moves WHERE game_id = ?"
    with CyberStrikeDB(db_file, apply_all_migrations=False) as db:
        db.apply_migrations(max_version=1)
        m_id = db.get_or_create_matchup("a", "d")
        map_id = db.get_or_create_map(1, 10, 0, 9, 15)
        db.insert_games_batch([GameRecordInput(m_id, map_id, 1, "attacker_wins", 7, 13, 930)])
        db.insert_moves_batch([(1, i, 1, "attacker", "Advance(1)", 1.0) for i in range(50)])

        before = " ".join(db.explain_query_plan(query, (1,)))
        assert "SCAN moves" in before

        db.apply_migrations()
        after = " ".join(db.explain_query_plan(query, (1,)))
        assert "SEARCH moves USING INDEX idx_moves_game_id" in after
