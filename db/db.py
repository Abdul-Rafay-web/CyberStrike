from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


@dataclass(frozen=True, slots=True)
class MatchupStatsRow:
    attacker_name: str
    defender_name: str
    sensor_name: str
    games: int
    attacker_win_rate: float
    mean_turns: float


@dataclass(frozen=True, slots=True)
class SlowMoveRow:
    move_id: int
    game_id: int
    turn: int
    side: str
    action: str
    duration_ms: float


@dataclass(frozen=True, slots=True)
class GameRecordInput:
    matchup_id: int
    map_id: int
    seed: int
    outcome: str
    turns: int
    plies: int
    utility: int
    experiment_id: int | None = None


class CyberStrikeDB:
    """SQLite database wrapper with WAL mode, enforced foreign keys, migrations, and analytics."""

    def __init__(self, db_path: Path | str, *, apply_all_migrations: bool = True) -> None:
        self.db_path = Path(db_path) if db_path != ":memory:" else None
        if self.db_path is not None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON;")
        if self.db_path is not None:
            self.conn.execute("PRAGMA journal_mode = WAL;")
            self.conn.execute("PRAGMA synchronous = NORMAL;")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.commit()
        if apply_all_migrations:
            self.apply_migrations()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> CyberStrikeDB:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def current_version(self) -> int:
        row = self.conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_version").fetchone()
        return int(row[0]) if row is not None else 0

    def apply_migrations(self, max_version: int | None = None) -> list[int]:
        """Apply numbered SQL migrations in `db/migrations/` strictly in ascending order."""
        current = self.current_version()
        applied: list[int] = []
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            ver = int(path.name.split("_", maxsplit=1)[0])
            if ver <= current:
                continue
            if max_version is not None and ver > max_version:
                break
            sql = path.read_text(encoding="utf-8")
            with self.conn:
                self.conn.executescript(sql)
                self.conn.execute("INSERT INTO schema_version(version) VALUES (?)", (ver,))
            applied.append(ver)
        return applied

    def get_or_create_experiment(self, name: str, description: str = "") -> int:
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO experiments(name, description) VALUES (?, ?)",
                (name, description),
            )
            row = self.conn.execute(
                "SELECT experiment_id FROM experiments WHERE name = ?", (name,)
            ).fetchone()
            assert row is not None
            return int(row[0])

    def get_or_create_matchup(
        self, attacker_name: str, defender_name: str, sensor_name: str = "rate_sensor"
    ) -> int:
        with self.conn:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO matchups(attacker_name, defender_name, sensor_name)
                VALUES (?, ?, ?)
                """,
                (attacker_name, defender_name, sensor_name),
            )
            row = self.conn.execute(
                """
                SELECT matchup_id FROM matchups
                WHERE attacker_name = ? AND defender_name = ? AND sensor_name = ?
                """,
                (attacker_name, defender_name, sensor_name),
            ).fetchone()
            assert row is not None
            return int(row[0])

    def get_or_create_map(
        self, seed: int, size: int, entry: int, jewel: int, edge_count: int
    ) -> int:
        with self.conn:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO maps(seed, size, entry, jewel, edge_count)
                VALUES (?, ?, ?, ?, ?)
                """,
                (seed, size, entry, jewel, edge_count),
            )
            row = self.conn.execute(
                "SELECT map_id FROM maps WHERE seed = ? AND size = ?", (seed, size)
            ).fetchone()
            assert row is not None
            return int(row[0])

    def insert_games_batch(self, records: Sequence[GameRecordInput]) -> int:
        """Insert many game rows inside a single ACID transaction."""
        rows = [
            (
                r.experiment_id,
                r.matchup_id,
                r.map_id,
                r.seed,
                r.outcome,
                r.turns,
                r.plies,
                r.utility,
            )
            for r in records
        ]
        with self.conn:
            cur = self.conn.executemany(
                """
                INSERT INTO games(
                    experiment_id, matchup_id, map_id, seed, outcome, turns, plies, utility
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            return int(cur.rowcount)

    def insert_moves_batch(self, moves: Iterable[tuple[int, int, int, str, str, float]]) -> int:
        """Batch-insert `(game_id, ply, turn, side, action, duration_ms)` rows in one tx."""
        with self.conn:
            cur = self.conn.executemany(
                """
                INSERT INTO moves(game_id, ply, turn, side, action, duration_ms)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                moves,
            )
            return int(cur.rowcount)

    def insert_alerts_batch(self, alerts: Iterable[tuple[int, int, int, int]]) -> int:
        with self.conn:
            cur = self.conn.executemany(
                "INSERT INTO alerts(game_id, turn, machine, fired) VALUES (?, ?, ?, ?)",
                alerts,
            )
            return int(cur.rowcount)

    def insert_beliefs_batch(self, beliefs: Iterable[tuple[int, int, int, float]]) -> int:
        with self.conn:
            cur = self.conn.executemany(
                "INSERT INTO beliefs(game_id, turn, machine, probability) VALUES (?, ?, ?, ?)",
                beliefs,
            )
            return int(cur.rowcount)

    def win_rate_by_matchup(self) -> list[MatchupStatsRow]:
        """SQL GROUP BY analytics query: attacker win rate and mean turns per matchup."""
        rows = self.conn.execute(
            """
            SELECT
                m.attacker_name,
                m.defender_name,
                m.sensor_name,
                COUNT(g.game_id) AS games,
                AVG(CASE WHEN g.outcome = 'attacker_wins' THEN 1.0 ELSE 0.0 END) AS win_rate,
                AVG(g.turns) AS mean_turns
            FROM games g
            JOIN matchups m ON g.matchup_id = m.matchup_id
            GROUP BY g.matchup_id
            ORDER BY m.attacker_name, m.defender_name
            """
        ).fetchall()
        return [
            MatchupStatsRow(
                attacker_name=str(r["attacker_name"]),
                defender_name=str(r["defender_name"]),
                sensor_name=str(r["sensor_name"]),
                games=int(r["games"]),
                attacker_win_rate=float(r["win_rate"]),
                mean_turns=float(r["mean_turns"]),
            )
            for r in rows
        ]

    def rolling_win_rate(self, matchup_id: int, window: int = 10) -> list[tuple[int, float]]:
        """SQL window-function rolling average of attacker win rate over the last `window` games."""
        if window < 1:
            raise ValueError("window must be at least 1")
        preceding = window - 1
        rows = self.conn.execute(
            f"""
            SELECT
                game_id,
                AVG(CASE WHEN outcome = 'attacker_wins' THEN 1.0 ELSE 0.0 END) OVER (
                    ORDER BY game_id
                    ROWS BETWEEN {preceding} PRECEDING AND CURRENT ROW
                ) AS rolling_avg
            FROM games
            WHERE matchup_id = ?
            ORDER BY game_id
            """,
            (matchup_id,),
        ).fetchall()
        return [(int(r["game_id"]), float(r["rolling_avg"])) for r in rows]

    def slowest_moves(self, limit: int = 10) -> list[SlowMoveRow]:
        """Return the top `limit` slowest individual moves ordered by duration_ms DESC."""
        rows = self.conn.execute(
            """
            SELECT move_id, game_id, turn, side, action, duration_ms
            FROM moves
            ORDER BY duration_ms DESC, move_id ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [
            SlowMoveRow(
                move_id=int(r["move_id"]),
                game_id=int(r["game_id"]),
                turn=int(r["turn"]),
                side=str(r["side"]),
                action=str(r["action"]),
                duration_ms=float(r["duration_ms"]),
            )
            for r in rows
        ]

    def explain_query_plan(self, sql: str, params: Sequence[Any] = ()) -> list[str]:
        """Run SQLite EXPLAIN QUERY PLAN and return human-readable plan detail lines."""
        rows = self.conn.execute(f"EXPLAIN QUERY PLAN {sql}", params).fetchall()
        return [str(r["detail"]) for r in rows]

    def game_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM games").fetchone()
        return int(row[0]) if row is not None else 0
