-- Migration 001: Base normalized schema

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS matchups (
    matchup_id INTEGER PRIMARY KEY AUTOINCREMENT,
    attacker_name TEXT NOT NULL,
    defender_name TEXT NOT NULL,
    sensor_name TEXT NOT NULL DEFAULT 'rate_sensor',
    UNIQUE(attacker_name, defender_name, sensor_name)
);

CREATE TABLE IF NOT EXISTS maps (
    map_id INTEGER PRIMARY KEY AUTOINCREMENT,
    seed INTEGER NOT NULL,
    size INTEGER NOT NULL CHECK (size >= 2 AND size <= 100),
    entry INTEGER NOT NULL CHECK (entry >= 0),
    jewel INTEGER NOT NULL CHECK (jewel >= 0 AND jewel != entry),
    edge_count INTEGER NOT NULL CHECK (edge_count >= 1),
    UNIQUE(seed, size)
);

CREATE TABLE IF NOT EXISTS games (
    game_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER REFERENCES experiments(experiment_id) ON DELETE SET NULL,
    matchup_id INTEGER NOT NULL REFERENCES matchups(matchup_id) ON DELETE CASCADE,
    map_id INTEGER NOT NULL REFERENCES maps(map_id) ON DELETE CASCADE,
    seed INTEGER NOT NULL,
    outcome TEXT NOT NULL CHECK (outcome IN ('attacker_wins', 'defender_wins', 'ongoing')),
    turns INTEGER NOT NULL CHECK (turns >= 0),
    plies INTEGER NOT NULL CHECK (plies >= 0),
    utility INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS moves (
    move_id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
    ply INTEGER NOT NULL CHECK (ply >= 0),
    turn INTEGER NOT NULL CHECK (turn >= 1),
    side TEXT NOT NULL CHECK (side IN ('attacker', 'defender')),
    action TEXT NOT NULL,
    duration_ms REAL NOT NULL CHECK (duration_ms >= 0.0)
);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
    turn INTEGER NOT NULL CHECK (turn >= 1),
    machine INTEGER NOT NULL CHECK (machine >= 0),
    fired INTEGER NOT NULL CHECK (fired IN (0, 1))
);

CREATE TABLE IF NOT EXISTS beliefs (
    belief_id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
    turn INTEGER NOT NULL CHECK (turn >= 1),
    machine INTEGER NOT NULL CHECK (machine >= 0),
    probability REAL NOT NULL CHECK (probability >= 0.0 AND probability <= 1.0)
);

