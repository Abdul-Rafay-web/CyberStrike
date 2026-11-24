-- Migration 002: Performance indexes on game_id, seed, matchup_id, and duration_ms

CREATE INDEX IF NOT EXISTS idx_games_seed ON games(seed);
CREATE INDEX IF NOT EXISTS idx_games_matchup ON games(matchup_id);
CREATE INDEX IF NOT EXISTS idx_moves_game_id ON moves(game_id);
CREATE INDEX IF NOT EXISTS idx_moves_duration ON moves(duration_ms DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_game_id ON alerts(game_id);
CREATE INDEX IF NOT EXISTS idx_beliefs_game_id ON beliefs(game_id);

