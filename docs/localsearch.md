# Phase 9: Local Search Weight Tuning & Post-Game Coach

## 1. Hill Climbing with Random Restarts & Simulated Annealing (`evo/localsearch.py`)
- **Search space:** The four Minimax evaluation weights `EvalWeights(patch, firewall, exits, honeypot)`.
- **Neighbourhood:** Each move nudges one weight up or down by its configured step size `(5.0, 10.0, 0.5, 4.0)` while clamping weights to valid non-negative bounds.
- **Escaping local maxima:**
  - `hill_climb` runs steepest-ascent local search with multiple random restarts (first restart begins from `DEFAULT_WEIGHTS`, subsequent restarts sample fresh starting vectors).
  - `simulated_annealing` accepts worse neighbours with probability $\exp(\Delta / T)$ as temperature $T$ cools geometrically.
- **Strict train/test separation:** `SeedSplit` enforces disjoint seed sets at construction — training uses development seeds (`0..`) and final evaluation touches unseen test seeds (`1000..`) once at the end.

## 2. Post-Game Defender Coach (`agents/coach.py`)
After a game finishes, `analyze_defender_game(history)` replays every Defender turn, tests nearby legal alternative placements along the attacker's active shortest route, simulates the resulting outcome, and reports the top missed moves ranked by turns gained. The one-line headline integrates cleanly with the bottom diagnostics tray (`ui/panels.py`).

