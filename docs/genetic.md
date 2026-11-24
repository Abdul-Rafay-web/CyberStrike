# Phase 8: Genetic Algorithm Map Evolution & Linear Regression Surrogate

## 1. Chromosome & Repair Operator
- **Chromosome (`evo/genetic.py`):** encodes a `size`-machine map via `(hardness, edge_mask, entry, jewel)`, where `edge_mask` is a boolean tuple over all $\binom{N}{2}$ canonical unordered node pairs.
- **Repair (`evo/repair.py`):** every child produced by single-point crossover or bit/hardness mutation passes through `repair_chromosome`, which:
  1. Clamps hardness to `[1, 3]` with `hardness[entry] = 1` and `hardness[jewel] = 3`.
  2. Reconnects any disconnected components with bridge links.
  3. Enforces minimum entry-to-jewel hop separation.
  4. Eliminates single-machine chokepoints and 2-firewall cut vulnerabilities (`can_be_cut_off`), while capping edge density to avoid trivial cliques.

## 2. Ordinary Least-Squares Difficulty Surrogate (`ml/regression.py`)
Playing full games to score every chromosome inside a 40-generation GA is far too slow. Instead, we extract 7 structural graph features (`machines`, `links`, `mean_hardness`, `cheapest_cost`, `route_count`, `entry_jewel_hops`, `avg_degree`) across 500 simulated maps, fit ordinary least-squares (`np.linalg.lstsq`) linear regressors in NumPy, and report $R^2$ and RMSE strictly on the 20% held-out test maps.

## 3. GA Fitness & Menu Integration
- **Fitness:** $\text{fitness}(G) = \big(1 - 2\lvert \hat{w}(G) - 0.5\rvert\big) + 0.15 \cdot \frac{\text{routes}(G)}{5}$, rewarding maps whose predicted attacker win rate sits near 50% while encouraging route variety.
- **Selection & Variation:** tournament selection (`k=3`), single-point crossover, edge/hardness mutation, and elitism (`top 2`).
- **Menu Export:** the top evolved maps are exported to `data/maps/evolved_*.json` and automatically loaded into the Map Library / Game Setup menu.

