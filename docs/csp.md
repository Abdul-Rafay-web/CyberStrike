# Phase 7: CSP Defender Planner & 4-Region Map Coloring Dry Run

## 1. CyberStrike Defender CSP Formulation
- **Variables:** Available defense units from current stock: `P1..Pk` (patches), `F1..Fk` (firewalls), `H1..Hk` (honeypots).
- **Domains:**
  - `P_i`: unpatched machines without an existing honeypot.
  - `F_i`: unfirewalled network links.
  - `H_i`: interior machines (excluding `entry`, `jewel`, patched nodes, and existing honeypots).
- **Constraints:**
  1. All-different within each type (`P_i != P_j`, `F_i != F_j`, `H_i != H_j`).
  2. No honeypot on `entry` or `crown jewel` (enforced by domain).
  3. No honeypot on a patched machine (`H_j.machine != P_i.machine`).
  4. Route coverage: each of the up to 3 cheapest routes from the top-belief machine to the Crown Jewel holds at least one assigned patch or firewall.
- **Heuristics & Propagation:**
  - **MRV** selects the variable with the fewest remaining legal values (tie-broken by highest unassigned degree).
  - **LCV (belief-guided)** prefers values that eliminate the fewest neighbour options while covering uncovered high-belief routes.
  - **Forward Checking (FC)** prunes direct neighbour conflicts after each assignment.
  - **MAC (AC-3)** enforces full arc consistency across all binary constraints and forced route-cover singletons after each assignment.

---

## 2. Backtracking Dry Run on a 4-Region Map Coloring CSP

### Problem Setup
- **Regions (Variables):** `A, B, C, D` (in order)
- **Borders (Constraints `!=`):** `(A, B), (A, C), (B, C), (B, D), (C, D)` — note `A` and `D` do **not** share a border.
- **Colors (Domains):** `{R, G, B}` (3 colors), plus a 2-color unsolvable check `{R, G}` to show dead-end detection.

### Dry Run A — 3 Colors `{R, G, B}` (Satisfiable)
| Step | Action | Assignment | Remaining Domains (FC / MAC) | Notes |
|---|---|---|---|---|
| 0 | Start | `{}` | `A:{R,G,B}, B:{R,G,B}, C:{R,G,B}, D:{R,G,B}` | Initial state |
| 1 | Assign `A = R` | `{A:R}` | `B:{G,B}, C:{G,B}, D:{R,G,B}` | `R` removed from neighbours `B, C` |
| 2 | Assign `B = G` | `{A:R, B:G}` | `C:{B}, D:{R,B}` | `G` removed from neighbours `C, D`; under MAC, `C={B}` forces `D={R}` immediately via AC-3 on `(D, C)`! |
| 3 | Assign `C = B` | `{A:R, B:G, C:B}` | `D:{R}` | `B` removed from `D` |
| 4 | Assign `D = R` | `{A:R, B:G, C:B, D:R}` | — | **Solved!** All 5 borders satisfied (`A!=B, A!=C, B!=C, B!=D, C!=D`). |

### Dry Run B — 2 Colors `{R, G}` (Detecting a Dead-End: Plain BT vs. FC vs. MAC)
Because `{A, B, C}` forms a triangle (`K_3`), 2 colors cannot satisfy the map:

1. **Plain Backtracking:**
   - Assigns `A = R` -> tries `B = R` (fails `A!=B`), assigns `B = G` -> tries `C = R` (fails `A!=C`), tries `C = G` (fails `B!=C`) -> backtracks to `B`, tries nothing left -> backtracks to `A`, repeats symmetrically for `A = G`.
   - Total: **8 assignments tried** across both subtrees before proving UNSAT.
2. **Forward Checking:**
   - Assigns `A = R` -> prunes `B:{G}, C:{G}` (non-empty, keeps going) -> assigns `B = G` -> prunes `C:{}` -> **wipeout!** Immediately backtracks without trying `C`.
   - Repeats symmetrically for `A = G`. Total: **4 assignments tried**.
3. **MAC with AC-3:**
   - Assigns `A = R` -> FC prunes `B:{G}, C:{G}` -> AC-3 revises arc `(C, B)` using `C != B` -> since `B` only has `{G}`, `G` has no support in `C` -> `C` becomes `{}` **immediately at depth 1** before ever assigning `B`!
   - Repeats symmetrically for `A = G`. Total: **2 assignments tried** — proving UNSAT 4x earlier than plain backtracking!

