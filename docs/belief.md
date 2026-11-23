# Phase 6: Bayesian Belief Tracker & Markov Blankets

## 1. Hand-Worked Bayes Update
Consider two machines $A$ and $B$ with uniform prior $P(X = A) = P(X = B) = 0.5$, detection rate $d = 0.8$, and false alarm rate $f = 0.1$:

- **Single-machine evidence ($A$ alerts):**
  $$P(X = A \mid A_A = \text{True}) = \frac{0.5 \times 0.8}{0.5 \times 0.8 + 0.5 \times 0.1} = \frac{0.40}{0.45} = 0.8889$$
  This matches `BeliefTracker.update_single(0, True)` and `exact_single_evidence_posterior` to 4+ decimals.

- **Full-vector evidence ($A$ alerts, $B$ is quiet):**
  Observing silence on $B$ adds extra evidence because $P(A_B = \text{False} \mid X = A) = 1 - f = 0.9$ while $P(A_B = \text{False} \mid X = B) = 1 - d = 0.2$:
  $$P(X = A \mid A_A = T, A_B = F) = \frac{0.5 \times (0.8 \times 0.9)}{0.5 \times (0.8 \times 0.9) + 0.5 \times (0.1 \times 0.2)} = \frac{0.36}{0.37} = 0.9730$$
  This matches `BeliefTracker.update((True, False))` to 4+ decimals.

## 2. Predict-Then-Correct Filtering
Each Defender turn runs two steps using Phase 5's measured sensor rates from `data/sensor_model.json` (`detection_rate = 0.725`, `false_alarm_rate = 0.119`):
1. **Predict (`b' = b @ T`):** Hard or patched machines hold the attacker with higher stay probability $(h_i - 1 + s_0)/h_i$, while outbound transition mass spreads across open (non-firewalled) neighbours weighted by $1 / h_j$.
2. **Update (`b_t ∝ b'_t * L(alerts)`):** Multiply each hypothesis by the exact joint sensor likelihood in log space, apply a tiny probability floor (`1e-6`) so noisy observations never permanently zero out a machine, and normalize to sum to 1.
3. **Honeypot certainty rule:** Whenever `observation.intruder_known_at` reports a triggered honeypot on machine $k$, belief collapses to exact certainty ($b_t(k) = 1.0$).

## 3. Bayesian Network & Markov Blanket
- In `prob/blanket.py`, `network_causal_dag(network)` orients edges from `network.entry` outward by `(bfs_distance, machine_id)` (guaranteed acyclic), and `markov_blanket(network, m)` returns the exact triple `(parents, children, co_parents)`. On `diamond_network` ($0 \to 1, 0 \to 2, 1 \to 3, 2 \to 3$), `markov_blanket(net, 1)` returns parents `{0}`, children `{3}`, and co-parents `{2}` (explaining-away through the Crown Jewel).
- In the unrolled temporal DBN (`dbn_dag`), `MB(X_t) = {X_{t-1}, X_{t+1}, A_{t,0..N-1}}`. Exact enumeration verifies $P(X_t \mid \text{MB}(X_t), Z) = P(X_t \mid \text{MB}(X_t))$ for any outside-blanket evidence $Z$.

## 4. Exact Enumeration Proof
On a 6-machine map across 4 turns ($6^5 = 7{,}776$ full hidden trajectories), `prob/bayesnet.py` (`exact_enumeration_posterior`) sums the full joint distribution over every trajectory directly without matrix filtering. `BeliefTracker` with `floor = 0.0` agrees to machine precision ($< 10^{-12}$), and with default `floor = 1e-6` agrees to $< 10^{-4}$.

