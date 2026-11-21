# Phase 4: Alpha-Beta Pruning Trace (`diamond_network`, Depth 3)

## Printed Game Trees

### Plain Minimax (`pruning=False` — 35 nodes, 0 cutoffs)

```text
pruning False | nodes 35 | cutoffs 0 | action Advance(target=1)
    attacker Advance(target=0) -> 847
  defender Firewall(a=1, b=3) -> 847
    attacker Advance(target=3) -> 896
    attacker Advance(target=0) -> 877
  defender Patch(machine=3) -> 896
    attacker Advance(target=3) -> 980
    attacker Advance(target=0) -> 897
  defender Firewall(a=2, b=3) -> 980
    attacker Advance(target=3) -> 980
    attacker Advance(target=0) -> 897
  defender Patch(machine=2) -> 980
    attacker Advance(target=3) -> 980
    attacker Advance(target=0) -> 896
  defender Firewall(a=0, b=2) -> 980
    attacker Advance(target=3) -> 980
    attacker Advance(target=0) -> 897
  defender Pass() -> 980
attacker Advance(target=1) -> 847
    attacker Advance(target=1) -> 907
  defender Firewall(a=0, b=2) -> 907
    attacker Advance(target=2) -> 846
  defender Firewall(a=2, b=3) -> 846
    attacker Advance(target=2) -> 826
  defender Patch(machine=2) -> 826
    attacker Advance(target=2) -> 887
  defender Patch(machine=3) -> 887
    attacker Advance(target=2) -> 907
  defender Firewall(a=1, b=3) -> 907
    attacker Advance(target=2) -> 907
  defender Patch(machine=1) -> 907
    attacker Advance(target=2) -> 907
  defender Firewall(a=0, b=1) -> 907
    attacker Advance(target=2) -> 907
  defender Pass() -> 907
attacker Advance(target=2) -> 826
```

### Alpha-Beta Pruning (`pruning=True` — 18 nodes, 6 cutoffs)

```text
pruning True | nodes 18 | cutoffs 6 | action Advance(target=1)
    attacker Advance(target=0) -> 847
  defender Firewall(a=1, b=3) -> 847
    attacker Advance(target=3) -> 896
      pruned 1 more
  defender Patch(machine=3) -> 896
    attacker Advance(target=3) -> 980
      pruned 1 more
  defender Firewall(a=2, b=3) -> 980
    attacker Advance(target=3) -> 980
      pruned 1 more
  defender Patch(machine=2) -> 980
    attacker Advance(target=3) -> 980
      pruned 1 more
  defender Firewall(a=0, b=2) -> 980
    attacker Advance(target=3) -> 980
      pruned 1 more
  defender Pass() -> 980
attacker Advance(target=1) -> 847
    attacker Advance(target=1) -> 907
  defender Firewall(a=0, b=2) -> 907
    attacker Advance(target=2) -> 846
  defender Firewall(a=2, b=3) -> 846
    pruned 6 more
attacker Advance(target=2) -> 846
```

## Why `defender Patch(machine=3)` -> `pruned 1 more` Cannot Change the Answer

1. After `attacker Advance(target=1)`, the defender's first option (`Firewall(a=1, b=3)`) already holds the attacker to **847**, so the defender's upper bound becomes **beta = 847**.
2. Next, the defender evaluates `Patch(machine=3)`, where the attacker tries `Advance(target=3)` first and gets **896**, raising the attacker's lower bound to **alpha = 896**.
3. Since the attacker is maximizing, any remaining reply under `Patch(machine=3)` (`Advance(target=0)`) can only keep the branch value at **896 or higher**—never lower.
4. Because **alpha (896) >= beta (847)**, the minimizing defender will never pick `Patch(machine=3)` when it already has `Firewall(a=1, b=3)` at 847.
5. Therefore skipping the remaining attacker move (`pruned 1 more`) saves work without ever affecting the defender's or root attacker's optimal move.

