from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from typing import Any

BinaryConstraint = Callable[[Any, Any], bool]


def revise(
    domains: dict[str, list[Any]],
    xi: str,
    xj: str,
    constraint: BinaryConstraint,
) -> bool:
    """Remove values from `domains[xi]` that have no supporting value in `domains[xj]`."""
    dom_j = domains[xj]
    kept = [vi for vi in domains[xi] if any(constraint(vi, vj) for vj in dom_j)]
    if len(kept) < len(domains[xi]):
        domains[xi] = kept
        return True
    return False


def ac3(
    domains: dict[str, list[Any]],
    neighbors: dict[str, tuple[str, ...]],
    constraints: dict[tuple[str, str], BinaryConstraint],
    initial_arcs: Iterable[tuple[str, str]] | None = None,
) -> tuple[bool, int]:
    """Maintain arc consistency across all binary constraints via the AC-3 queue algorithm.

    Returns `(consistent, revisions)` where `consistent` is False if any domain becomes empty.
    """
    queue: deque[tuple[str, str]] = deque(
        constraints.keys() if initial_arcs is None else initial_arcs
    )
    in_queue: set[tuple[str, str]] = set(queue)
    revisions = 0

    while queue:
        xi, xj = queue.popleft()
        in_queue.discard((xi, xj))
        constraint = constraints.get((xi, xj))
        if constraint is None:
            continue
        revisions += 1
        if revise(domains, xi, xj, constraint):
            if not domains[xi]:
                return False, revisions
            for xk in neighbors[xi]:
                if xk != xj and (xk, xi) not in in_queue and (xk, xi) in constraints:
                    queue.append((xk, xi))
                    in_queue.add((xk, xi))
    return True, revisions

