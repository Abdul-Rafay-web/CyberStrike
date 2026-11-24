from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from config import CONFIG
from core.actions import Action
from csp.arc import ac3
from csp.model import CSPProblem, RouteRequirement


@dataclass(frozen=True, slots=True)
class SolverConfig:
    """Feature switches controlling variable ordering, value ordering, and propagation."""

    name: str
    use_mrv: bool = False
    use_lcv: bool = False
    use_fc: bool = False
    use_mac: bool = False
    max_steps: int = CONFIG.csp.max_search_steps


PLAIN_BT = SolverConfig("plain_bt", use_mrv=False, use_lcv=False, use_fc=False, use_mac=False)
FORWARD_CHECKING = SolverConfig("fc", use_mrv=False, use_lcv=False, use_fc=True, use_mac=False)
MRV_LCV_FC = SolverConfig("mrv_lcv_fc", use_mrv=True, use_lcv=True, use_fc=True, use_mac=False)
MAC_MRV_LCV = SolverConfig("mac_mrv_lcv", use_mrv=True, use_lcv=True, use_fc=False, use_mac=True)

ALL_SOLVER_CONFIGS: tuple[SolverConfig, ...] = (
    PLAIN_BT,
    FORWARD_CHECKING,
    MRV_LCV_FC,
    MAC_MRV_LCV,
)


@dataclass(frozen=True, slots=True)
class SolverStats:
    """Search effort counters collected during one CSP solve."""

    assignments_tried: int
    backtracks: int
    arc_revisions: int
    seconds: float
    solved: bool


@dataclass(frozen=True, slots=True)
class SolveResult:
    """Solved variable assignment (or None if unsatisfiable / step budget hit) plus stats."""

    assignment: dict[str, Any] | None
    stats: SolverStats


@dataclass(slots=True)
class _Counters:
    assignments_tried: int = 0
    backtracks: int = 0
    arc_revisions: int = 0
    hit_limit: bool = False


def _route_satisfied(
    req: RouteRequirement, assignment: dict[str, Any], cover_vars: frozenset[str]
) -> bool:
    for var in cover_vars:
        if var in assignment:
            val = assignment[var]
            if isinstance(val, Action) and req.covered_by(val):
                return True
    return False


def _uncovered_routes(
    problem: CSPProblem, assignment: dict[str, Any]
) -> list[RouteRequirement]:
    return [
        req
        for req in problem.route_requirements
        if not _route_satisfied(req, assignment, problem.cover_variables)
    ]


def _check_route_feasibility(
    problem: CSPProblem,
    assignment: dict[str, Any],
    domains: dict[str, list[Any]],
) -> bool:
    """Verify every still-uncovered route can still be covered by unassigned cover variables."""
    if not problem.route_requirements:
        return True
    unassigned_cover = [v for v in problem.cover_variables if v not in assignment]
    uncovered = _uncovered_routes(problem, assignment)
    if not uncovered:
        return True
    if not unassigned_cover:
        return False

    # Every uncovered route must have at least one covering value still alive in some domain
    for req in uncovered:
        if not any(
            isinstance(val, Action) and req.covered_by(val)
            for v in unassigned_cover
            for val in domains[v]
        ):
            return False
    return True


def _tighten_forced_routes(
    problem: CSPProblem,
    assignment: dict[str, Any],
    domains: dict[str, list[Any]],
) -> tuple[bool, list[str]]:
    """For MAC: if only one unassigned cover variable can satisfy an uncovered route, filter it."""
    if not problem.route_requirements:
        return True, []
    unassigned_cover = [v for v in problem.cover_variables if v not in assignment]
    uncovered = _uncovered_routes(problem, assignment)
    if not uncovered:
        return True, []
    if not unassigned_cover:
        return False, []

    changed: list[str] = []
    for req in uncovered:
        capable = [
            v
            for v in unassigned_cover
            if any(isinstance(val, Action) and req.covered_by(val) for val in domains[v])
        ]
        if not capable:
            return False, changed
        if len(capable) == 1:
            sole_var = capable[0]
            filtered = [
                val for val in domains[sole_var] if isinstance(val, Action) and req.covered_by(val)
            ]
            if not filtered:
                return False, changed
            if len(filtered) < len(domains[sole_var]):
                domains[sole_var] = filtered
                changed.append(sole_var)
    return True, changed


def _select_variable(
    problem: CSPProblem,
    assignment: dict[str, Any],
    domains: dict[str, list[Any]],
    cfg: SolverConfig,
) -> str:
    unassigned = [v for v in problem.variables if v not in assignment]
    if not cfg.use_mrv:
        return unassigned[0]
    # Minimum Remaining Values (MRV), breaking ties by degree (most unassigned neighbours)
    return min(
        unassigned,
        key=lambda v: (
            len(domains[v]),
            -sum(1 for nbr in problem.neighbors[v] if nbr not in assignment),
            problem.variables.index(v),
        ),
    )


def _order_values(
    problem: CSPProblem,
    var: str,
    assignment: dict[str, Any],
    domains: dict[str, list[Any]],
    cfg: SolverConfig,
) -> list[Any]:
    values = list(domains[var])
    if not cfg.use_lcv or len(values) <= 1:
        return values

    uncovered = _uncovered_routes(problem, assignment)
    unassigned_nbrs = [nbr for nbr in problem.neighbors[var] if nbr not in assignment]

    def rank(val: Any) -> tuple[int, float]:
        # Least constraining: fewer neighbour values ruled out
        eliminated = 0
        for nbr in unassigned_nbrs:
            fn = problem.binary_constraints.get((var, nbr))
            if fn is not None:
                eliminated += sum(1 for v2 in domains[nbr] if not fn(val, v2))
        # Prefer values covering currently uncovered routes and high belief-guided priority
        route_bonus = (
            sum(1.0 for req in uncovered if isinstance(val, Action) and req.covered_by(val))
            if var in problem.cover_variables
            else 0.0
        )
        belief_score = problem.value_scores.get((var, val), 0.0)
        return (eliminated, -(route_bonus * 5.0 + belief_score))

    values.sort(key=rank)
    return values


def _consistent_with_assigned(
    problem: CSPProblem, var: str, val: Any, assignment: dict[str, Any]
) -> bool:
    for nbr in problem.neighbors[var]:
        if nbr in assignment and not problem.is_consistent_pair(var, val, nbr, assignment[nbr]):
            return False
    return True


def _backtrack(
    problem: CSPProblem,
    assignment: dict[str, Any],
    domains: dict[str, list[Any]],
    cfg: SolverConfig,
    counters: _Counters,
) -> dict[str, Any] | None:
    if len(assignment) == len(problem.variables):
        return dict(assignment) if problem.is_valid_assignment(assignment) else None

    if counters.assignments_tried >= cfg.max_steps:
        counters.hit_limit = True
        return None

    var = _select_variable(problem, assignment, domains, cfg)
    for val in _order_values(problem, var, assignment, domains, cfg):
        if counters.assignments_tried >= cfg.max_steps:
            counters.hit_limit = True
            return None
        counters.assignments_tried += 1

        if not _consistent_with_assigned(problem, var, val, assignment):
            continue

        assignment[var] = val

        if cfg.use_fc or cfg.use_mac:
            next_domains = {k: list(v) for k, v in domains.items()}
            next_domains[var] = [val]
            ok = True

            if cfg.use_mac:
                # Textbook MAC: seed AC-3 with arcs pointing into `var`, letting AC-3 prune
                # direct neighbours and cascade every domain reduction across the constraint graph.
                seeds: list[tuple[str, str]] = [
                    (nbr, var) for nbr in problem.neighbors[var] if nbr not in assignment
                ]
                ac_ok, revs = ac3(
                    next_domains, problem.neighbors, problem.binary_constraints, seeds
                )
                counters.arc_revisions += revs
                if not ac_ok:
                    ok = False
                else:
                    ok_routes, tightened = _tighten_forced_routes(
                        problem, assignment, next_domains
                    )
                    if not ok_routes:
                        ok = False
                    elif tightened:
                        extra_seeds = [
                            (nbr, tv)
                            for tv in tightened
                            for nbr in problem.neighbors[tv]
                            if nbr not in assignment
                        ]
                        ac_ok2, revs2 = ac3(
                            next_domains,
                            problem.neighbors,
                            problem.binary_constraints,
                            extra_seeds,
                        )
                        counters.arc_revisions += revs2
                        if not ac_ok2 or not _check_route_feasibility(
                            problem, assignment, next_domains
                        ):
                            ok = False
                    elif not _check_route_feasibility(problem, assignment, next_domains):
                        ok = False
            else:
                # Forward checking: prune only direct unassigned neighbours
                for nbr in problem.neighbors[var]:
                    if nbr in assignment:
                        continue
                    fn = problem.binary_constraints.get((var, nbr))
                    if fn is None:
                        continue
                    next_domains[nbr] = [v2 for v2 in next_domains[nbr] if fn(val, v2)]
                    if not next_domains[nbr]:
                        ok = False
                        break
                if ok and not _check_route_feasibility(problem, assignment, next_domains):
                    ok = False

            if ok:
                result = _backtrack(problem, assignment, next_domains, cfg, counters)
                if result is not None:
                    return result
        else:
            result = _backtrack(problem, assignment, domains, cfg, counters)
            if result is not None:
                return result

        del assignment[var]
        counters.backtracks += 1

    return None


def solve_csp(problem: CSPProblem, config: SolverConfig = MAC_MRV_LCV) -> SolveResult:
    """Solve `problem` using the selected backtracking / propagation configuration."""
    started = time.perf_counter()
    counters = _Counters()
    domains: dict[str, list[Any]] = {v: list(problem.domains[v]) for v in problem.variables}

    if any(not dom for dom in domains.values()):
        return SolveResult(
            None,
            SolverStats(0, 0, 0, time.perf_counter() - started, False),
        )

    if config.use_mac:
        ok, revs = ac3(domains, problem.neighbors, problem.binary_constraints)
        counters.arc_revisions += revs
        if not ok:
            return SolveResult(
                None,
                SolverStats(0, 0, counters.arc_revisions, time.perf_counter() - started, False),
            )

    solution = _backtrack(problem, {}, domains, config, counters)
    elapsed = time.perf_counter() - started
    return SolveResult(
        assignment=solution,
        stats=SolverStats(
            assignments_tried=counters.assignments_tried,
            backtracks=counters.backtracks,
            arc_revisions=counters.arc_revisions,
            seconds=elapsed,
            solved=solution is not None,
        ),
    )
