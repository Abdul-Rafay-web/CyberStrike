from pathlib import Path

from config import CONFIG
from core.actions import Action, Firewall, Honeypot, Patch
from core.generator import generate_network
from core.state import Stock
from csp.arc import ac3
from csp.model import build_defense_csp, k_cheapest_routes, map_coloring_csp
from csp.plots import CSPBenchRow, plot_csp_backtracks, plot_csp_defender_matchup
from csp.solver import (
    ALL_SOLVER_CONFIGS,
    FORWARD_CHECKING,
    MAC_MRV_LCV,
    MRV_LCV_FC,
    PLAIN_BT,
    solve_csp,
)
from tests.factories import chain_network, diamond_network

PNG = b"\x89PNG"


def _four_region_borders() -> list[tuple[str, str]]:
    return [("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"), ("C", "D")]


def test_four_region_map_coloring_matches_docs_dry_run() -> None:
    sat_csp = map_coloring_csp(("A", "B", "C", "D"), _four_region_borders(), ("R", "G", "B"))
    for cfg in ALL_SOLVER_CONFIGS:
        res = solve_csp(sat_csp, cfg)
        assert res.assignment is not None
        assert sat_csp.is_valid_assignment(res.assignment)
        assert res.assignment["A"] == res.assignment["D"]

    # Unsatisfiable 2-color triangle: MAC detects dead-end earlier than FC, which beats plain BT
    unsat_csp = map_coloring_csp(("A", "B", "C", "D"), _four_region_borders(), ("R", "G"))
    res_bt = solve_csp(unsat_csp, PLAIN_BT)
    res_fc = solve_csp(unsat_csp, FORWARD_CHECKING)
    res_mac = solve_csp(unsat_csp, MAC_MRV_LCV)
    assert res_bt.assignment is None
    assert res_fc.assignment is None
    assert res_mac.assignment is None
    assert res_mac.stats.assignments_tried < res_fc.stats.assignments_tried
    assert res_fc.stats.assignments_tried < res_bt.stats.assignments_tried


def test_ac3_prunes_singleton_cascades() -> None:
    csp = map_coloring_csp(("A", "B", "C"), [("A", "B"), ("B", "C"), ("A", "C")], ("R", "G"))
    domains = {"A": ["R"], "B": ["R", "G"], "C": ["R", "G"]}
    ok, revisions = ac3(domains, csp.neighbors, csp.binary_constraints)
    assert not ok
    assert revisions > 0


def test_k_cheapest_routes_finds_distinct_routes_in_cost_order() -> None:
    routes = k_cheapest_routes(diamond_network(), 0, k=3)
    assert routes == ((0, 1, 3), (0, 2, 3))

    chain_routes = k_cheapest_routes(chain_network(), 0, k=3)
    assert chain_routes == ((0, 1, 2),)


def test_all_four_solver_settings_return_legal_plans_across_map_sizes() -> None:
    stock = Stock(
        CONFIG.game.stock.patches,
        CONFIG.game.stock.firewalls,
        CONFIG.game.stock.honeypots,
    )
    for size in (10, 15, 20):
        net = generate_network(seed=size, size=size)
        problem = build_defense_csp(net, stock)
        for cfg in ALL_SOLVER_CONFIGS:
            result = solve_csp(problem, cfg)
            assert result.assignment is not None, f"{cfg.name} failed on size {size}"
            assert problem.is_valid_assignment(result.assignment)

            # Verify every domain rule explicitly
            patches: set[int] = set()
            firewalls: set[tuple[int, int]] = set()
            honeypots: set[int] = set()
            for val in result.assignment.values():
                assert isinstance(val, Action)
                if isinstance(val, Patch):
                    assert val.machine not in patches
                    patches.add(val.machine)
                elif isinstance(val, Firewall):
                    edge = (val.a, val.b)
                    assert edge not in firewalls
                    firewalls.add(edge)
                elif isinstance(val, Honeypot):
                    assert val.machine not in (net.entry, net.jewel)
                    assert val.machine not in honeypots
                    honeypots.add(val.machine)
            assert patches.isdisjoint(honeypots)


def test_mac_needs_fewer_backtracks_than_plain_backtracking_at_every_map_size() -> None:
    stock = Stock(
        CONFIG.game.stock.patches,
        CONFIG.game.stock.firewalls,
        CONFIG.game.stock.honeypots,
    )
    for size in (10, 15, 20):
        bt_backtracks = 0
        mac_backtracks = 0
        for seed in range(25):
            problem = build_defense_csp(generate_network(seed, size), stock)
            res_bt = solve_csp(problem, PLAIN_BT)
            res_mac = solve_csp(problem, MAC_MRV_LCV)
            assert res_bt.assignment is not None
            assert res_mac.assignment is not None
            bt_backtracks += res_bt.stats.backtracks
            mac_backtracks += res_mac.stats.backtracks
        assert mac_backtracks < bt_backtracks, (
            f"Expected MAC ({mac_backtracks}) < plain BT ({bt_backtracks}) at size {size}"
        )


def test_tiny_and_low_stock_csps_remain_solvable() -> None:
    # Must never crash or become unsolvable on tiny networks or depleted stock
    net = diamond_network()
    for stock in (Stock(1, 0, 0), Stock(0, 1, 0), Stock(0, 0, 1), Stock(1, 1, 1), Stock(0, 0, 0)):
        problem = build_defense_csp(net, stock)
        res = solve_csp(problem, MRV_LCV_FC)
        assert res.assignment is not None
        assert problem.is_valid_assignment(res.assignment)


def test_csp_plots_write_valid_pngs(tmp_path: Path) -> None:
    rows = [
        CSPBenchRow(10, "plain_bt", 120.0, 95.0, 0.8, 1.0),
        CSPBenchRow(10, "fc", 45.0, 22.0, 0.4, 1.0),
        CSPBenchRow(10, "mrv_lcv_fc", 12.0, 2.0, 0.3, 1.0),
        CSPBenchRow(10, "mac_mrv_lcv", 9.0, 0.2, 0.4, 1.0),
        CSPBenchRow(15, "plain_bt", 450.0, 390.0, 3.1, 1.0),
        CSPBenchRow(15, "fc", 90.0, 50.0, 0.9, 1.0),
        CSPBenchRow(15, "mrv_lcv_fc", 14.0, 1.5, 0.5, 1.0),
        CSPBenchRow(15, "mac_mrv_lcv", 9.0, 0.1, 0.6, 1.0),
    ]
    p1 = tmp_path / "csp_backtracks.png"
    plot_csp_backtracks(rows, p1)
    assert p1.exists() and p1.read_bytes()[:4] == PNG and p1.stat().st_size > 2000

    p2 = tmp_path / "nested" / "csp_matchup.png"
    plot_csp_defender_matchup([("vs Random", 0.55, 0.78), ("vs Greedy A*", 0.02, 0.25)], p2)
    assert p2.exists() and p2.read_bytes()[:4] == PNG and p2.stat().st_size > 2000

