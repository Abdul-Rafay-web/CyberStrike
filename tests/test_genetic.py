from pathlib import Path

import numpy as np

from config import CONFIG, EvoConfig
from core.chokepoints import can_be_cut_off, chokepoints
from core.generator import generate_network
from evo.genetic import (
    Chromosome,
    evolve_maps,
    load_evolved_map,
    mutate_chromosome,
    repair_chromosome,
    save_evolved_map,
    single_point_crossover,
)
from evo.plots import plot_ga_fitness_and_balance, plot_regression_fit
from ml.regression import fit_difficulty_surrogate
from utils.rng import STREAM_EVO, make_rng

PNG = b"\x89PNG"


def test_repair_reconnects_disconnected_chromosome_and_removes_chokepoints() -> None:
    rng = make_rng(11, STREAM_EVO)
    size = 12
    n_pairs = size * (size - 1) // 2
    # Start from a completely empty (disconnected) edge mask
    broken = Chromosome(
        size=size,
        hardness=(3,) * size,
        edge_mask=(False,) * n_pairs,
        entry=0,
        jewel=11,
    )
    fixed = repair_chromosome(broken, rng)
    net = fixed.to_network()
    assert net.is_connected()
    assert not chokepoints(net)
    assert not can_be_cut_off(net, CONFIG.game.stock.firewalls)
    assert net.hardness[net.entry] == CONFIG.game.min_hardness
    assert net.hardness[net.jewel] == CONFIG.game.max_hardness


def test_crossover_and_mutation_always_produce_valid_networks() -> None:
    rng = make_rng(22, STREAM_EVO)
    p1 = Chromosome.from_network(generate_network(1, size=12))
    p2 = Chromosome.from_network(generate_network(2, size=12))
    c1, c2 = single_point_crossover(p1, p2, rng)
    for child in (c1, c2, mutate_chromosome(c1, rng)):
        net = child.to_network()
        assert net.is_connected()
        assert not chokepoints(net)
        assert not can_be_cut_off(net, CONFIG.game.stock.firewalls)


def test_ga_improves_fitness_and_exports_loadable_maps(tmp_path: Path) -> None:
    surrogate = fit_difficulty_surrogate(num_maps=40, trials_per_map=5)
    evo_cfg = EvoConfig(population_size=12, generations=15, elite_count=2)
    res = evolve_maps(surrogate, size=12, seed=42, evo=evo_cfg)

    assert len(res.history) == 15
    # Elitism guarantees best fitness never drops; mean fitness improves from gen 0
    assert res.history[-1].best_fitness >= res.history[0].best_fitness
    assert res.history[-1].mean_fitness >= res.history[0].mean_fitness
    assert len(res.best_networks) >= 1

    best_net = res.best_networks[0]
    assert best_net.is_connected()
    assert not chokepoints(best_net)
    assert not can_be_cut_off(best_net, CONFIG.game.stock.firewalls)

    out = tmp_path / "evolved_01.json"
    save_evolved_map(
        best_net,
        out,
        name="EVOLVED_TEST_01",
        fitness=res.best_scores[0],
        predicted_win_rate=surrogate.predict_win_rate(best_net),
    )
    loaded_name, loaded_net = load_evolved_map(out)
    assert loaded_name == "EVOLVED_TEST_01"
    assert loaded_net == best_net


def test_evo_plots_create_valid_pngs(tmp_path: Path) -> None:
    surrogate = fit_difficulty_surrogate(num_maps=30, trials_per_map=4)
    res = evolve_maps(surrogate, size=12, seed=42, evo=EvoConfig(population_size=8, generations=6))
    p1 = tmp_path / "ga_fitness.png"
    plot_ga_fitness_and_balance(res.history, [0.8, 0.75, 0.9], [0.55, 0.48, 0.52], p1)
    assert p1.exists() and p1.read_bytes()[:4] == PNG and p1.stat().st_size > 2000

    p2 = tmp_path / "regression_fit.png"
    plot_regression_fit(
        surrogate.test_true_turns,
        surrogate.test_pred_turns,
        surrogate.turns_report.test_r2,
        p2,
    )
    assert p2.exists() and p2.read_bytes()[:4] == PNG and p2.stat().st_size > 2000


def test_chromosome_round_trip_preserves_network() -> None:
    net = generate_network(5, size=15)
    decoded = Chromosome.from_network(net).to_network()
    assert decoded.size == net.size
    assert len(decoded.edges()) == len(net.edges())
    assert decoded.is_connected()
    assert decoded.entry == 0 and decoded.jewel == net.size - 1
    assert np.isfinite(float(len(decoded.edges())))
