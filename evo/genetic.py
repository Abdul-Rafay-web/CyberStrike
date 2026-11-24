from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import CONFIG, AppConfig, EvoConfig
from core.generator import generate_network
from core.network import Network, make_edge
from csp.model import k_cheapest_routes
from evo.repair import all_edge_pairs, repair_network_edges
from ml.regression import DifficultySurrogate
from utils.rng import STREAM_EVO, make_rng


@dataclass(frozen=True, slots=True)
class Chromosome:
    """Genetic encoding of a CyberStrike map: hardness list, edge bitmask, entry, and jewel."""

    size: int
    hardness: tuple[int, ...]
    edge_mask: tuple[bool, ...]
    entry: int
    jewel: int

    def to_network(self) -> Network:
        pairs = all_edge_pairs(self.size)
        edges = [edge for edge, active in zip(pairs, self.edge_mask, strict=True) if active]
        return Network.from_edges(self.size, edges, self.hardness, self.entry, self.jewel)

    @classmethod
    def from_network(cls, network: Network) -> Chromosome:
        # Relabel so entry is always 0 and jewel is always size-1 for aligned GA crossover
        size = network.size
        perm = list(range(size))
        if network.entry != 0:
            perm[0], perm[network.entry] = perm[network.entry], perm[0]
        jewel_pos = perm.index(network.jewel)
        if jewel_pos != size - 1:
            perm[size - 1], perm[jewel_pos] = perm[jewel_pos], perm[size - 1]
        inv = {old: new for new, old in enumerate(perm)}

        edge_set = {make_edge(inv[a], inv[b]) for a, b in network.edges()}
        hardness = tuple(network.hardness[old] for old in perm)
        pairs = all_edge_pairs(size)
        mask = tuple(edge in edge_set for edge in pairs)
        return cls(
            size=size,
            hardness=hardness,
            edge_mask=mask,
            entry=0,
            jewel=size - 1,
        )


def repair_chromosome(
    chrom: Chromosome, rng: np.random.Generator, config: AppConfig = CONFIG
) -> Chromosome:
    """Repair a chromosome so its decoded `Network` is always connected and fair."""
    from core.chokepoints import can_be_cut_off, chokepoints

    pairs = all_edge_pairs(chrom.size)
    edges = {edge for edge, active in zip(pairs, chrom.edge_mask, strict=True) if active}
    game = config.game
    # Fast path: preserve already-valid chromosomes without random perturbations
    if (
        chrom.entry != chrom.jewel
        and chrom.hardness[chrom.entry] == game.min_hardness
        and chrom.hardness[chrom.jewel] == game.max_hardness
        and make_edge(chrom.entry, chrom.jewel) not in edges
    ):
        try:
            net = Network.from_edges(chrom.size, edges, chrom.hardness, chrom.entry, chrom.jewel)
            if (
                net.is_connected()
                and not chokepoints(net)
                and not can_be_cut_off(net, game.stock.firewalls)
            ):
                return chrom
        except ValueError:
            pass

    fixed_edges, fixed_h, entry, jewel = repair_network_edges(
        chrom.size, edges, list(chrom.hardness), chrom.entry, chrom.jewel, rng, config
    )
    mask = tuple(edge in fixed_edges for edge in pairs)
    return Chromosome(
        size=chrom.size,
        hardness=tuple(fixed_h),
        edge_mask=mask,
        entry=entry,
        jewel=jewel,
    )


def single_point_crossover(
    parent_a: Chromosome,
    parent_b: Chromosome,
    rng: np.random.Generator,
    config: AppConfig = CONFIG,
) -> tuple[Chromosome, Chromosome]:
    """Single-point crossover on the edge bitmask (plus hardness splice), followed by repair."""
    if parent_a.size != parent_b.size:
        raise ValueError("parents must have the same map size")
    n_edges = len(parent_a.edge_mask)
    cut_e = int(rng.integers(1, max(2, n_edges)))
    cut_h = int(rng.integers(1, max(2, parent_a.size)))

    child1 = Chromosome(
        size=parent_a.size,
        hardness=parent_a.hardness[:cut_h] + parent_b.hardness[cut_h:],
        edge_mask=parent_a.edge_mask[:cut_e] + parent_b.edge_mask[cut_e:],
        entry=parent_a.entry,
        jewel=parent_a.jewel,
    )
    child2 = Chromosome(
        size=parent_b.size,
        hardness=parent_b.hardness[:cut_h] + parent_a.hardness[cut_h:],
        edge_mask=parent_b.edge_mask[:cut_e] + parent_a.edge_mask[cut_e:],
        entry=parent_b.entry,
        jewel=parent_b.jewel,
    )
    return repair_chromosome(child1, rng, config), repair_chromosome(child2, rng, config)


def mutate_chromosome(
    chrom: Chromosome,
    rng: np.random.Generator,
    evo: EvoConfig = CONFIG.evo,
    config: AppConfig = CONFIG,
) -> Chromosome:
    """Balanced edge toggle (equal add/remove chance) and hardness nudge, then repair."""
    game = config.game
    mask = list(chrom.edge_mask)
    if float(rng.random()) < evo.edge_mutation_rate:
        present = [i for i, active in enumerate(mask) if active]
        absent = [i for i, active in enumerate(mask) if not active]
        if present and (not absent or float(rng.random()) < 0.5):
            idx = present[int(rng.integers(0, len(present)))]
            mask[idx] = False
        elif absent:
            idx = absent[int(rng.integers(0, len(absent)))]
            mask[idx] = True

    hardness = list(chrom.hardness)
    interior = [m for m in range(chrom.size) if m not in (chrom.entry, chrom.jewel)]
    if interior and float(rng.random()) < evo.hardness_mutation_rate:
        m = interior[int(rng.integers(0, len(interior)))]
        step = -1 if float(rng.random()) < 0.5 else 1
        hardness[m] = max(game.min_hardness, min(game.max_hardness, hardness[m] + step))

    mutated = Chromosome(
        size=chrom.size,
        hardness=tuple(hardness),
        edge_mask=tuple(mask),
        entry=chrom.entry,
        jewel=chrom.jewel,
    )
    return repair_chromosome(mutated, rng, config)


def tournament_select(
    population: list[Chromosome],
    scores: list[float],
    tournament_size: int,
    rng: np.random.Generator,
) -> Chromosome:
    """Pick the fittest chromosome among `tournament_size` random competitors."""
    k = min(tournament_size, len(population))
    contenders = rng.choice(len(population), size=k, replace=False)
    winner = max((int(i) for i in contenders), key=lambda idx: scores[idx])
    return population[winner]


def evaluate_fitness(network: Network, surrogate: DifficultySurrogate) -> float:
    """Score a map: predicted attacker win rate near 50% plus a route-variety bonus."""
    pred_win = surrogate.predict_win_rate(network)
    balance = 1.0 - 2.0 * abs(pred_win - 0.5)
    routes = len(k_cheapest_routes(network, network.entry, k=5))
    variety_bonus = 0.15 * (routes / 5.0)
    return balance + variety_bonus


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    """Per-generation GA telemetry for plotting and live diagnostics."""

    generation: int
    best_fitness: float
    mean_fitness: float
    best_predicted_win_rate: float


@dataclass(frozen=True, slots=True)
class EvolutionResult:
    """Final evolved population (sorted fittest first) and generation history."""

    best_networks: tuple[Network, ...]
    best_scores: tuple[float, ...]
    history: tuple[GenerationRecord, ...]


def evolve_maps(
    surrogate: DifficultySurrogate,
    *,
    size: int = CONFIG.game.default_machines,
    seed: int = CONFIG.seeds.master_seed,
    evo: EvoConfig = CONFIG.evo,
    config: AppConfig = CONFIG,
) -> EvolutionResult:
    """Run the Genetic Algorithm over `evo.generations` generations using the fast surrogate."""
    rng = make_rng(seed, STREAM_EVO)
    population: list[Chromosome] = [
        Chromosome.from_network(generate_network(seed * 100 + i, size=size, config=config))
        for i in range(evo.population_size)
    ]

    history: list[GenerationRecord] = []

    for gen in range(evo.generations):
        networks = [c.to_network() for c in population]
        scores = [evaluate_fitness(net, surrogate) for net in networks]
        ranked = sorted(range(len(population)), key=lambda i: -scores[i])
        best_idx = ranked[0]
        history.append(
            GenerationRecord(
                generation=gen,
                best_fitness=scores[best_idx],
                mean_fitness=float(np.mean(scores)),
                best_predicted_win_rate=surrogate.predict_win_rate(networks[best_idx]),
            )
        )

        # Elitism: preserve top `elite_count` unchanged
        next_pop: list[Chromosome] = [population[i] for i in ranked[: evo.elite_count]]
        while len(next_pop) < evo.population_size:
            p1 = tournament_select(population, scores, evo.tournament_size, rng)
            p2 = tournament_select(population, scores, evo.tournament_size, rng)
            if float(rng.random()) < evo.crossover_rate:
                c1, c2 = single_point_crossover(p1, p2, rng, config)
            else:
                c1, c2 = p1, p2
            next_pop.append(mutate_chromosome(c1, rng, evo, config))
            if len(next_pop) < evo.population_size:
                next_pop.append(mutate_chromosome(c2, rng, evo, config))
        population = next_pop

    final_nets = [c.to_network() for c in population]
    final_scores = [evaluate_fitness(net, surrogate) for net in final_nets]
    order = sorted(range(len(population)), key=lambda i: -final_scores[i])

    # Deduplicate top exported networks while preserving fitness ranking
    unique_nets: list[Network] = []
    unique_scores: list[float] = []
    seen: set[Network] = set()
    for idx in order:
        net = final_nets[idx]
        if net not in seen:
            seen.add(net)
            unique_nets.append(net)
            unique_scores.append(final_scores[idx])

    return EvolutionResult(
        best_networks=tuple(unique_nets),
        best_scores=tuple(unique_scores),
        history=tuple(history),
    )


def save_evolved_map(
    network: Network,
    path: Path,
    *,
    name: str,
    fitness: float,
    predicted_win_rate: float,
) -> None:
    """Export an evolved network map to JSON in `data/maps/` for menu loading."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "size": network.size,
        "entry": network.entry,
        "jewel": network.jewel,
        "hardness": list(network.hardness),
        "edges": [[a, b] for a, b in network.edges()],
        "fitness": round(fitness, 4),
        "predicted_win_rate": round(predicted_win_rate, 4),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_evolved_map(path: Path) -> tuple[str, Network]:
    """Load an evolved map JSON file back into a validated `Network`."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    edges = [make_edge(int(a), int(b)) for a, b in payload["edges"]]
    hardness = [int(h) for h in payload["hardness"]]
    net = Network.from_edges(
        int(payload["size"]),
        edges,
        hardness,
        int(payload["entry"]),
        int(payload["jewel"]),
    )
    return str(payload["name"]), net
