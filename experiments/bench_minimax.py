from __future__ import annotations

from dataclasses import dataclass

from core.generator import generate_network
from core.state import GameState, new_game
from experiments.scenarios import defended_state
from search.minimax import search_to_depth


@dataclass(frozen=True, slots=True)
class DepthRow:
    depth: int
    plain_nodes: float
    pruned_nodes: float
    pruned_seconds: float
    same_move_rate: float


def positions(count: int, size: int | None = None) -> list[GameState]:
    states: list[GameState] = []
    for seed in range(count):
        states.append(new_game(generate_network(seed, size)))
        states.append(defended_state(seed, size))
    return states


def compare_depths(states: list[GameState], depths: range, plain_limit: int) -> list[DepthRow]:
    rows: list[DepthRow] = []
    for depth in depths:
        pruned = [search_to_depth(s, depth, pruning=True) for s in states]
        if depth <= plain_limit:
            plain = [search_to_depth(s, depth, pruning=False) for s in states]
            plain_nodes = sum(r.nodes for r in plain) / len(plain)
            same = sum(a.action == b.action for a, b in zip(plain, pruned, strict=True)) / len(
                states
            )
        else:
            plain_nodes = float("nan")
            same = float("nan")
        rows.append(
            DepthRow(
                depth,
                plain_nodes,
                sum(r.nodes for r in pruned) / len(pruned),
                sum(r.seconds for r in pruned) / len(pruned),
                same,
            )
        )
    return rows


def format_table(rows: list[DepthRow]) -> str:
    header = f"{'depth':>5}{'minimax':>11}{'alpha beta':>12}{'saved':>8}"
    header += f"{'seconds':>9}{'same move':>11}"
    lines = [header]
    for row in rows:
        if row.plain_nodes == row.plain_nodes:
            saved = f"{1 - row.pruned_nodes / row.plain_nodes:.0%}"
            plain = f"{row.plain_nodes:.0f}"
            same = f"{row.same_move_rate:.0%}"
        else:
            saved, plain, same = "", "skipped", ""
        lines.append(
            f"{row.depth:>5}{plain:>11}{row.pruned_nodes:>12.0f}{saved:>8}"
            f"{row.pruned_seconds:>9.3f}{same:>11}"
        )
    return "\n".join(lines)


def main() -> None:
    states = positions(25)
    print("nodes expanded per move, average over", len(states), "positions")
    print(format_table(compare_depths(states, range(1, 7), plain_limit=6)))


if __name__ == "__main__":
    main()

