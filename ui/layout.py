from __future__ import annotations

from dataclasses import dataclass

import pygame

from core.network import Edge, Network, bfs_distances


@dataclass(frozen=True, slots=True)
class NodeVisualMeta:
    """Visual metadata and normalized [0..1] layout coordinates for one machine."""

    index: int
    code: str
    label: str
    sub_label: str
    role: str
    nx: float
    ny: float
    cvss: float


@dataclass(frozen=True, slots=True)
class MapPreset:
    """Curated network scenario available in Operation Setup and Map Library."""

    id: str
    name: str
    sector_code: str
    seed: int
    node_count: int
    badge: str
    description: str
    is_canonical_13: bool = False


MAP_PRESETS: tuple[MapPreset, ...] = (
    MapPreset(
        id="intra_net_alpha_09",
        name="INTRA_NET_ALPHA_09",
        sector_code="INTRA_NET_ALPHA_09",
        seed=42,
        node_count=13,
        badge="REFERENCE CANON",
        description=(
            "Canonical 5-tier corporate enterprise topology with balanced dual-gateway "
            "vault routing into Crown Jewel DB-01."
        ),
        is_canonical_13=True,
    ),
    MapPreset(
        id="vault_zero_dmz_15",
        name="VAULT_ZERO_DMZ_15",
        sector_code="VAULT_ZERO_DMZ_15",
        seed=108,
        node_count=15,
        badge="STANDARD 15",
        description=(
            "Standard 15-machine enterprise benchmark network generated with "
            "chokepoint-free redundant routing."
        ),
    ),
    MapPreset(
        id="chokepoint_omega_18",
        name="CHOKEPOINT_OMEGA_18",
        sector_code="CHOKEPOINT_OMEGA_18",
        seed=777,
        node_count=18,
        badge="LAYERED 18",
        description=(
            "Segmented DMZ, internal domain tier, and hardened gateway bridge across "
            "18 interconnected hosts."
        ),
    ),
    MapPreset(
        id="hydra_grid_22",
        name="HYDRA_GRID_22",
        sector_code="HYDRA_GRID_22",
        seed=4096,
        node_count=22,
        badge="LARGE MESH",
        description=(
            "Dense 22-node lateral-movement mesh offering wide multi-path flanking "
            "corridors toward the Crown Jewel."
        ),
    ),
)


_CANONICAL_13_SPECS: tuple[tuple[str, str, str, str, int, float, float, float], ...] = (
    ("SVR-01", "SVR-01", "ENTRY POINT", "entry", 1, 0.00, 0.49, 8.8),
    ("SVR-02", "SVR-02", "", "web_dmz", 2, 0.22, 0.00, 6.4),
    ("SVR-04", "SVR-04", "", "workstation", 1, 0.24, 0.50, 8.1),
    ("SVR-07", "SVR-07", "", "web_dmz", 2, 0.17, 0.99, 5.9),
    ("SVR-03", "SVR-03", "", "router", 2, 0.44, 0.00, 7.5),
    ("SVR-06", "SVR-06", "", "router", 2, 0.52, 0.50, 6.8),
    ("SVR-08", "SVR-08", "", "workstation", 2, 0.39, 1.00, 6.1),
    ("SVR-05", "SVR-05", "", "domain_ctrl", 2, 0.65, 0.00, 6.9),
    ("SVR-12", "SVR-12", "", "domain_ctrl", 2, 0.72, 0.48, 4.2),
    ("SVR-09", "SVR-09", "", "workstation", 2, 0.61, 1.00, 6.5),
    ("SVR-10", "SVR-10", "", "domain_ctrl", 3, 0.85, 0.00, 5.4),
    ("FW-01", "FW-01", "GATEWAY", "gateway", 3, 0.82, 0.99, 5.1),
    ("DB-01", "CROWN JEWEL", "(DB-01)", "crown_jewel", 3, 1.00, 0.50, 9.8),
)

_CANONICAL_13_EDGES: tuple[Edge, ...] = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 4),
    (2, 4),
    (2, 5),
    (2, 3),
    (2, 6),
    (3, 6),
    (4, 7),
    (4, 5),
    (5, 7),
    (5, 8),
    (5, 6),
    (5, 9),
    (5, 11),
    (6, 9),
    (7, 10),
    (7, 8),
    (8, 10),
    (8, 12),
    (8, 11),
    (9, 11),
    (10, 12),
    (11, 12),
)


def build_canonical_13_network() -> Network:
    """Build the 13-machine reference enterprise network using core.network.Network."""
    hardness = [spec[4] for spec in _CANONICAL_13_SPECS]
    return Network.from_edges(
        size=len(_CANONICAL_13_SPECS),
        edges=_CANONICAL_13_EDGES,
        hardness=hardness,
        entry=0,
        jewel=12,
    )


def is_canonical_13_network(network: Network) -> bool:
    """Return True when network matches the canonical 13-machine topology."""
    if network.size != 13 or network.entry != 0 or network.jewel != 12:
        return False
    expected = build_canonical_13_network()
    return network.adjacency == expected.adjacency


def _cvss_for_hardness(hardness: int, index: int, role: str) -> float:
    if role == "crown_jewel":
        return 9.8
    if role == "entry":
        return 8.7
    base = 9.1 - hardness * 1.35 + ((index * 17) % 9) * 0.08
    return round(max(3.5, min(9.5, base)), 1)


def compute_network_layout(network: Network) -> tuple[NodeVisualMeta, ...]:
    """Compute clean, non-overlapping normalized [0..1] coordinates for any Network."""
    if is_canonical_13_network(network):
        return tuple(
            NodeVisualMeta(
                index=idx,
                code=spec[0],
                label=spec[1],
                sub_label=spec[2],
                role=spec[3],
                nx=spec[5],
                ny=spec[6],
                cvss=spec[7],
            )
            for idx, spec in enumerate(_CANONICAL_13_SPECS)
        )

    size = network.size
    entry = network.entry
    jewel = network.jewel
    dist_entry = bfs_distances(network.adjacency, entry)
    dist_jewel = bfs_distances(network.adjacency, jewel)

    # Place entry on the far left (layer 0), jewel on far right (max_layer),
    # and distribute intermediate nodes by shortest-hop distance from entry.
    max_entry_dist = max((d for i, d in enumerate(dist_entry) if i != jewel and d >= 0), default=1)
    mid_cols = max(2, min(5, max_entry_dist))

    layers: list[list[int]] = [[] for _ in range(mid_cols + 2)]
    layers[0].append(entry)
    layers[-1].append(jewel)

    for m in range(size):
        if m in (entry, jewel):
            continue
        d = max(1, dist_entry[m])
        # Spread nodes across interior columns 1..mid_cols proportionally.
        if max_entry_dist <= 1:
            col = 1
        else:
            ratio = (d - 1) / max(1, max_entry_dist - 1)
            col = 1 + min(mid_cols - 1, round(ratio * (mid_cols - 1)))
        layers[col].append(m)

    # Rebalance empty middle layers if any occurred.
    non_empty_mid = [col for col in layers[1:-1] if col]
    if not non_empty_mid:
        non_empty_mid = [[m for m in range(size) if m not in (entry, jewel)]]
    ordered_layers = [layers[0], *non_empty_mid, layers[-1]]
    n_cols = len(ordered_layers)

    # Initial vertical ordering: sort each column by (dist_jewel, -degree, id),
    # then refine with one barycenter pass from left to right to minimize link crossings.
    y_by_node: dict[int, float] = {entry: 0.5, jewel: 0.5}
    for col_idx in range(1, n_cols - 1):
        layer_nodes = ordered_layers[col_idx]
        layer_nodes.sort(key=lambda m: (dist_jewel[m], -len(network.neighbours(m)), m))
        count = len(layer_nodes)
        for row_idx, m in enumerate(layer_nodes):
            y_by_node[m] = 0.5 if count == 1 else row_idx / (count - 1)

    for _ in range(2):
        for col_idx in range(1, n_cols - 1):
            layer_nodes = ordered_layers[col_idx]

            def barycenter(node: int) -> tuple[float, int]:
                placed = [y_by_node[nb] for nb in network.neighbours(node) if nb in y_by_node]
                avg = sum(placed) / len(placed) if placed else y_by_node[node]
                return (avg, node)

            layer_nodes.sort(key=barycenter)
            count = len(layer_nodes)
            for row_idx, m in enumerate(layer_nodes):
                if count == 1:
                    y_by_node[m] = 0.5
                elif count == 2:
                    y_by_node[m] = 0.22 if row_idx == 0 else 0.78
                else:
                    y_by_node[m] = row_idx / (count - 1)

    # Build final NodeVisualMeta table in machine index order 0..size-1.
    nx_by_node: dict[int, float] = {}
    for col_idx, layer_nodes in enumerate(ordered_layers):
        base_nx = col_idx / max(1, n_cols - 1)
        count = len(layer_nodes)
        for row_idx, m in enumerate(layer_nodes):
            # Subtle horizontal stagger inside dense columns prevents collinear overlap.
            stagger = 0.0
            if 0 < col_idx < n_cols - 1 and count >= 3:
                stagger = -0.025 if row_idx % 2 == 0 else 0.025
            nx_by_node[m] = max(0.0, min(1.0, base_nx + stagger))

    # Pick one node directly adjacent to jewel (in last interior layer) as Gateway FW-01.
    gateway_node: int | None = None
    jewel_nbs = [nb for nb in network.neighbours(jewel) if nb != entry]
    if jewel_nbs:
        gateway_node = max(
            jewel_nbs, key=lambda nb: (nx_by_node.get(nb, 0.0), y_by_node.get(nb, 0.0))
        )

    role_by_col = ("web_dmz", "workstation", "router", "domain_ctrl", "gateway")
    col_lookup: dict[int, int] = {}
    for col_idx, layer_nodes in enumerate(ordered_layers):
        for m in layer_nodes:
            col_lookup[m] = col_idx

    metas: list[NodeVisualMeta] = []
    svr_counter = 2
    code_by_idx: dict[int, str] = {entry: "SVR-01", jewel: "DB-01"}
    if gateway_node is not None:
        code_by_idx[gateway_node] = "FW-01"
    for m in range(size):
        if m not in code_by_idx:
            code_by_idx[m] = f"SVR-{svr_counter:02d}"
            svr_counter += 1

    for m in range(size):
        code = code_by_idx[m]
        if m == entry:
            role = "entry"
            label = code
            sub_label = "ENTRY POINT"
        elif m == jewel:
            role = "crown_jewel"
            label = "CROWN JEWEL"
            sub_label = "(DB-01)"
        elif m == gateway_node:
            role = "gateway"
            label = code
            sub_label = "GATEWAY"
        else:
            c_idx = max(0, min(len(role_by_col) - 1, col_lookup[m] - 1))
            role = role_by_col[c_idx]
            label = code
            sub_label = ""

        metas.append(
            NodeVisualMeta(
                index=m,
                code=code,
                label=label,
                sub_label=sub_label,
                role=role,
                nx=nx_by_node[m],
                ny=y_by_node[m],
                cvss=_cvss_for_hardness(network.hardness[m], m, role),
            )
        )

    return tuple(metas)


def compute_safe_rect(
    rect: pygame.Rect,
    pad_left: int = 92,
    pad_right: int = 92,
    pad_top: int = 88,
    pad_bottom: int = 82,
) -> pygame.Rect:
    """Return the padded interior safe zone inside a battlefield/preview rect."""
    w = max(24, rect.width - pad_left - pad_right)
    h = max(24, rect.height - pad_top - pad_bottom)
    return pygame.Rect(rect.x + pad_left, rect.y + pad_top, w, h)


def project_node_to_screen(
    nx: float,
    ny: float,
    rect: pygame.Rect,
    pad_left: int = 92,
    pad_right: int = 92,
    pad_top: int = 88,
    pad_bottom: int = 82,
) -> tuple[int, int]:
    """Map normalized [0..1] node coordinates into a single unified screen point inside rect."""
    safe = compute_safe_rect(
        rect,
        pad_left=pad_left,
        pad_right=pad_right,
        pad_top=pad_top,
        pad_bottom=pad_bottom,
    )
    clamped_x = max(0.0, min(1.0, nx))
    clamped_y = max(0.0, min(1.0, ny))
    px = int(round(safe.x + clamped_x * safe.width))
    py = int(round(safe.y + clamped_y * safe.height))
    return (px, py)
