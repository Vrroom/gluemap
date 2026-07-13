"""Maximum-spanning-tree initialization of global camera centres and scales.

Given the per-star relative-pose predictions and an initial estimate of the
global rotations, this module builds a graph over images, picks an MST that
maximises the per-edge confidence, and walks the tree to seed each image's
global centre and scale before similarity averaging refines them.
"""

from collections.abc import Callable

import networkx as nx
import numpy as np
import torch

from gluemap.utils.rigs import (
    BoundRig,
    add_member_centers,
    compute_world_space_rig_offset,
    fold_graph,
)

# Minimum median triangulation angle (degrees) for an edge's relative scale
# to be considered reliable; below this we fall back to a unit scale ratio.
MIN_TRI_ANGLE = 1


def initialize_mst_structures(
    predictions_dict: dict,
    global_rotations: dict[int, np.ndarray],
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    """Seed global centres and scales by walking an MST over star edges.

    The function reads ``predictions_dict["indexes"]``,
    ``["points3d_virtual"]``, ``["extrinsics"]``, and ``["pose_scores"]``,
    and writes ``predictions_dict["median_tri_angle"]`` as a side effect.
    Edges whose median triangulation angle is below ``MIN_TRI_ANGLE`` get a
    unit relative scale; otherwise the relative scale is the ratio of
    translation norms between the two directions of the edge.

    Args:
        predictions_dict: Per-star prediction dict produced upstream. Must
            contain ``"indexes"``, ``"points3d_virtual"``, ``"extrinsics"``,
            and ``"pose_scores"``. ``"median_tri_angle"`` is added in place.
        global_rotations: Mapping image-index → ``(3, 3)`` rotation matrix.

    Returns:
        Tuple of:
            * ``global_centers`` keyed by image index, each ``(3,)`` float64.
            * ``global_scales`` keyed by star index.
    """
    N = max(global_rotations.keys()) + 1
    indexes = range(len(predictions_dict["indexes"]))

    rel_poses = {}
    scales = {}  # (i,j): s_j / s_i
    node_idx_to_star_idx = {}
    predictions_dict["median_tri_angle"] = {}

    for star_idx, idx in enumerate(indexes):
        node_idx_to_star_idx[predictions_dict["indexes"][idx][0]] = star_idx

        # Compute max of median triangulation angle across edges
        points3d = predictions_dict["points3d_virtual"][idx][0]  # (K, 3)
        extr = predictions_dict["extrinsics"][idx]  # (1, N, 3, 4)

        ray_center = points3d / torch.clamp(
            points3d.norm(dim=-1, keepdim=True), min=1e-8
        )  # (K, 3)

        # Vectorized over all neighbor views
        R_all = extr[0, 1:, :3, :3]  # (M, 3, 3)
        t_all = extr[0, 1:, :3, 3]  # (M, 3)
        c_all = -torch.einsum("mji,mj->mi", R_all, t_all)  # (M, 3)

        ray_all = points3d.unsqueeze(0) - c_all.unsqueeze(1)  # (M, K, 3)
        ray_all = ray_all / torch.clamp(
            ray_all.norm(dim=-1, keepdim=True), min=1e-8
        )
        cos_angles = torch.clamp(
            torch.einsum("kd,mkd->mk", ray_center, ray_all),
            -1.0 + 1e-6,
            1.0 - 1e-6,
        )  # (M, K)
        angles = torch.acos(cos_angles)  # (M, K)
        median_angles = (
            angles.median(dim=-1).values
            if angles.numel() > 0
            else torch.tensor([])
        )  # (M,)

        predictions_dict["median_tri_angle"][idx] = (
            np.rad2deg(median_angles)
            if median_angles.numel() > 0
            else np.array([])
        )

    for idx in indexes: # means star index
        poses = predictions_dict["extrinsics"][idx]
        pose_scores = predictions_dict["pose_scores"][idx]
        N_poses = poses.shape[1]
        idx_i = predictions_dict["indexes"][idx][0]
        for i in range(N_poses):
            if i == 0:
                continue
            idx_j = predictions_dict["indexes"][idx][i]
            status = (idx_j, idx_i) not in rel_poses
            star_idx_i = node_idx_to_star_idx[idx_i]
            star_idx_j = node_idx_to_star_idx[idx_j]
            score = pose_scores[0, i].item()
            if not status:
                # If either edge has a small triangulation angle, scale is
                # unreliable
                angle_current = predictions_dict["median_tri_angle"][
                    star_idx_i
                ][i - 1]
                reverse_pos = rel_poses[(idx_j, idx_i)][2]
                angle_reverse = predictions_dict["median_tri_angle"][
                    star_idx_j
                ][reverse_pos - 1]
                if (
                    angle_current < MIN_TRI_ANGLE
                    or angle_reverse < MIN_TRI_ANGLE
                ):
                    scales[(star_idx_j, star_idx_i)] = 1.0
                    scales[(star_idx_i, star_idx_j)] = 1.0
                else:
                    scales[(star_idx_j, star_idx_i)] = (
                        poses[0, i, :3, 3:].norm().item()
                        / rel_poses[(idx_j, idx_i)][0][:3, 3:].norm().item()
                    )
                    scales[(star_idx_i, star_idx_j)] = (
                        1.0 / scales[(star_idx_j, star_idx_i)]
                    )
                score *= (
                    min(20, angle_current, angle_reverse) / 20
                )  # downweight the edge if the triangulation angle is small

            rel_poses[(idx_i, idx_j)] = (poses[0, i].cpu(), idx, i, score)

    # Only consider the two side edges
    invalid_edges = []
    for (i, j), _ in rel_poses.items():
        if (node_idx_to_star_idx[i], node_idx_to_star_idx[j]) not in scales:
            invalid_edges.append((i, j))

    for i, j in invalid_edges:
        del rel_poses[(i, j)]

    G = nx.Graph()
    G.add_nodes_from(np.arange(N))
    for (i, j), (_pose, _idx, _i_pos, score) in rel_poses.items():
        G.add_edge(i, j, weight=score)
    nx.set_edge_attributes(
        G,
        {(i, j): score for (i, j), (_, _, _, score) in rel_poses.items()},
        "weight",
    )
    mst = nx.maximum_spanning_tree(G)

    global_centers = {}
    global_scales = {}

    global_centers[0] = np.zeros((3,), dtype=np.float64)
    global_scales[node_idx_to_star_idx[0]] = 1.0

    def visit(node, neighbor):
        idx_node = node_idx_to_star_idx[neighbor]
        idx_parent = node_idx_to_star_idx[node]
        pose, idx, i_pos, _ = rel_poses[(neighbor, node)]

        if idx_node in global_scales:
            global_scales[idx_parent] = (
                global_scales[idx_node] * scales[(idx_node, idx_parent)]
            )
        else:
            global_scales[idx_node] = (
                global_scales[idx_parent] * scales[(idx_parent, idx_node)]
            )

        # s_i * (c_j - c_i) = -R_j^T * t_ij
        # ==> c_i = c_j + R_j^T * t_ij / s_i
        global_centers[neighbor] = (
            global_centers[node]
            + (global_rotations[node].T @ pose[:3, 3:].numpy()).flatten()
            / global_scales[idx_node]
        )

    walk_tree(mst, 0, visit)

    return global_centers, global_scales


def walk_tree(
    tree: nx.Graph, root: int, visit: Callable[[int, int], None]
) -> None:
    """Depth-first walk of a tree from ``root``, calling ``visit(parent, child)``
    exactly once per tree edge, parents before children. Iterative to avoid
    RecursionError on large graphs."""
    visited = {root}
    stack = [(root, iter(tree.neighbors(root)))]

    while stack:
        node, neighbors_iter = stack[-1]
        try:
            neighbor = next(neighbors_iter)
        except StopIteration:
            stack.pop()
            continue

        if neighbor in visited:
            continue

        visited.add(neighbor)
        visit(node, neighbor)
        stack.append((neighbor, iter(tree.neighbors(neighbor))))


def find_metric_anchors(
    rel_poses: dict, rig: BoundRig, components: list[set[int]]
) -> list[tuple[bool, int, float]]:
    """Per-component metric anchor, aligned with ``components``.

    A component is metric when one of its stars observed a same-frame rig pair
    with nonzero baseline; its entry is ``(True, root_node, root_scale)`` where
    root_node is that star's center image and root_scale the star-over-metric
    ratio ``measured_baseline / known_baseline``. Components without such an
    edge get ``(False, min(component), 1.0)``: their scale is a free gauge.
    """
    node_to_component = {n: k for k, comp in enumerate(components) for n in comp}
    anchors: list[tuple[bool, int, float] | None] = [None] * len(components)
    for (i, j), (pose, _star_idx, _slot, _score) in rel_poses.items():
        if i == j or rig.ref_of[i] != rig.ref_of[j]:
            continue
        k = node_to_component[i]
        if anchors[k] is not None:
            continue
        Mi = rig.sensor_from_ref_of(i)
        Mj = rig.sensor_from_ref_of(j)
        ci = -Mi[:3, :3].T @ Mi[:3, 3]
        cj = -Mj[:3, :3].T @ Mj[:3, 3]
        L = np.linalg.norm(ci - cj)
        if L <= 1e-6:
            continue
        anchors[k] = (True, i, float(np.linalg.norm(pose[:3, 3].numpy()) / L))
    return [
        a if a is not None else (False, min(comp), 1.0)
        for a, comp in zip(anchors, components)
    ]


def seed_star_scales(
    G: nx.Graph,
    scales: dict,
    node_idx_to_star_idx: dict,
    components: list[set[int]],
    anchors: list[tuple[bool, int, float]],
) -> tuple[dict[int, float], set[int]]:
    """Give every star a global scale, even on a disconnected star graph.

    Each connected component is seeded at its anchor (the measured metric
    scale when the component contains a rig baseline, 1.0 otherwise) and the
    measured pairwise ratios are chained outward through the component's
    spanning tree. Components without a metric anchor are then rescaled by
    the first metric component's anchor scale, so their overall size is at
    least in metric ballpark; that overall size is unobserved (no edge ties
    the components together), so this is a gauge choice, not a measurement.
    When no component is metric everything stays in its own 1.0 gauge.

    Returns ``(global_scales, metric_stars)`` where metric_stars holds the
    stars of metric components; all other stars carry a chosen gauge, not
    a measured link to metric.
    """
    assert len(components) == len(anchors), (
        f"(seed_star_scales): {len(components)} components but "
        f"{len(anchors)} anchors"
    )
    forest = nx.maximum_spanning_tree(G)

    global_scales = {}

    def visit(node, neighbor):
        idx_node = node_idx_to_star_idx[neighbor]
        idx_parent = node_idx_to_star_idx[node]

        if idx_node in global_scales:
            global_scales[idx_parent] = (
                global_scales[idx_node] * scales[(idx_node, idx_parent)]
            )
        else:
            global_scales[idx_node] = (
                global_scales[idx_parent] * scales[(idx_parent, idx_node)]
            )

    # Seed each component's root with its anchor scale and chain the measured
    # ratios outward; the forest walk stays inside the root's component.
    for _is_metric, root, root_scale in anchors:
        global_scales[node_idx_to_star_idx[root]] = root_scale
        walk_tree(forest, root, visit)

    metric_stars = {
        node_idx_to_star_idx[n]
        for (is_metric, _root, _s), comp in zip(anchors, components)
        if is_metric
        for n in comp
    }

    # A component without a metric anchor has an unobserved overall scale.
    # Rescale it by the first metric component's anchor scale so its gauge
    # lands in metric ballpark (no-op when nothing is metric).
    metric_anchors = [a for a in anchors if a[0]]
    if metric_anchors:
        S = metric_anchors[0][2]
        for (is_metric, _root, _s), comp in zip(anchors, components):
            if is_metric:
                continue
            for n in comp:
                global_scales[node_idx_to_star_idx[n]] *= S

    return global_scales, metric_stars


def walk_reference_centers(
    rel_poses: dict,
    global_rotations: dict[int, np.ndarray],
    global_scales: dict[int, float],
    rig: BoundRig,
    metric_stars: set[int],
) -> dict[int, np.ndarray]:
    """Walk the folded reference MST, setting each reference centre from its
    parent's. ``metric_stars`` marks the stars whose scale is measured against
    a rig baseline; when a reference pair offers both, their poses are
    preferred over gauge-scaled ones for placement."""
    # Build the image graph, edges weighted by pose score
    G = nx.Graph()
    G.add_nodes_from(rig.ref_of.keys())
    for (i, j), (_pose, _idx, _i_pos, score) in rel_poses.items():
        G.add_edge(i, j, weight=score, pair=(i, j))
    nx.set_edge_attributes(
        G,
        {(i, j): score for (i, j), (_, _, _, score) in rel_poses.items()},
        "weight",
    )

    # Collapse each image onto its rig reference
    folded_multigraph = fold_graph(G, rig.ref_of)

    # Keep one edge per reference pair. The tree "weight" stays the best score
    # seen for the pair, so the spanning tree is unchanged by this choice. The
    # pose used for placement ("pair") prefers an edge whose star scale was
    # measured against a rig baseline over one carrying a chosen gauge, and
    # breaks ties by score: placing a frame with a gauge scale when a measured
    # one exists would throw away a real measurement.
    folded_graph = nx.Graph()
    folded_graph.add_nodes_from(folded_multigraph.nodes())
    for ra, rb, data in folded_multigraph.edges(data=True):
        star = rel_poses[data["pair"]][1]
        candidate = (star in metric_stars, data["weight"])
        if not folded_graph.has_edge(ra, rb):
            folded_graph.add_edge(
                ra,
                rb,
                weight=data["weight"],
                pair=data["pair"],
                choice=candidate,
            )
            continue
        edge = folded_graph[ra][rb]
        edge["weight"] = max(edge["weight"], data["weight"])
        if candidate > edge["choice"]:
            edge["pair"] = data["pair"]
            edge["choice"] = candidate

    assert nx.is_connected(folded_graph), (
        f"(walk_reference_centers): folded reference graph is disconnected, "
        f"{nx.number_connected_components(folded_graph)} components"
    )

    # Max spanning tree over the references
    mst = nx.maximum_spanning_tree(folded_graph)

    root = rig.ref_of[0] 

    # Walk each reference from its parent: c_child = c_parent + t_hat/s + delta_i - delta_j
    global_centers = {}
    global_centers[root] = np.zeros((3,), dtype=np.float64)

    def visit(node, neighbor):
        i, j = mst[node][neighbor]["pair"]
        if rig.ref_of[i] != node:
            i, j = j, i

        # Diagram in my head: node -> i -> j -> neighbor
        pose, star, _slot, _score = rel_poses[(i, j)]
        t_hat = -global_rotations[j].T @ pose[:3, 3].numpy()
        s = global_scales[star]

        delta_i = compute_world_space_rig_offset(rig, global_rotations, i)
        delta_j = compute_world_space_rig_offset(rig, global_rotations, j)

        global_centers[neighbor] = (
            global_centers[node] + t_hat / s + delta_i - delta_j
        )

    walk_tree(mst, root, visit)

    return global_centers

def initialize_mst_structures_with_rig(
    predictions_dict: dict,
    global_rotations: dict[int, np.ndarray],
    rig: BoundRig,
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    """Rig-based variant of :func:`initialize_mst_structures`."""
    indexes = range(len(predictions_dict["indexes"]))

    rel_poses = {}
    scales = {}  # (i,j): s_j / s_i
    node_idx_to_star_idx = {}
    predictions_dict["median_tri_angle"] = {}

    for star_idx, idx in enumerate(indexes):
        node_idx_to_star_idx[predictions_dict["indexes"][idx][0]] = star_idx

        # Compute max of median triangulation angle across edges
        points3d = predictions_dict["points3d_virtual"][idx][0]  # (K, 3)
        extr = predictions_dict["extrinsics"][idx]  # (1, N, 3, 4)

        ray_center = points3d / torch.clamp(
            points3d.norm(dim=-1, keepdim=True), min=1e-8
        )  # (K, 3)

        # Vectorized over all neighbor views
        R_all = extr[0, 1:, :3, :3]  # (M, 3, 3)
        t_all = extr[0, 1:, :3, 3]  # (M, 3)
        c_all = -torch.einsum("mji,mj->mi", R_all, t_all)  # (M, 3)

        ray_all = points3d.unsqueeze(0) - c_all.unsqueeze(1)  # (M, K, 3)
        ray_all = ray_all / torch.clamp(
            ray_all.norm(dim=-1, keepdim=True), min=1e-8
        )
        cos_angles = torch.clamp(
            torch.einsum("kd,mkd->mk", ray_center, ray_all),
            -1.0 + 1e-6,
            1.0 - 1e-6,
        )  # (M, K)
        angles = torch.acos(cos_angles)  # (M, K)
        median_angles = (
            angles.median(dim=-1).values
            if angles.numel() > 0
            else torch.tensor([])
        )  # (M,)

        predictions_dict["median_tri_angle"][idx] = (
            np.rad2deg(median_angles)
            if median_angles.numel() > 0
            else np.array([])
        )

    for idx in indexes: # means star index
        poses = predictions_dict["extrinsics"][idx]
        pose_scores = predictions_dict["pose_scores"][idx]
        N_poses = poses.shape[1]
        idx_i = predictions_dict["indexes"][idx][0]
        for i in range(N_poses):
            if i == 0:
                continue
            idx_j = predictions_dict["indexes"][idx][i]
            status = (idx_j, idx_i) not in rel_poses
            star_idx_i = node_idx_to_star_idx[idx_i]
            star_idx_j = node_idx_to_star_idx[idx_j]
            score = pose_scores[0, i].item()
            if not status:
                # If either edge has a small triangulation angle, scale is
                # unreliable
                angle_current = predictions_dict["median_tri_angle"][
                    star_idx_i
                ][i - 1]
                reverse_pos = rel_poses[(idx_j, idx_i)][2]
                angle_reverse = predictions_dict["median_tri_angle"][
                    star_idx_j
                ][reverse_pos - 1]
                if (
                    angle_current < MIN_TRI_ANGLE
                    or angle_reverse < MIN_TRI_ANGLE
                ):
                    scales[(star_idx_j, star_idx_i)] = 1.0
                    scales[(star_idx_i, star_idx_j)] = 1.0
                else:
                    scales[(star_idx_j, star_idx_i)] = (
                        poses[0, i, :3, 3:].norm().item()
                        / rel_poses[(idx_j, idx_i)][0][:3, 3:].norm().item()
                    )
                    scales[(star_idx_i, star_idx_j)] = (
                        1.0 / scales[(star_idx_j, star_idx_i)]
                    )
                # A small triangulation angle makes the pair unreliable. That
                # is a property of the pair, not of one direction, so the
                # penalty must land on both stored directions. The reverse
                # entry was inserted before the angles were known; update it.
                factor = float(min(20, angle_current, angle_reverse) / 20)
                score *= factor
                if factor < 1.0:
                    rev = rel_poses[(idx_j, idx_i)]
                    rel_poses[(idx_j, idx_i)] = (*rev[:3], rev[3] * factor)

            rel_poses[(idx_i, idx_j)] = (poses[0, i].cpu(), idx, i, score)

    # Only consider the two side edges
    invalid_edges = []
    for (i, j), _ in rel_poses.items():
        if (node_idx_to_star_idx[i], node_idx_to_star_idx[j]) not in scales:
            invalid_edges.append((i, j))

    for i, j in invalid_edges:
        del rel_poses[(i, j)]

    # Graph over images: an edge means a relative pose between the two images
    # survived filtering. When matching is sparse this graph can split into
    # several disconnected components; each component is scale-seeded on its
    # own, so downstream code must not assume the graph is connected.
    G = nx.Graph()
    G.add_nodes_from(node_idx_to_star_idx.keys())
    for (i, j), (_pose, _idx, _i_pos, score) in rel_poses.items():
        G.add_edge(i, j, weight=score)

    components = list(nx.connected_components(G))
    anchors = find_metric_anchors(rel_poses, rig, components)
    global_scales, metric_stars = seed_star_scales(
        G, scales, node_idx_to_star_idx, components, anchors
    )
    reference_centers = walk_reference_centers(
        rel_poses, global_rotations, global_scales, rig, metric_stars
    )
    global_centers = add_member_centers(reference_centers, global_rotations, rig)

    return global_centers, global_scales
