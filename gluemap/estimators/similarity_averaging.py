import logging

import numpy as np
import pyceres
import pygluemap
import torch

from gluemap.utils.rigs import (
    BoundRig,
    add_member_centers,
    compute_world_space_rig_offset,
    is_metric_edge,
)

logger = logging.getLogger(__name__)
# Minimum angle (in degrees) for a triangle to be considered valid for
# scale estimation
MIN_TRI_ANGLE = 1


def _initialize_parameters(
    predictions_dict: dict,
    global_rotations: dict[int, np.ndarray],
    global_centers: dict[int, np.ndarray] | None,
    global_scales: list[np.ndarray] | dict[int, float] | None,
) -> tuple[int, dict[int, np.ndarray], list[np.ndarray]]:
    """
    Allocate per-image centers and per-ministar scale parameter buffers.

    Args:
        predictions_dict: Star inference output; only ``"indexes"`` is read
            here to determine the number of ministars.
        global_rotations: Per-image rotation matrices ``{image_id: R}``.
        global_centers: Optional initial per-image camera centers; missing
            images get random initialization.
        global_scales: Optional initial per-ministar scales as a list, or as
            a ``{ministar_idx: scale}`` mapping; ``None`` defaults to ones.

    Returns:
        ``(num_ministar, global_centers, global_scales)`` where the latter
        two are normalized into the canonical containers used downstream.
    """
    num_ministar = len(predictions_dict["indexes"])
    if global_centers is None:
        global_centers = {
            idx: np.random.rand(3).astype(np.float64)
            for idx in global_rotations
        }
    else:
        for idx in global_rotations:
            if idx not in global_centers:
                global_centers[idx] = np.random.rand(3).astype(np.float64)
    if global_scales is None:
        global_scales = [
            np.ones((1,)).astype(np.float64) for i in range(num_ministar)
        ]
    elif isinstance(global_scales, dict):
        temp_scales = []
        for idx_star in range(num_ministar):
            if idx_star in global_scales:
                temp_scales.append(
                    np.ones((1,)).astype(np.float64) * global_scales[idx_star]
                )
            else:
                temp_scales.append(np.ones((1,)).astype(np.float64))
        global_scales = temp_scales

    return num_ministar, global_centers, global_scales


def _initialize_parameters_with_rig(
    predictions_dict: dict,
    global_rotations: dict[int, np.ndarray],
    global_centers: dict[int, np.ndarray] | None,
    global_scales: list[np.ndarray] | dict[int, float] | None,
    rig: BoundRig,
) -> tuple[int, dict[int, np.ndarray], list[np.ndarray]]:
    """Allocate per-reference centers and per-ministar scales."""
    num_ministar = len(predictions_dict["indexes"])
    references = sorted({rig.ref_of[idx] for idx in global_rotations})
    if global_centers is None:
        reference_centers = {
            r: np.random.rand(3).astype(np.float64) for r in references
        }
    else:
        reference_centers = {}
        for r in references:
            if r in global_centers:
                reference_centers[r] = np.asarray(
                    global_centers[r], dtype=np.float64
                )
            else:
                reference_centers[r] = np.random.rand(3).astype(np.float64)
    if global_scales is None:
        global_scales = [
            np.ones((1,)).astype(np.float64) for i in range(num_ministar)
        ]
    elif isinstance(global_scales, dict):
        temp_scales = []
        for idx_star in range(num_ministar):
            if idx_star in global_scales:
                temp_scales.append(
                    np.ones((1,)).astype(np.float64) * global_scales[idx_star]
                )
            else:
                temp_scales.append(np.ones((1,)).astype(np.float64))
        global_scales = temp_scales

    return num_ministar, reference_centers, global_scales


def _add_star_edge_error(
    prob: pyceres.Problem,
    predictions_dict: dict,
    global_rotations: dict[int, np.ndarray],
    global_centers: dict[int, np.ndarray],
    global_scales: list[np.ndarray],
    num_ministar: int,
    costs: list,
    losses: list,
) -> int:
    """
    Add pairwise direction residuals for every valid star edge.

    For each ministar's center image and its neighbours, attaches a
    ``PairwiseDirectionError`` cost block linking the two camera centers
    and the ministar scale. Edges with score <= 0 or median triangulation
    angle below :data:`MIN_TRI_ANGLE` are skipped. Pins the first chosen
    center and its ministar's scale to remove the gauge.

    Args:
        prob: Ceres problem to append residuals to.
        predictions_dict: Star inference output with ``"indexes"``,
            ``"pose_scores"``, ``"median_tri_angle"`` and ``"extrinsics"``.
        global_rotations: Per-image rotation matrices.
        global_centers: Per-image camera centers (parameter blocks).
        global_scales: Per-ministar scale parameter blocks.
        num_ministar: Number of ministars to iterate over.
        costs: Output list to which created cost objects are appended (kept
            alive by the caller).
        losses: Output list of created loss functions (kept alive too).

    Returns:
        The image id whose center was fixed for gauge, or ``-1`` if no edge
        was added.
    """
    center = -1
    for idx_star in range(num_ministar):
        scores = predictions_dict["pose_scores"][idx_star][0]
        idx1 = predictions_dict["indexes"][idx_star][0]
        valid_j = torch.where(scores > 0.0)[0].tolist()

        # Now, only consider the scale of the center image
        for idx in valid_j:
            idx2 = predictions_dict["indexes"][idx_star][idx]

            if idx1 == idx2:
                continue

            if (
                predictions_dict["median_tri_angle"][idx_star][idx - 1].item()
                < MIN_TRI_ANGLE
            ):
                continue  # Skip low-confidence edges

            # s_i * (c_j - c_i) = -R_j^T * t_ij
            t_ij_rotated = (
                -global_rotations[idx2].T
                @ predictions_dict["extrinsics"][idx_star][0, idx, :3, 3:]
                .cpu()
                .numpy()
            )
            loss_scaled = pyceres.LossFunction(
                {"name": "huber", "params": [1e-2], "magnitude": scores[idx]}
            )

            cost = pygluemap.PairwiseDirectionError(t_ij_rotated)
            prob.add_residual_block(
                cost,
                loss_scaled,
                [
                    global_centers[idx1],
                    global_centers[idx2],
                    global_scales[idx_star],
                ],
            )

            costs.append(cost)
            losses.append(loss_scaled)

            if center < 0:
                prob.set_parameter_block_constant(global_centers[idx1])
                prob.set_parameter_block_constant(global_scales[idx_star])
                center = idx1

    return center

def _add_star_edge_error_with_rig(
    prob: pyceres.Problem,
    predictions_dict: dict,
    global_rotations: dict[int, np.ndarray],
    global_centers: dict[int, np.ndarray],
    global_scales: list[np.ndarray],
    num_ministar: int,
    costs: list,
    losses: list,
    rig: BoundRig,
    metric_stars: set,
) -> int:
    """
    Add pairwise direction residuals for every valid star edge with rig.

    For each ministar's center image and its neighbours, attaches a
    ``PairwiseDirectionError`` cost block linking the two camera centers
    and the ministar scale. Edges with score <= 0 or median triangulation
    angle below :data:`MIN_TRI_ANGLE` are skipped. Pins the first chosen
    center and its ministar's scale to remove the gauge.

    Args:
        prob: Ceres problem to append residuals to.
        predictions_dict: Star inference output with ``"indexes"``,
            ``"pose_scores"``, ``"median_tri_angle"`` and ``"extrinsics"``.
        global_rotations: Per-image rotation matrices.
        global_centers: Per-reference camera centers (parameter blocks).
        global_scales: Per-ministar scale parameter blocks.
        num_ministar: Number of ministars to iterate over.
        costs: Output list to which created cost objects are appended (kept
            alive by the caller).
        losses: Output list of created loss functions (kept alive too).
        rig: Holds the known relationships
        metric_stars: Output set; ministar indices given a scale-only
            (calibration) edge are added here.

    Returns:
        The image id whose center was fixed for gauge, or ``-1`` if no edge
        was added.
    """
    center = -1
    first_scale_star = None
    for idx_star in range(num_ministar):
        scores = predictions_dict["pose_scores"][idx_star][0]
        idx1 = predictions_dict["indexes"][idx_star][0]
        valid_j = torch.where(scores > 0.0)[0].tolist()

        # Now, only consider the scale of the center image
        for idx in valid_j:
            idx2 = predictions_dict["indexes"][idx_star][idx]

            if idx1 == idx2:
                continue

            is_intra = rig.ref_of[idx1] == rig.ref_of[idx2]
            metric = is_metric_edge(rig, idx1, idx2)  # metric => intra

            # intra-rig with a shared optical centre carries no scale info
            if is_intra and not metric:
                continue

            # inter-rig edges obey the tri-angle gate; metric edges bypass it
            # (their scale comes from calibration, not triangulation)
            if not is_intra and (
                predictions_dict["median_tri_angle"][idx_star][idx - 1].item()
                < MIN_TRI_ANGLE
            ):
                continue

            # Let r1 be the ref of i & let r2 be the ref of j
            # delta_i = world space offset of i wrt i1, same for j
            # Substitute: c_i = c_r1 + delta_i, same for j: 
            #    s_i * (c_j - c_i) = -R_j^T * t_ij
            # => s_i * (c_r2 - c_r1 + (delta_j - delta_i)) = ...
            t_ij_rotated = (
                -global_rotations[idx2].T
                @ predictions_dict["extrinsics"][idx_star][0, idx, :3, 3:]
                .cpu()
                .numpy()
            )
            # computing (delta_j - delta_i)
            offset = (
                compute_world_space_rig_offset(rig, global_rotations, idx2)
                - compute_world_space_rig_offset(rig, global_rotations, idx1)
            )
            loss_scaled = pyceres.LossFunction(
                {"name": "huber", "params": [1e-2], "magnitude": scores[idx]}
            )

            if metric: # by default also intra
                # centers cancel; only the scale is optimized
                cost = pygluemap.ScaleOnlyDirectionError(t_ij_rotated, offset)
                prob.add_residual_block(
                    cost,
                    loss_scaled,
                    [global_scales[idx_star]],
                )
                metric_stars.add(idx_star)
            else: # since not metric, intra can't be true here, because of continue above.
                # centers also optimized
                cost = pygluemap.PairwiseDirectionErrorWithOffset(
                    t_ij_rotated, offset
                )
                prob.add_residual_block(
                    cost,
                    loss_scaled,
                    [
                        global_centers[rig.ref_of[idx1]],
                        global_centers[rig.ref_of[idx2]],
                        global_scales[idx_star],
                    ],
                )

            costs.append(cost)
            losses.append(loss_scaled)

            if center < 0 and not is_intra:
                prob.set_parameter_block_constant(
                    global_centers[rig.ref_of[idx1]]
                )
                center = idx1
            if first_scale_star is None:
                first_scale_star = idx_star

    # scale gauge: pin one scale only when no metric edge fixes it
    if not metric_stars and first_scale_star is not None:
        prob.set_parameter_block_constant(global_scales[first_scale_star])

    return center

def _update_points3d(
    prob: pyceres.Problem,
    predictions_dict: dict,
    global_rotations: dict[int, np.ndarray],
    global_centers: dict[int, np.ndarray],
    global_scales: list[np.ndarray],
    num_ministar: int,
) -> None:
    """
    Apply solved similarity transforms back to predicted 3D content.

    Rescales/rotates ``world_points``, ``cam_points`` and
    ``points3d_virtual`` (when present) and rescales ministar extrinsics so
    everything is expressed in the new global frame. Mutates
    ``predictions_dict`` in place.

    Args:
        prob: Ceres problem used to detect which ministar scales were
            actually parameter blocks (others are skipped).
        predictions_dict: Star inference dict; relevant tensors are updated
            in place.
        global_rotations: Per-image rotation matrices used to rotate world
            points back into the global frame.
        global_centers: Per-image solved centers used as translations.
        global_scales: Per-ministar solved scales (length-1 numpy arrays).
        num_ministar: Number of ministars to process.
    """
    if (
        "world_points" in predictions_dict
        and len(predictions_dict["world_points"]) > 0
    ):
        # Rescale the world points
        for idx in range(num_ministar):
            idx_center = predictions_dict["indexes"][idx][0]
            predictions_dict["world_points"][idx] = (
                predictions_dict["world_points"][idx]
                @ global_rotations[idx_center]
                / global_scales[idx]
                + global_centers[idx_center]
            )
    # Since it is in camera frame, we only need to change the scale
    if (
        "cam_points" in predictions_dict
        and len(predictions_dict["cam_points"]) > 0
        and predictions_dict["cam_points"][0] is not None
    ):
        for idx in range(num_ministar):
            idx_center = predictions_dict["indexes"][idx][0]
            predictions_dict["cam_points"][idx] = (
                predictions_dict["cam_points"][idx] / global_scales[idx]
            )

    # Rescale the points and extrinsics
    if (
        "points3d_virtual" in predictions_dict
        and len(predictions_dict["points3d_virtual"]) > 0
    ):
        # Rescale the virtual points
        for idx in range(num_ministar):
            predictions_dict["points3d_virtual"][idx] = (
                predictions_dict["points3d_virtual"][idx] / global_scales[idx]
            )

    for idx_star in range(num_ministar):
        if not prob.has_parameter_block(global_scales[idx_star]):
            continue
        predictions_dict["extrinsics"][idx_star][:, :, :3, 3:] = (
            predictions_dict["extrinsics"][idx_star][:, :, :3, 3:]
            / global_scales[idx_star][0]
        )


def similarity_averaging(
    predictions_dict: dict,
    global_rotations: dict[int, np.ndarray],
    global_centers: dict[int, np.ndarray] | None = None,
    global_scales: list[np.ndarray] | dict[int, float] | None = None,
    max_num_iterations: int = 50,
    fix_scales: bool = False,
    rig: BoundRig | None = None,
) -> dict[int, np.ndarray]:
    """
    Solve for global per-image translations and per-ministar scales.

    Builds a Ceres problem from pairwise direction residuals over all valid
    ministar edges, fixes the gauge, solves, then back-substitutes the
    similarity transform into ``predictions_dict``.

    Args:
        predictions_dict: Star inference output; mutated in place by
            :func:`_update_points3d`.
        global_rotations: Per-image rotations (assumed already solved by
            rotation averaging).
        global_centers: Optional initial centers; missing entries are
            randomized via :func:`_initialize_parameters`.
        global_scales: Optional initial scales (list or dict).
        max_num_iterations: Max Ceres iterations.
        fix_scales: If True, freeze every ministar scale (useful when only
            translations should move).
        rig: Optional bound rig spec.

    Returns:
        The updated ``global_centers`` mapping (also written in place).
    """
    logger.info("Performing similarity averaging...")
    if rig is None:
        num_ministar, global_centers, global_scales = _initialize_parameters(
            predictions_dict,
            global_rotations,
            global_centers,
            global_scales,
        )
    else:
        num_ministar, global_centers, global_scales = (
            _initialize_parameters_with_rig(
                predictions_dict,
                global_rotations,
                global_centers,
                global_scales,
                rig,
            )
        )

    prob = pyceres.Problem()

    costs = []
    losses = []
    metric_stars = set()

    if rig is None :
        _add_star_edge_error(
            prob,
            predictions_dict,
            global_rotations,
            global_centers,
            global_scales,
            num_ministar,
            costs,
            losses,
        )
    else : 
        _add_star_edge_error_with_rig(
            prob,
            predictions_dict,
            global_rotations,
            global_centers,
            global_scales,
            num_ministar,
            costs,
            losses,
            rig,
            metric_stars,
        )

    for idx_star in range(num_ministar):
        if not prob.has_parameter_block(global_scales[idx_star]):
            continue
        if fix_scales or (
            idx_star not in metric_stars
            and predictions_dict["median_tri_angle"][idx_star].max().item()
            < MIN_TRI_ANGLE
        ):
            prob.set_parameter_block_constant(global_scales[idx_star])
        else:
            prob.set_parameter_lower_bound(global_scales[idx_star], 0, 1e-5)
            prob.set_parameter_upper_bound(global_scales[idx_star], 0, 1e5)

    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
    options.num_threads = 32
    options.max_num_iterations = max_num_iterations
    options.minimizer_progress_to_stdout = False

    logger.info("Solving the optimization problem...")
    summary = pyceres.SolverSummary()
    pyceres.solve(options, prob, summary)

    logger.info(summary.BriefReport())

    if rig is not None:
        global_centers = add_member_centers(
            global_centers, global_rotations, rig
        )

    _update_points3d(
        prob,
        predictions_dict,
        global_rotations,
        global_centers,
        global_scales,
        num_ministar,
    )

    return global_centers
