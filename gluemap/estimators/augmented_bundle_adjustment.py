import logging

import numpy as np
import pyceres
import pycolmap
import pygluemap

from gluemap.utils.rigs import BoundRig

logger = logging.getLogger(__name__)


def _update_poses_from_reconstruction(
    source_recon: pycolmap.Reconstruction,
    target_recon: pycolmap.Reconstruction,
) -> None:
    """
    Copy BA-optimized poses and camera intrinsics from source to target
    reconstruction. Matches images by name.

    We propagate the rig reference poses. This works because when there is 
    no rig, everyone is a rig. 
    """
    source_by_name = {
        img.name: (img_id, img) for img_id, img in source_recon.images.items()
    }
    for target_id, target_img in target_recon.images.items():
        if not target_img.is_ref_in_frame():
            continue
        if target_img.name in source_by_name:
            src_id, src_img = source_by_name[target_img.name]
            target_recon.frames[
                target_img.frame_id
            ].rig_from_world = src_img.cam_from_world()
    # Copy camera intrinsics
    for cam_id, cam in source_recon.cameras.items():
        if cam_id in target_recon.cameras:
            target_recon.cameras[cam_id].params = cam.params


def _pycolmap_loss_type(name: str) -> pycolmap.LossFunctionType:
    """
    Map a loss type name to a pycolmap.LossFunctionType enum value.

    Args:
        name: One of ``"trivial"``, ``"huber"``, ``"cauchy"``.

    Returns:
        Matching ``pycolmap.LossFunctionType`` enum value.
    """
    mapping = {
        "trivial": pycolmap.LossFunctionType.TRIVIAL,
        "huber": pycolmap.LossFunctionType.HUBER,
        "cauchy": pycolmap.LossFunctionType.CAUCHY,
    }
    if name not in mapping:
        raise ValueError(
            f"Unknown loss type '{name}', "
            f"expected one of {list(mapping.keys())}"
        )
    return mapping[name]


def _pyceres_loss_function(name: str) -> pyceres.LossFunction | None:
    """
    Map a loss type name to a pyceres.LossFunction (or None for trivial).

    Args:
        name: One of ``"trivial"``, ``"huber"``, ``"arctan"``, ``"cauchy"``.

    Returns:
        Configured ``pyceres.LossFunction``, or ``None`` for the trivial
        (squared) loss.
    """
    configs = {
        "trivial": None,
        "huber": {"name": "huber", "params": [1.0], "magnitude": 1.0},
        "arctan": {"name": "arctan", "params": [5.0], "magnitude": 1.0},
        "cauchy": {"name": "cauchy", "params": [1.0], "magnitude": 1.0},
    }
    if name not in configs:
        raise ValueError(
            f"Unknown loss type '{name}', "
            f"expected one of {list(configs.keys())}"
        )
    cfg = configs[name]
    return pyceres.LossFunction(cfg) if cfg is not None else None


# Sentinel: when callers pass no explicit loss_function, fall back to Arctan.
# ``None`` itself is a valid Ceres value (trivial / squared loss), so we need
# a distinct sentinel to distinguish "caller wants trivial" from "caller wants
# the default".
_DEFAULT_LOSS = object()

def _add_virtual_track_residuals(
    problem: pyceres.Problem,
    virtual_reconstruction: pycolmap.Reconstruction | None,
    reference_reconstruction: pycolmap.Reconstruction,
    negative_depth_observations: dict[int, set[int]],
    loss_function: pyceres.LossFunction | None | object = _DEFAULT_LOSS,
) -> None:
    """
    Add reprojection residuals for virtual tracks to an existing ceres problem. (reconstruction used a rig)

    Pose and intrinsic parameter blocks are resolved through
    ``reference_reconstruction`` (the real reconstruction handed to
    ``pycolmap.create_default_ceres_bundle_adjuster``) so that their numpy
    buffers are the same ones pycolmap is already optimizing -- the virtual
    residuals thus contribute to the same parameter blocks rather than
    detached copies.

    Virtual points3D are read from ``virtual_reconstruction``; their xyz
    arrays become new parameter blocks in ``problem`` via the residual block.

    Args:
        problem: Ceres problem owned by the active bundle adjuster; new
            residual blocks are appended to it in place.
        virtual_reconstruction: Reconstruction whose points3D are virtual.
            ``None`` or empty is a no-op.
        reference_reconstruction: Real reconstruction whose pose and
            intrinsics buffers are already parameter blocks of ``problem``.
        negative_depth_observations: ``{image_id: {point2D_idx, ...}}``
            marking observations that should use the negative-depth cost.
        loss_function: ``pyceres.LossFunction`` to apply to virtual
            residuals, or ``None`` for the trivial (squared) loss. If left
            at the sentinel ``_DEFAULT_LOSS``, defaults to Arctan for
            backward compatibility.
    """
    if (
        virtual_reconstruction is None
        or len(virtual_reconstruction.points3D) == 0
    ):
        return

    # Default to Arctan loss for backward compatibility.
    if loss_function is _DEFAULT_LOSS:
        loss_function = _pyceres_loss_function("arctan")

    # Match virtual images to reference images by name so the function is
    # independent of the image-ID convention used by each reconstruction.
    name_to_ref_id = {
        img.name: img_id
        for img_id, img in reference_reconstruction.images.items()
    }

    num_constraints = 0
    num_negative = 0
    num_skipped = 0
    num_none = 0

    for point3D in virtual_reconstruction.points3D.values():
        world_point = point3D.xyz
        if world_point is None or np.all(world_point == 0):
            num_none += 1
            continue

        for elem in point3D.track.elements:
            image_id, pt_idx = elem.image_id, elem.point2D_idx

            if image_id not in virtual_reconstruction.images:
                num_skipped += 1
                continue

            image = virtual_reconstruction.images[image_id]
            ref_id = name_to_ref_id.get(image.name)
            if ref_id is None:
                num_skipped += 1
                continue

            if pt_idx >= len(image.points2D):
                num_skipped += 1
                continue
            point2D = image.points2D[pt_idx].xy

            camera_id = reference_reconstruction.images[ref_id].camera_id

            # Pose & intrinsics come from the reference reconstruction so the
            # underlying numpy buffers are shared with pycolmap's residuals.
            frame = reference_reconstruction.frames[
                reference_reconstruction.images[ref_id].frame_id
            ]
            cam_pose = frame.rig_from_world.params
            camera_params = reference_reconstruction.cameras[camera_id].params
            active_model_id = reference_reconstruction.cameras[camera_id].model

            # Fixed sensor_from_rig for this image's sensor, read from the
            # reconstruction's rig (identity for the reference sensor).
            sid = pycolmap.sensor_t(pycolmap.SensorType.CAMERA, camera_id)
            rig = reference_reconstruction.rigs[frame.rig_id]
            sensor_from_rig = (
                pycolmap.Rigid3d()
                if rig.is_ref_sensor(sid)
                else rig.sensor_from_rig(sid)
            )
            sfr_R = np.array(sensor_from_rig.rotation.matrix())
            sfr_t = np.array(sensor_from_rig.translation)

            is_negative = (
                image_id in negative_depth_observations
                and pt_idx in negative_depth_observations[image_id]
            )
            if is_negative:
                cost = pygluemap.RigReprojErrorCostWithNegativeDepth(
                    active_model_id, point2D, sfr_R, sfr_t
                )
                num_negative += 1
            else:
                cost = pygluemap.RigReprojErrorCost(
                    active_model_id, point2D, sfr_R, sfr_t
                )

            problem.add_residual_block(
                cost,
                loss_function,
                [world_point, cam_pose, camera_params],
            )
            num_constraints += 1

    logger.info(
        f"Added {num_constraints} virtual reprojection constraints "
        f"({num_negative} with negative depth, "
        f"{num_skipped} skipped, {num_none} with no xyz)"
    )


def _add_rig_prior_residuals(
    problem: pyceres.Problem,
    reconstruction: pycolmap.Reconstruction,
    rig: BoundRig,
) -> None:
    """
    A soft prior says: sensor a (in one rig) and sensor b (in a different
    rig) are mounted on the same physical object, and calibration measured
    their relative pose as b_from_a = N, trusted up to weights (w_rot,
    w_trans). The rigs stay separate in this reconstruction, so at every
    capture instant each has its own free frame pose (rig_from_world), and
    without this function nothing ties them together.

    This function adds, per instant, one residual that pulls the two frame
    poses toward the calibrated relative pose. N relates the two member
    cameras, while the parameter blocks are the frames of their reference
    cameras, so the members' fixed within-rig offsets are composed into a
    frame-level nominal first. The nominal is applied at its metric value:
    the reconstruction is expected to arrive metrically scaled (hard-mode
    averaging anchors scale from these same nominals). The observed/nominal
    baseline ratio s* is only measured, and a warning fires when the
    mismatch alone would cost more than one sigma per instant.

    The residuals attach to the same pose buffers pycolmap is optimizing,
    so the pull happens inside the existing BA solve.
    """
    index_at = {
        (rig.sensor_of[idx], rig.frame_of[idx]): idx for idx in rig.sensor_of
    }

    for prior in rig.spec.soft_priors:
        M_a = rig.sensor_from_ref_of(index_at[(prior.a, 0)])
        M_b = rig.sensor_from_ref_of(index_at[(prior.b, 0)])
        N_folded = np.linalg.inv(M_b) @ prior.b_from_a @ M_a

        n_instants = len(rig.spec.images[prior.a])
        assert n_instants == len(rig.spec.images[prior.b]), (
            f"(_add_rig_prior_residuals): prior ({prior.a}, {prior.b}) links "
            f"sensors with {n_instants} vs {len(rig.spec.images[prior.b])} frames"
        )
        pairs = []
        num_missing = 0
        for instant in range(n_instants):
            fa = rig.frame_id_of(index_at[(prior.a, instant)])
            fb = rig.frame_id_of(index_at[(prior.b, instant)])
            assert fa != fb, (
                f"(_add_rig_prior_residuals): prior ({prior.a}, {prior.b}) "
                f"resolves to one frame {fa} at instant {instant}"
            )
            if fa not in reconstruction.frames or fb not in reconstruction.frames:
                num_missing += 1
                continue
            pairs.append((fa, fb))
        assert pairs, (
            f"(_add_rig_prior_residuals): no registered instant for "
            f"prior ({prior.a}, {prior.b})"
        )

        t_norm = np.linalg.norm(N_folded[:3, 3])
        assert t_norm > 1e-6, (
            f"(_add_rig_prior_residuals): prior ({prior.a}, {prior.b}) has a "
            f"co-located nominal, cannot infer reconstruction scale"
        )

        def frame_center(fid):
            pose = reconstruction.frames[fid].rig_from_world
            R = np.array(pose.rotation.matrix())
            return -R.T @ np.array(pose.translation)

        ratios = [
            np.linalg.norm(frame_center(fb) - frame_center(fa)) / t_norm
            for fa, fb in pairs
        ]
        s_star = float(np.median(ratios))

        bias_sigma = abs(s_star - 1.0) * t_norm * prior.w_trans
        if bias_sigma > 1.0:
            logger.warning(
                f"Soft prior ({prior.a}, {prior.b}): observed/nominal "
                f"baseline ratio s*={s_star:.4f}, the scale mismatch alone "
                f"costs {bias_sigma:.1f} sigma per instant; motion averaging "
                f"likely did not pin metric scale"
            )

        num_added = 0
        num_unblocked = 0
        for fa, fb in pairs:
            buf_a = reconstruction.frames[fa].rig_from_world.params
            buf_b = reconstruction.frames[fb].rig_from_world.params
            if not (
                problem.has_parameter_block(buf_a)
                and problem.has_parameter_block(buf_b)
            ):
                num_unblocked += 1
                continue
            cost = pygluemap.RelativePosePriorError(
                N_folded[:3, :3],
                N_folded[:3, 3],
                prior.w_rot,
                prior.w_trans,
            )
            problem.add_residual_block(cost, None, [buf_a, buf_b])
            num_added += 1

        logger.info(
            f"Soft prior ({prior.a}, {prior.b}): {num_added} residuals, "
            f"s*={s_star:.4f} ({min(ratios):.4f}..{max(ratios):.4f} over "
            f"{len(ratios)} instants), {num_missing} unregistered, "
            f"{num_unblocked} without pose blocks"
        )


def bundle_adjustment(
    reconstruction: pycolmap.Reconstruction,
    virtual_reconstruction: pycolmap.Reconstruction | None,
    negative_depth_observations: dict[int, set[int]],
    max_num_iterations: int = 200,
    loss_type_normal: str = "huber",
    loss_type_virtual: str = "arctan",
    known_camera_ids: set[int] | None = None,
    rig: "BoundRig | None" = None,
) -> tuple[
    pycolmap.Reconstruction,
    pycolmap.Reconstruction | None,
    pyceres.SolverSummary,
]:
    """
    Bundle adjustment over real + virtual reconstructions.

    The real reconstruction is optimized via pycolmap's built-in ceres
    bundle adjuster (handles manifolds, gauge fixing, solver selection).
    Virtual residuals are appended manually to the same ceres problem via
    ``_add_virtual_track_residuals`` so that they share the pose/intrinsic
    parameter blocks with the real residuals.

    Args:
        reconstruction: pycolmap.Reconstruction holding the real tracks
            plus authoritative poses and intrinsics. Optimized in-place.
        virtual_reconstruction: pycolmap.Reconstruction whose points3D
            are virtual; may be None or empty for a pure real BA. Its
            points3D.xyz values are optimized in-place as part of the
            joint solve.
        negative_depth_observations: Dict[image_id, Set[point2D_idx]]
            marking observations that should use the negative-depth cost.
        max_num_iterations: Max Ceres iterations.
        loss_type_normal: Loss function for real tracks. One of
            ``"trivial"``, ``"huber"``, ``"cauchy"``.
        loss_type_virtual: Loss function for virtual tracks. One of
            ``"trivial"``, ``"huber"``, ``"arctan"``, ``"cauchy"``.

    Returns:
        (reconstruction, virtual_reconstruction, summary) with parameters
        updated in-place and the Ceres solver summary.
    """
    num_virtual = (
        len(virtual_reconstruction.points3D)
        if virtual_reconstruction is not None
        else 0
    )
    logger.info(
        f"Bundle adjustment: {len(reconstruction.points3D)} real tracks, "
        f"{num_virtual} virtual tracks"
    )

    # --- Build pycolmap BA over the real reconstruction --------------------
    ba_options = pycolmap.BundleAdjustmentOptions()
    # Hard rig: freeze the fixed sensor_from_rig offsets (pycolmap defaults this
    # to True). No-op for trivial rigs, whose sensors are all reference sensors.
    ba_options.refine_sensor_from_rig = False
    # Restore stock Ceres convergence tolerances.
    ba_options.ceres.solver_options = pyceres.SolverOptions()
    ba_options.ceres.solver_options.max_num_iterations = max_num_iterations
    ba_options.ceres.auto_select_solver_type = True
    ba_options.ceres.loss_function_type = _pycolmap_loss_type(loss_type_normal)

    ba_config = pycolmap.BundleAdjustmentConfig()
    for image_id in reconstruction.images:
        ba_config.add_image(image_id)
    for point3D_id in reconstruction.points3D:
        ba_config.add_variable_point(point3D_id)
    ba_config.fix_gauge(pycolmap.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD)
    # This only freezes cameras with at least one real observation. A known
    # camera observed solely by virtual tracks enters the problem as a fresh,
    # variable block and can still drift; accepted as too rare to handle.
    for camera_id in sorted(known_camera_ids or ()):
        assert camera_id in reconstruction.cameras, (
            f"(bundle_adjustment): known camera id {camera_id} not in "
            f"reconstruction, have {sorted(reconstruction.cameras)}"
        )
        ba_config.set_constant_cam_intrinsics(camera_id)

    bundle_adjuster = pycolmap.create_default_ceres_bundle_adjuster(
        ba_options, ba_config, reconstruction
    )
    problem = bundle_adjuster.problem

    logger.info(
        f"After pycolmap BA construction: "
        f"{problem.num_residual_blocks()} residual blocks, "
        f"{problem.num_parameter_blocks()} parameter blocks, "
        f"{problem.num_residuals()} residuals"
    )

    # --- Append virtual residuals to the same problem ----------------------
    _add_virtual_track_residuals(
        problem,
        virtual_reconstruction=virtual_reconstruction,
        reference_reconstruction=reconstruction,
        negative_depth_observations=negative_depth_observations,
        loss_function=_pyceres_loss_function(loss_type_virtual),
    )

    logger.info(
        f"After virtual residual add: "
        f"{problem.num_residual_blocks()} residual blocks, "
        f"{problem.num_parameter_blocks()} parameter blocks, "
        f"{problem.num_residuals()} residuals"
    )

    if rig is not None and rig.spec.soft_priors:
        _add_rig_prior_residuals(problem, reconstruction, rig)
        logger.info(
            f"After rig prior add: "
            f"{problem.num_residual_blocks()} residual blocks, "
            f"{problem.num_residuals()} residuals"
        )

    # --- Solve -------------------------------------------------------------
    solver_options = ba_options.ceres.create_solver_options(ba_config, problem)
    summary = pyceres.SolverSummary()
    pygluemap.solve_cuda(solver_options, problem, summary)
    logger.info(summary.BriefReport())

    # --- Sync poses/intrinsics into the virtual reconstruction -------------
    # Only the real reconstruction's numpy buffers flowed into the ceres
    # problem (see ``_add_virtual_track_residuals``); the virtual
    # reconstruction still holds the pre-solve values. Copy optimized
    # poses and per-camera intrinsics over so downstream consumers
    # reading from virtual_reconstruction observe consistent state.
    if virtual_reconstruction is not None:
        # Lazy import to avoid a circular estimators -> controllers import
        # at module load time.

        _update_poses_from_reconstruction(
            reconstruction, virtual_reconstruction
        )

    return reconstruction, virtual_reconstruction, summary
