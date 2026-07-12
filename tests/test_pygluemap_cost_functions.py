"""Unit tests for the pygluemap cost-function bindings."""

import numpy as np
import pyceres
import pycolmap
import pygluemap
from scipy.spatial.transform import Rotation


def solve_for_c2(cost, c1_val, s_val, c2_init):
    """Solve a single-residual problem for c2 with c1 and scale held fixed."""
    c1 = np.array(c1_val, dtype=np.float64)
    c2 = np.array(c2_init, dtype=np.float64)
    s = np.array([s_val], dtype=np.float64)
    prob = pyceres.Problem()
    prob.add_residual_block(cost, pyceres.TrivialLoss(), [c1, c2, s])
    prob.set_parameter_block_constant(c1)
    prob.set_parameter_block_constant(s)
    opts = pyceres.SolverOptions()
    opts.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
    opts.max_num_iterations = 50
    opts.minimizer_progress_to_stdout = False
    pyceres.solve(opts, prob, pyceres.SolverSummary())
    return c2


def test_offset_cost_recovers_center():
    # residual t_hat - s*(c2 - c1 + offset) is zero at c2 = c1 + t_hat/s - offset
    c1 = np.array([0.0, 0.0, 0.0])
    s = 2.0
    offset = np.array([1.0, 0.0, 0.0])
    t_hat = np.array([4.0, 2.0, 0.0])
    expected = c1 + t_hat / s - offset

    cost = pygluemap.PairwiseDirectionErrorWithOffset(
        translation_obs=t_hat, offset=offset
    )
    rng = np.random.default_rng(0)
    got = solve_for_c2(cost, c1, s, c2_init=rng.random(3))
    np.testing.assert_allclose(got, expected, atol=1e-7)


def test_offset_zero_matches_base():
    c1 = np.array([0.0, 0.0, 0.0])
    s = 2.0
    t_hat = np.array([4.0, 2.0, 0.0])
    rng = np.random.default_rng(1)
    init = rng.random(3)

    base = pygluemap.PairwiseDirectionError(translation_obs=t_hat)
    off0 = pygluemap.PairwiseDirectionErrorWithOffset(
        translation_obs=t_hat, offset=np.zeros(3)
    )
    c_base = solve_for_c2(base, c1, s, c2_init=init.copy())
    c_off0 = solve_for_c2(off0, c1, s, c2_init=init.copy())
    np.testing.assert_allclose(c_base, c_off0, atol=1e-10)


def solve_for_scale(cost, s_init):
    """Solve a single-residual problem for the scale parameter."""
    s = np.array([s_init], dtype=np.float64)
    prob = pyceres.Problem()
    prob.add_residual_block(cost, pyceres.TrivialLoss(), [s])
    opts = pyceres.SolverOptions()
    opts.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
    opts.max_num_iterations = 50
    opts.function_tolerance = 1e-14
    opts.parameter_tolerance = 1e-14
    opts.gradient_tolerance = 1e-16
    opts.minimizer_progress_to_stdout = False
    pyceres.solve(opts, prob, pyceres.SolverSummary())
    return s[0]


def test_scale_only_cost_recovers_scale():
    # residual t_hat - s*offset is zero at s_true when t_hat = s_true*offset
    offset = np.array([2.0, 0.0, 0.0])
    s_true = 3.0
    t_hat = s_true * offset
    cost = pygluemap.ScaleOnlyDirectionError(
        translation_obs=t_hat, offset=offset
    )
    got = solve_for_scale(cost, s_init=1.0)
    np.testing.assert_allclose(got, s_true, atol=1e-8)


def test_scale_only_cost_least_squares():
    # non-parallel: minimizer is the projection s = (offset . t_hat) / |offset|^2
    offset = np.array([1.0, 0.0, 0.0])
    t_hat = np.array([4.0, 2.0, 0.0])
    expected = offset @ t_hat / (offset @ offset)
    cost = pygluemap.ScaleOnlyDirectionError(
        translation_obs=t_hat, offset=offset
    )
    got = solve_for_scale(cost, s_init=0.5)
    np.testing.assert_allclose(got, expected, atol=1e-8)


def qmul(a, b):  # Hamilton product, quaternions in [x, y, z, w] order
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def qconj(a):
    return np.array([-a[0], -a[1], -a[2], a[3]])


def pose_block(R, t):
    return np.concatenate(
        [Rotation.from_matrix(R).as_quat(), np.asarray(t, float)]
    ).astype(np.float64)


def rel_prior_residual(cost, a, b):
    out = cost.evaluate(a, b)
    res = out[0] if isinstance(out, tuple) else out
    return np.asarray(res, float).ravel()


def test_relative_pose_prior_zero_at_prior():
    R_a = Rotation.from_euler("xyz", [15, -25, 40], degrees=True).as_matrix()
    t_a = np.array([0.3, -0.7, 1.1])
    R_prior = Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix()
    t_prior = np.array([1.0, 2.0, 3.0])
    # b = prior . a  ->  predicted b_from_a == prior  ->  residual is zero
    R_b = R_prior @ R_a
    t_b = R_prior @ t_a + t_prior
    cost = pygluemap.RelativePosePriorError(
        b_R_a=R_prior, b_t_a=t_prior, w_rot=1.0, w_trans=1.0
    )
    r = rel_prior_residual(cost, pose_block(R_a, t_a), pose_block(R_b, t_b))
    np.testing.assert_allclose(r, 0.0, atol=1e-9)


def test_relative_pose_prior_matches_hand_computation():
    R_a = Rotation.from_euler("xyz", [15, -25, 40], degrees=True).as_matrix()
    t_a = np.array([0.3, -0.7, 1.1])
    R_prior = Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix()
    t_prior = np.array([1.0, 2.0, 3.0])
    R_b = (
        Rotation.from_euler("xyz", [12, 18, 35], degrees=True).as_matrix() @ R_a
    )
    t_b = R_prior @ t_a + t_prior + np.array([0.05, -0.1, 0.2])
    w_rot, w_trans = 3.0, 0.5

    q_ba = qmul(
        Rotation.from_matrix(R_b).as_quat(),
        qconj(Rotation.from_matrix(R_a).as_quat()),
    )
    t_ba = t_b - Rotation.from_quat(q_ba).as_matrix() @ t_a
    q_err = qmul(qconj(Rotation.from_matrix(R_prior).as_quat()), q_ba)
    expected = np.concatenate(
        [2 * w_rot * q_err[:3], w_trans * (t_ba - t_prior)]
    )

    cost = pygluemap.RelativePosePriorError(
        b_R_a=R_prior, b_t_a=t_prior, w_rot=w_rot, w_trans=w_trans
    )
    got = rel_prior_residual(cost, pose_block(R_a, t_a), pose_block(R_b, t_b))
    np.testing.assert_allclose(got, expected, atol=1e-9)


def test_relative_pose_prior_weight_scaling():
    R_a = Rotation.from_euler("xyz", [15, -25, 40], degrees=True).as_matrix()
    t_a = np.array([0.3, -0.7, 1.1])
    R_prior = Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix()
    t_prior = np.array([1.0, 2.0, 3.0])
    R_b = (
        Rotation.from_euler("xyz", [12, 18, 35], degrees=True).as_matrix() @ R_a
    )
    t_b = R_prior @ t_a + t_prior + np.array([0.05, -0.1, 0.2])
    a, b = pose_block(R_a, t_a), pose_block(R_b, t_b)

    r1 = rel_prior_residual(
        pygluemap.RelativePosePriorError(R_prior, t_prior, 1.0, 1.0), a, b
    )
    r2 = rel_prior_residual(
        pygluemap.RelativePosePriorError(R_prior, t_prior, 2.0, 5.0), a, b
    )
    np.testing.assert_allclose(r2[:3], 2.0 * r1[:3], atol=1e-9)
    np.testing.assert_allclose(r2[3:], 5.0 * r1[3:], atol=1e-9)


def rot_err_deg(quat, R_target):
    R = Rotation.from_quat(quat / np.linalg.norm(quat)).as_matrix()
    c = np.clip((np.trace(R @ R_target.T) - 1) / 2, -1.0, 1.0)
    return np.degrees(np.arccos(c))


class SplitPoseBAdapter(pyceres.CostFunction):
    """Present pose-b as separate quaternion(4) + translation(3) blocks so the
    built-in EigenQuaternionManifold / EuclideanManifold can be attached, while
    delegating the residual and Jacobian to RelativePosePriorError. pyceres 2.6
    cannot attach a Python-subclassed Manifold, so this split is how we run a
    real solve. Parameter blocks: [a (7, const), q_b (4), t_b (3)]."""

    def __init__(self, cost):
        super().__init__()
        self.cost = cost
        self.set_num_residuals(6)
        self.set_parameter_block_sizes([7, 4, 3])

    def Evaluate(self, parameters, residuals, jacobians):
        a = np.asarray(parameters[0], float)
        b = np.concatenate(
            [np.asarray(parameters[1], float), np.asarray(parameters[2], float)]
        )
        res, jac = self.cost.evaluate(a, b)
        residuals[:] = np.asarray(res, float).ravel()
        if jacobians is not None:
            j_a = np.asarray(jac[0], float)  # 6x7 wrt a
            j_b = np.asarray(jac[1], float)  # 6x7 wrt (q_b | t_b)
            if jacobians[0] is not None:
                jacobians[0][:] = j_a.reshape(jacobians[0].shape)
            if jacobians[1] is not None:
                jacobians[1][:] = j_b[:, 0:4].reshape(jacobians[1].shape)
            if jacobians[2] is not None:
                jacobians[2][:] = j_b[:, 4:7].reshape(jacobians[2].shape)
        return True


def solve_pose_b_to_prior(cost, a_block, b_init):
    a = a_block.copy()
    q_b = b_init[0:4].copy()
    t_b = b_init[4:7].copy()
    prob = pyceres.Problem()
    prob.add_residual_block(
        SplitPoseBAdapter(cost), pyceres.TrivialLoss(), [a, q_b, t_b]
    )
    prob.set_parameter_block_constant(a)
    prob.set_manifold(q_b, pyceres.EigenQuaternionManifold())
    prob.set_manifold(t_b, pyceres.EuclideanManifold(3))
    opts = pyceres.SolverOptions()
    opts.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
    opts.max_num_iterations = 100
    opts.function_tolerance = 1e-16
    opts.parameter_tolerance = 1e-16
    pyceres.solve(opts, prob, pyceres.SolverSummary())
    return np.concatenate([q_b, t_b])


def test_relative_pose_prior_full_solve():
    R_a = Rotation.from_euler("xyz", [15, -25, 40], degrees=True).as_matrix()
    t_a = np.array([0.3, -0.7, 1.1])
    R_prior = Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix()
    t_prior = np.array([1.0, 2.0, 3.0])
    a_block = pose_block(R_a, t_a)

    # optimum: b_from_world = prior . a
    R_b_target = R_prior @ R_a
    t_b_target = R_prior @ t_a + t_prior

    # start well off target in both rotation and translation
    R_b_init = (
        Rotation.from_euler("xyz", [35, -28, 22], degrees=True).as_matrix()
        @ R_b_target
    )
    b_init = pose_block(R_b_init, t_b_target + np.array([0.8, -0.6, 0.9]))

    for w_rot, w_trans in [(1.0, 1.0), (0.5, 3.0)]:
        cost = pygluemap.RelativePosePriorError(
            b_R_a=R_prior, b_t_a=t_prior, w_rot=w_rot, w_trans=w_trans
        )
        b = solve_pose_b_to_prior(cost, a_block, b_init)
        assert rot_err_deg(b[0:4], R_b_target) < 1e-5
        np.testing.assert_allclose(b[4:7], t_b_target, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(b[0:4]), 1.0, atol=1e-9)


def pinhole_camera():
    return pycolmap.Camera(
        camera_id=1, model="PINHOLE", width=640, height=480,
        params=[500.0, 500.0, 320.0, 240.0],
    )


def reproj_residual(cost, point3D, pose, params):
    out = cost.evaluate(
        np.asarray(point3D, float), pose, np.asarray(params, float)
    )
    return np.asarray(out[0], float).ravel()


def test_rig_reproj_zero_at_true_observation():
    cam = pinhole_camera()
    R_sfr = Rotation.from_euler("xyz", [5, -8, 3], degrees=True).as_matrix()
    t_sfr = np.array([0.2, -0.1, 0.05])
    R_rig = Rotation.from_euler("xyz", [10, -5, 20], degrees=True).as_matrix()
    t_rig = np.array([0.3, 0.1, 0.5])
    pose = pose_block(R_rig, t_rig)

    # cam_from_world = sensor_from_rig o rig_from_world
    R_cw = R_sfr @ R_rig
    t_cw = R_sfr @ t_rig + t_sfr
    # negative-depth cost is for BEHIND-camera points: pick point_in_cam z < 0
    p_cam = np.array([0.1, -0.2, -5.0])
    point3D = R_cw.T @ (p_cam - t_cw)
    # cost negates before projecting, so the true observation is proj(-p_cam)
    obs = np.asarray(cam.img_from_cam(np.asarray(-p_cam, float)))

    cost = pygluemap.RigReprojErrorCostWithNegativeDepth(
        cam.model, obs, R_sfr, t_sfr
    )
    r = reproj_residual(cost, point3D, pose, cam.params)
    np.testing.assert_allclose(r, 0.0, atol=1e-8)

    # shifting the observation shifts the residual by exactly -delta
    delta = np.array([1.5, -2.0])
    cost2 = pygluemap.RigReprojErrorCostWithNegativeDepth(
        cam.model, obs + delta, R_sfr, t_sfr
    )
    r2 = reproj_residual(cost2, point3D, pose, cam.params)
    np.testing.assert_allclose(r2, -delta, atol=1e-8)


def test_rig_reproj_identity_matches_plain():
    cam = pinhole_camera()
    R_rig = Rotation.from_euler("xyz", [10, -5, 20], degrees=True).as_matrix()
    t_rig = np.array([0.3, 0.1, 0.5])
    pose = pose_block(R_rig, t_rig)
    # behind-camera point so both costs produce a real (nonzero) residual
    p_cam = np.array([0.2, 0.15, -4.0])
    point3D = R_rig.T @ (p_cam - t_rig)
    obs = np.array([310.0, 250.0])  # arbitrary -> nonzero residual

    plain = pygluemap.ReprojErrorCostWithNegativeDepth(cam.model, obs)
    rig = pygluemap.RigReprojErrorCostWithNegativeDepth(
        cam.model, obs, np.eye(3), np.zeros(3)
    )
    r_plain = reproj_residual(plain, point3D, pose, cam.params)
    r_rig = reproj_residual(rig, point3D, pose, cam.params)
    assert np.abs(r_plain).max() > 1.0  # guard: not a vacuous zero-vs-zero match
    np.testing.assert_allclose(r_rig, r_plain, atol=1e-12)
