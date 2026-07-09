"""Unit tests for gluemap.math.geometry primitives."""

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from gluemap.math.geometry import (
    average_rotations_so3,
    bilinear_interpolate_value,
    project,
    project_tracks,
    quaternion_to_rotation_matrix,
    restore_identity,
    rotation_matrix_to_quaternion,
    unproject,
)


@pytest.mark.parametrize(
    "R_input",
    [
        np.eye(3),
        Rotation.from_euler("z", 90, degrees=True).as_matrix(),
        Rotation.from_euler("xyz", [30, 45, 60], degrees=True).as_matrix(),
        Rotation.from_rotvec([np.pi, 0, 0]).as_matrix(),  # 180° about x
    ],
)
@pytest.mark.parametrize("w_first", [True, False])
def test_quaternion_rotation_matrix_round_trip(R_input, w_first):
    q = rotation_matrix_to_quaternion(R_input, w_first=w_first)
    R_recovered = quaternion_to_rotation_matrix(q, w_first=w_first)
    np.testing.assert_allclose(R_recovered, R_input, atol=1e-12)


def test_quaternion_w_first_layout():
    R = Rotation.from_euler("y", 30, degrees=True).as_matrix()
    q_first = rotation_matrix_to_quaternion(R, w_first=True)
    q_last = rotation_matrix_to_quaternion(R, w_first=False)
    np.testing.assert_allclose(q_first[0], q_last[3], atol=1e-12)
    np.testing.assert_allclose(q_first[1:], q_last[:3], atol=1e-12)


def test_restore_identity_rebases_first_frame():
    rng = np.random.default_rng(0)
    B, N = 2, 4
    extrinsics = torch.zeros(B, N, 4, 4, dtype=torch.float64)
    extrinsics[..., 3, 3] = 1.0
    for b in range(B):
        for n in range(N):
            R = torch.from_numpy(Rotation.random(random_state=rng).as_matrix())
            t = torch.from_numpy(rng.normal(0, 1, size=3))
            extrinsics[b, n, :3, :3] = R
            extrinsics[b, n, :3, 3] = t

    extrinsics_orig = extrinsics.clone()
    restored = restore_identity(extrinsics.clone())

    for b in range(B):
        np.testing.assert_allclose(
            restored[b, 0, :3, :3].numpy(), np.eye(3), atol=1e-12
        )
        np.testing.assert_allclose(
            restored[b, 0, :3, 3].numpy(), np.zeros(3), atol=1e-12
        )

    # Pairwise relative pose (frame i seen from frame 0 of *new* coords) must
    # equal the original (frame i seen from frame 0 of *old* coords).
    for b in range(B):
        R0 = extrinsics_orig[b, 0, :3, :3]
        t0 = extrinsics_orig[b, 0, :3, 3]
        for i in range(1, N):
            Ri = extrinsics_orig[b, i, :3, :3]
            ti = extrinsics_orig[b, i, :3, 3]
            R_rel_old = Ri @ R0.T
            t_rel_old = ti - R_rel_old @ t0

            R_rel_new = restored[b, i, :3, :3]
            t_rel_new = restored[b, i, :3, 3]

            np.testing.assert_allclose(
                R_rel_new.numpy(), R_rel_old.numpy(), atol=1e-12
            )
            np.testing.assert_allclose(
                t_rel_new.numpy(), t_rel_old.numpy(), atol=1e-12
            )


def test_project_unproject_round_trip():
    K = torch.tensor(
        [[[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]],
        dtype=torch.float64,
    )
    rng = np.random.default_rng(1)
    uvs_np = rng.uniform(low=[10, 10], high=[630, 470], size=(50, 2))
    uvs = torch.from_numpy(uvs_np).unsqueeze(0)

    rays = unproject(uvs, K)
    uvs_recovered = project(rays, K)
    np.testing.assert_allclose(uvs_recovered.numpy(), uvs.numpy(), atol=1e-9)


def test_bilinear_interpolate_value_constant_grid():
    H, W, C = 8, 10, 3
    value_map = torch.full((1, C, H, W), 7.5, dtype=torch.float32)
    coords = torch.tensor(
        [[[1.5, 2.0], [4.5, 3.7], [9.0, 7.0]]], dtype=torch.float32
    )
    sampled = bilinear_interpolate_value(value_map, coords, align_corners=True)
    assert sampled.shape == (1, 3, C)
    # grid_sample runs at float32 precision internally.
    np.testing.assert_allclose(sampled.numpy(), 7.5, atol=1e-4)


def test_bilinear_interpolate_value_linear_ramp_align_corners():
    # value_map[..., y, x] = x: sampling at integer (x, y) with
    # align_corners=True should return x within float32 grid_sample precision.
    H, W = 8, 10
    x_grid = (
        torch.arange(W, dtype=torch.float32)
        .view(1, 1, 1, W)
        .expand(1, 1, H, W)
        .contiguous()
    )
    coords = torch.tensor(
        [[[0.0, 0.0], [3.0, 2.0], [9.0, 7.0]]], dtype=torch.float32
    )
    sampled = bilinear_interpolate_value(
        x_grid, coords.clone(), align_corners=True
    )
    expected = torch.tensor([[[0.0], [3.0], [9.0]]], dtype=torch.float32)
    np.testing.assert_allclose(sampled.numpy(), expected.numpy(), atol=1e-4)


def test_project_tracks_angle_thresholding():
    # Identity extrinsics -> camera_points are world points. With
    # angle_threshold=5°, points along +z are valid; points at very large
    # off-axis angles are filtered. camera_points has the (B, H, W, C) layout.
    B, N, H, W = 1, 1, 2, 2
    extrinsics = torch.zeros(B, N, 4, 4, dtype=torch.float64)
    extrinsics[..., :3, :3] = torch.eye(3, dtype=torch.float64)
    extrinsics[..., 3, 3] = 1.0
    intrinsic = (
        torch.eye(3, dtype=torch.float64).expand(B, N, 3, 3).contiguous()
    )
    intrinsic[..., 0, 0] = 100.0
    intrinsic[..., 1, 1] = 100.0

    # Four points laid out on a 2×2 grid. sin(5°) ≈ 0.0872.
    # The angle filter rejects only points where BOTH directions are invalid
    # (invalid_i2j AND invalid_j2i), so a point is rejected only when its
    # |z_normalized| < sin(threshold) — i.e., near-grazing rays.
    pts = torch.tensor(
        [
            [
                [[0.0, 0.0, 1.0], [5.0, 0.0, 1.0]],  # row 0
                [[100.0, 0.0, 1.0], [0.0, 0.0, -1.0]],  # row 1
            ]
        ],
        dtype=torch.float64,
    )
    assert pts.shape == (B, H, W, 3)

    _, _, valid_mask, is_negative = project_tracks(
        pts, extrinsics, intrinsic, angle_threshold=5
    )
    # valid_mask is (B, N, H*W) flattened in row-major order over (H, W).
    valid = valid_mask[0, 0].tolist()
    neg = is_negative[0, 0].tolist()
    assert valid[0] is True  # (0, 0, 1)   straight ahead
    assert valid[1] is True  # (5, 0, 1)   off-axis but z_norm > sin(5°)
    assert valid[2] is False  # (100, 0, 1) near-grazing (z_norm ≈ 0.01)
    assert valid[3] is True  # (0, 0, -1)  behind camera but z_norm = -1
    # is_negative flag tracks behind-camera points (z < 0).
    assert neg[3] is True
    assert neg[0] is False


def plot_rotated_x_axes(angles: list, Rs: np.ndarray, mean: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for a, R in zip(angles, Rs):
        ax.annotate("", xytext=(0, 0), xy=(R[0, 0], R[1, 0]),
                    arrowprops=dict(arrowstyle="->", color="0.55", lw=1.4))
        ax.text(1.06 * R[0, 0], 1.06 * R[1, 0], f"{a:g}\N{DEGREE SIGN}",
                fontsize=9, ha="center", color="0.35")
    ax.annotate("", xytext=(0, 0), xy=(mean[0, 0], mean[1, 0]),
                arrowprops=dict(arrowstyle="->", color="#0E7A6B", lw=2.4))
    ax.text(1.12 * mean[0, 0], 1.12 * mean[1, 0], "mean", fontsize=10,
            ha="center", color="#0E7A6B", weight="bold")
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="0.85", lw=0.8))
    ax.set_xlim(-0.15, 1.2)
    ax.set_ylim(-0.15, 1.2)
    ax.set_aspect("equal")
    ax.set_title("rotated x-axes: Rz(30/32/34) and their SO(3) mean")
    out = Path(__file__).parent.parent / "Debug" / "so3_mean_directions.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def test_average_rotations_so3_z_axis():
    angles = [30.0, 32.0, 34.0]
    Rs = np.stack(
        [Rotation.from_euler("z", a, degrees=True).as_matrix() for a in angles]
    )
    mean, residuals = average_rotations_so3(Rs, np.ones(3))
    expected = Rotation.from_euler("z", 32.0, degrees=True).as_matrix()
    print("Rz(30):")
    print(Rs[0])
    print("mean:")
    print(mean)
    assert np.allclose(mean, expected, atol=1e-9), mean
    assert np.allclose(residuals, [2.0, 0.0, 2.0], atol=1e-9), residuals
    plot_rotated_x_axes(angles, Rs, mean)
