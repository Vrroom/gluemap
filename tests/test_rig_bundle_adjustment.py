"""Tests for rig-aware augmented bundle adjustment."""

import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation

from gluemap.controllers.augmented_bundle_adjustment import (
    build_reconstruction_for_ba,
    extract_results_from_reconstruction,
)
from gluemap.math.reprojection_error import (
    ReprojectionErrorType,
    compute_all_errors_from_reconstruction,
    filter_reconstruction_by_reprojection_error,
)
from gluemap.utils.rigs import Rig, RigSpec, bind_rig_spec


def rigid(R, t):
    return pycolmap.Rigid3d(
        pycolmap.Rotation3d(np.asarray(R, float)), np.asarray(t, float)
    )


def build_rig_reconstruction():
    """Build a 1-rig, 2-sensor, 3-frame reconstruction with a known,
    non-identity sensor_from_rig.

    Returns ``(reconstruction, expected)`` where ``expected`` maps each
    ``image_id`` to its ground-truth ``(R, center)`` for the composed
    ``cam_from_world = sensor_from_rig . rig_from_world``.
    """
    CAM = pycolmap.SensorType.CAMERA
    W, H, f = 640, 480, 500.0

    # one camera per sensor (a sensor IS a camera_id)
    cam0 = pycolmap.Camera(camera_id=1, model="PINHOLE", width=W, height=H,
                           params=[f, f, W / 2, H / 2])
    cam1 = pycolmap.Camera(camera_id=2, model="PINHOLE", width=W, height=H,
                           params=[f, f, W / 2, H / 2])
    sid_ref = pycolmap.sensor_t(CAM, 1)
    sid_mem = pycolmap.sensor_t(CAM, 2)
    sfr = rigid(
        Rotation.from_euler("xyz", [5, -8, 3], degrees=True).as_matrix(),
        [0.2, -0.1, 0.05],
    )

    rec = pycolmap.Reconstruction()
    rec.add_camera(cam0)
    rec.add_camera(cam1)
    rig = pycolmap.Rig()
    rig.rig_id = 1
    rig.add_ref_sensor(sid_ref)
    rig.add_sensor(sid_mem, sfr)

    specs = [
        (Rotation.from_euler("xyz", [10, -5, 20], degrees=True).as_matrix(),
         [0.0, 0.0, 3.0]),
        (Rotation.from_euler("xyz", [-15, 12, -8], degrees=True).as_matrix(),
         [0.5, -0.3, 2.7]),
        (Rotation.from_euler("xyz", [7, 22, 4], degrees=True).as_matrix(),
         [-0.4, 0.6, 3.2]),
    ]
    rfw = {}
    frames = []
    for fid, (R, t) in enumerate(specs, start=1):
        rfw[fid] = rigid(R, t)
        fr = pycolmap.Frame()
        fr.frame_id = fid
        fr.rig_id = 1
        fr.rig_from_world = rfw[fid]
        fr.add_data_id(pycolmap.data_t(sid_ref, 10 * fid + 0))
        fr.add_data_id(pycolmap.data_t(sid_mem, 10 * fid + 1))
        frames.append(fr)
    rec.set_rigs_and_frames([rig], frames)

    pts = np.random.default_rng(0).uniform(
        [-0.5, -0.5, 4.0], [0.5, 0.5, 6.0], size=(8, 3)
    )
    expected = {}
    for fid in rfw:
        for slot, (sid, cam) in enumerate([(sid_ref, cam0), (sid_mem, cam1)]):
            iid = 10 * fid + slot
            s_from_r = pycolmap.Rigid3d() if slot == 0 else sfr
            sfw = s_from_r * rfw[fid]  # composed cam_from_world
            R = np.asarray(sfw.rotation.matrix())
            t = np.asarray(sfw.translation)
            expected[iid] = (R, -R.T @ t)
            uv = np.array(
                [cam.img_from_cam(np.asarray(sfw * X, float)) for X in pts]
            )
            im = pycolmap.Image(name=f"f{fid}s{slot}", keypoints=uv,
                                camera_id=cam.camera_id, image_id=iid)
            im.frame_id = fid
            rec.add_image(im)

    for j, X in enumerate(pts):
        tr = pycolmap.Track()
        for fid in rfw:
            for slot in (0, 1):
                tr.add_element(10 * fid + slot, j)
        rec.add_point3D(X, tr)

    return rec, expected


def test_extract_results_recovers_composed_rig_poses():
    rec, expected = build_rig_reconstruction()
    global_rotations, global_centers, _intrinsics, _points3D = (
        extract_results_from_reconstruction(rec)
    )
    assert set(global_rotations) == set(expected)
    assert set(global_centers) == set(expected)
    for iid, (R_exp, c_exp) in expected.items():
        np.testing.assert_allclose(global_rotations[iid], R_exp, atol=1e-9)
        np.testing.assert_allclose(global_centers[iid], c_exp, atol=1e-9)


def test_filter_removes_member_sensor_outlier():
    """Regression: the reprojection filter must project through the composed
    ``sensor_from_rig . rig_from_world`` pose. A rig-unaware filter would
    misjudge a member (non-reference) sensor, so exact observations must give
    near-zero error and a corrupted member observation must be the one removed.
    """
    rec, _ = build_rig_reconstruction()

    errors = compute_all_errors_from_reconstruction(
        rec, ReprojectionErrorType.PIXEL
    )
    assert max(e for track in errors.values() for _, _, e in track) < 1e-6

    member_id = min(
        iid for iid, im in rec.images.items() if im.camera_id == 2
    )
    im = rec.images[member_id]
    im.points2D[0].xy = np.asarray(im.points2D[0].xy, float) + [25.0, -18.0]

    obs_removed, tracks_removed = filter_reconstruction_by_reprojection_error(
        rec, ReprojectionErrorType.PIXEL, 5.0, min_track_length=2
    )
    assert (obs_removed, tracks_removed) == (1, 0)
    assert not any(
        e.image_id == member_id and e.point2D_idx == 0
        for p in rec.points3D.values()
        for e in p.track.elements
    )


def test_build_reconstruction_for_ba_rig_shared_intrinsics():
    """Sensors sharing one intrinsics group must still become distinct COLMAP
    cameras (a rig requires distinct sensors), and a member image's composed
    cam_from_world must equal sensor_from_rig . rig_from_world.
    """
    def rigid4(R, t):
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = t
        return M

    sfr_R = Rotation.from_euler("xyz", [5, -8, 3], degrees=True).as_matrix()
    sfr_t = np.array([0.2, -0.1, 0.05])
    rigs = [Rig("rig0", ["cam0", "cam1"], [np.eye(4), rigid4(sfr_R, sfr_t)])]
    p0 = ["c0_f0.png", "c0_f1.png"]
    p1 = ["c1_f0.png", "c1_f1.png"]
    spec = RigSpec(["cam0", "cam1"], rigs, [], {"cam0": p0, "cam1": p1})
    image_paths = p0 + p1
    brig = bind_rig_spec(spec, image_paths)

    # One shared intrinsics group for all four images (the real-pipeline case
    # that crashed: two sensors would have collapsed onto one camera_id).
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    global_intrinsics = [K[None]]
    intrinsics_mapping = {0: 0, 1: 0, 2: 0, 3: 0}
    R0 = Rotation.from_euler("xyz", [10, -5, 20], degrees=True).as_matrix()
    R1 = Rotation.from_euler("xyz", [-15, 12, -8], degrees=True).as_matrix()
    c0 = np.array([0.0, 0.0, 3.0])
    c1 = np.array([0.5, -0.3, 2.7])
    global_rotations = {0: R0, 1: R1, 2: np.eye(3), 3: np.eye(3)}
    global_centers = {0: c0, 1: c1, 2: np.zeros(3), 3: np.zeros(3)}
    keypoints = {i: np.array([[100.0, 100.0], [200.0, 150.0]]) for i in range(4)}
    image_sizes = [(480, 640)] * 4

    rec = build_reconstruction_for_ba(
        global_rotations, global_centers, global_intrinsics,
        intrinsics_mapping, {}, keypoints,
        image_sizes=image_sizes, images_list=image_paths,
        camera_model="PINHOLE", rig=brig,
    )

    # Two sensors sharing K still yield two distinct cameras.
    assert len(rec.cameras) == 2
    assert rec.images[1].camera_id != rec.images[3].camera_id
    assert (len(rec.rigs), len(rec.frames), rec.num_images()) == (1, 2, 4)
    # images 1 (cam0 ref) and 3 (cam1 member) share the same frame
    assert rec.images[1].frame_id == rec.images[3].frame_id

    # Reference image keeps its own pose.
    ref = rec.images[1].cam_from_world()
    np.testing.assert_allclose(ref.rotation.matrix(), R0, atol=1e-9)
    np.testing.assert_allclose(ref.translation, -R0 @ c0, atol=1e-9)

    # Member image pose is the composition sensor_from_rig . rig_from_world.
    sfr = pycolmap.Rigid3d(pycolmap.Rotation3d(sfr_R), sfr_t)
    rfw0 = pycolmap.Rigid3d(pycolmap.Rotation3d(R0), -R0 @ c0)
    expected = sfr * rfw0
    got = rec.images[3].cam_from_world()
    np.testing.assert_allclose(
        got.rotation.matrix(), expected.rotation.matrix(), atol=1e-9
    )
    np.testing.assert_allclose(got.translation, expected.translation, atol=1e-9)
