"""End-to-end rig-consistency test, one per scene.

For each scene in ``SCENES`` this runs the full ``gluemap-demo`` pipeline to a
temp dir and then checks that the output reconstruction's rig is *rigid*: for
every frame, each member sensor's pose relative to its frame's reference image
equals the schema's fixed ``sensor_from_ref`` -- constant across all frames and
matching the calibration.

Add a scene by dropping ``Debug/<Scene>/schema.yaml`` + ``Debug/<Scene>/images/``
and appending its name to ``SCENES``; nothing else is needed.

These are SLOW integration tests (each runs the full neural SfM). They are
skipped by default; run them with::

    RUN_SCENE_TESTS=1 pytest tests/test_rig_scenes.py -s
"""

import os
from collections import defaultdict

import numpy as np
import pycolmap
import pytest
import yaml

from gluemap.datasets.utils import get_image_list
from gluemap.utils.rigs import bind_rig_spec, load_rig_spec

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCENES = ["Bathroom", "CornellBox", "LegoTractor", "Bulldozer"]

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_SCENE_TESTS"),
    reason="slow end-to-end scene test; set RUN_SCENE_TESTS=1 to run",
)


def scene_paths(scene: str) -> tuple[str, str]:
    """Absolute (schema.yaml, images/) paths for a scene, by convention."""
    schema = os.path.join(REPO, "Debug", scene, "schema.yaml")
    images = os.path.join(REPO, "Debug", scene, "images") + "/"
    return schema, images


def run_demo(scene: str, out_dir: str) -> str:
    """Run the full gluemap-demo end to end for a scene into out_dir.

    rig_config_path is not a CLI flag, so we hand the demo a per-scene config
    (absolute ``_base_`` so it resolves from anywhere). Returns the output
    reconstruction dir (``out_dir/gluemap_aba``).
    """
    import subprocess

    schema, images = scene_paths(scene)
    os.makedirs(out_dir, exist_ok=True)
    config = {
        "_base_": os.path.join(REPO, "configs", "base.yaml"),
        "images_path": images,
        "write_path": out_dir + "/",
        "rig_config_path": schema,
        "gt_intrinsics_path": None,
    }
    config_path = os.path.join(out_dir, "scene_config.yaml")
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)

    subprocess.run(
        ["gluemap-demo", "--config", config_path], check=True, cwd=REPO
    )
    return os.path.join(out_dir, "gluemap_aba")


def check_rig_consistency(
    rec: pycolmap.Reconstruction,
    schema: str,
    images: str,
    rot_tol_deg: float = 0.1,
    trans_tol: float = 1e-3,
    spread_tol_deg: float = 1e-2,
) -> dict:
    """Assert the reconstruction's rig is rigid and matches the schema.

    For every member image, the observed offset from its frame reference is
    ``member.cam_from_world() . reference.cam_from_world()^-1``, which for a
    hard rig must equal that sensor's ``sensor_from_ref``. We check both that
    it matches the schema (``rot_tol_deg`` / ``trans_tol``) and that it is
    constant across frames (``spread_tol_deg``). Returns a per-sensor report;
    raises AssertionError on any violation. Images map by basename, so it is
    independent of the reconstruction's id convention.
    """
    image_paths = get_image_list(images)
    brig = bind_rig_spec(load_rig_spec(schema), image_paths)
    cfw = {os.path.basename(im.name): im.cam_from_world() for im in rec.images.values()}

    per_sensor: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for idx, ref_idx in brig.ref_of.items():
        if idx == ref_idx:  # reference sensor: offset is identity by definition
            continue
        member = os.path.basename(image_paths[idx])
        reference = os.path.basename(image_paths[ref_idx])
        if member not in cfw or reference not in cfw:  # dropped in filtering
            continue

        observed = cfw[member] * cfw[reference].inverse()
        M = brig.sensor_from_ref_of(idx)
        expected = pycolmap.Rigid3d(
            pycolmap.Rotation3d(np.ascontiguousarray(M[:3, :3])), M[:3, 3]
        )
        dR = np.asarray(observed.rotation.matrix()) @ np.asarray(
            expected.rotation.matrix()
        ).T
        ang = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
        dt = float(
            np.linalg.norm(
                np.asarray(observed.translation) - np.asarray(expected.translation)
            )
        )
        per_sensor[brig.sensor_of[idx]].append((ang, dt))

    assert per_sensor, "no member observations survived to check"

    report = {}
    for sensor, devs in per_sensor.items():
        angs = np.array([d[0] for d in devs])
        dts = np.array([d[1] for d in devs])
        spread = float(angs.max() - angs.min())
        report[sensor] = {
            "n_frames": len(devs),
            "max_rot_deg": float(angs.max()),
            "rot_spread_deg": spread,
            "max_trans": float(dts.max()),
        }
        assert angs.max() < rot_tol_deg, (
            f"{sensor}: offset differs from schema by {angs.max():.4g} deg "
            f"(> {rot_tol_deg})"
        )
        assert spread < spread_tol_deg, (
            f"{sensor}: offset not constant across frames, spread "
            f"{spread:.4g} deg (> {spread_tol_deg}); rig is not rigid"
        )
        assert dts.max() < trans_tol, (
            f"{sensor}: translation offset differs from schema by "
            f"{dts.max():.4g} (> {trans_tol})"
        )
    return report


def check_declared_intrinsics(
    rec: pycolmap.Reconstruction,
    schema: str,
    param_tol: float = 1e-4,
    reproj_tol_px: float = 2.0,
) -> int:
    """Assert cameras with schema-declared intrinsics carry them unchanged.

    The tolerance only absorbs the float32 quantization of the injection;
    any BA drift is orders of magnitude larger and fails. Returns the number
    of sensors checked (0 when the schema declares no intrinsics).
    """
    spec = load_rig_spec(schema)
    checked = 0
    for i, sensor in enumerate(spec.sensors):
        rig = spec.rig_of_sensor(sensor)
        if rig.intrinsics is None:
            continue
        K = rig.intrinsics[rig.members.index(sensor)]
        km = rec.cameras[i + 1].calibration_matrix()
        est = (km[0, 0], km[1, 1], km[0, 2], km[1, 2])
        for e, g, name in zip(est, (K.fx, K.fy, K.cx, K.cy), ["fx", "fy", "cx", "cy"]):
            assert abs(e - g) < param_tol, (
                f"{sensor} {name}: {e} differs from schema {g} by {abs(e - g):.4g}; "
                f"intrinsics leaked into BA"
            )
        checked += 1
    if checked:
        err = rec.compute_mean_reprojection_error()
        assert err < reproj_tol_px, (
            f"mean reprojection error {err:.3f}px > {reproj_tol_px}px "
            f"with known intrinsics"
        )
    return checked


@pytest.mark.parametrize("scene", SCENES)
def test_rig_consistent_end_to_end(scene, tmp_path):
    schema, images = scene_paths(scene)
    assert os.path.isfile(schema), f"{scene}: missing {schema}"
    assert os.path.isdir(os.path.dirname(images)), f"{scene}: missing {images}"

    recon_dir = run_demo(scene, str(tmp_path / scene))
    rec = pycolmap.Reconstruction(recon_dir)
    report = check_rig_consistency(rec, schema, images)
    print(f"\n[{scene}] rig consistency OK: {report}")
    n_checked = check_declared_intrinsics(rec, schema)
    print(f"[{scene}] declared intrinsics held for {n_checked} sensors")
