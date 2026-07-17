"""Tests for BoundRig.averaging_view / glue_soft_priors: promoting soft
priors to hard rig structure for motion averaging, without disturbing the
original BoundRig that BA and the database writer consume."""

import numpy as np
import pytest

from gluemap.utils.rigs import (
    Rig,
    RigSpec,
    SensorIntrinsics,
    SoftPrior,
    bind_rig_spec,
    glue_soft_priors,
    is_metric_edge,
)


def rotz(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rigid(R: np.ndarray, t) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


I4 = np.eye(4)
M_CAM1 = rigid(rotz(30.0), [0.1, 0.0, 0.0])
N_PHONE = rigid(rotz(10.0), [0.0, 0.2, 0.0])


def fake_images(sensors: list[str], n_frames: int) -> dict[str, list[str]]:
    return {
        s: [f"/fake/{s}_f{f}.png" for f in range(n_frames)] for s in sensors
    }


def two_rig_spec(soft_priors=None, intrinsics=(None, None)) -> RigSpec:
    """A fan2 rig (cam0 ref, cam1 offset) plus a phone singleton, with the
    soft prior declared on the fan2's NON-reference member to exercise the
    member-matrix folding."""
    if soft_priors is None:
        soft_priors = [SoftPrior("cam1", "phone", N_PHONE, 1.0, 1.0)]
    return RigSpec(
        sensors=["cam0", "cam1", "phone"],
        rigs=[
            Rig("fan2", ["cam0", "cam1"], [I4, M_CAM1], intrinsics[0]),
            Rig("phone", ["phone"], [I4], intrinsics[1]),
        ],
        soft_priors=soft_priors,
        images=fake_images(["cam0", "cam1", "phone"], 2),
    )


def bind(spec: RigSpec, soft_averaging: str = "free"):
    paths = [p for s in spec.sensors for p in spec.images[s]]
    return bind_rig_spec(spec, paths, soft_averaging=soft_averaging)


def test_free_mode_view_is_self():
    rig = bind(two_rig_spec(), "free")
    assert rig.averaging_view() is rig


def test_hard_mode_without_priors_is_self():
    spec = two_rig_spec(soft_priors=[])
    rig = bind(spec, "hard")
    assert rig.averaging_view() is rig


def test_bad_mode_rejected():
    with pytest.raises(AssertionError):
        bind(two_rig_spec(), "soft")


def test_glue_composes_through_non_reference_member():
    glued = glue_soft_priors(two_rig_spec())
    assert len(glued.rigs) == 1
    rig = glued.rigs[0]
    assert rig.name == "fan2"
    assert rig.members == ["cam0", "cam1", "phone"]
    np.testing.assert_allclose(rig.sensor_from_ref[0], I4)
    np.testing.assert_allclose(rig.sensor_from_ref[1], M_CAM1)
    # cam_phone = N o cam_cam1 = N o M_cam1 o cam_cam0.
    np.testing.assert_allclose(rig.sensor_from_ref[2], N_PHONE @ M_CAM1)
    assert glued.soft_priors == []


def test_view_rebinds_refs_and_leaves_original_untouched():
    rig = bind(two_rig_spec(), "hard")
    view = rig.averaging_view()
    assert view is not rig

    # Image order is sensor-major: cam0 f0, cam0 f1, cam1 ..., phone ...
    idx = {(rig.sensor_of[i], rig.frame_of[i]): i for i in rig.sensor_of}
    for f in range(2):
        assert view.ref_of[idx[("phone", f)]] == idx[("cam0", f)]
        assert view.ref_of[idx[("cam1", f)]] == idx[("cam0", f)]
        # The original keeps the phone as its own singleton reference.
        assert rig.ref_of[idx[("phone", f)]] == idx[("phone", f)]
    np.testing.assert_allclose(
        view.sensor_from_ref_of(idx[("phone", 0)]), N_PHONE @ M_CAM1
    )
    np.testing.assert_allclose(rig.sensor_from_ref_of(idx[("phone", 0)]), I4)
    assert len(rig.spec.rigs) == 2 and len(rig.spec.soft_priors) == 1

    # The glued spec has no priors left, so the view is a fixed point.
    assert view.averaging_view() is view


def test_view_exposes_metric_edges():
    rig = bind(two_rig_spec(), "hard")
    view = rig.averaging_view()
    idx = {(rig.sensor_of[i], rig.frame_of[i]): i for i in rig.sensor_of}
    assert not is_metric_edge(rig, idx[("cam0", 0)], idx[("phone", 0)])
    assert is_metric_edge(view, idx[("cam0", 0)], idx[("phone", 0)])


def test_chain_matches_hand_constructed_world_poses():
    """Three rigs glued by a chain of priors: every member's composed
    sensor_from_ref must map the root reference pose to the same world pose
    the nominal relations construct directly."""
    M_a1 = rigid(rotz(-20.0), [0.05, 0.02, 0.0])
    M_c1 = rigid(rotz(45.0), [0.0, 0.0, 0.3])
    N1 = rigid(rotz(5.0), [0.0, 0.15, 0.0])
    N2 = rigid(rotz(-8.0), [0.1, 0.0, 0.05])
    spec = RigSpec(
        sensors=["a0", "a1", "b0", "c0", "c1"],
        rigs=[
            Rig("A", ["a0", "a1"], [I4, M_a1]),
            Rig("B", ["b0"], [I4]),
            Rig("C", ["c0", "c1"], [I4, M_c1]),
        ],
        # The second prior points from c1 INTO the chain, so the glue has
        # to walk it against the declared direction.
        soft_priors=[
            SoftPrior("a1", "b0", N1, 1.0, 1.0),
            SoftPrior("c1", "b0", N2, 1.0, 1.0),
        ],
        images=fake_images(["a0", "a1", "b0", "c0", "c1"], 1),
    )
    glued = glue_soft_priors(spec)
    assert len(glued.rigs) == 1
    rig = glued.rigs[0]
    assert rig.members == ["a0", "a1", "b0", "c0", "c1"]

    P = rigid(rotz(70.0), [1.0, -2.0, 0.5])  # arbitrary root cam_from_world
    world = {"a0": P, "a1": M_a1 @ P}
    world["b0"] = N1 @ world["a1"]  # cam_b0 = N1 o cam_a1
    world["c1"] = np.linalg.inv(N2) @ world["b0"]  # cam_b0 = N2 o cam_c1
    world["c0"] = np.linalg.inv(M_c1) @ world["c1"]
    for member, S in zip(rig.members, rig.sensor_from_ref):
        np.testing.assert_allclose(S @ P, world[member], atol=1e-12)


def test_prior_cycle_rejected():
    spec = RigSpec(
        sensors=["a", "b", "c"],
        rigs=[Rig("A", ["a"], [I4]), Rig("B", ["b"], [I4]), Rig("C", ["c"], [I4])],
        soft_priors=[
            SoftPrior("a", "b", N_PHONE, 1.0, 1.0),
            SoftPrior("b", "c", N_PHONE, 1.0, 1.0),
            SoftPrior("c", "a", N_PHONE, 1.0, 1.0),
        ],
        images=fake_images(["a", "b", "c"], 1),
    )
    with pytest.raises(AssertionError, match="cycle"):
        glue_soft_priors(spec)


def test_duplicate_prior_between_two_rigs_rejected():
    spec = two_rig_spec(
        soft_priors=[
            SoftPrior("cam1", "phone", N_PHONE, 1.0, 1.0),
            SoftPrior("cam0", "phone", N_PHONE, 1.0, 1.0),
        ]
    )
    with pytest.raises(AssertionError, match="two priors"):
        glue_soft_priors(spec)


def test_mixed_intrinsics_declaration_rejected():
    K = SensorIntrinsics(100.0, 100.0, 50.0, 50.0, 100, 100)
    spec = two_rig_spec(intrinsics=([K, K], None))
    with pytest.raises(AssertionError, match="intrinsics"):
        glue_soft_priors(spec)


def test_intrinsics_concatenate_in_member_order():
    Ka = SensorIntrinsics(100.0, 100.0, 50.0, 50.0, 100, 100)
    Kb = SensorIntrinsics(200.0, 200.0, 50.0, 50.0, 100, 100)
    Kp = SensorIntrinsics(300.0, 300.0, 50.0, 50.0, 100, 100)
    spec = two_rig_spec(intrinsics=([Ka, Kb], [Kp]))
    glued = glue_soft_priors(spec)
    assert glued.rigs[0].intrinsics == [Ka, Kb, Kp]


def test_untouched_singleton_component_passes_through():
    """A rig with no prior attached must survive gluing unchanged while an
    unrelated pair merges."""
    spec = RigSpec(
        sensors=["cam0", "cam1", "phone", "lone"],
        rigs=[
            Rig("fan2", ["cam0", "cam1"], [I4, M_CAM1]),
            Rig("phone", ["phone"], [I4]),
            Rig("lone", ["lone"], [I4]),
        ],
        soft_priors=[SoftPrior("cam1", "phone", N_PHONE, 1.0, 1.0)],
        images=fake_images(["cam0", "cam1", "phone", "lone"], 1),
    )
    glued = glue_soft_priors(spec)
    assert [r.name for r in glued.rigs] == ["fan2", "lone"]
    assert glued.rigs[1] is spec.rigs[2]
