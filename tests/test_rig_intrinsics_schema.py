"""Schema-level tests for per-sensor rig intrinsics in load_rig_spec.

The happy path loads the permanent Bulldozer fixture (tests/data/Bulldozer).
Failure cases copy its parsed YAML, mutate one field, write the result to
tmp_path, and assert the loader's validation fires.
"""

import os

import pytest
import torch
import yaml

from gluemap.utils.rigs import (
    bind_rig_spec,
    inject_rig_intrinsics,
    load_rig_spec,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(REPO, "tests", "data", "Bulldozer", "schema.yaml")


def fixture_raw() -> dict:
    with open(FIXTURE) as fh:
        return yaml.safe_load(fh)


def write_schema(tmp_path, raw: dict) -> str:
    path = str(tmp_path / "schema.yaml")
    with open(path, "w") as fh:
        yaml.safe_dump(raw, fh)
    return path


def test_bulldozer_round_trip():
    spec = load_rig_spec(FIXTURE)
    rig = spec.rigs[0]
    assert spec.sensors == ["cam0", "cam1", "cam2", "cam3"]
    assert len(spec.images[rig.members[0]]) == 8
    assert rig.name == "fan4"
    expected_fx = [256.0, 1024.0 / 3, 1280.0 / 3, 512.0]
    for K, fx in zip(rig.intrinsics, expected_fx):
        assert K.fx == pytest.approx(fx, abs=1e-6)
        assert K.fy == pytest.approx(fx, abs=1e-6)
        assert (K.cx, K.cy) == (256.0, 256.0)
        assert (K.width, K.height) == (512, 512)
    paths = [p for s in spec.sensors for p in spec.images[s]]
    assert bind_rig_spec(spec, paths).intrinsics_of("cam3").fx == 512.0


def test_intrinsics_key_set_enforced(tmp_path):
    raw = fixture_raw()
    del raw["rigs"][0]["intrinsics"][0]["fx"]
    with pytest.raises(AssertionError, match="as_sensor_intrinsics.*keys"):
        load_rig_spec(write_schema(tmp_path, raw))


def test_nonpositive_focal_rejected(tmp_path):
    raw = fixture_raw()
    raw["rigs"][0]["intrinsics"][1]["fy"] = -341.0
    with pytest.raises(AssertionError, match="focal lengths must be positive"):
        load_rig_spec(write_schema(tmp_path, raw))


def test_principal_point_outside_image_rejected(tmp_path):
    raw = fixture_raw()
    raw["rigs"][0]["intrinsics"][2]["cx"] = 700.0
    with pytest.raises(AssertionError, match="principal point"):
        load_rig_spec(write_schema(tmp_path, raw))


def test_width_mismatch_with_file_rejected(tmp_path):
    raw = fixture_raw()
    raw["rigs"][0]["intrinsics"][0]["width"] = 1024
    with pytest.raises(AssertionError, match="declares 1024x512"):
        load_rig_spec(write_schema(tmp_path, raw))


def test_intrinsics_count_must_match_members(tmp_path):
    raw = fixture_raw()
    raw["rigs"][0]["intrinsics"].pop()
    with pytest.raises(AssertionError, match="4 members but 3 intrinsics"):
        load_rig_spec(write_schema(tmp_path, raw))


def test_inject_rig_intrinsics_partial_declaration(tmp_path):
    raw = fixture_raw()
    rig = raw["rigs"][0]
    raw["rigs"] = [
        {
            "name": "declared",
            "members": ["cam0", "cam1"],
            "sensor_from_ref": rig["sensor_from_ref"][:2],
            "intrinsics": rig["intrinsics"][:2],
        },
        {
            "name": "undeclared",
            "members": ["cam2", "cam3"],
            "sensor_from_ref": rig["sensor_from_ref"][:2],
        },
    ]
    spec = load_rig_spec(write_schema(tmp_path, raw))
    paths = [p for s in spec.sensors for p in spec.images[s]]
    bound = bind_rig_spec(spec, paths)
    buckets = [torch.eye(3).unsqueeze(0) for _ in range(4)]
    ids = inject_rig_intrinsics(buckets, bound, "SIMPLE_PINHOLE")
    assert ids == {1, 2}
    assert buckets[0][0, 0, 0] == 256.0
    assert buckets[1][0, 0, 0] == pytest.approx(1024.0 / 3)
    assert torch.equal(buckets[2], torch.eye(3).unsqueeze(0))
    assert torch.equal(buckets[3], torch.eye(3).unsqueeze(0))


def test_inject_rejects_fx_ne_fy_for_simple_model(tmp_path):
    raw = fixture_raw()
    raw["rigs"][0]["intrinsics"][0]["fy"] = 300.0
    spec = load_rig_spec(write_schema(tmp_path, raw))
    paths = [p for s in spec.sensors for p in spec.images[s]]
    bound = bind_rig_spec(spec, paths)
    buckets = [torch.eye(3).unsqueeze(0) for _ in range(4)]
    with pytest.raises(AssertionError, match="forces fx == fy"):
        inject_rig_intrinsics(buckets, bound, "SIMPLE_PINHOLE")


def test_intrinsics_of_none_when_undeclared(tmp_path):
    raw = fixture_raw()
    del raw["rigs"][0]["intrinsics"]
    spec = load_rig_spec(write_schema(tmp_path, raw))
    assert spec.rigs[0].intrinsics is None
    paths = [p for s in spec.sensors for p in spec.images[s]]
    assert bind_rig_spec(spec, paths).intrinsics_of("cam2") is None
