import os
from dataclasses import dataclass

import networkx as nx
import numpy as np
import yaml


@dataclass(frozen=True)
class Rig:
    name: str
    members: list[str]
    sensor_from_ref: list[np.ndarray]


@dataclass(frozen=True)
class SoftPrior:
    a: str
    b: str
    b_from_a: np.ndarray
    w_rot: float
    w_trans: float


@dataclass(frozen=True)
class RigSpec:
    sensors: list[str]
    rigs: list[Rig]
    soft_priors: list[SoftPrior]
    images: dict[str, list[str]]

    @property
    def n_frames(self) -> int:
        return len(self.images[self.sensors[0]])

    def rig_of_sensor(self, sensor: str) -> Rig:
        rig = next((r for r in self.rigs if sensor in r.members), None)
        assert rig is not None, f"(rig_of_sensor): unknown sensor {sensor}"
        return rig

    def sensor_of_image(self, image_path: str) -> tuple[str, int]:
        for sensor, paths in self.images.items():
            if image_path in paths:
                return sensor, paths.index(image_path)
        raise AssertionError(f"(sensor_of_image): unknown image {image_path}")


def fold_graph(G: nx.Graph, ref_of: dict) -> nx.MultiGraph:
    for n in G.nodes:
        assert n in ref_of, f"(fold_graph): node {n} has no reference"
    F = nx.MultiDiGraph() if G.is_directed() else nx.MultiGraph()
    F.add_nodes_from(sorted({ref_of[n] for n in G.nodes}))
    for i, j, data in G.edges(data=True):
        ri, rj = ref_of[i], ref_of[j]
        if ri == rj:
            continue
        F.add_edge(ri, rj, **data)
    return F


@dataclass(frozen=True)
class BoundRig:
    spec: RigSpec
    ref_of: dict[int, int]
    sensor_of: dict[int, str]
    frame_of: dict[int, int]

    def sensor_from_ref_of(self, idx: int) -> np.ndarray:
        sensor = self.sensor_of[idx]
        rig = self.spec.rig_of_sensor(sensor)
        return rig.sensor_from_ref[rig.members.index(sensor)]


def fold_relative_poses(
    edges: dict[tuple[int, int], tuple[np.ndarray, float]],
    rig: BoundRig,
) -> dict[tuple[int, int], list[tuple[np.ndarray, float]]]:
    raise NotImplementedError


def bind_rig_spec(spec: RigSpec, image_paths: list[str]) -> BoundRig:
    spec_paths = {p for paths in spec.images.values() for p in paths}
    assert set(image_paths) == spec_paths, (
        f"(bind_rig_spec): dataset images differ from spec images, "
        f"only_dataset={sorted(set(image_paths) - spec_paths)[:3]}, "
        f"only_spec={sorted(spec_paths - set(image_paths))[:3]}"
    )
    assert len(image_paths) == len(spec_paths), (
        f"(bind_rig_spec): duplicate dataset paths, {len(image_paths)} paths, {len(spec_paths)} unique"
    )
    sensor_of, frame_of, index_at = {}, {}, {}
    for idx, p in enumerate(image_paths):
        sensor, frame = spec.sensor_of_image(p)
        sensor_of[idx] = sensor
        frame_of[idx] = frame
        index_at[(sensor, frame)] = idx
    ref_of = {
        idx: index_at[(spec.rig_of_sensor(sensor_of[idx]).members[0], frame_of[idx])]
        for idx in sensor_of
    }
    return BoundRig(spec, ref_of, sensor_of, frame_of)


def as_rigid_matrix(rows: list[list[float]]) -> np.ndarray:
    M = np.array(rows, dtype=np.float64)
    assert M.shape == (4, 4), f"(as_rigid_matrix): expected 4x4, got {M.shape}"
    assert np.array_equal(M[3], [0.0, 0.0, 0.0, 1.0]), (
        f"(as_rigid_matrix): bottom row must be [0,0,0,1], got {M[3]}"
    )
    R = M[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-5), (
        f"(as_rigid_matrix): R is not orthogonal, R @ R.T =\n{R @ R.T}"
    )
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-5), (
        f"(as_rigid_matrix): det(R) must be +1, got {np.linalg.det(R)}"
    )
    return M


def load_rig_spec(path: str) -> RigSpec:
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    expected = {"n_sensors", "sensors", "rigs", "soft_priors", "images"}
    assert set(raw) == expected, (
        f"(load_rig_spec): top-level keys {sorted(set(raw))} != {sorted(expected)}"
    )
    sensors = raw["sensors"]
    assert len(sensors) == raw["n_sensors"], (
        f"(load_rig_spec): n_sensors={raw['n_sensors']}, but {len(sensors)} sensors listed"
    )
    assert len(set(sensors)) == len(sensors), (
        f"(load_rig_spec): duplicate sensor names in {sensors}"
    )
    rigs = []
    for i, entry in enumerate(raw["rigs"]):
        assert {"members", "sensor_from_ref"} <= set(entry) <= {"name", "members", "sensor_from_ref"}, (
            f"(load_rig_spec): rig {i} has bad keys {sorted(entry)}"
        )
        members = entry["members"]
        mats = [as_rigid_matrix(m) for m in entry["sensor_from_ref"]]
        assert len(mats) == len(members), (
            f"(load_rig_spec): rig {i} has {len(members)} members but {len(mats)} matrices"
        )
        assert np.allclose(mats[0], np.eye(4)), (
            f"(load_rig_spec): rig {i} reference matrix must be identity, got\n{mats[0]}"
        )
        rigs.append(Rig(entry.get("name", f"rig_{i}"), members, mats))
    claimed = [m for rig in rigs for m in rig.members]
    assert sorted(claimed) == sorted(sensors), (
        f"(load_rig_spec): rigs must partition sensors exactly, {sorted(claimed)} != {sorted(sensors)}"
    )
    rig_of = {m: i for i, rig in enumerate(rigs) for m in rig.members}
    soft_priors = []
    for i, entry in enumerate(raw["soft_priors"]):
        assert set(entry) == {"pair", "b_from_a", "w_rot", "w_trans"}, (
            f"(load_rig_spec): soft prior {i} has bad keys {sorted(entry)}"
        )
        a, b = entry["pair"]
        assert a in rig_of and b in rig_of and a != b, (
            f"(load_rig_spec): soft prior {i} needs two distinct known sensors, got {entry['pair']}"
        )
        assert rig_of[a] != rig_of[b], (
            f"(load_rig_spec): soft prior {i} pair ({a}, {b}) lies inside one hard rig"
        )
        w_rot, w_trans = float(entry["w_rot"]), float(entry["w_trans"])
        assert w_rot > 0 and w_trans > 0, (
            f"(load_rig_spec): soft prior {i} weights must be positive, got {w_rot}, {w_trans}"
        )
        soft_priors.append(SoftPrior(a, b, as_rigid_matrix(entry["b_from_a"]), w_rot, w_trans))
    images = raw["images"]
    assert set(images) == set(sensors), (
        f"(load_rig_spec): images keys {sorted(images)} != sensors {sorted(sensors)}"
    )
    n_frames = len(images[sensors[0]])
    assert n_frames > 0, f"(load_rig_spec): sensor {sensors[0]} has an empty image list"
    all_paths = []
    for s in sensors:
        paths = images[s]
        assert len(paths) == n_frames, (
            f"(load_rig_spec): sensor {s} has {len(paths)} images, expected {n_frames}"
        )
        for p in paths:
            assert isinstance(p, str), f"(load_rig_spec): sensor {s} has non-string image entry {p!r}"
            assert os.path.isfile(p), f"(load_rig_spec): missing image file {p}"
        all_paths += paths
    assert len(set(all_paths)) == len(all_paths), (
        "(load_rig_spec): duplicate image paths across sensors"
    )
    return RigSpec(sensors, rigs, soft_priors, images)


if __name__ == "__main__":
    spec = load_rig_spec("/home/salmonuser/DigitalTwins/gluemap/Debug/Bathroom/schema.yaml")
    rig = spec.rigs[0]
    print(f"{len(spec.sensors)} sensors, {spec.n_frames} frames, rig {rig.name}, ref {rig.members[0]}")
    for member, M in zip(rig.members, rig.sensor_from_ref):
        print(f"{member} sensor_from_ref[:3] = {M[:3].tolist()}")
    last = spec.sensors[-1]
    image = spec.images[last][spec.n_frames - 1]
    print(f"{image} -> {spec.sensor_of_image(image)}")
    print(f"rig_of_sensor({last}) -> {spec.rig_of_sensor(last).name}")
