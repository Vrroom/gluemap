import logging
import os
from dataclasses import dataclass

import imagesize
import networkx as nx
import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SensorIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass(frozen=True)
class Rig:
    name: str
    members: list[str]
    sensor_from_ref: list[np.ndarray]
    intrinsics: list[SensorIntrinsics] | None = None


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


def glue_soft_priors(spec: RigSpec) -> RigSpec:
    """Merge hard rigs connected by soft priors into single hard rigs, with
    the nominal prior transforms treated as exact. This is the pretend-hard
    structure that motion averaging folds with when soft rigs are promoted to
    hard there; BA must keep working from the original spec, where the glued
    rigs stay separate and the priors become residuals.

    The priors must form a forest over the rigs: a cycle would carry two
    composition paths between the same rigs with no guarantee they agree.
    """
    rig_index_of = {m: i for i, r in enumerate(spec.rigs) for m in r.members}
    G = nx.Graph()
    G.add_nodes_from(range(len(spec.rigs)))
    for p in spec.soft_priors:
        ra, rb = rig_index_of[p.a], rig_index_of[p.b]
        assert not G.has_edge(ra, rb), (
            f"(glue_soft_priors): two priors between rigs "
            f"{spec.rigs[ra].name} and {spec.rigs[rb].name}"
        )
        G.add_edge(ra, rb, prior=p)
    assert nx.is_forest(G), "(glue_soft_priors): soft priors form a cycle"

    def member_matrix(sensor: str) -> np.ndarray:
        r = spec.rigs[rig_index_of[sensor]]
        return r.sensor_from_ref[r.members.index(sensor)]

    merged = []
    for comp in sorted(nx.connected_components(G), key=min):
        rig_ids = sorted(comp)
        root = rig_ids[0]
        if len(rig_ids) == 1:
            merged.append(spec.rigs[root])
            continue

        # Walk the prior tree outward from the root rig, composing each
        # rig's reference pose onto the root reference: a prior (a, b, N)
        # means cam_b = N o cam_a, so with member matrices M_a, M_b the
        # unknown side's reference follows from the known side's.
        ref_from_root = {root: np.eye(4)}
        for u, v in nx.bfs_edges(G, root):
            p = G.edges[u, v]["prior"]
            ra, rb = rig_index_of[p.a], rig_index_of[p.b]
            assert {ra, rb} == {u, v}, (
                f"(glue_soft_priors): edge ({u}, {v}) carries prior for "
                f"rigs ({ra}, {rb})"
            )
            M_a, M_b = member_matrix(p.a), member_matrix(p.b)
            if ra == u:
                ref_from_root[v] = (
                    np.linalg.inv(M_b) @ p.b_from_a @ M_a @ ref_from_root[u]
                )
            else:
                ref_from_root[v] = (
                    np.linalg.inv(M_a)
                    @ np.linalg.inv(p.b_from_a)
                    @ M_b
                    @ ref_from_root[u]
                )

        # Intrinsics concatenate in member order, so within one component
        # the rigs must agree on declaring them.
        declared = [spec.rigs[i].intrinsics is not None for i in rig_ids]
        assert all(declared) or not any(declared), (
            f"(glue_soft_priors): rigs "
            f"{[spec.rigs[i].name for i in rig_ids]} mix declared and "
            f"undeclared intrinsics"
        )
        members, mats, intrinsics = [], [], []
        for i in rig_ids:
            r = spec.rigs[i]
            for m, M in zip(r.members, r.sensor_from_ref):
                members.append(m)
                mats.append(M @ ref_from_root[i])
            if r.intrinsics is not None:
                intrinsics += list(r.intrinsics)
        assert np.allclose(mats[0], np.eye(4)), (
            f"(glue_soft_priors): merged rig {spec.rigs[root].name} lost its "
            f"identity reference"
        )
        merged.append(
            Rig(
                spec.rigs[root].name,
                members,
                mats,
                intrinsics if intrinsics else None,
            )
        )

    return RigSpec(spec.sensors, merged, [], spec.images)


@dataclass(frozen=True)
class BoundRig:
    spec: RigSpec
    ref_of: dict[int, int]
    sensor_of: dict[int, str]
    frame_of: dict[int, int]
    soft_averaging: str = "free"

    def averaging_view(self) -> "BoundRig":
        """The rig structure motion averaging should fold with. In "free"
        mode soft priors stay out of averaging (they act only as BA
        residuals) and the view is this instance itself. In "hard" mode the
        priors are applied as exact transforms: rigs glued along them become
        one hard rig at the nominal, so averaging couples their evidence and
        their nominal baselines can anchor metric scale. BA and the database
        writer must keep using this original instance, never the view."""
        if self.soft_averaging == "free" or not self.spec.soft_priors:
            return self
        glued = glue_soft_priors(self.spec)
        logger.info(
            f"soft_rig_averaging=hard: treating "
            f"{len(self.spec.soft_priors)} soft prior(s) as exact in motion "
            f"averaging, gluing rigs {[r.name for r in self.spec.rigs]} "
            f"-> {[r.name for r in glued.rigs]}; BA still sees them as soft"
        )
        index_at = {
            (self.sensor_of[idx], self.frame_of[idx]): idx
            for idx in self.sensor_of
        }
        ref_of = {
            idx: index_at[
                (
                    glued.rig_of_sensor(self.sensor_of[idx]).members[0],
                    self.frame_of[idx],
                )
            ]
            for idx in self.sensor_of
        }
        return BoundRig(
            glued, ref_of, self.sensor_of, self.frame_of, self.soft_averaging
        )

    def sensor_from_ref_of(self, idx: int) -> np.ndarray:
        sensor = self.sensor_of[idx]
        rig = self.spec.rig_of_sensor(sensor)
        return rig.sensor_from_ref[rig.members.index(sensor)]

    def camera_id_of(self, sensor: str) -> int:
        """
        1-indexed COLMAP camera_id for a sensor. Single source of truth so
        the DB writer and the reconstruction builder cannot number cameras
        differently. 

        I think this is to keep track of the colmap Camera type. 
        """
        return self.spec.sensors.index(sensor) + 1

    def intrinsics_of(self, sensor: str) -> SensorIntrinsics | None:
        """Declared intrinsics for a sensor, or None when its rig declares none."""
        rig = self.spec.rig_of_sensor(sensor)
        if rig.intrinsics is None:
            return None
        return rig.intrinsics[rig.members.index(sensor)]

    def rig_id_of(self, sensor: str) -> int:
        """
        1-indexed COLMAP rig_id for the rig that owns a sensor. Shared source
        of truth so the DB writer and the builder agree on rig numbering.

        In the single rig setting, this will be just 1. But if we used two image sets: 
         
        + rig
        + dslr 

        Then they'll get different ids.
        """
        rig_id = next(
            (i + 1 for i, r in enumerate(self.spec.rigs) if sensor in r.members),
            None,
        )
        assert rig_id is not None, f"(rig_id_of): unknown sensor {sensor}"
        return rig_id

    def frame_id_of(self, idx: int) -> int:
        """
        1-indexed COLMAP frame_id for an image's frame, which is the frame's
        reference image id. Shared source of truth so the DB writer and the
        builder group images into frames identically.
        """
        return self.ref_of[idx] + 1


def inject_rig_intrinsics(
    global_intrinsics: list,
    rig: BoundRig,
    camera_model: str,
) -> set[int]:
    """Overwrite declared sensors' buckets in place; returns their COLMAP camera ids."""
    sensors = rig.spec.sensors
    assert len(global_intrinsics) == len(sensors), (
        f"(inject_rig_intrinsics): {len(global_intrinsics)} buckets, {len(sensors)} sensors"
    )
    known_camera_ids = set()
    for bucket, sensor in enumerate(sensors):
        K = rig.intrinsics_of(sensor)
        if K is None:
            continue
        assert not camera_model.startswith("SIMPLE") or K.fx == K.fy, (
            f"(inject_rig_intrinsics): {sensor} declares fx={K.fx} fy={K.fy}, "
            f"but camera model {camera_model} forces fx == fy"
        )
        global_intrinsics[bucket] = torch.tensor(
            [[K.fx, 0.0, K.cx], [0.0, K.fy, K.cy], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).unsqueeze(0)
        known_camera_ids.add(rig.camera_id_of(sensor))
    return known_camera_ids


def fold_relative_poses(
    edges: dict[tuple[int, int], tuple[np.ndarray, float]],
    rig: BoundRig,
) -> dict[tuple[int, int], list[tuple[np.ndarray, float]]]:
    raise NotImplementedError


def bind_rig_spec(
    spec: RigSpec,
    image_paths: list[str],
    soft_averaging: str = "free",
) -> BoundRig:
    assert soft_averaging in ("free", "hard"), (
        f"(bind_rig_spec): soft_averaging must be 'free' or 'hard', "
        f"got {soft_averaging!r}"
    )
    if soft_averaging == "free":
        for p in spec.soft_priors:
            assert p.w_rot == 1.0 and p.w_trans == 1.0, (
                f"(bind_rig_spec): prior ({p.a}, {p.b}) carries calibrated weights "
                f"w_rot={p.w_rot} w_trans={p.w_trans} but soft_rig_averaging is "
                f"'free', so the prior never anchors averaging and scale stays a "
                f"free gauge; set soft_rig_averaging: hard or use weights 1.0"
            )
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
    return BoundRig(spec, ref_of, sensor_of, frame_of, soft_averaging)


def compute_world_space_rig_offset(
    rig: BoundRig,
    global_rotations: dict[int, np.ndarray],
    idx: int,
) -> np.ndarray:
    """World-space offset of image idx from its rig reference centre."""
    r = rig.ref_of[idx]
    M = rig.sensor_from_ref_of(idx)
    return global_rotations[r].T @ (-M[:3, :3].T @ M[:3, 3])


def add_member_centers(
    reference_centers: dict[int, np.ndarray],
    global_rotations: dict[int, np.ndarray],
    rig: BoundRig,
) -> dict[int, np.ndarray]:
    """Place every image at its reference centre plus the known rig offset."""
    global_centers = {}
    for idx in global_rotations:
        r = rig.ref_of[idx]
        delta = compute_world_space_rig_offset(rig, global_rotations, idx)
        global_centers[idx] = reference_centers[r] + delta
    return global_centers


def is_metric_edge(rig: BoundRig, idx1: int, idx2: int, eps: float = 1e-6) -> bool:
    """True if idx1 and idx2 are the same rig-frame with a nonzero baseline."""
    if rig.ref_of[idx1] != rig.ref_of[idx2]:
        return False
    M1 = rig.sensor_from_ref_of(idx1)
    M2 = rig.sensor_from_ref_of(idx2)
    c1 = -M1[:3, :3].T @ M1[:3, 3]
    c2 = -M2[:3, :3].T @ M2[:3, 3]
    return np.linalg.norm(c1 - c2) > eps


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


def as_sensor_intrinsics(raw: dict) -> SensorIntrinsics:
    expected = {"fx", "fy", "cx", "cy", "width", "height"}
    assert set(raw) == expected, (
        f"(as_sensor_intrinsics): keys {sorted(raw)} != {sorted(expected)}"
    )
    fx, fy, cx, cy = (float(raw[k]) for k in ("fx", "fy", "cx", "cy"))
    width, height = raw["width"], raw["height"]
    assert isinstance(width, int) and width > 0, (
        f"(as_sensor_intrinsics): width must be a positive int, got {width!r}"
    )
    assert isinstance(height, int) and height > 0, (
        f"(as_sensor_intrinsics): height must be a positive int, got {height!r}"
    )
    assert fx > 0 and fy > 0, (
        f"(as_sensor_intrinsics): focal lengths must be positive, got {fx}, {fy}"
    )
    assert 0 < cx < width and 0 < cy < height, (
        f"(as_sensor_intrinsics): principal point ({cx}, {cy}) lies outside {width}x{height}"
    )
    return SensorIntrinsics(fx, fy, cx, cy, width, height)


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
        assert {"members", "sensor_from_ref"} <= set(entry) <= {"name", "members", "sensor_from_ref", "intrinsics"}, (
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
        intrinsics = None
        if "intrinsics" in entry:
            intrinsics = [as_sensor_intrinsics(k) for k in entry["intrinsics"]]
            assert len(intrinsics) == len(members), (
                f"(load_rig_spec): rig {i} has {len(members)} members but {len(intrinsics)} intrinsics"
            )
        rigs.append(Rig(entry.get("name", f"rig_{i}"), members, mats, intrinsics))
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
    frames_of = [len(images[r.members[0]]) for r in rigs]
    for r, n in zip(rigs, frames_of):
        assert n > 0, f"(load_rig_spec): rig {r.name} has an empty image list"
    for p in soft_priors:
        ra, rb = rig_of[p.a], rig_of[p.b]
        assert frames_of[ra] == frames_of[rb], (
            f"(load_rig_spec): soft prior ({p.a}, {p.b}) links rigs with "
            f"{frames_of[ra]} vs {frames_of[rb]} frames, priors pair instants "
            f"so their rigs must be frame-aligned"
        )
    all_paths = []
    for s in sensors:
        paths = images[s]
        n_frames = frames_of[rig_of[s]]
        assert len(paths) == n_frames, (
            f"(load_rig_spec): sensor {s} has {len(paths)} images, "
            f"expected {n_frames} for rig {rigs[rig_of[s]].name}"
        )
        for p in paths:
            assert isinstance(p, str), f"(load_rig_spec): sensor {s} has non-string image entry {p!r}"
            assert os.path.isfile(p), f"(load_rig_spec): missing image file {p}"
        rig = rigs[rig_of[s]]
        if rig.intrinsics is not None:
            K = rig.intrinsics[rig.members.index(s)]
            for p in paths:
                w, h = imagesize.get(p)
                assert (w, h) == (K.width, K.height), (
                    f"(load_rig_spec): sensor {s} declares {K.width}x{K.height}, "
                    f"but {p} is {w}x{h}"
                )
        all_paths += paths
    assert len(set(all_paths)) == len(all_paths), (
        "(load_rig_spec): duplicate image paths across sensors"
    )
    return RigSpec(sensors, rigs, soft_priors, images)


if __name__ == "__main__":
    spec = load_rig_spec("/home/salmonuser/DigitalTwins/gluemap/Debug/Bathroom/schema.yaml")
    rig = spec.rigs[0]
    n_frames = len(spec.images[rig.members[0]])
    print(f"{len(spec.sensors)} sensors, {n_frames} frames, rig {rig.name}, ref {rig.members[0]}")
    for member, M in zip(rig.members, rig.sensor_from_ref):
        print(f"{member} sensor_from_ref[:3] = {M[:3].tolist()}")
    last = spec.sensors[-1]
    image = spec.images[last][len(spec.images[last]) - 1]
    print(f"{image} -> {spec.sensor_of_image(image)}")
    print(f"rig_of_sensor({last}) -> {spec.rig_of_sensor(last).name}")
