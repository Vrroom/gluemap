"""Merge several exported Soft takes of one scene into a single gluemap scene.

Run from the gluemap root in the gluemap env:

    conda run -n gluemap python captureTools/merge_takes.py <out_dir> \
        --take sun Debug/SSS3SunSoft \
        --take shade Debug/SSS3ShadeSoft \
        --take light Debug/SSS3LightSoft
"""

import argparse
import json
import os

import numpy as np
import yaml

from gluemap.datasets.utils import get_image_list
from gluemap.utils.rigs import bind_rig_spec, load_rig_spec


def load_takes(pairs):
    """[(label, scene_dir)] -> [{label, scene_dir, spec, poses}] in the given order."""
    takes = []
    for label, scene_dir in pairs:
        assert label and "/" not in label, \
            f"(load_takes): take label {label!r} must be a bare directory name"
        scene_dir = os.path.abspath(scene_dir)
        spec = load_rig_spec(os.path.join(scene_dir, "schema.yaml"))
        with open(os.path.join(scene_dir, "poses.json")) as f:
            poses = json.load(f)
        n_frames = len(spec.images[spec.rigs[0].members[0]])
        assert len(poses["poses"]) == n_frames, \
            f"(load_takes): take {label} has {len(poses['poses'])} poses but {n_frames} frames"
        takes.append({"label": label, "scene_dir": scene_dir, "spec": spec, "poses": poses})
    assert len(takes) >= 2, f"(load_takes): need at least two takes, got {len(takes)}"
    labels = [t["label"] for t in takes]
    assert len(set(labels)) == len(labels), f"(load_takes): duplicate take labels {labels}"
    return takes


def pose_deviation(m_ref, m):
    """(rotation deg, sensor centre mm) between two sensor_from_ref matrices."""
    assert m_ref.shape == (4, 4) and m.shape == (4, 4), \
        f"(pose_deviation): expected a 4x4 pair, got {m_ref.shape} and {m.shape}"
    r_rel = m[:3, :3] @ m_ref[:3, :3].T
    cos = (np.trace(r_rel) - 1.0) / 2.0
    deg = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    centre_ref = -m_ref[:3, :3].T @ m_ref[:3, 3]
    centre = -m[:3, :3].T @ m[:3, 3]
    return deg, float(np.linalg.norm(centre - centre_ref) * 1000.0)


def assert_rigs_agree(takes, max_rot_deg, max_trans_mm):
    """Stop unless every take declares the same sensors, rigs, sensor_from_ref,
    intrinsics and soft priors as the first one."""
    ref = takes[0]
    for t in takes[1:]:
        assert t["spec"].sensors == ref["spec"].sensors, \
            f"(assert_rigs_agree): take {t['label']} sensors {t['spec'].sensors} != " \
            f"take {ref['label']} sensors {ref['spec'].sensors}"
        assert len(t["spec"].rigs) == len(ref["spec"].rigs), \
            f"(assert_rigs_agree): take {t['label']} has {len(t['spec'].rigs)} rigs, " \
            f"take {ref['label']} has {len(ref['spec'].rigs)}"
        devs = []
        for r0, r in zip(ref["spec"].rigs, t["spec"].rigs):
            assert (r.name, r.members) == (r0.name, r0.members), \
                f"(assert_rigs_agree): take {t['label']} rig {r.name} members {r.members} != " \
                f"take {ref['label']} rig {r0.name} members {r0.members}"
            assert (r.intrinsics is None) == (r0.intrinsics is None), \
                f"(assert_rigs_agree): take {t['label']} rig {r.name} declares " \
                f"intrinsics={r.intrinsics is not None}, take {ref['label']} declares " \
                f"{r0.intrinsics is not None}"
            if r0.intrinsics is not None:
                for member, k0, k in zip(r0.members, r0.intrinsics, r.intrinsics):
                    px = max(abs(getattr(k, f) - getattr(k0, f))
                             for f in ("fx", "fy", "cx", "cy"))
                    if px > 0:
                        print(f"(assert_rigs_agree): {t['label']} {member} intrinsics differ "
                              f"from {ref['label']} by up to {px:.6f} px, kept per-take")
            for member, m0, m in zip(r0.members, r0.sensor_from_ref, r.sensor_from_ref):
                devs.append((member,) + pose_deviation(m0, m))
        worst_rot = max(devs, key=lambda d: d[1])
        worst_trans = max(devs, key=lambda d: d[2])
        print(f"(assert_rigs_agree): {t['label']} vs {ref['label']}: max "
              f"{worst_rot[1]:.4f} deg at {worst_rot[0]}, {worst_trans[2]:.3f} mm at {worst_trans[0]}")
        assert worst_rot[1] <= max_rot_deg, \
            f"(assert_rigs_agree): take {t['label']} sensor {worst_rot[0]} rotates " \
            f"{worst_rot[1]:.4f} deg from take {ref['label']}, limit {max_rot_deg} deg"
        assert worst_trans[2] <= max_trans_mm, \
            f"(assert_rigs_agree): take {t['label']} sensor {worst_trans[0]} centre moves " \
            f"{worst_trans[2]:.3f} mm from take {ref['label']}, limit {max_trans_mm} mm"
        assert len(t["spec"].soft_priors) == len(ref["spec"].soft_priors), \
            f"(assert_rigs_agree): take {t['label']} has {len(t['spec'].soft_priors)} soft " \
            f"priors, take {ref['label']} has {len(ref['spec'].soft_priors)}"
        for p0, p in zip(ref["spec"].soft_priors, t["spec"].soft_priors):
            assert (p.a, p.b, p.w_rot, p.w_trans) == (p0.a, p0.b, p0.w_rot, p0.w_trans), \
                f"(assert_rigs_agree): take {t['label']} prior ({p.a}, {p.b}) w_rot={p.w_rot} " \
                f"w_trans={p.w_trans} != take {ref['label']} ({p0.a}, {p0.b}) w_rot={p0.w_rot} " \
                f"w_trans={p0.w_trans}"
            deg, mm = pose_deviation(p0.b_from_a, p.b_from_a)
            assert deg <= max_rot_deg and mm <= max_trans_mm, \
                f"(assert_rigs_agree): take {t['label']} prior ({p.a}, {p.b}) b_from_a differs " \
                f"from take {ref['label']} by {deg:.4f} deg / {mm:.3f} mm, limits " \
                f"{max_rot_deg} deg / {max_trans_mm} mm"


def link_take_images(takes, images_dir):
    """Symlink each take's images/ into images_dir/<label>; label -> link path."""
    os.makedirs(images_dir, exist_ok=True)
    linked = {}
    for t in takes:
        src = os.path.join(t["scene_dir"], "images")
        assert os.path.isdir(src), f"(link_take_images): {src} is not a directory"
        dst = os.path.join(images_dir, t["label"])
        if os.path.islink(dst):
            assert os.readlink(dst) == src, \
                f"(link_take_images): {dst} already points at {os.readlink(dst)}, not {src}"
        else:
            assert not os.path.exists(dst), \
                f"(link_take_images): {dst} exists and is not a symlink"
            os.symlink(src, dst)
        linked[t["label"]] = dst
    return linked


def take_sensor_names(take):
    """Original sensor name -> its per-take name, e.g. cam_phone -> cam_phone_sun."""
    names = {s: f"{s}_{take['label']}" for s in take["spec"].sensors}
    assert len(set(names.values())) == len(names), \
        f"(take_sensor_names): take {take['label']} renaming collides, {sorted(names.values())}"
    return names


def merged_image_lists(takes, linked):
    """Per-take sensor name -> that take's paths, rewritten under its symlink dir."""
    images = {}
    for t in takes:
        names = take_sensor_names(t)
        src_root = os.path.join(t["scene_dir"], "images")
        for sensor, paths in t["spec"].images.items():
            out = []
            for p in paths:
                assert p.startswith(src_root + os.sep), \
                    f"(merged_image_lists): take {t['label']} image {p} lies outside {src_root}"
                out.append(os.path.join(linked[t["label"]], os.path.relpath(p, src_root)))
            images[names[sensor]] = out
    total = sum(len(t["spec"].sensors) for t in takes)
    assert len(images) == total, \
        f"(merged_image_lists): produced {len(images)} sensors, expected {total}"
    return images


def merged_schema_doc(takes, images):
    """Per-take sensors, per-take rigs, and each take's own soft priors."""
    sensors, rig_docs, priors = [], [], []
    for t in takes:
        names = take_sensor_names(t)
        sensors += [names[s] for s in t["spec"].sensors]
        for r in t["spec"].rigs:
            entry = {"name": f"{r.name}_{t['label']}",
                     "members": [names[m] for m in r.members],
                     "sensor_from_ref": [np.round(m, 6).tolist() for m in r.sensor_from_ref]}
            if r.intrinsics is not None:
                entry["intrinsics"] = [{"fx": k.fx, "fy": k.fy, "cx": k.cx, "cy": k.cy,
                                        "width": k.width, "height": k.height}
                                       for k in r.intrinsics]
            rig_docs.append(entry)
        for p in t["spec"].soft_priors:
            priors.append({"pair": [names[p.a], names[p.b]],
                           "b_from_a": np.round(p.b_from_a, 6).tolist(),
                           "w_rot": p.w_rot, "w_trans": p.w_trans})
    assert len(set(sensors)) == len(sensors) and set(sensors) == set(images), \
        f"(merged_schema_doc): {len(sensors)} sensors, {len(set(sensors))} unique, " \
        f"{len(images)} image lists"
    return {"n_sensors": len(sensors), "sensors": sensors, "rigs": rig_docs,
            "soft_priors": priors, "images": images}


def merged_poses_doc(takes):
    """One poses doc, every instant tagged with the take it came from."""
    convention = takes[0]["poses"]["convention"]
    entries, index = [], []
    for order, t in enumerate(takes):
        assert t["poses"]["convention"] == convention, \
            f"(merged_poses_doc): take {t['label']} declares a different pose convention"
        index.append({"take": t["label"], "order": order, "scene_dir": t["scene_dir"],
                      "n_frames": len(t["poses"]["poses"])})
        for frame, p in enumerate(t["poses"]["poses"]):
            assert p["instant"] == frame, \
                f"(merged_poses_doc): take {t['label']} pose {frame} carries instant {p['instant']}"
            entries.append({"take": t["label"], "frame": frame, **p})
    return {"convention": convention + " Each take has its own ARKit world, so world_from_ref "
                                       "is comparable only within one take, never across takes.",
            "takes": index, "poses": entries}


def write_merged(out_dir, schema_doc, poses_doc):
    os.makedirs(out_dir, exist_ok=True)
    schema_path = os.path.join(out_dir, "schema.yaml")
    with open(schema_path, "w") as f:
        yaml.safe_dump(schema_doc, f, default_flow_style=None, sort_keys=False, width=100000)
    poses_path = os.path.join(out_dir, "poses.json")
    with open(poses_path, "w") as f:
        json.dump(poses_doc, f, indent=2)
    print(f"(write_merged): wrote {schema_path} and {poses_path}")


def verify_merged(out_dir, takes):
    """Reload the written scene through gluemap's own loader and binder."""
    spec = load_rig_spec(os.path.join(out_dir, "schema.yaml"))
    expected = {f"{r.name}_{t['label']}": len(t["spec"].images[r.members[0]])
                for t in takes for r in t["spec"].rigs}
    counts = {r.name: len(spec.images[r.members[0]]) for r in spec.rigs}
    assert counts == expected, f"(verify_merged): frame counts {counts} != {expected}"
    paths = get_image_list(os.path.join(out_dir, "images") + "/")
    bound = bind_rig_spec(spec, paths, soft_averaging="hard")
    assert len(bound.ref_of) == len(paths), \
        f"(verify_merged): bound {len(bound.ref_of)} of {len(paths)} images"
    bound.averaging_view()
    print(f"(verify_merged): {len(spec.sensors)} sensors, {len(paths)} images, rigs={counts}")


def write_and_check_config(out_dir, configs_dir="configs"):
    """write_demo_config for the merged scene; it owns soft_rig_averaging, we verify it."""
    from pipeline import write_demo_config
    path = write_demo_config(out_dir, configs_dir)
    with open(path) as f:
        doc = yaml.safe_load(f)
    got = doc.get("soft_rig_averaging")
    assert got == "hard", \
        f"(write_and_check_config): {path} carries soft_rig_averaging={got!r}, expected 'hard' " \
        f"- write_demo_config is the only writer of this key"
    print(f"(write_and_check_config): {path} carries soft_rig_averaging=hard")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--take", nargs=2, action="append", metavar=("LABEL", "SCENE_DIR"),
                    required=True)
    ap.add_argument("--max-rot-deg", type=float, default=0.5)
    ap.add_argument("--max-trans-mm", type=float, default=5.0)
    args = ap.parse_args()
    out_dir = os.path.abspath(args.out_dir)
    takes = load_takes([tuple(pair) for pair in args.take])
    assert_rigs_agree(takes, args.max_rot_deg, args.max_trans_mm)
    linked = link_take_images(takes, os.path.join(out_dir, "images"))
    images = merged_image_lists(takes, linked)
    write_merged(out_dir, merged_schema_doc(takes, images), merged_poses_doc(takes))
    verify_merged(out_dir, takes)
    write_and_check_config(out_dir)


if __name__ == "__main__":
    main()
