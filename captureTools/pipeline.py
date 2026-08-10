"""Calibrate an iPhone+360 video take, export it as gluemap Debug scenes (a hard
rig and a 2-rig variant with a soft prior), and optionally reconstruct them.

Run from the gluemap root in the gluemap env:

    conda run -n gluemap python captureTools/pipeline.py \
        <run_dir> <out_root> [--name AKWLab] [--n 50] [--run Hard MiniHard]
"""

import argparse
import json
import os
import shutil
import subprocess

import cv2
import numpy as np
import yaml

# equilib imports torch, whose conda CUDA 12.4 libraries win the library load race and
# break jax's CUDA 12.9 cusolver. Initializing jax's solver first keeps both working.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax.numpy as jnp

jnp.linalg.solve(jnp.eye(2), jnp.ones(2))

from relative_pose_solver import IPhoneAlignmentCapture, LeveledRelativePoseSolver, SolverConfig
from rig_capture import VideoTake, measure_audio_offset
from rig_simulation import (azimuth_top, cube_faces, cube_low, face_intrinsics,
                            rebase_sensor_from_ref, sensor_from_ref_matrices,
                            simulate_rigid_faces, yaw_of)


def rig_spec(faces, name360):
    sensors = ["cam_phone"] + [f"cam_{name}" for name, _, _ in faces]
    return {"faces": faces, "sensors": sensors, "name360": name360}
UPRIGHT_ROT = {1: cv2.ROTATE_90_COUNTERCLOCKWISE, 2: cv2.ROTATE_180,
               3: cv2.ROTATE_90_CLOCKWISE}  # image rotation undoing k camera quarter-turns about the look axis


def upright_image(img, k):
    return img if k == 0 else cv2.rotate(img, UPRIGHT_ROT[k])


def upright_intrinsics(intr, k):
    if k == 0:
        return dict(intr)
    w, h = intr["width"], intr["height"]
    wv, hv = (h, w) if k % 2 else (w, h)
    t = np.radians(90.0 * k)
    r_inv = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
    d = r_inv @ np.array([intr["cx"] - (w - 1) / 2, intr["cy"] - (h - 1) / 2])
    fx, fy = (intr["fy"], intr["fx"]) if k % 2 else (intr["fx"], intr["fy"])
    return {"fx": float(fx), "fy": float(fy), "cx": float(round(d[0] + (wv - 1) / 2, 6)),
            "cy": float(round(d[1] + (hv - 1) / 2, 6)), "width": int(wv), "height": int(hv)}


def calibrate(run_dir, cache_path):
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            d = json.load(f)
        assert "psi_rel" in d, f"(calibrate): stale cache {cache_path}, delete it and rerun"
        return np.array(d["c"]), d["psi_rel"]
    cap = IPhoneAlignmentCapture(os.path.join(run_dir, "alignment"))
    solver = LeveledRelativePoseSolver(SolverConfig(boards=[cap.board_spec()]))
    res = solver.fit(*cap.solver_inputs())
    yaws = [yaw_of(LeveledRelativePoseSolver.iphone_c2w_opencv(cap.load_measurement(m)[4])[0])
            for m in cap.measurements]
    assert np.ptp(yaws) < np.radians(2.0), \
        f"(calibrate): measurement yaws spread {np.degrees(np.ptp(yaws)):.2f}deg, shared psi_rel invalid"
    psi_rel = float(res["psi"] - np.mean(yaws))
    with open(cache_path, "w") as f:
        json.dump({"c": res["c"].tolist(), "psi_rel": psi_rel}, f)
    return res["c"], psi_rel


def iphone_pose(meta):
    """ARKit frame meta -> (r1, t1), OpenCV camera->world in the z-up world."""
    pose = IPhoneAlignmentCapture.convert_pose(meta["poseWorldTCam"])
    return LeveledRelativePoseSolver.iphone_c2w_opencv(pose)


def iphone_poses(take, step=1):
    return [iphone_pose(meta) for meta in take.frames[::step]]


def measure_body_roll(r1s):
    """Best cube-aligned roll about the cam look axis: (score, quarter_turns, rz)."""
    best = None
    for k in range(4):
        t = np.radians(90.0 * k)
        rz = np.array([[np.cos(t), -np.sin(t), 0.0], [np.sin(t), np.cos(t), 0.0], [0.0, 0.0, 1.0]])
        score = np.mean([(r @ rz @ np.array([0.0, -1.0, 0.0]))[2] for r in r1s])
        if best is None or score > best[0]:
            best = (score, k, rz)
    return best


def phone_intrinsics(take):
    ks = np.array([[m["fx"], m["fy"], m["cx"], m["cy"]] for m in take.frames])
    w, h = take.frames[0]["imageWidth"], take.frames[0]["imageHeight"]
    assert all(m["imageWidth"] == w and m["imageHeight"] == h for m in take.frames), \
        "(phone_intrinsics): image size varies across frames"
    mean, spread = ks.mean(axis=0), np.abs(ks - ks.mean(axis=0)).max()
    print(f"(phone_intrinsics): mean fx={mean[0]:.2f} cx={mean[2]:.2f}, "
          f"max |dev| {spread:.2f} px over {len(ks)} frames")
    return {"fx": float(round(mean[0], 6)), "fy": float(round(mean[1], 6)),
            "cx": float(round(mean[2], 6)), "cy": float(round(mean[3], 6)),
            "width": int(w), "height": int(h)}


def export_frame_images(take, psi, r_roll, k, n, images_dir, rig):
    os.makedirs(images_dir, exist_ok=True)
    images = {s: [] for s in rig["sensors"]}
    poses = []
    for j, inst in enumerate(take.sample_instants(n)):
        r1, t1 = iphone_pose(inst["meta"])
        T = np.eye(4)
        T[:3, :3] = r1 @ r_roll
        T[:3, 3] = t1
        poses.append({"instant": j, "phone_index": inst["phone_index"],
                      "timestamp": inst["meta"]["timestamp"],
                      "world_from_ref": np.round(T, 8).tolist()})
        phone = upright_image(cv2.imread(os.path.join(take.phone_dir, inst["meta"]["imagePath"])), k)
        face_imgs = simulate_rigid_faces(inst["equirect"], psi, r1 @ r_roll, rig["faces"])
        for sensor, img in [("cam_phone", phone)] + [(f"cam_{f}", v) for f, v in face_imgs.items()]:
            path = os.path.join(images_dir, f"{sensor}_f{j}.png")
            cv2.imwrite(path, img)
            images[sensor].append(path)
    return images, poses


def write_schema(path, sensors, rigs, soft_priors, images):
    rig_docs = []
    for r in rigs:
        entry = {"name": r["name"], "members": r["members"],
                 "sensor_from_ref": [np.round(m, 6).tolist() for m in r["sensor_from_ref"]]}
        if "intrinsics" in r:
            entry["intrinsics"] = [dict(k) for k in r["intrinsics"]]  # copies keep repeated entries from becoming yaml anchors
        rig_docs.append(entry)
    doc = {"n_sensors": len(sensors), "sensors": sensors,
           "rigs": rig_docs,
           "soft_priors": [{"pair": list(sp["pair"]),
                            "b_from_a": np.round(sp["b_from_a"], 6).tolist(),
                            "w_rot": sp["w_rot"], "w_trans": sp["w_trans"]}
                           for sp in soft_priors],
           "images": images}
    with open(path, "w") as f:
        yaml.safe_dump(doc, f, default_flow_style=None, sort_keys=False, width=100000)


def write_poses(path, poses):
    doc = {"convention": "world_from_ref: OpenCV cam->world, z-up ARKit-derived world, row-major 4x4. "
                         "ref = the upright phone camera (schema sensor_from_ref identity member); "
                         "world_from_sensor = world_from_ref @ inv(sensor_from_ref)",
           "poses": poses}
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)


def export_scene_pair(take, c, psi, k, r_roll, phone_k, n, out_root, name, w_rot, w_trans, rig):
    hard_dir = os.path.join(out_root, f"{name}Hard")
    soft_dir = os.path.join(out_root, f"{name}Soft")
    hard_images, poses = export_frame_images(take, psi, r_roll, k, n, os.path.join(hard_dir, "images"), rig)
    shutil.copytree(os.path.join(hard_dir, "images"), os.path.join(soft_dir, "images"),
                    dirs_exist_ok=True)
    write_poses(os.path.join(hard_dir, "poses.json"), poses)
    write_poses(os.path.join(soft_dir, "poses.json"), poses)
    soft_images = {s: [p.replace(hard_dir, soft_dir) for p in paths]
                   for s, paths in hard_images.items()}

    sensors = rig["sensors"]
    mats = sensor_from_ref_matrices(c, r_roll, rig["faces"])
    by_sensor = {"cam_phone": mats["phone"], **{f"cam_{nm}": mats[nm] for nm, _, _ in rig["faces"]}}
    intr = {"cam_phone": phone_k,
            **{f"cam_{nm}": face_intrinsics(fov_deg=fov) for nm, _, fov in rig["faces"]}}
    hard_rigs = [{"name": "iphone360", "members": sensors,
                  "sensor_from_ref": [by_sensor[s] for s in sensors],
                  "intrinsics": [intr[s] for s in sensors]}]
    write_schema(os.path.join(hard_dir, "schema.yaml"), sensors, hard_rigs, [], hard_images)

    ring = sensors[1:]
    rebased = rebase_sensor_from_ref(by_sensor, ring)
    soft_rigs = [{"name": "iphone", "members": [sensors[0]], "sensor_from_ref": [np.eye(4)],
                  "intrinsics": [phone_k]},
                 {"name": rig["name360"], "members": ring,
                  "sensor_from_ref": [rebased[s] for s in ring],
                  "intrinsics": [intr[s] for s in ring]}]
    prior = {"pair": [sensors[0], ring[0]], "b_from_a": by_sensor[ring[0]],
             "w_rot": w_rot, "w_trans": w_trans}
    write_schema(os.path.join(soft_dir, "schema.yaml"), sensors, soft_rigs, [prior], soft_images)
    print(f"(export_scene_pair): wrote {hard_dir} + {soft_dir} ({n} instants)")


def write_demo_config(scene_dir, configs_dir="configs"):
    scene = os.path.basename(os.path.normpath(scene_dir))
    doc = {"_base_": "base.yaml",
           "images_path": os.path.abspath(os.path.join(scene_dir, "images")) + "/",
           "write_path": f"results/scenes/{scene}/",
           "rig_config_path": os.path.abspath(os.path.join(scene_dir, "schema.yaml")),
           "gt_intrinsics_path": None}
    path = os.path.join(configs_dir, f"{scene.lower()}.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(doc, f, default_flow_style=False, sort_keys=False)
    print(f"(write_demo_config): wrote {path}")
    return path


def run_gluemap(config_path, num_workers=None):
    cmd = ["gluemap-demo", "--config", config_path]
    if num_workers is not None:
        cmd += ["--num_workers", str(num_workers)]
    print(f"(run_gluemap): {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("out_root")
    ap.add_argument("--name", default="AKWLab")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--n-mini", type=int, default=5)
    ap.add_argument("--w-rot", type=float, default=1.0)
    ap.add_argument("--w-trans", type=float, default=1.0)
    ap.add_argument("--run", nargs="*", default=[],
                    choices=["Hard", "Soft", "MiniHard", "MiniSoft"])
    ap.add_argument("--rig", default="cube", choices=["cube", "azimuth_top", "cube_low"])
    ap.add_argument("--num-workers", type=int, default=None)
    args = ap.parse_args()
    args.out_root = os.path.abspath(args.out_root)
    c, psi_rel = calibrate(args.run_dir, os.path.join(args.out_root, f"{args.name}_calibration.json"))
    sync = measure_audio_offset(args.run_dir, os.path.join(args.out_root, f"{args.name}_sync_offset.json"))
    take = VideoTake(args.run_dir, sync)
    with open(take.video_path.replace(".mp4", ".json")) as f:
        stab = json.load(f)["stabilizationMode"]
    assert stab == "zDirectional", f"(main): stabilizationMode {stab!r}, export assumes zDirectional"
    score, k, r_roll = measure_body_roll([r for r, _ in iphone_poses(take, step=20)])
    print(f"(main): c={np.round(c, 4)} psi_rel={np.degrees(psi_rel):.2f}deg "
          f"roll={90 * k}deg (up-score {score:.3f})")
    phone_k = upright_intrinsics(phone_intrinsics(take), k)
    rig = {"cube": lambda: rig_spec(cube_faces(include_down=False), "cube360"),
           "azimuth_top": lambda: rig_spec(azimuth_top(), "aztop360"),
           "cube_low": lambda: rig_spec(cube_low(), "cubelow360")}[args.rig]()
    for name, n in [(args.name, args.n), (args.name + "Mini", args.n_mini)]:
        export_scene_pair(take, c, psi_rel, k, r_roll, phone_k, n, args.out_root, name,
                          args.w_rot, args.w_trans, rig)
    for variant in args.run:
        cfg = write_demo_config(os.path.join(args.out_root, args.name + variant))
        run_gluemap(cfg, args.num_workers)


if __name__ == "__main__":
    main()
