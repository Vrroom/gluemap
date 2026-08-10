"""Relative pose between an iPhone (LiDAR depth) and a 360 camera from shared
AprilTag boards.

Blender-free: consumes images + depth + intrinsics and returns T_360<-iPhone.
"""

import json
import os
from dataclasses import dataclass, field
from PIL import Image

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # don't preallocate the GPU

import cv2
import jax
import jax.numpy as jnp
import jaxls
import numpy as np
from equilib import equi2pers
from equilib.numpy_utils.rotation import create_global2camera_rotation_matrix, create_rotation_matrix

G2C = create_global2camera_rotation_matrix()
ARUCO_DICT = {
    "tag16h5": cv2.aruco.DICT_APRILTAG_16H5,
    "tag25h9": cv2.aruco.DICT_APRILTAG_25H9,
    "tag36h10": cv2.aruco.DICT_APRILTAG_36H10,
    "tag36h11": cv2.aruco.DICT_APRILTAG_36H11,
}
DEFAULT_FACE_GRID = ([(y, p) for y in range(0, 360, 45) for p in (-60, -30, 0, 30, 60)]
                     + [(0, 90), (0, -90)])


@dataclass
class BoardSpec:
    family: str
    tag_ids: list
    tag_size_m: float = 0.04


@dataclass
class BaselinePrior:
    # weak prior on the iPhone->360 baseline from the iPhone pose: assume a vertical stick
    # (360 straight above). Normalize b = R_iph_world @ R^T @ t and push its up-component to
    # -1 (baseline points straight up); magnitude-independent, nothing from the 360.
    r_iphone_world: list  # 3x3 iPhone camera->world rotation (OpenCV frame)
    weight: float = 0.1


@dataclass
class SolverConfig:
    boards: list
    face_fov_deg: float = 60.0
    face_size: int = 1536
    face_grid: list = field(default_factory=lambda: list(DEFAULT_FACE_GRID))
    corner_refine: bool = True
    baseline: BaselinePrior = None  # None -> no prior, solver stays device-agnostic


# 360 camera frame (equilib convention) produced by the bearing functions below:
#   equirect CENTER -> +z (forward),  BOTTOM / below-horizon -> +y (down),  TOP -> -y (up),
#   LEFT -> -x,  RIGHT -> +x,  far edges (yaw 180) -> -z (behind).
def face_pixel_to_bearing(px, K, yaw, pitch):
    # Map pixels detected in a perspective "face" (cropped from the equirect at
    # orientation yaw/pitch, radians) to unit ray directions in the 360 camera frame.
    # 1. Unproject pixels through the face pinhole K -> rays in the face camera
    #    frame (OpenCV axes: x right, y down, z forward).
    rays = np.c_[(px[:, 0] - K[0, 2]) / K[0, 0],
                 (px[:, 1] - K[1, 2]) / K[1, 1],
                 np.ones(len(px))]
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    # 2. R_crop rotates a face-camera direction into the equirect (360) frame.
    #    equilib orients the face by (yaw, pitch); G2C converts between its global
    #    and camera axis conventions (z_down=False).
    R_crop = G2C.T @ create_rotation_matrix(roll=0.0, pitch=pitch, yaw=yaw, z_down=False) @ G2C
    # 3. Rotate every ray into the 360 camera frame -> the direction toward each
    #    pixel's scene point, in the 360 camera's own coordinates (not world).
    return rays @ R_crop.T

def show_image(image, name):
    # saves the array as-is (raw bytes); pass RGB if you want true colors.
    out = name
    Image.fromarray(np.ascontiguousarray(image).astype(np.uint8)).save(out)

def draw_detections(image, detections, name):
    vis = image.copy()
    t = max(2, image.shape[1] // 500)
    for tag_id, corners in detections.items():
        pts = corners.astype(np.int32)
        cv2.polylines(vis, [pts], True, (0, 255, 0), t)
        ctr = tuple(pts.mean(axis=0).astype(int))
        cv2.putText(vis, str(tag_id), ctr, cv2.FONT_HERSHEY_SIMPLEX, t / 3.0, (0, 0, 255), t)
    scale = 1000.0 / vis.shape[1]
    if scale < 1.0:
        vis = cv2.resize(vis, None, fx=scale, fy=scale)
    show_image(vis[..., ::-1], name)   # BGR -> RGB (show_image saves raw)

def draw_detections_equirect(image, detections, name):
    # detections: {key: 4x3 unit bearings in the 360 frame}. Project to equirect
    # pixels (center=+z, +x right, +y down).
    vis = image.copy()
    h, w = vis.shape[:2]
    t = max(2, w // 500)
    for tag_id, bearings in detections.items():
        col = (w / 2.0) * (1.0 + np.arctan2(bearings[:, 0], bearings[:, 2]) / np.pi)
        row = h * (0.5 + np.arcsin(np.clip(bearings[:, 1], -1.0, 1.0)) / np.pi)
        pts = np.c_[col, row].astype(np.int32)
        cv2.polylines(vis, [pts], True, (0, 255, 0), t)
        ctr = tuple(pts.mean(axis=0).astype(int))
        # cv2.putText(vis, str(tag_id), ctr, cv2.FONT_HERSHEY_SIMPLEX, t / 3.0, (0, 0, 255), t)
    scale = 1000.0 / w
    if scale < 1.0:
        vis = cv2.resize(vis, None, fx=scale, fy=scale)
    show_image(vis[..., ::-1], name)


def match_color(i, n):
    hue = int(180 * i / max(n, 1))
    bgr = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(c) for c in bgr)


def draw_matches(perspective, detections, equirect, bearings, name):
    # side-by-side: iPhone pixel detections | 360 bearings; each matched tag one color.
    left, right = perspective.copy(), equirect.copy()
    keys = [k for k in detections if k in bearings]
    h, w = right.shape[:2]
    tl, tr = max(2, left.shape[1] // 500), max(2, w // 500)
    for i, key in enumerate(keys):
        color = match_color(i, len(keys))
        pp = detections[key].astype(np.int32)
        col = (w / 2.0) * (1.0 + np.arctan2(bearings[key][:, 0], bearings[key][:, 2]) / np.pi)
        row = h * (0.5 + np.arcsin(np.clip(bearings[key][:, 1], -1.0, 1.0)) / np.pi)
        bp = np.c_[col, row].astype(np.int32)
        cv2.polylines(left, [pp], True, color, tl)
        cv2.polylines(right, [bp], True, color, tr)
        cv2.putText(left, str(key[1]), tuple(pp.mean(0).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, tl / 2.0, color, tl)
        cv2.putText(right, str(key[1]), tuple(bp.mean(0).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, tr / 2.0, color, tr)
    big = 720
    left = cv2.resize(left, (round(left.shape[1] * big / left.shape[0]), big))
    right = cv2.resize(right, (round(right.shape[1] * big / right.shape[0]), big))
    show_image(np.concatenate([left, right], axis=1)[..., ::-1], name)


class BoardSolverBase:
    def __init__(self, config):
        self.config = config
        self.check_uniqueness()
        self.printed = {(b.family, t) for b in config.boards for t in b.tag_ids}
        self.families = sorted({b.family for b in config.boards})
        for fam in self.families:
            assert fam in ARUCO_DICT, f"(__init__): unknown tag family {fam!r}"

    def check_uniqueness(self):
        seen = set()
        for board in self.config.boards:
            for tid in board.tag_ids:
                key = (board.family, tid)
                assert key not in seen, f"(check_uniqueness): duplicate code {key} across boards"
                seen.add(key)

    def face_intrinsics(self):
        f = self.config.face_size / (2.0 * np.tan(np.radians(self.config.face_fov_deg) / 2.0))
        c = self.config.face_size / 2.0
        return np.array([[f, 0.0, c], [0.0, f, c], [0.0, 0.0, 1.0]])

    def detect(self, image):
        # image is assumed BGR (OpenCV order)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        params = cv2.aruco.DetectorParameters()
        if self.config.corner_refine:
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG  # subpixel corners
        out = {}
        for family in self.families:
            detector = cv2.aruco.ArucoDetector(
                cv2.aruco.getPredefinedDictionary(ARUCO_DICT[family]), params)
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is None:
                continue
            for c, i in zip(corners, ids.flatten()):
                key = (family, int(i))
                if key in self.printed:
                    out[key] = c.reshape(4, 2)
        # draw_detections(image, out, "A.png")
        return out

    def backproject(self, dets, depth, intrinsics):
        fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
        out = {}
        for key, corners in dets.items():
            pts = []
            for u, v in corners:
                z = float(depth[int(round(v)), int(round(u))])
                pts.append(((u - cx) * z / fx, (v - cy) * z / fy, z))
            out[key] = np.array(pts)
        return out

    def scan_bearings(self, equirect):
        # equirect assumed BGR. Sweep the face grid; first detection of a code wins.
        equi_chw = np.transpose(equirect, (2, 0, 1)).astype(np.float32)
        K, sz, fov = self.face_intrinsics(), self.config.face_size, self.config.face_fov_deg
        out = {}
        for yaw_deg, pitch_deg in self.config.face_grid:
            yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
            face = equi2pers(equi_chw, rots={"roll": 0.0, "pitch": pitch, "yaw": yaw},
                             height=sz, width=sz, fov_x=fov)
            face = np.clip(np.transpose(face, (1, 2, 0)), 0, 255).astype(np.uint8)
            for key, corners in self.detect(face).items():
                if key not in out:
                    out[key] = face_pixel_to_bearing(corners, K, yaw, pitch)
        return out

    def match(self, points, bearings):
        P, B = [], []
        for key in points:
            if key in bearings:
                P.append(points[key])
                B.append(bearings[key])
        return np.concatenate(P, axis=0), np.concatenate(B, axis=0)


class RelativePoseSolver(BoardSolverBase):
    def solve(self, points, bearings):
        T = jaxls.SE3Var(0)

        def residual(vals, var, P, b):
            # predicted bearing (T applied to the iPhone point, normalized) vs observed.
            pred = vals[var].apply(P)
            return pred / jnp.linalg.norm(pred) - b

        costs = [jaxls.Cost(residual, (T, jnp.asarray(P), jnp.asarray(b)))
                 for P, b in zip(points, bearings)]

        bp = self.config.baseline
        if bp is not None:
            sw = float(bp.weight) ** 0.5

            def baseline_residual(vals, var, r_iph):
                # b = R_iph_world @ R^T @ t = -(iPhone->360 baseline). Unit-direction prior:
                # push b's up-component to -1 (360 straight above), magnitude-independent.
                b = r_iph @ vals[var].rotation().inverse().apply(vals[var].translation())
                bz = b[2] / jnp.sqrt(b @ b + 1e-12)  # safe norm: t=0 at init -> b=0
                return jnp.atleast_1d(sw * (bz + 1.0))

            costs.append(jaxls.Cost(baseline_residual, (T, jnp.asarray(bp.r_iphone_world))))
        solution = jaxls.LeastSquaresProblem(costs, [T]).analyze().solve()
        return np.asarray(solution[T].as_matrix())

    def fit(self, iphone_rgb, iphone_depth, iphone_intrinsics, equirect_360, viz_prefix=None):
        # Each arg is a list of N measurements from a RIGID iphone+360 rig, so the
        # relative pose is shared: concatenate every view's correspondences, solve once.
        print(f"(fit): {len(iphone_rgb)} measurement(s); iphone_rgb/equirect_360 assumed BGR")
        Ps, Bs = [], []
        for i, (rgb, depth, intr, equi) in enumerate(zip(iphone_rgb, iphone_depth, iphone_intrinsics, equirect_360)):
            dets = self.detect(rgb)
            bearings = self.scan_bearings(equi)
            if viz_prefix is not None:
                draw_matches(rgb, dets, equi, bearings, f"{viz_prefix}_{i}.png")
            points = self.backproject(dets, depth, intr)
            P, B = self.match(points, bearings)
            Ps.append(P)
            Bs.append(B)
        return self.solve(np.concatenate(Ps), np.concatenate(Bs))

class CenterVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros(3)):
    """c: 360 optical center in the iPhone {up, look, cross} basis."""

class LeveledRelativePoseSolver(BoardSolverBase):
    """
    Gravity-leveled 360 + iPhone: recover (c, psi) for one measurement.

    Only the optical center c (iPhone {up, look, cross} basis) and heading psi are observable, 
    so solve those 4 DOF instead of a full SE3.
    """

    @staticmethod
    def iphone_c2w_opencv(gt_iphone):
        """Blender iPhone world matrix -> (R1, t1), camera->world in OpenCV axes."""
        c2w = np.asarray(gt_iphone, float) @ np.diag([1.0, -1.0, -1.0, 1.0])
        return c2w[:3, :3], c2w[:3, 3]

    @staticmethod
    def iphone_basis(R1):
        """B = [up | look | cross] in world from the OpenCV iPhone rotation."""
        look, up = R1[:, 2], -R1[:, 1]
        cross = np.cross(up, look)
        return np.column_stack([up, look, cross])

    @staticmethod
    def world_points(points_cam, R1, t1):
        """iPhone-frame tag corners -> metric world points."""
        return np.asarray(points_cam) @ R1.T + t1

    @staticmethod
    def center_world(c, B, t1):
        """360 optical center in world from c in the {up, look, cross} basis."""
        return t1 + B @ c

    def solve(self, X_world, bearings, B, t1):
        """Joint least-squares: shared centre c and shared heading psi.

        Each argument is a list with one entry per measurement: X_world[i] the Nx3 world
        points, bearings[i] the Nx3 leveled bearings, B[i] the {up,look,cross} basis, t1[i]
        the iPhone world position.
        """
        R_G0 = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])

        def leveled_residual(vals, c_var, heading_var, X, b, B, t1):
            """Predicted world ray (from c) minus the leveled bearing mapped to world."""
            c = vals[c_var]
            t2 = t1 + B @ c
            pred = X - t2
            pred = pred / jnp.linalg.norm(pred)
            w = jnp.asarray(R_G0) @ b
            xy = vals[heading_var].apply(w[:2])
            meas = jnp.array([xy[0], xy[1], w[2]])
            return pred - meas

        c_var, heading = CenterVar(0), jaxls.SO2Var(0)
        costs = []
        for Xm, bm, Bm, t1m in zip(X_world, bearings, B, t1):
            Bj, t1j = jnp.asarray(Bm), jnp.asarray(t1m)
            costs += [jaxls.Cost(leveled_residual, (c_var, heading, jnp.asarray(X), jnp.asarray(b), Bj, t1j))
                      for X, b in zip(Xm, bm)]
        sol = jaxls.LeastSquaresProblem(costs, [c_var, heading]).analyze().solve()
        return np.asarray(sol[c_var]), float(sol[heading].as_radians())

    def fit(self, iphone_rgb, iphone_depth, iphone_intrinsics, equirect_360, gt_iphone, viz_prefix=None):
        # Lists of N measurements from a RIGID iphone+360 rig: the 360 centre c (iPhone
        # {up,look,cross} basis) and the heading psi are shared across all measurements.
        print(f"(fit): {len(iphone_rgb)} measurement(s); leveled solve (shared centre + shared heading)")
        Xs, bs, Bs, t1s = [], [], [], []
        for i, (rgb, depth, intr, equi, gt) in enumerate(
                zip(iphone_rgb, iphone_depth, iphone_intrinsics, equirect_360, gt_iphone)):
            dets = self.detect(rgb)
            bearings = self.scan_bearings(equi)
            if viz_prefix is not None:
                draw_matches(rgb, dets, equi, bearings, f"{viz_prefix}_{i}.png")
            P_cam, b = self.match(self.backproject(dets, depth, intr), bearings)
            R1, t1 = self.iphone_c2w_opencv(gt)
            Xs.append(self.world_points(P_cam, R1, t1))
            bs.append(b)
            Bs.append(self.iphone_basis(R1))
            t1s.append(t1)
        c, psi = self.solve(Xs, bs, Bs, t1s)
        t2 = [self.center_world(c, Bm, t1m) for Bm, t1m in zip(Bs, t1s)]
        return {"c": c, "psi": psi, "t2": t2, "n_points": [len(X) for X in Xs]}


class IPhoneAlignmentCapture:
    ARKIT_TO_ZUP = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    def __init__(self, alignment_dir):
        self.dir = alignment_dir
        with open(os.path.join(alignment_dir, "solver_request.json")) as f:
            req = json.load(f)
        assert req["status"] == "ready_for_solver", f"(IPhoneAlignmentCapture::__init__): status {req['status']}"
        self.measurements = req["measurements"]
        self.marker_family = req["markerFamily"]
        self.tag_size_m = req["tagSizeMeters"]

    @staticmethod
    def convert_pose(pose_flat):
        pose = np.asarray(pose_flat, float).reshape(4, 4)
        assert np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5), f"(IPhoneAlignmentCapture::convert_pose): bad bottom row {pose[3]}"
        assert np.allclose(pose[:3, :3] @ pose[:3, :3].T, np.eye(3), atol=1e-4), "(IPhoneAlignmentCapture::convert_pose): R not orthonormal"
        return IPhoneAlignmentCapture.ARKIT_TO_ZUP @ pose

    def load_measurement(self, name):
        d = os.path.join(self.dir, name)
        with open(os.path.join(d, "measurement_meta.json")) as f:
            meta = json.load(f)
        assert meta["trackingState"] == "normal", f"(IPhoneAlignmentCapture::load_measurement): {name} trackingState {meta['trackingState']}"
        with open(os.path.join(d, meta["intrinsicsPath"])) as f:
            intr = json.load(f)
        rgb = cv2.imread(os.path.join(d, meta["phoneImagePath"]))
        depth = np.load(os.path.join(d, meta["phoneDepthPath"]))
        equirect = cv2.imread(os.path.join(d, meta["equirect360ImagePath"]))
        assert rgb.shape[:2] == (intr["height"], intr["width"]), f"(IPhoneAlignmentCapture::load_measurement): rgb {rgb.shape} vs intrinsics"
        assert depth.shape == rgb.shape[:2], f"(IPhoneAlignmentCapture::load_measurement): depth {depth.shape} vs rgb"
        assert equirect.shape[1] == 2 * equirect.shape[0], f"(IPhoneAlignmentCapture::load_measurement): equirect {equirect.shape} not 2:1"
        gt_iphone = self.convert_pose(meta["poseWorldTCam"])
        return rgb, depth, intr, equirect, gt_iphone

    def solver_inputs(self):
        loaded = [self.load_measurement(m) for m in self.measurements]
        rgbs, depths, intrs, equirects, gt_iphones = map(list, zip(*loaded))
        return rgbs, depths, intrs, equirects, gt_iphones

    def board_spec(self):
        n = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[self.marker_family]).bytesList.shape[0]
        return BoardSpec(family=self.marker_family, tag_ids=list(range(n)),
                         tag_size_m=self.tag_size_m)


def load_measurement(cache_dir):
    """(rgb, depth, intrinsics, equirect) read from a rendered cache dir (BGR images)."""
    rgb = cv2.imread(os.path.join(cache_dir, "rgb_iphone.png"))
    equirect = cv2.imread(os.path.join(cache_dir, "equirect_360.png"))
    depth = np.load(os.path.join(cache_dir, "depth_iphone.npy"))
    with open(os.path.join(cache_dir, "intrinsics.json")) as f:
        intrinsics = json.load(f)
    return rgb, depth, intrinsics, equirect


def load_alignment_measurement(folder):
    """Like load_measurement, but the 360 panorama is a JPG (equirect_360.jpg)."""
    rgb = cv2.imread(os.path.join(folder, "rgb_iphone.png"))
    equirect = cv2.imread(os.path.join(folder, "equirect_360.jpg"))
    depth = np.load(os.path.join(folder, "depth_iphone.npy"))
    with open(os.path.join(folder, "intrinsics.json")) as f:
        intrinsics = json.load(f)
    return rgb, depth, intrinsics, equirect


if __name__ == "__main__":
    root = "/home/salmonuser/Data/iPhone360Captures/AKWLab_2026-06-25_1200/alignment"
    dirs = sorted(os.path.join(root, x) for x in os.listdir(root)
                  if os.path.isdir(os.path.join(root, x))
                  and os.path.exists(os.path.join(root, x, "intrinsics.json")))
    print(f"(main): {len(dirs)} measurement(s): {[os.path.basename(d) for d in dirs]}")
    config = SolverConfig(boards=[BoardSpec("tag16h5", list(range(30)), 0.04)])
    solver = RelativePoseSolver(config)
    rgbs, depths, intrs, equis = zip(*(load_alignment_measurement(d) for d in dirs))
    T = solver.fit(list(rgbs), list(depths), list(intrs), list(equis))
    print("T_360<-iPhone =\n", T)
    print("translation t_360<-iPhone (m) =", T[:3, 3])

    # --- previous main: single rendered measurement with a gt-derived baseline prior ---
    # cache_dir = "Logs/Jun24_AprilTagRelativePoseExpts/distance/Cache/d00_s00"
    # with open(os.path.join(cache_dir, "gt_poses.json")) as f:
    #     gt = json.load(f)
    # R_iph = (np.array(gt["iPhoneCam"]) @ np.diag([1.0, -1.0, -1.0, 1.0]))[:3, :3]
    # config = SolverConfig(boards=[BoardSpec("tag16h5", list(range(30)))],
    #                       baseline=BaselinePrior(R_iph.tolist(), weight=1.0))
    # solver = RelativePoseSolver(config)
    # rgb, depth, intrinsics, equirect = load_measurement(cache_dir)
    # T = solver.fit([rgb], [depth], [intrinsics], [equirect])
    # print("T_360<-iPhone =\n", T)
