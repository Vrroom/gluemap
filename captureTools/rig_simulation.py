"""Simulate a rigid iPhone+360 rig from a z-leveled, yaw-following equirect
video (insta360 zDirectional): the equirect pans with the rig, so only the
phone's tilt is applied, against a constant rig-relative heading psi_rel.

Face resample rotation: R = equirect_c2w(psi_rel).T @ remove_yaw(r1) @ R_face
(face-camera rays -> stabilized-equirect-camera rays, all OpenCV cam axes).
"""

import cv2
import numpy as np

R_G0 = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
FACE_NAMES = ["front", "right", "back", "left", "up", "down"]


def cube_face_rotations():
    def ry(deg):
        t = np.radians(deg)
        return np.array([[np.cos(t), 0.0, np.sin(t)], [0.0, 1.0, 0.0], [-np.sin(t), 0.0, np.cos(t)]])

    def rx(deg):
        t = np.radians(deg)
        return np.array([[1.0, 0.0, 0.0], [0.0, np.cos(t), -np.sin(t)], [0.0, np.sin(t), np.cos(t)]])

    return {"front": np.eye(3), "right": ry(90), "back": ry(180), "left": ry(-90),
            "up": rx(90), "down": rx(-90)}


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def face_rotation(azimuth_deg, elevation_deg):
    # center ray points at azimuth (from front toward +x) and elevation (up-positive)
    return rot_y(np.radians(azimuth_deg)) @ rot_x(np.radians(elevation_deg))


def cube_faces(fov_deg=90.0, include_down=True):
    rots = cube_face_rotations()
    names = FACE_NAMES if include_down else [n for n in FACE_NAMES if n != "down"]
    return [(name, rots[name], fov_deg) for name in names]


def azimuth_top(n_azimuth=8, ring_fov_deg=55.0, ring_elevation_deg=0.0,
                n_top=3, top_elevation_deg=58.0, top_fov_deg=70.0,
                top_azimuth_offset_deg=60.0, add_zenith=False, zenith_fov_deg=70.0):
    faces = []
    for i in range(n_azimuth):
        az = round(i * 360.0 / n_azimuth) % 360
        faces.append((f"az{az:03d}", face_rotation(az, ring_elevation_deg), ring_fov_deg))
    for i in range(n_top):
        az = round(top_azimuth_offset_deg + i * 360.0 / n_top) % 360
        faces.append((f"top{az:03d}", face_rotation(az, top_elevation_deg), top_fov_deg))
    if add_zenith:
        faces.append(("zenith", face_rotation(0.0, 90.0), zenith_fov_deg))
    names = [f[0] for f in faces]
    assert len(set(names)) == len(names), f"(azimuth_top): duplicate face names {names}"
    return faces


def cube_low(fov_deg=90.0, low_elevation_deg=-45.0, low_fov_deg=None):
    faces = cube_faces(fov_deg, include_down=False)
    low_fov = fov_deg if low_fov_deg is None else low_fov_deg
    for az in (0, 90, 180, 270):
        faces.append((f"low{az:03d}", face_rotation(az, low_elevation_deg), low_fov))
    names = [f[0] for f in faces]
    assert len(set(names)) == len(names), f"(cube_low): duplicate face names {names}"
    return faces


def rot_z(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def equirect_c2w(psi):
    return rot_z(psi) @ R_G0


def yaw_of(r_c2w):
    look = np.asarray(r_c2w)[:, 2]
    assert np.hypot(look[0], look[1]) > 1e-3, f"(yaw_of): look axis near vertical, {look}"
    return np.arctan2(look[1], look[0])


def remove_yaw(r_c2w):
    return rot_z(-yaw_of(r_c2w)) @ r_c2w


def equirect_to_face(equirect, R, face_size, fov_deg=90.0):
    # pinhole focal for a square face_size spanning fov_deg on each axis; fov 90 gives f = size/2
    h, w = equirect.shape[:2]
    f = (face_size / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    c = (face_size - 1) / 2.0
    u, v = np.meshgrid(np.arange(face_size), np.arange(face_size))
    rays = np.stack([(u - c) / f, (v - c) / f, np.ones((face_size, face_size))], axis=-1)
    d = rays @ R.T
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    map_x = (w / 2.0) * (1.0 + np.arctan2(d[..., 0], d[..., 2]) / np.pi) - 0.5
    map_y = h * (0.5 + np.arcsin(d[..., 1]) / np.pi) - 0.5
    padded = np.concatenate([equirect, equirect[:, :1]], axis=1)
    return cv2.remap(padded, np.mod(map_x, w).astype(np.float32), map_y.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def face_intrinsics(face_size=960, fov_deg=90.0):
    # matches equirect_to_face: f = (size/2)/tan(fov/2), principal point at pixel centers
    f = round(float((face_size / 2.0) / np.tan(np.radians(fov_deg) / 2.0)), 6)
    return {"fx": f, "fy": f,
            "cx": (face_size - 1) / 2.0, "cy": (face_size - 1) / 2.0,
            "width": face_size, "height": face_size}


def iphone_cam_offset(c):
    # {up, look, cross} coords -> iPhone OpenCV cam axes (up=-y, look=+z, cross=-x)
    c = np.asarray(c, float)
    return np.array([-c[2], -c[0], c[1]])


def sensor_from_ref_matrices(c, r_roll, faces):
    # T_sensor<-ref (pycolmap sensor_from_rig) per rig member. Reference = the upright
    # phone (body frame); phone images must be pixel-rotated by r_roll to match.
    offset = np.asarray(r_roll).T @ iphone_cam_offset(c)
    mats = {"phone": np.eye(4)}
    for name, rf, _ in faces:
        T = np.eye(4)
        T[:3, :3] = rf.T
        T[:3, 3] = -rf.T @ offset
        mats[name] = T
    return mats


def rebase_sensor_from_ref(mats, members):
    # re-reference a subset so its first member becomes the identity reference
    ref_inv = np.linalg.inv(mats[members[0]])
    return {m: mats[m] @ ref_inv for m in members}


def simulate_rigid_faces(equirect, psi_rel, r_iphone_world, faces, face_size=960):
    assert equirect.shape[1] == 2 * equirect.shape[0], f"(simulate_rigid_faces): equirect {equirect.shape} not 2:1"
    r_iphone_world = np.asarray(r_iphone_world)
    assert r_iphone_world.shape == (3, 3), f"(simulate_rigid_faces): r_iphone_world {r_iphone_world.shape}"
    r_e2w = equirect_c2w(psi_rel)
    r_tilt = remove_yaw(r_iphone_world)
    return {name: equirect_to_face(equirect, r_e2w.T @ r_tilt @ r_face, face_size, fov)
            for name, r_face, fov in faces}
