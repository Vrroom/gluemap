"""End-to-end RealityScan pipeline for a Debug scene folder.

Per scene: align+mesh in RealityScan, headless model + registration export,
frame-rotation check, dense points into points3D, binary COLMAP conversion,
staging and rsync to bouchet with checksum verification.

Run: python realityscan_pipeline.py <scene_dir> [<scene_dir> ...] --dest <bouchet_folder>
"""

import argparse
import os
import subprocess
import time
from itertools import permutations, product

import numpy as np
import pycolmap
import trimesh
from scipy.spatial import cKDTree

WINE = '/opt/realityscan/bin/wine'
RSEXE = 'C:/Program Files/Epic Games/RealityScan/RealityScan.exe'
META = '/home/salmonuser/DigitalTwins/gluemap/realityScriptMeta'
REG_PARAMS = f'{META}/registration_export_colmap_undistorted.xml'
MODEL_PARAMS = f'{META}/model_export_ply.xml'
BOUCHET_GSPLAT = '/home/sc3344/project_pi_jd374/sc3344/Data/Gsplat'
STAGING_ROOT = '/home/salmonuser/DigitalTwins/gluemap/results'


def to_windows_path(path):
    assert os.path.isabs(path), f'(to_windows_path): need absolute path, got {path}'
    assert ' ' not in path, f'(to_windows_path): RealityScan rejects spaces: {path}'
    return 'Z:' + path.replace('/', '\\')


def run_realityscan(args_list):
    # Wine processes from our own previous invocation linger for a few
    # seconds after the app quits, so wait for them before concluding a
    # genuinely concurrent instance is running.
    for _ in range(30):
        probe = subprocess.run(['pgrep', '-f', 'RealityScan.exe'], capture_output=True)
        if probe.returncode != 0:
            break
        time.sleep(1)
    assert probe.returncode != 0, \
        '(run_realityscan): another RealityScan instance is running, close it first'
    cmd = [WINE, '--bottle=default', '--wait-children',
           '--workdir', 'C:/Program Files/Epic Games/RealityScan', RSEXE]
    subprocess.run(cmd + args_list + ['-quit'], check=True)


def align_and_mesh(scene_dir):
    images = os.path.join(scene_dir, 'images')
    assert os.path.isdir(images), f'(align_and_mesh): no images folder in {scene_dir}'
    project = os.path.join(scene_dir, 'RealityScanOut', 'scan.rsproj')
    if os.path.exists(project):
        print(f'(align_and_mesh): {project} exists, skipping align')
        return project
    os.makedirs(os.path.dirname(project), exist_ok=True)
    run_realityscan([
        '-newScene',
        '-addFolder', to_windows_path(images),
        '-align',
        '-selectMaximalComponent',
        '-setReconstructionRegionAuto',
        '-calculateNormalModel',
        '-calculateVertexColors',
        '-save', to_windows_path(project)])
    assert os.path.exists(project), f'(align_and_mesh): {project} not written'
    return project


def export_model(project_path, out_ply):
    if os.path.exists(out_ply):
        print(f'(export_model): {out_ply} exists, skipping')
        return
    run_realityscan([
        '-load', to_windows_path(project_path),
        '-exportModel', 'Model 1', to_windows_path(out_ply),
        to_windows_path(MODEL_PARAMS)])
    assert os.path.exists(out_ply), f'(export_model): {out_ply} not written'


def export_registration(project_path, out_dir):
    sparse = os.path.join(out_dir, 'sparse', '0')
    if os.path.isdir(sparse):
        print(f'(export_registration): {sparse} exists, skipping')
        return sparse
    os.makedirs(out_dir, exist_ok=True)
    anchor = os.path.join(out_dir, 'reg.txt')
    run_realityscan([
        '-load', to_windows_path(project_path),
        '-exportRegistration', to_windows_path(anchor), to_windows_path(REG_PARAMS)])
    assert os.path.isdir(sparse), f'(export_registration): {sparse} not created'
    return sparse


def axis_aligned_rotations():
    rots = []
    for perm in permutations(range(3)):
        for signs in product([1, -1], repeat=3):
            R = np.zeros((3, 3))
            R[np.arange(3), perm] = signs
            if np.linalg.det(R) > 0:
                rots.append(R)
    return rots


def find_reg_to_mesh_rotation(sparse_dir, ply_path):
    rec = pycolmap.Reconstruction(sparse_dir)
    pts = np.array([p.xyz for p in rec.points3D.values()])
    radii = np.linalg.norm(pts - np.median(pts, axis=0), axis=1)
    pts = pts[radii < 3 * np.percentile(radii, 90)]
    rng = np.random.default_rng(0)
    sample = pts[rng.choice(len(pts), min(20000, len(pts)), replace=False)]
    mv = np.asarray(trimesh.load(ply_path, process=False).vertices)
    mv = mv[rng.choice(len(mv), min(300000, len(mv)), replace=False)]
    tree = cKDTree(mv)
    diag = float(np.linalg.norm(
        np.percentile(mv, 95, axis=0) - np.percentile(mv, 5, axis=0)))
    scores = [(float(np.median(tree.query(sample @ R.T, workers=8)[0])), R)
              for R in axis_aligned_rotations()]
    scores.sort(key=lambda s: s[0])
    (d0, R0), (d1, _) = scores[0], scores[1]
    print(f'(find_reg_to_mesh_rotation): best {d0:.4f}, second {d1:.4f}, '
          f'mesh diag {diag:.2f}')
    assert d0 < 0.01 * diag, \
        f'(find_reg_to_mesh_rotation): best residual {d0:.4f} above 1% of diag {diag:.2f}'
    assert d1 > 3 * d0, \
        f'(find_reg_to_mesh_rotation): not decisive, second {d1:.4f} < 3x best {d0:.4f}'
    return R0


def replace_points3d_with_dense(sparse_dir, ply_path, rotation):
    points_txt = os.path.join(sparse_dir, 'points3D.txt')
    with open(points_txt) as f:
        for line in f:
            if not line.startswith('#'):
                break
    if len(line.split()) == 8:
        print('(replace_points3d_with_dense): points3D.txt probably already dense '
              '(first point has no track), skipping')
        return
    mesh = trimesh.load(ply_path, process=False)
    xyz = np.asarray(mesh.vertices, dtype=np.float64) @ rotation
    rgb = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.int64)
    assert len(rgb) == len(xyz), '(replace_points3d_with_dense): color count mismatch'
    assert rgb.std(axis=0).max() > 1, '(replace_points3d_with_dense): mesh has no vertex colors'
    n = len(xyz)
    rows = np.column_stack([np.arange(1, n + 1), xyz, rgb, np.zeros(n)])
    with open(points_txt, 'w') as f:
        f.write('# 3D point list with one line of data per point:\n'
                '#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n'
                f'# Number of points: {n}, mean track length: 0\n')
        np.savetxt(f, rows, fmt=['%d', '%.6f', '%.6f', '%.6f', '%d', '%d', '%d', '%d'])
    print(f'(replace_points3d_with_dense): wrote {n} dense points')


def convert_to_binary(sparse_txt_dir, out_bin_dir):
    os.makedirs(out_bin_dir, exist_ok=True)
    rec = pycolmap.Reconstruction(sparse_txt_dir)
    rec.write_binary(out_bin_dir)
    back = pycolmap.Reconstruction(out_bin_dir)
    assert back.num_images() == rec.num_images(), \
        f'(convert_to_binary): image count {back.num_images()} != {rec.num_images()}'
    assert back.num_points3D() == rec.num_points3D(), \
        f'(convert_to_binary): point count {back.num_points3D()} != {rec.num_points3D()}'
    names = {im.name: i for i, im in back.images.items()}
    for im in rec.images.values():
        a, b = im.cam_from_world(), back.images[names[im.name]].cam_from_world()
        assert np.allclose(a.rotation.quat, b.rotation.quat) and \
            np.allclose(a.translation, b.translation), \
            f'(convert_to_binary): pose mismatch for {im.name}'
    print(f'(convert_to_binary): {back.num_points3D()} points, '
          f'{back.num_images()} images round-trip clean')


def stage_scene(scene_dir, dest_name):
    scene_name = os.path.basename(scene_dir).lower() + '_realityscan'
    sparse = os.path.join(scene_dir, 'RealityScanOut', 'sparse', '0')
    assert os.path.isdir(sparse), f'(stage_scene): missing {sparse}'
    staging = os.path.join(STAGING_ROOT, f'export_{dest_name}', scene_name)
    convert_to_binary(sparse, os.path.join(staging, 'sparse'))
    return staging


def rsync_and_verify(staging_dir, images_dir, dest_name, scene_name):
    dest = f'bouchet:{BOUCHET_GSPLAT}/{dest_name}/{scene_name}'
    subprocess.run(['rsync', '-a', '--stats', staging_dir + '/', dest + '/'],
                   check=True)
    subprocess.run(['rsync', '-a', '--stats', images_dir + '/', dest + '/images/'],
                   check=True)
    checks = [(staging_dir + '/sparse/', dest + '/sparse/'),
              (images_dir + '/', dest + '/images/')]
    for src, dst in checks:
        out = subprocess.run(['rsync', '-rcin', '--delete', src, dst],
                             capture_output=True, text=True, check=True)
        assert out.stdout.strip() == '', \
            f'(rsync_and_verify): checksum mismatch {src} vs {dst}:\n{out.stdout}'
    print(f'(rsync_and_verify): {dest} sparse+images checksum clean')


def scene_report(scene_dir, sparse_dir):
    total = len([f for f in os.listdir(os.path.join(scene_dir, 'images'))
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    scan = os.path.join(scene_dir, 'RealityScanOut', 'scan')
    comps = len([f for f in os.listdir(scan)
                 if f.startswith('sfm') and f.endswith('.dat')])
    n = pycolmap.Reconstruction(sparse_dir).num_images()
    print(f'(scene_report): {n}/{total} images in maximal component '
          f'({100 * n / total:.1f}%), {comps} components in project')


def run_scene(scene_dir, dest_name):
    scene_dir = os.path.abspath(scene_dir)
    print(f'=== (run_scene): {scene_dir}')
    project = align_and_mesh(scene_dir)
    rs_out = os.path.dirname(project)
    ply = os.path.join(rs_out, 'model.ply')
    export_model(project, ply)
    sparse = export_registration(project, rs_out)
    # RealityScan drops a metadata db into the source images folder on every
    # project open; remove it after the last RealityScan stage of this run.
    rsmeta = os.path.join(scene_dir, 'images', 'rsmeta.db')
    if os.path.exists(rsmeta):
        os.remove(rsmeta)
    rotation = find_reg_to_mesh_rotation(sparse, ply)
    print(f'(run_scene): reg->mesh rotation\n{rotation.astype(int)}')
    replace_points3d_with_dense(sparse, ply, rotation)
    scene_report(scene_dir, sparse)
    staging = stage_scene(scene_dir, dest_name)
    rsync_and_verify(staging, os.path.join(rs_out, 'images'),
                     dest_name, os.path.basename(staging))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('scene_dirs', nargs='+')
    ap.add_argument('--dest', required=True,
                    help='bouchet Data/Gsplat subfolder, e.g. auditorium_jul30')
    args = ap.parse_args()
    for scene_dir in args.scene_dirs:
        run_scene(scene_dir, args.dest)


if __name__ == '__main__':
    main()
