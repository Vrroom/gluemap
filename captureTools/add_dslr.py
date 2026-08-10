"""Add a DSLR image set to an exported scene as a single-sensor rig with
unknown intrinsics and no link to the existing rig.

Run: python captureTools/add_dslr.py <scene_dir> <dslr_dir> [--take N]
"""

import argparse
import os
import shutil

import numpy as np
import yaml
from PIL import Image, ImageOps


def copy_dslr_images(dslr_dir, images_dir, take):
    src = sorted(f for f in os.listdir(dslr_dir)
                 if f.lower().endswith((".jpg", ".jpeg", ".png")))
    assert src, f"(copy_dslr_images): no images in {dslr_dir}"
    if take is not None:
        assert 0 < take <= len(src), f"(copy_dslr_images): take={take} of {len(src)}"
        src = [src[i] for i in np.round(np.linspace(0, len(src) - 1, take)).astype(int)]
    out = []
    n_rotated = 0
    for j, name in enumerate(src):
        dst = os.path.join(images_dir, f"cam_dslr_f{j}{os.path.splitext(name)[1]}")
        assert not os.path.exists(dst), f"(copy_dslr_images): {dst} already exists"
        im = Image.open(os.path.join(dslr_dir, name))
        if (im.getexif().get(274, 1)) == 1:
            shutil.copy2(os.path.join(dslr_dir, name), dst)
        else:
            # bake the EXIF orientation into the pixels and drop the tag, so
            # every loader (cv2 applies it, PIL/imagesize ignore it) agrees
            ImageOps.exif_transpose(im).save(dst, quality=100)
            n_rotated += 1
        out.append(dst)
    print(f"(copy_dslr_images): {len(out)} of {len(os.listdir(dslr_dir))} images -> {images_dir}")
    return out


def extend_schema(schema_path, dslr_paths):
    with open(schema_path) as f:
        doc = yaml.safe_load(f)
    assert "cam_dslr" not in doc["sensors"], f"(extend_schema): cam_dslr already in {schema_path}"
    doc["n_sensors"] += 1
    # the exporter anchors rigs[0].members to the sensors list, so appending in
    # place would grow the rig too; a new list breaks the aliasing
    doc["sensors"] = doc["sensors"] + ["cam_dslr"]
    doc["rigs"].append({"name": "dslr", "members": ["cam_dslr"],
                        "sensor_from_ref": [np.eye(4).tolist()]})
    doc["images"]["cam_dslr"] = dslr_paths
    with open(schema_path, "w") as f:
        yaml.safe_dump(doc, f, default_flow_style=None, sort_keys=False, width=100000)
    print(f"(extend_schema): {schema_path} now has {doc['n_sensors']} sensors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir")
    ap.add_argument("dslr_dir")
    ap.add_argument("--take", type=int, default=None)
    args = ap.parse_args()

    images_dir = os.path.join(os.path.abspath(args.scene_dir), "images")
    schema_path = os.path.join(os.path.abspath(args.scene_dir), "schema.yaml")
    assert os.path.isdir(images_dir), f"(main): missing {images_dir}"
    assert os.path.isfile(schema_path), f"(main): missing {schema_path}"

    paths = copy_dslr_images(args.dslr_dir, images_dir, args.take)
    extend_schema(schema_path, paths)

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from gluemap.utils.rigs import load_rig_spec
    spec = load_rig_spec(schema_path)
    counts = {r.name: len(spec.images[r.members[0]]) for r in spec.rigs}
    print(f"(main): round-trip OK, rigs={counts}")


if __name__ == "__main__":
    main()
