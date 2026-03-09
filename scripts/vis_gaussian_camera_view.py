#!/usr/bin/env python3
"""
Project trained Gaussian centers to one camera view (single image output).

This is NOT renderer output. It is a geometric debug image:
  - input: point_cloud.ply
  - camera: head_camera_calib.json (w2c + intrinsics)
  - output: one image with projected points

Example:
  PY=/home/SENSETIME/yanzichen/anaconda3/envs/gggs/bin/python
  ROOT=/home/SENSETIME/yanzichen/data/file/Geometry-Grounded-Gaussian-Splatting
  "$PY" "$ROOT/scripts/vis_gaussian_camera_view.py" \
    --ply "$ROOT/gggs_run/robot_fk_out_orange_left_1f_12k_w2c/point_cloud/iteration_3000/point_cloud.ply" \
    --calib "$ROOT/robotics_world_model/world_model/head_camera_calib.json" \
    --out "$ROOT/gggs_run/debug/gaussian_head_view.png" \
    --point-radius 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData


def _load_calib(path: Path):
    cfg = json.loads(path.read_text())["head"]
    K = np.array(
        [
            [float(cfg["fx"]), 0.0, float(cfg["cx"])],
            [0.0, float(cfg["fy"]), float(cfg["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    T_w2c = np.array(cfg["w2c"], dtype=np.float32)
    W, H = int(cfg["width"]), int(cfg["height"])
    return K, T_w2c, W, H


def main() -> None:
    ap = argparse.ArgumentParser(description="Single camera-view Gaussian projection")
    ap.add_argument("--ply", required=True, help="point_cloud.ply")
    ap.add_argument("--calib", required=True, help="head_camera_calib.json")
    ap.add_argument("--out", required=True, help="output image path")
    ap.add_argument("--point-radius", type=int, default=1, help="point draw radius in px")
    args = ap.parse_args()

    ply_path = Path(args.ply)
    calib_path = Path(args.calib)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    K, T_w2c, W, H = _load_calib(calib_path)
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"].data
    xyz = np.stack(
        [np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])],
        axis=1,
    ).astype(np.float32)

    ones = np.ones((xyz.shape[0], 1), dtype=np.float32)
    xh = np.concatenate([xyz, ones], axis=1)
    xc = (T_w2c @ xh.T).T
    z = xc[:, 2]
    valid = z > 1e-6
    if not np.any(valid):
        Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8)).save(out_path)
        print(f"[saved] {out_path} (no points in front)")
        return

    xc = xc[valid]
    z = z[valid]
    u = K[0, 0] * (xc[:, 0] / z) + K[0, 2]
    vv = K[1, 1] * (xc[:, 1] / z) + K[1, 2]

    ui = np.round(u).astype(np.int32)
    vi = np.round(vv).astype(np.int32)
    inside = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui = ui[inside]
    vi = vi[inside]
    z = z[inside]

    img = np.zeros((H, W, 3), dtype=np.uint8)
    if ui.size > 0:
        zmin, zmax = float(z.min()), float(z.max())
        zn = (z - zmin) / (zmax - zmin + 1e-8)
        r = max(0, int(args.point_radius))
        for x, y, t in zip(ui, vi, zn):
            c = np.array([255 * (1.0 - t), 80, 255 * t], dtype=np.uint8)
            y0, y1 = max(0, y - r), min(H, y + r + 1)
            x0, x1 = max(0, x - r), min(W, x + r + 1)
            img[y0:y1, x0:x1] = c

    Image.fromarray(img).save(out_path)
    print(f"[saved] {out_path}")
    print(f"[stats] projected_inside={len(ui)} total={len(xyz)}")


if __name__ == "__main__":
    main()
