#!/usr/bin/env python3
"""
Visualize canonical prior Gaussian points projected to one frame.

Inputs:
  - init npz: points(link-local), link_ids, link_names
  - poses.json: frame pose + q
  - cameras.json: K, width, height

Output:
  - one RGB image of projected prior points
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from robotics_world_model.world_model.kinematics_common import (
        build_q_map_from_q32,
        compute_link_transforms,
    )
except Exception:
    from world_model.kinematics_common import (  # type: ignore
        build_q_map_from_q32,
        compute_link_transforms,
    )


def _find_frame(poses_json: Path, frame_id: str, cam_id: str) -> dict:
    payload = json.loads(poses_json.read_text())
    for fr in payload.get("frames", []):
        if fr.get("frame_id") == frame_id and fr.get("cam_id") == cam_id:
            return fr
    raise KeyError(f"frame not found: {frame_id}/{cam_id}")


def _find_cam(cameras_json: Path, cam_id: str) -> dict:
    payload = json.loads(cameras_json.read_text())
    for cam in payload.get("cameras", []):
        if cam.get("cam_id") == cam_id:
            return cam
    raise KeyError(f"camera not found: {cam_id}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Project init prior points to one frame")
    ap.add_argument("--init-npz", required=True)
    ap.add_argument("--poses-json", required=True)
    ap.add_argument("--cameras-json", required=True)
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--frame-id", default="frame_000000")
    ap.add_argument("--cam-id", default="head")
    ap.add_argument("--out", required=True)
    ap.add_argument("--point-radius", type=int, default=1)
    ap.add_argument("--sample-points", type=int, default=20000)
    args = ap.parse_args()

    npz = np.load(args.init_npz, allow_pickle=False)
    points = np.asarray(npz["points"], dtype=np.float32)
    link_ids = np.asarray(npz["link_ids"], dtype=np.int64)
    link_names = [str(x) for x in np.asarray(npz["link_names"]).tolist()]

    if args.sample_points > 0 and points.shape[0] > int(args.sample_points):
        idx = np.linspace(0, points.shape[0] - 1, int(args.sample_points)).astype(np.int64)
        points = points[idx]
        link_ids = link_ids[idx]

    fr = _find_frame(Path(args.poses_json), args.frame_id, args.cam_id)
    cam = _find_cam(Path(args.cameras_json), args.cam_id)
    q = np.asarray(fr["q"], dtype=np.float32)
    T_w2c = np.asarray(fr["pose"], dtype=np.float32)
    K = np.asarray(cam["K"], dtype=np.float32)
    W = int(cam["width"])
    H = int(cam["height"])

    T_links = compute_link_transforms(Path(args.urdf), q_map=build_q_map_from_q32(q))
    xyz_world = points.copy()
    for lid, ln in enumerate(link_names):
        m = link_ids == lid
        if not np.any(m):
            continue
        T = T_links.get(ln)
        if T is None:
            continue
        R = T[:3, :3].astype(np.float32)
        t = T[:3, 3].astype(np.float32)
        xyz_world[m] = xyz_world[m] @ R.T + t[None, :]

    xh = np.concatenate([xyz_world, np.ones((xyz_world.shape[0], 1), dtype=np.float32)], axis=1)
    xc = (T_w2c @ xh.T).T
    z = xc[:, 2]
    valid = z > 1e-6
    if not np.any(valid):
        img = np.zeros((H, W, 3), dtype=np.uint8)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(out)
        print(f"[saved] {out} (no valid points)")
        return

    xc = xc[valid]
    z = xc[:, 2]
    u = K[0, 0] * (xc[:, 0] / z) + K[0, 2]
    v = K[1, 1] * (xc[:, 1] / z) + K[1, 2]
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
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

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out)
    print(f"[saved] {out}")
    print(f"[stats] inside={len(ui)} total={len(points)}")


if __name__ == "__main__":
    main()

