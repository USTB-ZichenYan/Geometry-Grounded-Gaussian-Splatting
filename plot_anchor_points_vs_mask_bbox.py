#!/usr/bin/env python3
"""
Plot selected anchor points and mask bbox on top of a robot mask frame.

This is a small diagnostic helper for visually checking where a few robot-link
centroids project relative to the true mask extent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
WM_DIR = ROOT / "robotics_world_model" / "world_model"
if str(WM_DIR) not in sys.path:
    sys.path.insert(0, str(WM_DIR))

from pose_pipeline import PoseConfig, PosePipeline, build_default_intrinsics  # type: ignore


ANCHOR_POINTS_BASE = {
    "shoulder_yaw_l_link": [0.193137, 0.290020, 0.925774],
    "shoulder_yaw_r_link": [0.173744, -0.249277, 0.899430],
    "elbow_yaw_l_link": [0.391704, 0.357519, 0.907100],
    "elbow_yaw_r_link": [0.351607, -0.343743, 0.821253],
    "shoulder_pitch_l_link": [0.049552, 0.209877, 1.061915],
    "shoulder_pitch_r_link": [0.050770, -0.210822, 1.065894],
    "shoulder_roll_l_link": [0.079526, 0.233091, 1.037721],
    "shoulder_roll_r_link": [0.073467, -0.225181, 1.032724],
    "right_tcp_link": [0.507430, -0.430075, 0.866545],
    "left_tcp_link": [0.572404, 0.389997, 0.957290],
}


def _mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot anchor points vs mask bbox")
    parser.add_argument("--poses-json", required=True)
    parser.add_argument("--cameras-json", required=True)
    parser.add_argument("--robot-mask", required=True)
    parser.add_argument("--frame-id", default="frame_000000")
    parser.add_argument("--cam-id", default="head")
    parser.add_argument(
        "--recompute-head-pose",
        action="store_true",
        help="Ignore stored poses.json head pose and recompute it from q using pose_pipeline semantics",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    poses = json.loads(Path(args.poses_json).read_text())["frames"]
    frame = next(f for f in poses if f["frame_id"] == args.frame_id and f["cam_id"] == args.cam_id)
    if args.recompute_head_pose and args.cam_id == "head":
        pose_cfg = PoseConfig()
        q = np.asarray(frame["q"], dtype=np.float32)
        head_K, wrist_K = build_default_intrinsics()
        pipeline = PosePipeline(pose_cfg, head_K, wrist_K)
        T = pipeline.compute_poses(q)["head"]
    else:
        T = np.asarray(frame["pose"], dtype=np.float32)

    cams = json.loads(Path(args.cameras_json).read_text())["cameras"]
    cam = next(c for c in cams if c["cam_id"] == args.cam_id)
    K = np.asarray(cam["K"], dtype=np.float32)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

    mask = np.asarray(Image.open(args.robot_mask).convert("L"), dtype=np.uint8)
    mask_fg = mask > 0
    bbox = _mask_bbox(mask_fg)

    canvas = Image.new("RGB", (mask.shape[1], mask.shape[0]), (0, 0, 0))
    draw = ImageDraw.Draw(canvas, mode="RGBA")

    # Green translucent mask
    green = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    green[..., 1] = np.where(mask_fg, 255, 0).astype(np.uint8)
    green[..., 3] = np.where(mask_fg, 64, 0).astype(np.uint8)
    canvas = Image.alpha_composite(canvas.convert("RGBA"), Image.fromarray(green, mode="RGBA")).convert("RGB")
    draw = ImageDraw.Draw(canvas, mode="RGBA")

    if bbox is not None:
        draw.rectangle(bbox, outline=(0, 255, 255, 255), width=4)

    for name, p_base in ANCHOR_POINTS_BASE.items():
        p = np.asarray(p_base, dtype=np.float32)
        p_cam = T[:3, :3] @ p + T[:3, 3]
        if p_cam[2] <= 1e-6:
            continue
        u = fx * (float(p_cam[0]) / float(p_cam[2])) + cx
        v = fy * (float(p_cam[1]) / float(p_cam[2])) + cy
        r = 8
        draw.ellipse((u - r, v - r, u + r, v + r), fill=(255, 0, 0, 180), outline=(255, 255, 255, 255), width=2)
        draw.text((u + 10, v - 10), name, fill=(255, 255, 255, 255))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
