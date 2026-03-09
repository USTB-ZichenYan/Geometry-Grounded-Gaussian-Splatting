#!/usr/bin/env python3
"""
One-shot geometry consistency check for robot FK training/rendering.

It reports:
1) point projection visibility (in-front / in-image) under both pose conventions
2) alpha_in_robot / alpha_in_bg from render alpha + dataset mask
3) frame_id <-> q_ref alignment (if robot state is provided)

Usage example:
  PY=/home/SENSETIME/yanzichen/anaconda3/envs/gggs/bin/python
  ROOT=/home/SENSETIME/yanzichen/data/file/Geometry-Grounded-Gaussian-Splatting
  "$PY" "$ROOT/scripts/check_robot_geometry_consistency.py" \
    --poses-json "$ROOT/gggs_run/gs_dataset_orange_1f/poses.json" \
    --cameras-json "$ROOT/gggs_run/gs_dataset_orange_1f/cameras.json" \
    --frame-id frame_000000 \
    --cam-id head \
    --state "$ROOT/gggs_run/robot_fk_out_orange_left_1f/robot_fk_state_final.pth" \
    --mask "$ROOT/gggs_run/gs_colmap_orange_left_1f/masks/frame_000000_head.png" \
    --alpha "$ROOT/gggs_run/robot_fk_render_orange_left_1f/robot_alpha/00000.png" \
    --out "$ROOT/gggs_run/debug/geometry_check_000000.json"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _to_torch_matrix(x: Any) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32)


def _find_frame(poses_json: Path, frame_id: str, cam_id: str) -> dict[str, Any]:
    payload = _load_json(poses_json)
    frames = payload.get("frames", [])
    for fr in frames:
        if fr.get("frame_id") == frame_id and fr.get("cam_id") == cam_id:
            return fr
    raise KeyError(f"frame not found: frame_id={frame_id}, cam_id={cam_id}")


def _find_cam(cameras_json: Path, cam_id: str) -> dict[str, Any]:
    payload = _load_json(cameras_json)
    cams = payload.get("cameras", [])
    for cam in cams:
        if cam.get("cam_id") == cam_id:
            return cam
    raise KeyError(f"camera not found: cam_id={cam_id}")


def _load_alpha(path: Path) -> torch.Tensor:
    im = Image.open(path).convert("L")
    return torch.tensor(list(im.getdata()), dtype=torch.float32).reshape(im.height, im.width) / 255.0


def _load_mask(path: Path) -> torch.Tensor:
    # Input convention in this project: white=bg keep, black=robot
    im = Image.open(path).convert("L")
    return torch.tensor(list(im.getdata()), dtype=torch.float32).reshape(im.height, im.width) / 255.0


def _project_stats(points_base: torch.Tensor, K: torch.Tensor, pose_4x4: torch.Tensor) -> dict[str, float]:
    n = points_base.shape[0]
    ones = torch.ones((n, 1), dtype=torch.float32)
    xh = torch.cat([points_base, ones], dim=1)  # [N,4]

    xc_h = (pose_4x4 @ xh.T).T
    z = xc_h[:, 2].clamp_min(1e-8)
    in_front = xc_h[:, 2] > 1e-6

    u = K[0, 0] * (xc_h[:, 0] / z) + K[0, 2]
    v = K[1, 1] * (xc_h[:, 1] / z) + K[1, 2]
    return {
        "in_front_ratio": float(in_front.float().mean().item()),
        "u_min": float(u.min().item()),
        "u_max": float(u.max().item()),
        "v_min": float(v.min().item()),
        "v_max": float(v.max().item()),
    }


def _project_in_image_ratio(
    points_base: torch.Tensor,
    K: torch.Tensor,
    pose_4x4: torch.Tensor,
    width: int,
    height: int,
) -> float:
    n = points_base.shape[0]
    ones = torch.ones((n, 1), dtype=torch.float32)
    xh = torch.cat([points_base, ones], dim=1)
    xc_h = (pose_4x4 @ xh.T).T
    in_front = xc_h[:, 2] > 1e-6
    if not bool(in_front.any()):
        return 0.0
    xf = xc_h[in_front]
    z = xf[:, 2].clamp_min(1e-8)
    u = K[0, 0] * (xf[:, 0] / z) + K[0, 2]
    v = K[1, 1] * (xf[:, 1] / z) + K[1, 2]
    in_img = (u >= 0.0) & (u < float(width)) & (v >= 0.0) & (v < float(height))
    return float(in_img.float().mean().item())


def main() -> None:
    ap = argparse.ArgumentParser(description="Check geometry consistency for robot training/rendering.")
    ap.add_argument("--poses-json", required=True)
    ap.add_argument("--cameras-json", required=True)
    ap.add_argument("--frame-id", required=True)
    ap.add_argument("--cam-id", default="head")
    ap.add_argument("--state", required=True, help="robot_fk_state_final.pth")
    ap.add_argument("--mask", required=True, help="dataset/colmap mask png (white=bg, black=robot)")
    ap.add_argument("--alpha", required=True, help="rendered robot alpha png")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    frame = _find_frame(Path(args.poses_json), args.frame_id, args.cam_id)
    cam = _find_cam(Path(args.cameras_json), args.cam_id)
    K = _to_torch_matrix(cam["K"])
    width = int(cam["width"])
    height = int(cam["height"])
    pose = _to_torch_matrix(frame["pose"])
    q_frame = torch.tensor(frame["q"], dtype=torch.float32)

    state = torch.load(args.state, map_location="cpu")
    gauss = state["gaussians"]
    xyz = gauss[2].detach().float().cpu()  # capture() index 2 is xyz
    q_ref = torch.tensor(state.get("q_ref", []), dtype=torch.float32)

    # Convention check: frame pose might be w2c or c2w depending on data source
    pose_w2c = pose
    pose_c2w_inv = torch.linalg.inv(pose)
    in_img_w2c = _project_in_image_ratio(xyz, K, pose_w2c, width, height)
    in_img_c2w = _project_in_image_ratio(xyz, K, pose_c2w_inv, width, height)
    best_mode = "w2c" if in_img_w2c >= in_img_c2w else "c2w"

    pose_best = pose_w2c if best_mode == "w2c" else pose_c2w_inv
    proj_stats = _project_stats(xyz, K, pose_best)
    proj_stats["in_img_of_front"] = max(in_img_w2c, in_img_c2w)

    alpha = _load_alpha(Path(args.alpha))
    mask = _load_mask(Path(args.mask))
    if alpha.shape != mask.shape:
        raise ValueError(f"shape mismatch: alpha={tuple(alpha.shape)} mask={tuple(mask.shape)}")

    robot = (mask < 0.5).float()
    bg = 1.0 - robot
    alpha_in_robot = float((alpha * robot).sum().item() / max(float(robot.sum().item()), 1.0))
    alpha_in_bg = float((alpha * bg).sum().item() / max(float(bg.sum().item()), 1.0))

    q_l2 = None
    q_max_abs = None
    if q_ref.numel() > 0 and q_ref.numel() == q_frame.numel():
        diff = q_ref - q_frame
        q_l2 = float(torch.linalg.vector_norm(diff).item())
        q_max_abs = float(diff.abs().max().item())

    out = {
        "frame": {"frame_id": args.frame_id, "cam_id": args.cam_id},
        "camera": {"width": width, "height": height, "K": cam["K"]},
        "pose_convention_check": {
            "in_img_if_pose_is_w2c": in_img_w2c,
            "in_img_if_pose_is_c2w": in_img_c2w,
            "selected_for_report": best_mode,
        },
        "projection": proj_stats,
        "alpha": {
            "alpha_mean": float(alpha.mean().item()),
            "alpha_in_robot": alpha_in_robot,
            "alpha_in_bg": alpha_in_bg,
            "robot_pixel_ratio": float(robot.mean().item()),
        },
        "q_alignment": {
            "has_q_ref": bool(q_ref.numel() > 0),
            "same_dim": bool(q_ref.numel() == q_frame.numel()),
            "q_l2": q_l2,
            "q_max_abs": q_max_abs,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[saved] {out_path}")
    print(
        f"[summary] in_img={out['projection']['in_img_of_front']:.4f}, "
        f"alpha_in_robot={alpha_in_robot:.6f}, alpha_in_bg={alpha_in_bg:.6f}, "
        f"pose_mode={best_mode}, q_l2={q_l2}"
    )


if __name__ == "__main__":
    main()
