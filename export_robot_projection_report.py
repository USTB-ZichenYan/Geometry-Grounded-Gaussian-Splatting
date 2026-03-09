#!/usr/bin/env python3
"""
Export robot coordinate/projection report to JSON.

Includes:
  - intrinsics/extrinsics
  - link-local / base / camera centroids
  - uv projections
  - global xyz/cam/uv stats
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
WM_DIR = ROOT / "robotics_world_model" / "world_model"
if str(WM_DIR) not in sys.path:
    sys.path.insert(0, str(WM_DIR))

from kinematics_common import build_q_map_from_q32, compute_link_transforms  # type: ignore
from pose_pipeline import PoseConfig, PosePipeline, build_default_intrinsics  # type: ignore


def _project(points_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    z = points_cam[:, 2:3]
    eps = 1e-8
    z_safe = np.where(np.abs(z) < eps, np.sign(z) * eps + (z == 0) * eps, z)
    uvw = (K @ points_cam.T).T
    uv = uvw[:, :2] / z_safe
    return uv.astype(np.float32)


def _stats(arr: np.ndarray) -> dict[str, Any]:
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.shape[0]),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "p01": np.percentile(arr, 1, axis=0).tolist(),
        "p50": np.percentile(arr, 50, axis=0).tolist(),
        "p99": np.percentile(arr, 99, axis=0).tolist(),
    }


def _load_frame(poses_json: Path, frame_id: str | None, cam_id: str) -> dict[str, Any]:
    payload = json.loads(poses_json.read_text())
    frames = payload["frames"]
    if frame_id is None:
        for f in frames:
            if str(f.get("cam_id")) == cam_id:
                return f
        raise ValueError(f"No frame with cam_id={cam_id} in {poses_json}")
    for f in frames:
        if str(f.get("frame_id")) == frame_id and str(f.get("cam_id")) == cam_id:
            return f
    raise ValueError(f"Frame not found: frame_id={frame_id}, cam_id={cam_id}")


def _load_cam(cameras_json: Path, cam_id: str) -> tuple[np.ndarray, int, int]:
    payload = json.loads(cameras_json.read_text())
    for c in payload["cameras"]:
        if str(c.get("cam_id")) == cam_id:
            K = np.asarray(c["K"], dtype=np.float32)
            return K, int(c["width"]), int(c["height"])
    raise ValueError(f"Camera not found: {cam_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export robot projection report json")
    parser.add_argument("--init-npz", required=True)
    parser.add_argument("--poses-json", required=True)
    parser.add_argument("--cameras-json", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--cam-id", default="head")
    parser.add_argument("--frame-id", default=None, help="Default: first frame matching cam-id")
    parser.add_argument(
        "--use-recompute-head-pose",
        action="store_true",
        help="Use PosePipeline default T_head_base_to_cam for head pose",
    )
    args = parser.parse_args()

    init_npz = np.load(args.init_npz, allow_pickle=True)
    points_local = np.asarray(init_npz["points"], dtype=np.float32)
    link_ids = np.asarray(init_npz["link_ids"], dtype=np.int32)
    link_names = np.asarray(init_npz["link_names"])

    frame = _load_frame(Path(args.poses_json), args.frame_id, args.cam_id)
    q = np.asarray(frame["q"], dtype=np.float32)
    frame_id = str(frame["frame_id"])

    K, width, height = _load_cam(Path(args.cameras_json), args.cam_id)

    if args.use_recompute_head_pose and args.cam_id == "head":
        head_K, wrist_K = build_default_intrinsics()
        pose = PosePipeline(PoseConfig(), head_K, wrist_K).compute_poses(q)["head"]
    else:
        pose = np.asarray(frame["pose"], dtype=np.float32)

    q_map = build_q_map_from_q32(q)
    link_T = compute_link_transforms(Path(PoseConfig().urdf_path), q_map=q_map)

    n = points_local.shape[0]
    points_base = np.zeros((n, 3), dtype=np.float32)
    valid = np.zeros((n,), dtype=bool)

    per_link: list[dict[str, Any]] = []
    for li, lname in enumerate(link_names.tolist()):
        idx = np.where(link_ids == li)[0]
        if idx.size == 0:
            continue
        if lname not in link_T:
            continue
        T = np.asarray(link_T[lname], dtype=np.float32)
        pts = points_local[idx]
        pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
        pts_b = (T @ pts_h.T).T[:, :3]
        points_base[idx] = pts_b
        valid[idx] = True

        c_local = pts.mean(axis=0)
        c_base = pts_b.mean(axis=0)
        c_cam = (pose @ np.array([c_base[0], c_base[1], c_base[2], 1.0], dtype=np.float32))[:3]
        uv = _project(c_cam[None, :], K)[0]
        in_image = bool((c_cam[2] > 0) and (0 <= uv[0] < width) and (0 <= uv[1] < height))
        per_link.append(
            {
                "link": str(lname),
                "count": int(idx.size),
                "centroid_link_local": c_local.tolist(),
                "centroid_base": c_base.tolist(),
                "centroid_cam": c_cam.tolist(),
                "uv": uv.tolist(),
                "z_cam": float(c_cam[2]),
                "in_image": in_image,
            }
        )

    points_base = points_base[valid]
    points_cam = (pose @ np.concatenate([points_base, np.ones((points_base.shape[0], 1), dtype=np.float32)], axis=1).T).T[:, :3]
    uv = _project(points_cam, K)
    in_front = points_cam[:, 2] > 0
    in_image = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height) & in_front

    report = {
        "frame_id": frame_id,
        "cam_id": args.cam_id,
        "num_points_total": int(points_local.shape[0]),
        "num_points_valid_fk": int(points_base.shape[0]),
        "intrinsics": {
            "K": K.tolist(),
            "width": width,
            "height": height,
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
        },
        "extrinsics_w2c_used": np.asarray(pose, dtype=np.float32).tolist(),
        "q": q.tolist(),
        "global_stats": {
            "points_link_local_xyz": _stats(points_local),
            "points_base_xyz": _stats(points_base),
            "points_cam_xyz": _stats(points_cam),
            "points_uv": _stats(uv),
            "ratio_in_front": float(in_front.mean()) if in_front.size else 0.0,
            "ratio_in_image": float(in_image.mean()) if in_image.size else 0.0,
        },
        "per_link_centroids": sorted(per_link, key=lambda x: x["count"], reverse=True),
    }

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()

