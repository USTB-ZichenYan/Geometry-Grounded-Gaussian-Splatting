"""
Convert warm_start + waypoint images into GGGS dataset format.

Input directory example:
  images/
    warm_start.npy            # shape [N, D]
    waypoint_1.png ... png    # N images

Output directory:
  out_dir/
    images/head/frame_000000.png
    cameras.json
    poses.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from .pose_pipeline import build_default_intrinsics, PoseConfig, PosePipeline
except ImportError:  # pragma: no cover
    from pose_pipeline import build_default_intrinsics, PoseConfig, PosePipeline  # type: ignore


def _natural_key(p: Path) -> tuple[int, str]:
    m = re.search(r"(\d+)", p.stem)
    if m:
        return (int(m.group(1)), p.name)
    return (10**9, p.name)


def _parse_w2c(s: str) -> list[list[float]]:
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if len(vals) != 16:
        raise ValueError("--w2c must contain exactly 16 comma-separated floats")
    mat = np.asarray(vals, dtype=np.float64).reshape(4, 4)
    return mat.tolist()


def _default_w2c_from_pose_pipeline() -> str:
    T = np.asarray(PoseConfig().T_head_base_to_cam, dtype=np.float64).reshape(-1)
    return ",".join(f"{float(v):.10g}" for v in T.tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert warm_start + waypoint pngs to GGGS dataset format")
    parser.add_argument("--in-dir", required=True, help="Directory with warm_start.npy and waypoint_*.png")
    parser.add_argument("--out-dir", required=True, help="Output dataset directory")
    parser.add_argument("--warm-start", default="warm_start.npy", help="Warm start npy filename")
    parser.add_argument("--image-glob", default="waypoint_*.png", help="Input image glob pattern")
    parser.add_argument("--cam-id", default="head", help="Camera id")
    head_K, _ = build_default_intrinsics()
    parser.add_argument("--fx", type=float, default=float(head_K.fx))
    parser.add_argument("--fy", type=float, default=float(head_K.fy))
    parser.add_argument("--cx", type=float, default=float(head_K.cx))
    parser.add_argument("--cy", type=float, default=float(head_K.cy))
    parser.add_argument(
        "--w2c",
        default="",
        help="Optional fixed world-to-camera 4x4 matrix, row-major, 16 comma-separated values. "
        "If empty, pose is computed from pose_pipeline per frame q.",
    )
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of hard link")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    warm_path = in_dir / args.warm_start

    if not warm_path.exists():
        raise FileNotFoundError(f"Missing warm start: {warm_path}")

    q = np.load(warm_path, allow_pickle=False)
    if q.ndim != 2:
        raise ValueError(f"warm_start must be 2D [N, D], got shape={q.shape}")

    imgs = sorted(in_dir.glob(args.image_glob), key=_natural_key)
    if not imgs:
        raise FileNotFoundError(f"No images matched: {in_dir / args.image_glob}")

    n = min(len(imgs), int(q.shape[0]))
    if n == 0:
        raise ValueError("No valid frame to convert (image count or warm_start length is zero)")

    # Probe image size from first frame.
    with Image.open(imgs[0]) as im0:
        width, height = im0.size

    img_out_dir = out_dir / "images" / args.cam_id
    img_out_dir.mkdir(parents=True, exist_ok=True)

    fixed_w2c = _parse_w2c(args.w2c) if str(args.w2c).strip() else None
    pose_pipeline = None
    if fixed_w2c is None:
        head_K, wrist_K = build_default_intrinsics()
        pose_pipeline = PosePipeline(PoseConfig(), head_K, wrist_K)

    frames = []
    for i in range(n):
        frame_id = f"frame_{i:06d}"
        dst_rel = Path("images") / args.cam_id / f"{frame_id}.png"
        dst_abs = out_dir / dst_rel
        src_abs = imgs[i]

        if dst_abs.exists():
            dst_abs.unlink()

        if args.copy_images:
            shutil.copy2(src_abs, dst_abs)
        else:
            try:
                dst_abs.hardlink_to(src_abs)
            except Exception:
                shutil.copy2(src_abs, dst_abs)

        if fixed_w2c is not None:
            pose = fixed_w2c
        else:
            q_i = np.asarray(q[i], dtype=np.float32)
            pose = pose_pipeline.compute_poses(q_i)[args.cam_id].tolist()

        frames.append(
            {
                "frame_id": frame_id,
                "cam_id": args.cam_id,
                "pose": pose,
                "q": np.asarray(q[i], dtype=np.float32).tolist(),
                "image_path": str(dst_rel),
            }
        )

    cameras_json = {
        "cameras": [
            {
                "cam_id": args.cam_id,
                "K": [[float(args.fx), 0.0, float(args.cx)], [0.0, float(args.fy), float(args.cy)], [0.0, 0.0, 1.0]],
                "width": int(width),
                "height": int(height),
            }
        ]
    }

    poses_json = {"frames": frames}

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cameras.json").write_text(json.dumps(cameras_json, indent=2))
    (out_dir / "poses.json").write_text(json.dumps(poses_json, indent=2))

    print(f"[convert] in_dir={in_dir}")
    print(f"[convert] out_dir={out_dir}")
    print(f"[convert] frames={n}, q_dim={q.shape[1]}")
    if len(imgs) != int(q.shape[0]):
        print(f"[convert] warning: image_count={len(imgs)} != q_count={int(q.shape[0])}, used min={n}")


if __name__ == "__main__":
    main()
