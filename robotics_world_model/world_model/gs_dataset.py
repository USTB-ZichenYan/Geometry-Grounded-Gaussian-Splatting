"""
Convert parquet robot data to GS training dataset format.

Output directory:
  - images/head/*.png, images/left_wrist/*.png, images/right_wrist/*.png
  - cameras.json
  - poses.json

Copy-paste run:
  ROOT=/home/SENSETIME/yanzichen/data/file/Geometry-Grounded-Gaussian-Splatting
  WM_ROOT="$ROOT/robotics_world_model"
  DATA_ROOT="$WM_ROOT/dual_arm_grab_data"
  PYTHONPATH="$WM_ROOT" python3 -m world_model.gs_dataset \
    --data-root "$DATA_ROOT" \
    --out-dir "$ROOT/gs_dataset"

Optional:
  - --stride N for temporal downsampling.
  - --max-frames N to cap output size.
  - --q-delta-max to keep only near-static frames by ||q_t - q_{t-1}||_2.
  - --head-only to export only head camera (recommended baseline).
  - --black-mean-thr / --black-var-thr to drop black frames before writing.
  - --head-link-cam / --left-tcp-cam / --right-tcp-cam for extrinsics.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .pose_pipeline import (
    PoseConfig,
    PosePipeline,
    build_default_intrinsics,
    _load_T,  # reuse loader
)

# Optional deps
try:  # pragma: no cover
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # pragma: no cover
    pq = None  # type: ignore

try:  # pragma: no cover
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


# Parquet keys (same as dataset/data_test_wrist_cam.py)
CAM_HIGH_KEY = "observation.images.cam_high"
CAM_LEFT_WRIST_KEY = "observation.images.cam_left_wrist"
CAM_RIGHT_WRIST_KEY = "observation.images.cam_right_wrist"
Q_STATE_KEY = "observation.state"
Q_ACTION_KEY = "action"


def _require_pyarrow() -> None:
    if pq is None:
        raise ImportError("pyarrow is required. Install it to read parquet.")


def _require_cv2() -> None:
    if cv2 is None:
        raise ImportError("opencv-python is required. Install it to decode/save images.")


def _decode_image(img: Any) -> np.ndarray:
    _require_cv2()
    if isinstance(img, dict):
        if "bytes" in img and img["bytes"] is not None:
            return _decode_image(img["bytes"])
        if "path" in img and img["path"] is not None:
            decoded = cv2.imread(str(img["path"]), cv2.IMREAD_COLOR)
            if decoded is None:
                raise ValueError(f"Failed to read image from path: {img['path']}")
            return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        raise ValueError("Image dict missing 'bytes' or 'path'.")

    if isinstance(img, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(img, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("Failed to decode image bytes.")
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

    if isinstance(img, str):
        decoded = cv2.imread(img, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError(f"Failed to read image from path: {img}")
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

    if isinstance(img, np.ndarray):
        return img

    arr = np.array(img)
    if arr.dtype == object and arr.size == 1 and isinstance(arr.item(), (bytes, bytearray, memoryview)):
        return _decode_image(arr.item())
    return arr


def _save_image_rgb(img: np.ndarray, path: Path) -> None:
    _require_cv2()
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


@dataclass(frozen=True)
class ConvertConfig:
    data_root: Path
    out_dir: Path
    stride: int = 1
    max_frames: int | None = None
    batch_size: int = 64
    cameras: tuple[str, ...] = ("head", "left_wrist", "right_wrist")
    black_mean_thr: float = 1.0
    black_var_thr: float = 1.0
    q_delta_max: float | None = None
    q_source: str = "state"  # state|action


def _iter_parquet_files(root: Path) -> list[Path]:
    return sorted(root.rglob("episode_*.parquet"))


def _iter_frames(cfg: ConvertConfig):
    _require_pyarrow()
    files = _iter_parquet_files(cfg.data_root)
    if not files:
        raise FileNotFoundError(f"No parquet files under: {cfg.data_root}")

    q_key = Q_STATE_KEY if cfg.q_source == "state" else Q_ACTION_KEY
    columns = [CAM_HIGH_KEY, CAM_LEFT_WRIST_KEY, CAM_RIGHT_WRIST_KEY, q_key]
    stride = max(1, int(cfg.stride))
    frame_count = 0

    for pq_path in files:
        parquet = pq.ParquetFile(str(pq_path))
        for batch in parquet.iter_batches(columns=columns, batch_size=cfg.batch_size):
            data = batch.to_pydict()
            n = batch.num_rows
            for i in range(n):
                if (frame_count % stride) != 0:
                    frame_count += 1
                    continue
                q = np.asarray(data[q_key][i], dtype=np.float32)
                head = _decode_image(data[CAM_HIGH_KEY][i])
                left = _decode_image(data[CAM_LEFT_WRIST_KEY][i])
                right = _decode_image(data[CAM_RIGHT_WRIST_KEY][i])
                yield frame_count, q, head, left, right
                frame_count += 1
                if cfg.max_frames is not None and frame_count >= cfg.max_frames:
                    return


def _write_cameras_json(out_dir: Path, head_K, wrist_K, selected_cameras: tuple[str, ...]) -> None:
    cam_intrinsics = {
        "head": head_K,
        "left_wrist": wrist_K,
        "right_wrist": wrist_K,
    }
    cameras = []
    for cam_id in selected_cameras:
        intr = cam_intrinsics[cam_id]
        cameras.append(
            {
                "cam_id": cam_id,
                "K": intr.K(),
                "width": intr.width,
                "height": intr.height,
            }
        )
    payload = {"cameras": cameras}
    (out_dir / "cameras.json").write_text(json.dumps(payload, indent=2))


def _write_poses_json(out_dir: Path, frames: Iterable[dict[str, Any]]) -> None:
    payload = {"frames": list(frames)}
    (out_dir / "poses.json").write_text(json.dumps(payload, indent=2))


def _is_black_image(img: np.ndarray, mean_thr: float, var_thr: float) -> bool:
    img_f = np.asarray(img, dtype=np.float32)
    mean = float(np.mean(img_f))
    var = float(np.var(img_f))
    return mean <= mean_thr and var <= var_thr


def convert_dataset(cfg: ConvertConfig, pose_cfg: PoseConfig) -> None:
    out_dir = cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    head_K, wrist_K = build_default_intrinsics()
    pipeline = PosePipeline(pose_cfg, head_K, wrist_K)

    _write_cameras_json(out_dir, head_K, wrist_K, cfg.cameras)

    frames_out = []
    kept_count = {k: 0 for k in cfg.cameras}
    black_skip_count = {k: 0 for k in cfg.cameras}
    motion_skip_count = 0
    prev_q: np.ndarray | None = None
    for idx, q, head, left, right in _iter_frames(cfg):
        if cfg.q_delta_max is not None and prev_q is not None:
            q_delta = float(np.linalg.norm(q - prev_q))
            if q_delta > cfg.q_delta_max:
                motion_skip_count += 1
                prev_q = q
                continue
        prev_q = q

        frame_id = f"frame_{idx:06d}"
        poses = pipeline.compute_poses(q)
        img_map = {
            "head": head,
            "left_wrist": left,
            "right_wrist": right,
        }

        for cam_id in cfg.cameras:
            img = img_map[cam_id]
            if _is_black_image(img, cfg.black_mean_thr, cfg.black_var_thr):
                black_skip_count[cam_id] += 1
                continue

            img_path = Path("images") / cam_id / f"{frame_id}.png"
            _save_image_rgb(img, out_dir / img_path)
            frames_out.append(
                {
                    "frame_id": frame_id,
                    "cam_id": cam_id,
                    "pose": poses[cam_id].tolist(),
                    "q": q.tolist(),
                    "image_path": str(img_path),
                }
            )
            kept_count[cam_id] += 1

    _write_poses_json(out_dir, frames_out)
    print(f"[gs_dataset] selected cameras: {','.join(cfg.cameras)}")
    print(f"[gs_dataset] q source: {cfg.q_source}")
    print(f"[gs_dataset] written frames: {len(frames_out)}")
    print(f"[gs_dataset] kept per camera: {kept_count}")
    print(f"[gs_dataset] skipped black per camera: {black_skip_count}")
    if cfg.q_delta_max is not None:
        print(f"[gs_dataset] q-delta-max: {cfg.q_delta_max}")
        print(f"[gs_dataset] skipped by motion: {motion_skip_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GS dataset conversion")
    parser.add_argument(
        "--data-root",
        default="/home/SENSETIME/yanzichen/data/file/Geometry-Grounded-Gaussian-Splatting/robotics_world_model/dual_arm_grab_data",
        help="Parquet dataset root",
    )
    parser.add_argument("--out-dir", default="/tmp/gs_dataset", help="Output GS dataset dir")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames")
    parser.add_argument(
        "--q-delta-max",
        type=float,
        default=None,
        help="Keep frame only if ||q_t-q_{t-1}||_2 <= threshold",
    )
    parser.add_argument(
        "--cams",
        default="head,left_wrist,right_wrist",
        help="Comma-separated cameras to export: head,left_wrist,right_wrist",
    )
    parser.add_argument("--head-only", action="store_true", help="Shortcut for --cams head")
    parser.add_argument(
        "--black-mean-thr",
        type=float,
        default=1.0,
        help="Drop image if mean <= threshold (uint8 scale)",
    )
    parser.add_argument(
        "--black-var-thr",
        type=float,
        default=1.0,
        help="Drop image if variance <= threshold (uint8 scale)",
    )
    parser.add_argument("--head-link-cam", default=None, help="4x4 extrinsic (head_link->head cam) .json/.npy")
    parser.add_argument("--left-tcp-cam", default=None, help="4x4 extrinsic (tcp->left cam) .json/.npy")
    parser.add_argument("--right-tcp-cam", default=None, help="4x4 extrinsic (tcp->right cam) .json/.npy")
    parser.add_argument(
        "--q-source",
        choices=["state", "action"],
        default="state",
        help="Use joint state or action vector for FK/mask generation. Usually state is correct.",
    )

    args = parser.parse_args()

    valid_cams = {"head", "left_wrist", "right_wrist"}
    if args.head_only:
        cameras = ("head",)
    else:
        cameras = tuple(x.strip() for x in args.cams.split(",") if x.strip())
    if not cameras:
        raise ValueError("No cameras selected. Use --head-only or --cams ...")
    unknown = [x for x in cameras if x not in valid_cams]
    if unknown:
        raise ValueError(f"Unknown cameras: {unknown}. Valid: {sorted(valid_cams)}")

    pose_cfg = PoseConfig()
    if args.head_link_cam:
        pose_cfg = dataclasses.replace(pose_cfg, T_head_link_cam=_load_T(args.head_link_cam))
    if args.left_tcp_cam:
        pose_cfg = dataclasses.replace(pose_cfg, T_left_tcp_cam=_load_T(args.left_tcp_cam))
    if args.right_tcp_cam:
        pose_cfg = dataclasses.replace(pose_cfg, T_right_tcp_cam=_load_T(args.right_tcp_cam))

    convert_dataset(
        ConvertConfig(
            data_root=Path(args.data_root),
            out_dir=Path(args.out_dir),
            stride=args.stride,
            max_frames=args.max_frames,
            cameras=cameras,
            black_mean_thr=args.black_mean_thr,
            black_var_thr=args.black_var_thr,
            q_delta_max=args.q_delta_max,
            q_source=args.q_source,
        ),
        pose_cfg,
    )


if __name__ == "__main__":
    main()
