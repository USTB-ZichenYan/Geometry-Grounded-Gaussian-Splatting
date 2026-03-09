"""
Convert parquet robot data to GS training dataset format.

Output directory:
  - images/head/*.png, images/left_wrist/*.png, images/right_wrist/*.png
  - robot_masks/head/*.png, robot_masks/left_wrist/*.png, robot_masks/right_wrist/*.png
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
  - --left-tcp-cam / --right-tcp-cam for extrinsics.
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from .pose_pipeline import (
        PoseConfig,
        PosePipeline,
        build_default_intrinsics,
        _load_T,  # reuse loader
    )
except ImportError:  # pragma: no cover
    from pose_pipeline import (  # type: ignore
        PoseConfig,
        PosePipeline,
        build_default_intrinsics,
        _load_T,
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

try:  # pragma: no cover
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore


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


def _require_pil() -> None:
    if Image is None:
        raise ImportError("Pillow is required. Install it to decode/save images robustly.")


def _decode_image(img: Any) -> np.ndarray:
    if hasattr(img, "as_py"):
        return _decode_image(img.as_py())

    if isinstance(img, dict):
        if "bytes" in img and img["bytes"] is not None:
            return _decode_image(img["bytes"])
        if "path" in img and img["path"] is not None:
            return _decode_image(str(img["path"]))
        raise ValueError("Image dict missing 'bytes' or 'path'.")

    if isinstance(img, (bytes, bytearray, memoryview)):
        raw = img.tobytes() if isinstance(img, memoryview) else bytes(img)
        if Image is not None:
            with Image.open(io.BytesIO(raw)) as im:
                return np.asarray(im.convert("RGB"))
        _require_cv2()
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(-1)
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("Failed to decode image bytes.")
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

    if isinstance(img, str):
        path = Path(img)
        if Image is not None:
            with Image.open(path) as im:
                return np.asarray(im.convert("RGB"))
        _require_cv2()
        decoded = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError(f"Failed to read image from path: {img}")
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

    if isinstance(img, np.ndarray):
        if img.ndim == 1:
            arr = np.ascontiguousarray(img.astype(np.uint8, copy=False).reshape(-1))
            decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if decoded is not None:
                return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        return img

    arr = np.array(img)
    if arr.ndim == 1 and arr.size > 0 and arr.dtype != object:
        arr_u8 = np.ascontiguousarray(arr.astype(np.uint8, copy=False).reshape(-1))
        decoded = cv2.imdecode(arr_u8, cv2.IMREAD_COLOR)
        if decoded is not None:
            return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    if arr.dtype == object and arr.size == 1 and isinstance(arr.item(), (bytes, bytearray, memoryview)):
        return _decode_image(arr.item())
    return arr


def _save_image(img: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img_u8 = np.asarray(img, dtype=np.uint8)
    if Image is not None:
        mode = "RGBA" if (img_u8.ndim == 3 and img_u8.shape[2] == 4) else "RGB"
        Image.fromarray(img_u8, mode=mode).save(path)
        return
    _require_cv2()
    if img_u8.ndim == 3 and img_u8.shape[2] == 4:
        bgra = cv2.cvtColor(img_u8, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(str(path), bgra)
    else:
        bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr)


def _save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    if Image is not None:
        Image.fromarray(mask_u8, mode="L").save(path)
        return
    _require_cv2()
    cv2.imwrite(str(path), mask_u8)


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
    green_screen: bool = True
    green_g_min: int = 100
    green_margin: int = 30


def _iter_parquet_files(root: Path) -> list[Path]:
    return sorted(root.rglob("episode_*.parquet"))


def _natural_key(p: Path) -> tuple[int, str]:
    m = re.search(r"(\d+)", p.stem)
    if m:
        return (int(m.group(1)), p.name)
    return (10**9, p.name)


def _iter_warmstart_frames(cfg: ConvertConfig):
    warm = cfg.data_root / "warm_start.npy"
    imgs = sorted(cfg.data_root.glob("waypoint_*.png"), key=_natural_key)
    if not warm.exists() or not imgs:
        raise FileNotFoundError(f"No parquet and no warm_start/waypoint data under: {cfg.data_root}")

    q_all = np.load(warm, allow_pickle=False)
    if q_all.ndim != 2:
        raise ValueError(f"warm_start.npy must be 2D [N, D], got {q_all.shape}")

    n = min(len(imgs), int(q_all.shape[0]))
    stride = max(1, int(cfg.stride))
    emitted = 0
    for i in range(n):
        if (i % stride) != 0:
            continue
        if cfg.max_frames is not None and emitted >= cfg.max_frames:
            return
        q = np.asarray(q_all[i], dtype=np.float32)
        try:
            head = _decode_image(str(imgs[i]))
        except Exception as e:
            print(f"[gs_dataset] warning: skip frame {i} decode failed: {e}")
            continue
        # Warm-start format only contains head camera image.
        yield i, q, head, head, head
        emitted += 1


def _iter_frames(cfg: ConvertConfig):
    files = _iter_parquet_files(cfg.data_root)
    if not files:
        print(f"[gs_dataset] no parquet found, fallback to warm_start+waypoint format under {cfg.data_root}")
        yield from _iter_warmstart_frames(cfg)
        return

    _require_pyarrow()

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
                try:
                    head = _decode_image(data[CAM_HIGH_KEY][i])
                    left = _decode_image(data[CAM_LEFT_WRIST_KEY][i])
                    right = _decode_image(data[CAM_RIGHT_WRIST_KEY][i])
                except Exception as e:
                    print(f"[gs_dataset] warning: skip frame {frame_count} decode failed: {e}")
                    frame_count += 1
                    continue
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


def _extract_robot_mask_from_green(img: np.ndarray, g_min: int, margin: int) -> np.ndarray:
    img_u8 = np.asarray(img, dtype=np.uint8)
    if img_u8.ndim != 3 or img_u8.shape[2] != 3:
        h, w = img_u8.shape[:2]
        return np.full((h, w), 255, dtype=np.uint8)

    r = img_u8[..., 0].astype(np.int16)
    g = img_u8[..., 1].astype(np.int16)
    b = img_u8[..., 2].astype(np.int16)

    is_green = (g >= int(g_min)) & ((g - r) >= int(margin)) & ((g - b) >= int(margin))
    robot_mask = np.full(img_u8.shape[:2], 255, dtype=np.uint8)
    robot_mask[is_green] = 0
    return robot_mask


def _apply_robot_mask_transparent(img: np.ndarray, robot_mask: np.ndarray) -> np.ndarray:
    img_u8 = np.asarray(img, dtype=np.uint8)
    m = np.asarray(robot_mask, dtype=np.uint8) > 0
    alpha = np.where(m, 255, 0).astype(np.uint8)[..., None]
    out = np.concatenate([img_u8, alpha], axis=2)
    return out


def convert_dataset(cfg: ConvertConfig, pose_cfg: PoseConfig) -> None:
    out_dir = cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    head_K, wrist_K = build_default_intrinsics()
    pipeline = PosePipeline(pose_cfg, head_K, wrist_K)

    _write_cameras_json(out_dir, head_K, wrist_K, cfg.cameras)

    frames_out = []
    kept_count = {k: 0 for k in cfg.cameras}
    black_skip_count = {k: 0 for k in cfg.cameras}
    robot_mask_count = {k: 0 for k in cfg.cameras}
    motion_skip_count = 0
    green_applied_count = {k: 0 for k in cfg.cameras}
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
            img_out = img
            robot_mask: np.ndarray | None = None
            if cfg.green_screen:
                robot_mask = _extract_robot_mask_from_green(img, cfg.green_g_min, cfg.green_margin)
                green_applied_count[cam_id] += 1
            if _is_black_image(img, cfg.black_mean_thr, cfg.black_var_thr):
                black_skip_count[cam_id] += 1
                continue
            if cfg.green_screen and robot_mask is not None:
                img_out = _apply_robot_mask_transparent(img, robot_mask)

            img_path = Path("images") / cam_id / f"{frame_id}.png"
            _save_image(img_out, out_dir / img_path)

            if robot_mask is not None:
                mask_path = Path("robot_masks") / cam_id / f"{frame_id}.png"
                _save_mask(robot_mask, out_dir / mask_path)
                robot_mask_count[cam_id] += 1

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
    if cfg.green_screen:
        print(
            f"[gs_dataset] green screen mask extraction: on "
            f"(g_min={cfg.green_g_min}, margin={cfg.green_margin})"
        )
        print(f"[gs_dataset] green-mask processed per camera: {green_applied_count}")
        print(f"[gs_dataset] robot-mask exported per camera: {robot_mask_count}")
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
    parser.add_argument("--left-tcp-cam", default=None, help="4x4 extrinsic (tcp->left cam) .json/.npy")
    parser.add_argument("--right-tcp-cam", default=None, help="4x4 extrinsic (tcp->right cam) .json/.npy")
    parser.add_argument(
        "--q-source",
        choices=["state", "action"],
        default="state",
        help="Use joint state or action vector for FK/mask generation. Usually state is correct.",
    )
    parser.add_argument(
        "--no-green-screen",
        action="store_true",
        help="Disable green-background robot mask extraction during export",
    )
    parser.add_argument(
        "--green-g-min",
        type=int,
        default=100,
        help="Green-screen threshold: G channel minimum",
    )
    parser.add_argument(
        "--green-margin",
        type=int,
        default=30,
        help="Green-screen threshold: require G-R and G-B >= margin",
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
            green_screen=not args.no_green_screen,
            green_g_min=args.green_g_min,
            green_margin=args.green_margin,
        ),
        pose_cfg,
    )


if __name__ == "__main__":
    main()
