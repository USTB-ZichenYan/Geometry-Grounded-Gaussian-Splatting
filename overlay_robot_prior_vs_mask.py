#!/usr/bin/env python3
"""
Overlay canonical robot prior projection against dataset robot masks.

Inputs:
  - gs_init.npz      : canonical robot prior in link-local coordinates
  - poses.json       : per-frame q and camera pose
  - cameras.json     : per-camera intrinsics
  - robot_masks/     : per-frame robot masks (255 robot, 0 background)

Outputs:
  - out_dir/overlays/*.png   : green=true robot mask, red=projected prior
  - out_dir/summary.json     : per-frame metrics to diagnose alignment

Copy-paste:
  ROOT=/home/SENSETIME/yanzichen/data/file/Geometry-Grounded-Gaussian-Splatting
  PY=/home/SENSETIME/yanzichen/anaconda3/envs/gggs/bin/python

  "$PY" "$ROOT/overlay_robot_prior_vs_mask.py" \
    --init-npz "$ROOT/gggs_run/gs_init.npz" \
    --poses-json "$ROOT/gggs_run/gs_dataset/poses.json" \
    --cameras-json "$ROOT/gggs_run/gs_dataset/cameras.json" \
    --robot-mask-root "$ROOT/gggs_run/gs_dataset/robot_masks" \
    --out-dir "$ROOT/gggs_run/prior_overlay_check"

Usage:
  python overlay_robot_prior_vs_mask.py \
    --init-npz <gs_init.npz> \
    --poses-json <poses.json> \
    --cameras-json <cameras.json> \
    --robot-mask-root <robot_masks_dir> \
    --out-dir <out_dir> \
    --cam-id head

Example (left-arm single frame):
  ROOT=/home/SENSETIME/yanzichen/data/file/Geometry-Grounded-Gaussian-Splatting
  PY=/home/SENSETIME/yanzichen/anaconda3/envs/gggs/bin/python
  "$PY" "$ROOT/overlay_robot_prior_vs_mask.py" \
    --init-npz "$ROOT/gggs_run/gs_init_left.npz" \
    --poses-json "$ROOT/gggs_run/gs_dataset_raw_1f/poses.json" \
    --cameras-json "$ROOT/gggs_run/gs_dataset_raw_1f/cameras.json" \
    --robot-mask-root "$ROOT/gggs_run/gs_dataset_raw_1f/robot_masks" \
    --out-dir "$ROOT/gggs_run/overlay_left_1f" \
    --cam-id head \
    --sample-points 12000 \
    --point-radius 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
WM_DIR = ROOT / "robotics_world_model" / "world_model"
if str(WM_DIR) not in sys.path:
    sys.path.insert(0, str(WM_DIR))

from kinematics_common import DEFAULT_URDF, build_q_map_from_q32, compute_link_transforms  # type: ignore
from pose_pipeline import PoseConfig, PosePipeline, build_default_intrinsics  # type: ignore


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _load_camera_defs(cameras_json: Path) -> dict[str, dict[str, float]]:
    data = _load_json(cameras_json)
    cams: dict[str, dict[str, float]] = {}
    for cam in data["cameras"]:
        K = np.asarray(cam["K"], dtype=np.float32)
        cams[str(cam["cam_id"])] = {
            "width": int(cam["width"]),
            "height": int(cam["height"]),
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
        }
    return cams


def _load_init(init_npz: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    data = np.load(init_npz, allow_pickle=True)
    points = np.asarray(data["points"], dtype=np.float32)
    link_ids = np.asarray(data["link_ids"], dtype=np.int64).reshape(-1)
    raw_link_names = data["link_names"]
    if isinstance(raw_link_names, np.ndarray):
        link_names = [str(x) for x in raw_link_names.tolist()]
    else:
        link_names = [str(x) for x in raw_link_names]
    if points.shape[0] != link_ids.shape[0]:
        raise ValueError(f"points/link_ids mismatch: {points.shape[0]} vs {link_ids.shape[0]}")
    meta_path = init_npz.with_suffix(".json")
    if meta_path.exists():
        meta = _load_json(meta_path)
        output_frame = str(meta.get("output_frame", "unknown"))
        if output_frame != "link_local":
            raise ValueError(
                f"{init_npz} must be link_local for FK overlay; got output_frame={output_frame}"
            )
    return points, link_ids, link_names


def _make_link_buckets(points: np.ndarray, link_ids: np.ndarray, link_names: list[str]) -> dict[str, np.ndarray]:
    buckets: dict[str, list[np.ndarray]] = {}
    for idx, link_name in enumerate(link_names):
        sel = link_ids == idx
        if np.any(sel):
            buckets[link_name] = [points[sel]]
    return {k: np.concatenate(v, axis=0).astype(np.float32, copy=False) for k, v in buckets.items()}


def _project_points(
    link_points_local: dict[str, np.ndarray],
    link_T_world: dict[str, np.ndarray],
    pose_w2c: np.ndarray,
    cam: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    R_cw = pose_w2c[:3, :3]
    t_cw = pose_w2c[:3, 3]
    proj_chunks: list[np.ndarray] = []
    valid_chunks: list[np.ndarray] = []
    for link, pts_local in link_points_local.items():
        T = link_T_world.get(link)
        if T is None or pts_local.size == 0:
            continue
        pts_world = (T[:3, :3] @ pts_local.T).T + T[:3, 3][None, :]
        pts_cam = (R_cw @ pts_world.T).T + t_cw[None, :]
        z = pts_cam[:, 2]
        valid = z > 1e-6
        if not np.any(valid):
            continue
        pts_cam = pts_cam[valid]
        z = z[valid]
        u = cam["fx"] * (pts_cam[:, 0] / z) + cam["cx"]
        v = cam["fy"] * (pts_cam[:, 1] / z) + cam["cy"]
        proj_chunks.append(np.stack([u, v], axis=1).astype(np.float32))
        valid_chunks.append(np.ones((u.shape[0],), dtype=bool))
    if not proj_chunks:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=bool)
    proj = np.concatenate(proj_chunks, axis=0)
    valid = np.concatenate(valid_chunks, axis=0)
    return proj, valid


def _project_link_centroids(
    link_points_local: dict[str, np.ndarray],
    link_T_world: dict[str, np.ndarray],
    pose_w2c: np.ndarray,
    cam: dict[str, float],
    w: int,
    h: int,
) -> list[dict[str, Any]]:
    R_cw = pose_w2c[:3, :3]
    t_cw = pose_w2c[:3, 3]
    out: list[dict[str, Any]] = []
    for link, pts_local in link_points_local.items():
        if pts_local.size == 0:
            continue
        T = link_T_world.get(link)
        if T is None:
            continue
        c_local = pts_local.mean(axis=0)
        c_world = (T[:3, :3] @ c_local) + T[:3, 3]
        c_cam = (R_cw @ c_world) + t_cw
        z = float(c_cam[2])
        if z <= 1e-6:
            out.append(
                {
                    "link": link,
                    "uv": None,
                    "z_cam": z,
                    "in_image": False,
                }
            )
            continue
        u = float(cam["fx"] * (float(c_cam[0]) / z) + cam["cx"])
        v = float(cam["fy"] * (float(c_cam[1]) / z) + cam["cy"])
        in_image = (0.0 <= u <= float(w - 1)) and (0.0 <= v <= float(h - 1))
        out.append(
            {
                "link": link,
                "uv": [u, v],
                "z_cam": z,
                "in_image": bool(in_image),
            }
        )
    return out


def _bbox_from_bool(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _bbox_from_points(uv: np.ndarray, w: int, h: int) -> list[int] | None:
    if uv.size == 0:
        return None
    x = np.clip(np.rint(uv[:, 0]).astype(np.int32), 0, w - 1)
    y = np.clip(np.rint(uv[:, 1]).astype(np.int32), 0, h - 1)
    if x.size == 0:
        return None
    return [int(x.min()), int(y.min()), int(x.max()), int(y.max())]


def _rasterize_projection_mask(h: int, w: int, uv_in: np.ndarray, point_radius: int) -> np.ndarray:
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    r = max(0, int(point_radius))
    for u, v in uv_in.tolist():
        x = float(u)
        y = float(v)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=255)
    return np.asarray(mask, dtype=np.uint8) > 0


def _overlay_image(
    mask_robot: np.ndarray,
    uv_in: np.ndarray,
    proj_mask: np.ndarray,
    point_radius: int,
    link_centroids: list[dict[str, Any]] | None = None,
    centroid_limit: int = 40,
    marker_size: int = 10,
    font: ImageFont.ImageFont | None = None,
    cam_uv: tuple[float, float] | None = None,
) -> Image.Image:
    h, w = mask_robot.shape
    base = Image.new("RGBA", (w, h), (0, 0, 0, 255))

    green = np.zeros((h, w, 4), dtype=np.uint8)
    green[..., 1] = np.where(mask_robot, 255, 0).astype(np.uint8)
    green[..., 3] = np.where(mask_robot, 56, 0).astype(np.uint8)
    img = Image.alpha_composite(base, Image.fromarray(green, mode="RGBA"))

    overlap = np.zeros((h, w, 4), dtype=np.uint8)
    overlap_mask = mask_robot & proj_mask
    overlap[..., 0] = np.where(overlap_mask, 255, 0).astype(np.uint8)
    overlap[..., 1] = np.where(overlap_mask, 220, 0).astype(np.uint8)
    overlap[..., 3] = np.where(overlap_mask, 180, 0).astype(np.uint8)
    img = Image.alpha_composite(img, Image.fromarray(overlap, mode="RGBA"))

    draw = ImageDraw.Draw(img, mode="RGBA")
    r = max(0, int(point_radius))
    for u, v in uv_in.tolist():
        x = float(u)
        y = float(v)
        draw.ellipse(
            (x - r, y - r, x + r, y + r),
            fill=(255, 0, 0, 80),
            outline=(255, 40, 40, 220),
            width=max(2, r),
        )

    if link_centroids:
        shown = 0
        m = max(4, int(marker_size))
        for item in link_centroids:
            if shown >= int(max(0, centroid_limit)):
                break
            uv = item.get("uv")
            if uv is None:
                continue
            x = float(uv[0])
            y = float(uv[1])
            # blue cross marker
            draw.line((x - m, y, x + m, y), fill=(80, 180, 255, 255), width=max(2, m // 3))
            draw.line((x, y - m, x, y + m), fill=(80, 180, 255, 255), width=max(2, m // 3))
            # short label (remove common suffix to reduce clutter)
            name = str(item.get("link", ""))
            name = name.replace("_link", "")
            draw.text((x + m + 2, y - m - 2), name, fill=(120, 210, 255, 255), font=font)
            shown += 1

    # camera optical center marker (principal point on image)
    if cam_uv is not None:
        cx, cy = float(cam_uv[0]), float(cam_uv[1])
        csz = max(6, m + 2)
        draw.line((cx - csz, cy, cx + csz, cy), fill=(255, 220, 80, 255), width=max(2, csz // 4))
        draw.line((cx, cy - csz, cx, cy + csz), fill=(255, 220, 80, 255), width=max(2, csz // 4))
        draw.text((cx + csz + 4, cy - csz - 2), "CAM", fill=(255, 235, 120, 255), font=font)

    # top-right legend (larger for readability)
    legend_items = [
        ("mask", (0, 255, 0, 180)),
        ("prior", (255, 40, 40, 220)),
        ("centroid", (80, 180, 255, 255)),
        ("cam_center", (255, 220, 80, 255)),
    ]
    lx = w - 220
    ly = 12
    draw.rectangle((lx - 8, ly - 8, w - 8, ly + 34 * len(legend_items)), fill=(0, 0, 0, 140))
    for i, (label, color) in enumerate(legend_items):
        y = ly + i * 30
        draw.rectangle((lx, y + 6, lx + 18, y + 22), fill=color, outline=(255, 255, 255, 180), width=1)
        draw.text((lx + 26, y + 3), label, fill=(240, 240, 240, 255), font=font)
    return img.convert("RGB")


def _frame_metrics(mask_robot: np.ndarray, proj_mask: np.ndarray, uv_all: np.ndarray, uv_in: np.ndarray) -> dict[str, Any]:
    h, w = mask_robot.shape
    hit = 0
    if uv_in.size > 0:
        x = np.clip(np.rint(uv_in[:, 0]).astype(np.int32), 0, w - 1)
        y = np.clip(np.rint(uv_in[:, 1]).astype(np.int32), 0, h - 1)
        hit = int(mask_robot[y, x].sum())
    inside = int(uv_in.shape[0])
    total = int(uv_all.shape[0])
    proj_pixels = int(proj_mask.sum())
    mask_pixels = int(mask_robot.sum())
    overlap_pixels = int((mask_robot & proj_mask).sum())
    union_pixels = int((mask_robot | proj_mask).sum())
    precision = float(overlap_pixels / max(proj_pixels, 1))
    recall = float(overlap_pixels / max(mask_pixels, 1))
    iou = float(overlap_pixels / max(union_pixels, 1))
    return {
        "points_total": total,
        "points_inside_image": inside,
        "inside_image_ratio": float(inside / max(total, 1)),
        "hit_robot_mask": hit,
        "hit_robot_mask_ratio_over_all": float(hit / max(total, 1)),
        "hit_robot_mask_ratio_over_inside": float(hit / max(inside, 1)),
        "proj_pixels": proj_pixels,
        "mask_pixels": mask_pixels,
        "overlap_pixels": overlap_pixels,
        "union_pixels": union_pixels,
        "proj_precision": precision,
        "mask_recall": recall,
        "proj_mask_iou": iou,
        "mask_fg_ratio": float(mask_robot.mean()),
        "proj_fg_ratio": float(proj_mask.mean()),
        "mask_bbox": _bbox_from_bool(mask_robot),
        "proj_mask_bbox": _bbox_from_bool(proj_mask),
        "proj_bbox": _bbox_from_points(uv_in, w, h),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay projected robot prior vs dataset robot masks")
    parser.add_argument("--init-npz", required=True, help="Canonical gs_init.npz (link_local)")
    parser.add_argument("--poses-json", required=True, help="Dataset poses.json")
    parser.add_argument("--cameras-json", default=None, help="Dataset cameras.json (default: poses dir / cameras.json)")
    parser.add_argument("--robot-mask-root", default=None, help="Dataset robot_masks root (default: poses dir / robot_masks)")
    parser.add_argument("--urdf", default=DEFAULT_URDF, help="URDF used for FK")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--cam-id", default="head", help="Only inspect this camera")
    parser.add_argument(
        "--recompute-head-pose",
        action="store_true",
        help="Ignore stored poses.json head pose and recompute it from q using pose_pipeline semantics",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride")
    parser.add_argument("--sample-points", type=int, default=12000, help="Subsample prior points for faster overlays")
    parser.add_argument("--point-radius", type=int, default=2, help="Overlay point radius in pixels")
    parser.add_argument(
        "--centroid-limit",
        type=int,
        default=40,
        help="Maximum link-centroid labels per frame in overlay",
    )
    parser.add_argument("--marker-size", type=int, default=10, help="Centroid cross marker half-size in pixels")
    parser.add_argument("--font-size", type=int, default=22, help="Label/legend font size")
    parser.add_argument("--font-path", default=None, help="Optional .ttf font path")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for point subsampling")
    args = parser.parse_args()

    init_npz = Path(args.init_npz)
    poses_json = Path(args.poses_json)
    dataset_dir = poses_json.parent
    cameras_json = Path(args.cameras_json) if args.cameras_json else dataset_dir / "cameras.json"
    robot_mask_root = Path(args.robot_mask_root) if args.robot_mask_root else dataset_dir / "robot_masks"
    out_dir = Path(args.out_dir)
    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    cams = _load_camera_defs(cameras_json)
    if args.cam_id not in cams:
        raise KeyError(f"cam_id={args.cam_id} not found in {cameras_json}")
    cam = cams[args.cam_id]

    points, link_ids, link_names = _load_init(init_npz)
    if args.sample_points is not None and 0 < int(args.sample_points) < points.shape[0]:
        rng = np.random.default_rng(args.seed)
        keep = np.sort(rng.choice(points.shape[0], size=int(args.sample_points), replace=False))
        points = points[keep]
        link_ids = link_ids[keep]
    link_points_local = _make_link_buckets(points, link_ids, link_names)

    frames = [f for f in _load_json(poses_json)["frames"] if str(f.get("cam_id")) == args.cam_id]
    stride = max(1, int(args.stride))
    selected_frames = frames[::stride]
    if args.max_frames is not None:
        selected_frames = selected_frames[: int(args.max_frames)]

    def build_pose_pipeline() -> PosePipeline | None:
        if not args.recompute_head_pose:
            return None
        pose_cfg = PoseConfig()
        head_K, wrist_K = build_default_intrinsics()
        return PosePipeline(pose_cfg, head_K, wrist_K)

    def run_one(save_overlays: bool, mode_out_dir: Path) -> dict[str, Any]:
        pose_pipeline = build_pose_pipeline()
        mode_overlay_dir = mode_out_dir / "overlays"
        if save_overlays:
            mode_overlay_dir.mkdir(parents=True, exist_ok=True)
        if args.font_path:
            font = ImageFont.truetype(args.font_path, size=max(10, int(args.font_size)))
        else:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", size=max(10, int(args.font_size)))
            except Exception:
                font = ImageFont.load_default()

        summaries: list[dict[str, Any]] = []
        for frame in selected_frames:
            frame_id = str(frame["frame_id"])
            q = np.asarray(frame["q"], dtype=np.float32)
            if pose_pipeline is not None and args.cam_id == "head":
                pose = pose_pipeline.compute_poses(q)["head"]
            else:
                pose = np.asarray(frame["pose"], dtype=np.float32)
            mask_path = robot_mask_root / args.cam_id / f"{frame_id}.png"
            if not mask_path.exists():
                raise FileNotFoundError(mask_path)
            mask_robot = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0

            q_map = build_q_map_from_q32(q)
            link_T_world = compute_link_transforms(Path(args.urdf), q_map=q_map)
            uv_all, _ = _project_points(link_points_local, link_T_world, pose, cam)
            h, w = mask_robot.shape
            link_centroids = _project_link_centroids(link_points_local, link_T_world, pose, cam, w=w, h=h)
            if uv_all.size == 0:
                uv_in = np.zeros((0, 2), dtype=np.float32)
            else:
                keep = (
                    (uv_all[:, 0] >= 0.0)
                    & (uv_all[:, 0] <= float(w - 1))
                    & (uv_all[:, 1] >= 0.0)
                    & (uv_all[:, 1] <= float(h - 1))
                )
                uv_in = uv_all[keep]

            proj_mask = _rasterize_projection_mask(h, w, uv_in, args.point_radius)
            if save_overlays:
                overlay = _overlay_image(
                    mask_robot,
                    uv_in,
                    proj_mask,
                    args.point_radius,
                    link_centroids=link_centroids,
                    centroid_limit=args.centroid_limit,
                    marker_size=args.marker_size,
                    font=font,
                    cam_uv=(cam["cx"], cam["cy"]),
                )
                overlay.save(mode_overlay_dir / f"{frame_id}_{args.cam_id}.png")

            metrics = _frame_metrics(mask_robot, proj_mask, uv_all, uv_in)
            metrics["frame_id"] = frame_id
            metrics["cam_id"] = args.cam_id
            metrics["link_centroids"] = link_centroids
            # camera center in base/world coordinates, from w2c inverse.
            R = pose[:3, :3]
            t = pose[:3, 3]
            cam_center_base = (-R.T @ t).astype(np.float32)
            metrics["camera_center_base"] = cam_center_base.tolist()
            metrics["camera_optical_center_pixel"] = [float(cam["cx"]), float(cam["cy"])]
            summaries.append(metrics)
            print(
                f"[overlay] {frame_id} "
                f"inside={metrics['inside_image_ratio']:.3f} "
                f"hit_all={metrics['hit_robot_mask_ratio_over_all']:.3f} "
                f"hit_in={metrics['hit_robot_mask_ratio_over_inside']:.3f} "
                f"iou={metrics['proj_mask_iou']:.4f} "
                f"precision={metrics['proj_precision']:.4f} "
                f"recall={metrics['mask_recall']:.4f}"
            )

        mean_hit_inside = float(np.mean([m["hit_robot_mask_ratio_over_inside"] for m in summaries])) if summaries else 0.0
        mean_hit_all = float(np.mean([m["hit_robot_mask_ratio_over_all"] for m in summaries])) if summaries else 0.0
        mean_inside = float(np.mean([m["inside_image_ratio"] for m in summaries])) if summaries else 0.0
        mean_iou = float(np.mean([m["proj_mask_iou"] for m in summaries])) if summaries else 0.0
        mean_precision = float(np.mean([m["proj_precision"] for m in summaries])) if summaries else 0.0
        mean_recall = float(np.mean([m["mask_recall"] for m in summaries])) if summaries else 0.0
        payload = {
            "init_npz": str(init_npz),
            "poses_json": str(poses_json),
            "cameras_json": str(cameras_json),
            "robot_mask_root": str(robot_mask_root),
            "urdf": str(args.urdf),
            "cam_id": args.cam_id,
            "recompute_head_pose": bool(args.recompute_head_pose),
            "frames": len(summaries),
            "sample_points": int(points.shape[0]),
            "mean_inside_image_ratio": mean_inside,
            "mean_hit_robot_mask_ratio_over_all": mean_hit_all,
            "mean_hit_robot_mask_ratio_over_inside": mean_hit_inside,
            "mean_proj_mask_iou": mean_iou,
            "mean_proj_precision": mean_precision,
            "mean_mask_recall": mean_recall,
            "overlays_dir": str(mode_overlay_dir),
            "per_frame": summaries,
        }
        (mode_out_dir / "summary.json").write_text(json.dumps(payload, indent=2))
        if save_overlays:
            print(f"[saved] overlays: {mode_overlay_dir}")
        print(f"[saved] summary: {mode_out_dir / 'summary.json'}")
        return payload

    run_one(save_overlays=True, mode_out_dir=out_dir)


if __name__ == "__main__":
    main()
