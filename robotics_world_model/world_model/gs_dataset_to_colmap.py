"""
Convert gs_dataset output to COLMAP text format.

Output directory:
  out_dir/
    images/
    masks/                  # optional, per-image background keep mask from FK
    sparse/0/cameras.txt
    sparse/0/images.txt
    sparse/0/points3D.txt

Copy-paste run:
  ROOT=/home/SENSETIME/yanzichen/data/file/Geometry-Grounded-Gaussian-Splatting
  WM_ROOT="$ROOT/robotics_world_model"
  PYTHONPATH="$WM_ROOT" python3 -m world_model.gs_dataset_to_colmap \
    --in-dir "$ROOT/gs_dataset" \
    --out-dir "$ROOT/gs_colmap" \
    --init-npz "$ROOT/gs_init.npz" \
    --init-max-points 200000 \
    --init-color 128,128,128

Note:
  poses.json may be camera-to-world (T_wc) or world-to-camera (T_cw).
  This script supports --pose-convention {auto,c2w,w2c}.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:  # pragma: no cover
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    from .kinematics_common import build_q_map_from_q32, compute_link_transforms
    from .gs_init_from_urdf import (
        _apply_visual_transform,
        _mesh_area,
        _parse_urdf_meshes,
        _sample_points_on_mesh,
    )
except ImportError:  # pragma: no cover
    from kinematics_common import build_q_map_from_q32, compute_link_transforms  # type: ignore
    from gs_init_from_urdf import (  # type: ignore
        _apply_visual_transform,
        _mesh_area,
        _parse_urdf_meshes,
        _sample_points_on_mesh,
    )


@dataclass(frozen=True)
class CameraDef:
    cam_id: str
    camera_id: int
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _rot_to_qvec(R: np.ndarray) -> np.ndarray:
    # COLMAP expects qw, qx, qy, qz
    K = np.array(
        [
            [R[0, 0] - R[1, 1] - R[2, 2], 0.0, 0.0, 0.0],
            [R[1, 0] + R[0, 1], R[1, 1] - R[0, 0] - R[2, 2], 0.0, 0.0],
            [R[2, 0] + R[0, 2], R[2, 1] + R[1, 2], R[2, 2] - R[0, 0] - R[1, 1], 0.0],
            [R[1, 2] - R[2, 1], R[2, 0] - R[0, 2], R[0, 1] - R[1, 0], R[0, 0] + R[1, 1] + R[2, 2]],
        ],
        dtype=np.float64,
    )
    K /= 3.0
    w, V = np.linalg.eigh(K)
    q = V[:, np.argmax(w)]
    if q[3] < 0:
        q = -q
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def _write_cameras_txt(out_dir: Path, cams: list[CameraDef]) -> None:
    lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        "# Number of cameras: {}".format(len(cams)),
    ]
    for c in cams:
        params = f"{c.fx} {c.fy} {c.cx} {c.cy}"
        lines.append(f"{c.camera_id} PINHOLE {c.width} {c.height} {params}")
    (out_dir / "cameras.txt").write_text("\n".join(lines) + "\n")


def _write_images_txt(out_dir: Path, images_lines: list[str]) -> None:
    header = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(images_lines)//2}",
    ]
    (out_dir / "images.txt").write_text("\n".join(header + images_lines) + "\n")


def _write_points3D_txt(out_dir: Path, pts: np.ndarray, rgb: np.ndarray | None = None) -> None:
    lines = [
        "# 3D point list with one line of data per point:",
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]",
        "# Number of points: {}".format(len(pts)),
    ]
    if rgb is None:
        rgb = np.full((len(pts), 3), 128, dtype=np.uint8)
    if len(rgb) != len(pts):
        raise ValueError("rgb length must match pts length")
    for i, p in enumerate(pts, start=1):
        x, y, z = p
        r, g, b = rgb[i - 1]
        lines.append(f"{i} {x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)} 1.0")
    (out_dir / "points3D.txt").write_text("\n".join(lines) + "\n")


def _sample_points(camera_centers: np.ndarray, num_points: int) -> np.ndarray:
    if camera_centers.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    mn = camera_centers.min(axis=0)
    mx = camera_centers.max(axis=0)
    span = np.maximum(mx - mn, 1e-3)
    mn = mn - 0.5 * span
    mx = mx + 0.5 * span
    return np.random.uniform(mn, mx, size=(num_points, 3)).astype(np.float32)


def _sample_points_in_front_of_camera(
    camera: CameraDef,
    R_cw: np.ndarray,
    t_cw: np.ndarray,
    num_points: int,
    depth_min: float = 0.5,
    depth_max: float = 2.5,
) -> np.ndarray:
    if num_points <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    u = np.random.uniform(0.0, float(camera.width), size=(num_points,))
    v = np.random.uniform(0.0, float(camera.height), size=(num_points,))
    z = np.random.uniform(depth_min, depth_max, size=(num_points,))
    x = (u - camera.cx) / camera.fx * z
    y = (v - camera.cy) / camera.fy * z
    pts_cam = np.stack([x, y, z], axis=1).astype(np.float64)

    # camera -> world
    R_wc = R_cw.T
    C = -R_wc @ t_cw
    pts_world = (R_wc @ pts_cam.T).T + C[None, :]
    return pts_world.astype(np.float32)


def _allocate_counts(weights: np.ndarray, total: int) -> np.ndarray:
    n = int(weights.shape[0])
    if n == 0 or total <= 0:
        return np.zeros((n,), dtype=np.int64)
    w = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    if float(w.sum()) <= 0:
        w = np.full((n,), 1.0 / n, dtype=np.float64)
    else:
        w = w / float(w.sum())
    raw = w * float(total)
    cnt = np.floor(raw).astype(np.int64)
    rem = int(total - int(cnt.sum()))
    if rem > 0:
        order = np.argsort(raw - cnt)[::-1]
        cnt[order[:rem]] += 1
    return cnt


def _build_robot_link_local_points(
    urdf_path: Path,
    package_root: Path,
    total_points: int,
    seed: int = 0,
    include_regex: str | None = None,
    exclude_regex: str | None = None,
) -> dict[str, np.ndarray]:
    visuals = _parse_urdf_meshes(urdf_path, package_root)
    inc = re.compile(include_regex) if include_regex else None
    exc = re.compile(exclude_regex) if exclude_regex else None
    if inc is not None or exc is not None:
        filtered = []
        for v in visuals:
            link = str(v.link)
            if inc is not None and inc.search(link) is None:
                continue
            if exc is not None and exc.search(link) is not None:
                continue
            filtered.append(v)
        visuals = filtered
    if not visuals:
        return {}

    mesh_cache: dict[str, Any] = {}
    areas = np.array([max(_mesh_area(v.mesh_path, v, mesh_cache), 0.0) for v in visuals], dtype=np.float64)
    counts = _allocate_counts(areas, total_points)
    rng = np.random.default_rng(seed)

    by_link: dict[str, list[np.ndarray]] = defaultdict(list)
    for v, n in zip(visuals, counts):
        if int(n) <= 0:
            continue
        pts_mesh = _sample_points_on_mesh(v.mesh_path, int(n), rng=rng, mesh_cache=mesh_cache)
        pts_link = _apply_visual_transform(pts_mesh, v)
        by_link[v.link].append(pts_link.astype(np.float32, copy=False))

    return {
        link: np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        for link, chunks in by_link.items()
        if len(chunks) > 0
    }


def _build_background_keep_mask_from_fk(
    link_points_local: dict[str, np.ndarray],
    link_T_world: dict[str, np.ndarray],
    R_cw: np.ndarray,
    t_cw: np.ndarray,
    camera: CameraDef,
    radius_px: int = 3,
) -> np.ndarray:
    h, w = int(camera.height), int(camera.width)
    dyn = np.zeros((h, w), dtype=np.uint8)
    if not link_points_local:
        return np.full((h, w), 255, dtype=np.uint8)

    for link, pts_local in link_points_local.items():
        T = link_T_world.get(link)
        if T is None or pts_local.size == 0:
            continue
        R_lw = T[:3, :3]
        t_lw = T[:3, 3]
        pts_world = (R_lw @ pts_local.T).T + t_lw[None, :]
        pts_cam = (R_cw @ pts_world.T).T + t_cw[None, :]

        z = pts_cam[:, 2]
        valid = z > 1e-6
        if not np.any(valid):
            continue
        pts_cam = pts_cam[valid]
        u = camera.fx * (pts_cam[:, 0] / pts_cam[:, 2]) + camera.cx
        v = camera.fy * (pts_cam[:, 1] / pts_cam[:, 2]) + camera.cy
        ui = np.round(u).astype(np.int32)
        vi = np.round(v).astype(np.int32)
        inside = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        ui = ui[inside]
        vi = vi[inside]
        dyn[vi, ui] = 1

    if radius_px > 0 and np.any(dyn):
        if cv2 is None:
            pad = int(radius_px)
            ys, xs = np.nonzero(dyn)
            for y, x in zip(ys, xs):
                y0 = max(0, y - pad)
                y1 = min(h, y + pad + 1)
                x0 = max(0, x - pad)
                x1 = min(w, x + pad + 1)
                dyn[y0:y1, x0:x1] = 1
        else:
            k = np.ones((2 * int(radius_px) + 1, 2 * int(radius_px) + 1), np.uint8)
            dyn = cv2.dilate(dyn, k, iterations=1)

    keep = (1 - dyn) * 255
    return keep.astype(np.uint8, copy=False)


def _pose_to_w2c(pose: np.ndarray, convention: str) -> tuple[np.ndarray, np.ndarray]:
    if convention == "w2c":
        return pose[:3, :3], pose[:3, 3]
    if convention == "c2w":
        R_wc = pose[:3, :3]
        t_wc = pose[:3, 3]
        R_cw = R_wc.T
        t_cw = -R_cw @ t_wc
        return R_cw, t_cw
    raise ValueError(f"Unsupported pose convention: {convention}")


def _project_in_image_ratio(
    points: np.ndarray,
    R_cw: np.ndarray,
    t_cw: np.ndarray,
    camera: CameraDef,
    max_points: int = 4000,
) -> tuple[float, float]:
    if points.size == 0:
        return 0.0, 0.0
    pts = points
    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]
    Xc = (R_cw @ pts.T).T + t_cw[None, :]
    z = Xc[:, 2]
    in_front = z > 1e-6
    if not np.any(in_front):
        return 0.0, 0.0
    Xf = Xc[in_front]
    u = camera.fx * (Xf[:, 0] / Xf[:, 2]) + camera.cx
    v = camera.fy * (Xf[:, 1] / Xf[:, 2]) + camera.cy
    in_img = (u >= 0.0) & (u < camera.width) & (v >= 0.0) & (v < camera.height)
    return float(np.mean(in_front)), float(np.mean(in_img))


def _infer_pose_convention(
    poses: list[dict[str, Any]],
    cam_defs: list[CameraDef],
    points: np.ndarray,
) -> str:
    if points.size == 0 or len(poses) == 0:
        return "w2c"
    cam_by_id = {c.cam_id: c for c in cam_defs}
    # Score a handful of frames to avoid overhead.
    probe = poses[: min(8, len(poses))]
    scores: dict[str, list[float]] = {"w2c": [], "c2w": []}
    for frame in probe:
        cam = cam_by_id[frame["cam_id"]]
        pose = np.asarray(frame["pose"], dtype=np.float64)
        for conv in ("w2c", "c2w"):
            R_cw, t_cw = _pose_to_w2c(pose, conv)
            _, in_img = _project_in_image_ratio(points, R_cw, t_cw, cam)
            scores[conv].append(in_img)
    mean_w2c = float(np.mean(scores["w2c"])) if scores["w2c"] else 0.0
    mean_c2w = float(np.mean(scores["c2w"])) if scores["c2w"] else 0.0
    chosen = "w2c" if mean_w2c >= mean_c2w else "c2w"
    print(
        f"[gs_dataset_to_colmap] pose auto-detect: "
        f"w2c_inimg={mean_w2c:.4f}, c2w_inimg={mean_c2w:.4f} -> {chosen}"
    )
    return chosen


def convert(
    in_dir: Path,
    out_dir: Path,
    num_points: int,
    *,
    init_npz: Path | None = None,
    init_max_points: int | None = None,
    init_color: tuple[int, int, int] = (128, 128, 128),
    pose_convention: str = "auto",
    robot_mask_urdf: Path | None = None,
    robot_mask_package_root: Path | None = None,
    robot_mask_points: int = 60000,
    robot_mask_radius_px: int = 3,
    robot_mask_seed: int = 0,
    robot_mask_include_regex: str | None = "(shoulder|elbow|wrist|tcp|thumb|index|middle|ring|little)",
    robot_mask_exclude_regex: str | None = "(base|waist|torso|pelvis|head|camera|neck|chest)",
) -> None:
    cameras = _load_json(in_dir / "cameras.json")["cameras"]
    poses = _load_json(in_dir / "poses.json")["frames"]

    cam_defs = []
    for i, cam in enumerate(cameras, start=1):
        K = np.asarray(cam["K"], dtype=np.float32)
        cam_defs.append(
            CameraDef(
                cam_id=cam["cam_id"],
                camera_id=i,
                width=int(cam["width"]),
                height=int(cam["height"]),
                fx=float(K[0, 0]),
                fy=float(K[1, 1]),
                cx=float(K[0, 2]),
                cy=float(K[1, 2]),
            )
        )
    cam_id_map = {c.cam_id: c.camera_id for c in cam_defs}
    cam_by_id = {c.cam_id: c for c in cam_defs}

    init_points_for_detection = np.zeros((0, 3), dtype=np.float32)
    if init_npz is not None:
        init_data = np.load(init_npz)
        init_points_for_detection = np.asarray(init_data["points"], dtype=np.float32)
        if init_max_points is not None and init_points_for_detection.shape[0] > init_max_points:
            idx = np.random.choice(init_points_for_detection.shape[0], size=init_max_points, replace=False)
            init_points_for_detection = init_points_for_detection[idx]

    if pose_convention == "auto":
        pose_convention = _infer_pose_convention(poses, cam_defs, init_points_for_detection)
    elif pose_convention not in ("w2c", "c2w"):
        raise ValueError("pose_convention must be one of: auto, w2c, c2w")
    else:
        print(f"[gs_dataset_to_colmap] pose convention: {pose_convention}")

    images_dir = out_dir / "images"
    sparse_dir = out_dir / "sparse" / "0"
    masks_dir = out_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    link_points_local: dict[str, np.ndarray] | None = None
    frame_link_T_cache: dict[str, dict[str, np.ndarray]] = {}
    if robot_mask_urdf is not None:
        pkg_root = (
            robot_mask_package_root
            if robot_mask_package_root is not None
            else robot_mask_urdf.parent.parent
        )
        link_points_local = _build_robot_link_local_points(
            robot_mask_urdf,
            pkg_root,
            total_points=int(robot_mask_points),
            seed=int(robot_mask_seed),
            include_regex=robot_mask_include_regex,
            exclude_regex=robot_mask_exclude_regex,
        )
        masks_dir.mkdir(parents=True, exist_ok=True)
        total_local = int(sum(v.shape[0] for v in link_points_local.values()))
        print(
            f"[gs_dataset_to_colmap] FK mask enabled: links={len(link_points_local)}, "
            f"points={total_local}, radius={robot_mask_radius_px}px"
        )
        if robot_mask_include_regex:
            print(f"[gs_dataset_to_colmap] FK mask include_regex: {robot_mask_include_regex}")
        if robot_mask_exclude_regex:
            print(f"[gs_dataset_to_colmap] FK mask exclude_regex: {robot_mask_exclude_regex}")

    images_lines: list[str] = []
    camera_centers = []

    for image_id, frame in enumerate(poses, start=1):
        cam_id = frame["cam_id"]
        camera_id = cam_id_map[cam_id]

        pose = np.asarray(frame["pose"], dtype=np.float64)
        R_cw, t_cw = _pose_to_w2c(pose, pose_convention)
        # camera center in world for fallback point sampling
        camera_center = -R_cw.T @ t_cw
        camera_centers.append(camera_center)

        qvec = _rot_to_qvec(R_cw)
        tvec = t_cw

        src_path = in_dir / frame["image_path"]
        dst_name = f"{frame['frame_id']}_{cam_id}.png"
        dst_path = images_dir / dst_name
        if not dst_path.exists():
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_path, dst_path)

        images_lines.append(
            f"{image_id} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {tvec[0]} {tvec[1]} {tvec[2]} {camera_id} {dst_name}"
        )
        images_lines.append("")  # empty line for points2D

        if link_points_local is not None:
            q = frame.get("q")
            if q is None:
                raise ValueError(
                    "poses.json missing 'q' field required by FK mask. "
                    "Regenerate gs_dataset with current gs_dataset.py."
                )
            frame_id = str(frame["frame_id"])
            if frame_id not in frame_link_T_cache:
                q_arr = np.asarray(q, dtype=np.float32)
                q_map = build_q_map_from_q32(q_arr)
                frame_link_T_cache[frame_id] = compute_link_transforms(robot_mask_urdf, q_map=q_map)
            link_T_world = frame_link_T_cache[frame_id]
            keep_mask = _build_background_keep_mask_from_fk(
                link_points_local,
                link_T_world,
                R_cw,
                t_cw,
                cam_by_id[cam_id],
                radius_px=int(robot_mask_radius_px),
            )
            Image.fromarray(keep_mask, mode="L").save(masks_dir / dst_name)

    _write_cameras_txt(sparse_dir, cam_defs)
    _write_images_txt(sparse_dir, images_lines)

    if init_npz is not None:
        pts = init_points_for_detection
        if init_max_points is not None and pts.shape[0] > init_max_points:
            idx = np.random.choice(pts.shape[0], size=init_max_points, replace=False)
            pts = pts[idx]
        rgb = np.tile(np.array(init_color, dtype=np.uint8)[None, :], (pts.shape[0], 1))
        _write_points3D_txt(sparse_dir, pts, rgb=rgb)
    else:
        camera_centers = np.asarray(camera_centers, dtype=np.float32)
        span = (
            np.linalg.norm(camera_centers.max(axis=0) - camera_centers.min(axis=0))
            if camera_centers.shape[0] > 0
            else 0.0
        )
        if span < 0.05 and len(poses) > 0:
            # With near-static cameras, center-box sampling collapses around the camera and
            # can yield invisible initialization. Sample points inside the camera frustum.
            probe = poses[0]
            probe_cam = cam_by_id[probe["cam_id"]]
            probe_pose = np.asarray(probe["pose"], dtype=np.float64)
            probe_R_cw, probe_t_cw = _pose_to_w2c(probe_pose, pose_convention)
            pts = _sample_points_in_front_of_camera(
                probe_cam,
                probe_R_cw,
                probe_t_cw,
                num_points=num_points,
                depth_min=0.5,
                depth_max=2.5,
            )
            print(
                f"[gs_dataset_to_colmap] static-camera fallback: "
                f"span={span:.6f}m, sampled {len(pts)} points in frustum"
            )
        else:
            pts = _sample_points(camera_centers, num_points=num_points)
        _write_points3D_txt(sparse_dir, pts)


def main() -> None:
    parser = argparse.ArgumentParser(description="GS dataset -> COLMAP")
    parser.add_argument("--in-dir", default="/tmp/gs_dataset", help="Input gs_dataset dir")
    parser.add_argument("--out-dir", default="/tmp/gs_colmap", help="Output COLMAP dataset dir")
    parser.add_argument("--num-points", type=int, default=20000, help="Synthetic points for initialization")
    parser.add_argument("--init-npz", default=None, help="Optional gs_init.npz for points3D initialization")
    parser.add_argument("--init-max-points", type=int, default=None, help="Downsample init points")
    parser.add_argument("--init-color", default="128,128,128", help="RGB for init points, e.g. 128,128,128")
    parser.add_argument(
        "--pose-convention",
        default="auto",
        choices=["auto", "w2c", "c2w"],
        help="Interpretation of poses.json matrices",
    )
    parser.add_argument("--robot-mask-urdf", default=None, help="Enable FK robot mask with this URDF path")
    parser.add_argument("--robot-mask-package-root", default=None, help="Package root for URDF mesh resolving")
    parser.add_argument("--robot-mask-points", type=int, default=60000, help="Total sampled robot points for FK mask")
    parser.add_argument("--robot-mask-radius-px", type=int, default=3, help="Pixel dilation radius for FK mask")
    parser.add_argument("--robot-mask-seed", type=int, default=0, help="Seed for FK mask point sampling")
    parser.add_argument(
        "--robot-mask-include-regex",
        default="(shoulder|elbow|wrist|tcp|thumb|index|middle|ring|little)",
        help="Regex to include robot links for FK mask",
    )
    parser.add_argument(
        "--robot-mask-exclude-regex",
        default="(base|waist|torso|pelvis|head|camera|neck|chest)",
        help="Regex to exclude robot links for FK mask",
    )
    args = parser.parse_args()

    init_npz = Path(args.init_npz) if args.init_npz else None
    color_parts = [int(x) for x in args.init_color.replace(" ", "").split(",")]
    init_color = (color_parts[0], color_parts[1], color_parts[2])

    convert(
        Path(args.in_dir),
        Path(args.out_dir),
        num_points=args.num_points,
        init_npz=init_npz,
        init_max_points=args.init_max_points,
        init_color=init_color,
        pose_convention=args.pose_convention,
        robot_mask_urdf=Path(args.robot_mask_urdf) if args.robot_mask_urdf else None,
        robot_mask_package_root=Path(args.robot_mask_package_root) if args.robot_mask_package_root else None,
        robot_mask_points=args.robot_mask_points,
        robot_mask_radius_px=args.robot_mask_radius_px,
        robot_mask_seed=args.robot_mask_seed,
        robot_mask_include_regex=args.robot_mask_include_regex,
        robot_mask_exclude_regex=args.robot_mask_exclude_regex,
    )


if __name__ == "__main__":
    main()
