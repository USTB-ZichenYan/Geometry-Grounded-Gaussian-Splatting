"""
Generate 3DGS initialization points from URDF meshes.

Outputs:
  - <out>.npz with keys: points, sigmas, link_ids, link_names
  - <out>.json metadata

Copy-paste run:
  ROOT=/home/SENSETIME/yanzichen/data/file/Geometry-Grounded-Gaussian-Splatting
  WM_ROOT="$ROOT/robotics_world_model"
  URDF="$WM_ROOT/tianyi2_urdf-tianyi2.0/urdf/tianyi2.0_urdf_with_hands.urdf"
  PYTHONPATH="$WM_ROOT" python3 -m world_model.gs_init_from_urdf \
    --urdf "$URDF" \
    --out "$ROOT/gs_init.npz" \
    --total-points 200000 \
    --sigma-mm 2.0 \
    --alloc-mode area \
    --area-exp 0.7 \
    --output-frame base \
    --seed 0 \
    --verbose

Optional:
  - Add --q "comma,separated,32d" to place points at a specific joint state.
  - Add --plot-mesh for a quick mesh/points visual check (needs trimesh scene backend).
"""

from __future__ import annotations

import argparse
import json
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
try:
    from .kinematics_common import (
        DEFAULT_URDF as KIN_DEFAULT_URDF,
        build_q_map_from_q32,
        compute_link_transforms,
        load_q_input,
        rpy_to_rot,
    )
except ImportError:  # pragma: no cover
    from kinematics_common import (  # type: ignore
        DEFAULT_URDF as KIN_DEFAULT_URDF,
        build_q_map_from_q32,
        compute_link_transforms,
        load_q_input,
        rpy_to_rot,
    )

try:  # pragma: no cover
    import trimesh  # type: ignore
except Exception:  # pragma: no cover
    trimesh = None  # type: ignore


DEFAULT_URDF = KIN_DEFAULT_URDF
DEFAULT_PACKAGE_ROOT = "/home/SENSETIME/yanzichen/data/file/Geometry-Grounded-Gaussian-Splatting/robotics_world_model/tianyi2_urdf-tianyi2.0"
DEFAULT_INCLUDE_REGEX = r"(shoulder|elbow|wrist|tcp|left_|right_|L_base_link|R_base_link)"


def _require_trimesh() -> None:
    if trimesh is None:
        raise ImportError("trimesh is required. Install it with: pip install trimesh")


@dataclass(frozen=True)
class MeshVisual:
    link: str
    mesh_path: Path
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    scale: np.ndarray


def _rpy_to_rot(rpy: np.ndarray) -> np.ndarray:
    return rpy_to_rot(rpy)


def _resolve_mesh_path(mesh_filename: str, urdf_dir: Path, package_root: Path) -> Path:
    if mesh_filename.startswith("package://"):
        rel = mesh_filename[len("package://") :]
        # Drop the package name prefix (e.g., tianyi2_urdf/meshes/xxx.stl)
        parts = rel.split("/", 1)
        rel_path = parts[1] if len(parts) == 2 else parts[0]
        return package_root / rel_path
    p = Path(mesh_filename)
    if p.is_absolute():
        return p
    return urdf_dir / p


def _parse_urdf_meshes(urdf_path: Path, package_root: Path) -> List[MeshVisual]:
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()
    urdf_dir = urdf_path.parent
    visuals: List[MeshVisual] = []

    for link in root.findall("link"):
        link_name = link.attrib.get("name", "")
        for visual in link.findall("visual"):
            origin = visual.find("origin")
            if origin is not None:
                xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
                rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
            else:
                xyz = np.zeros(3, dtype=np.float32)
                rpy = np.zeros(3, dtype=np.float32)

            geom = visual.find("geometry")
            if geom is None:
                continue
            mesh = geom.find("mesh")
            if mesh is None:
                continue
            filename = mesh.attrib.get("filename")
            if not filename:
                continue

            scale_attr = mesh.attrib.get("scale", "1 1 1")
            scale = np.fromstring(scale_attr, sep=" ").astype(np.float32)
            mesh_path = _resolve_mesh_path(filename, urdf_dir, package_root)
            visuals.append(
                MeshVisual(
                    link=link_name,
                    mesh_path=mesh_path,
                    origin_xyz=xyz.astype(np.float32),
                    origin_rpy=rpy.astype(np.float32),
                    scale=scale,
                )
            )

    return visuals


def _filter_visuals(visuals: List[MeshVisual], include_regex: str | None, exclude_regex: str | None) -> List[MeshVisual]:
    import re

    if include_regex:
        include_re = re.compile(include_regex)
        visuals = [v for v in visuals if include_re.search(v.link)]
    if exclude_regex:
        exclude_re = re.compile(exclude_regex)
        visuals = [v for v in visuals if not exclude_re.search(v.link)]
    return visuals


def _compute_link_transforms(urdf_path: Path, q_map: dict[str, float] | None = None) -> dict[str, np.ndarray]:
    return compute_link_transforms(urdf_path, q_map=q_map)


def _load_q(arg: str) -> np.ndarray:
    return load_q_input(arg)


def _build_q_map(q: np.ndarray | None) -> dict[str, float]:
    return build_q_map_from_q32(q)




def _load_mesh(mesh_path: Path, mesh_cache: dict[str, Any] | None = None) -> Any:
    _require_trimesh()
    key = str(mesh_path)
    if mesh_cache is not None and key in mesh_cache:
        return mesh_cache[key]
    mesh = trimesh.load_mesh(str(mesh_path), force="mesh")
    if mesh_cache is not None:
        mesh_cache[key] = mesh
    return mesh


def _sample_points_on_mesh(
    mesh_path: Path,
    n: int,
    *,
    rng: np.random.Generator | None = None,
    mesh_cache: dict[str, Any] | None = None,
) -> np.ndarray:
    _require_trimesh()
    mesh = _load_mesh(mesh_path, mesh_cache)
    if mesh.is_empty:
        raise ValueError(f"Empty mesh: {mesh_path}")
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    rng = rng or np.random.default_rng()
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if faces.size == 0:
        raise ValueError(f"Mesh has no faces: {mesh_path}")

    face_areas = np.asarray(mesh.area_faces, dtype=np.float64)
    face_areas = np.maximum(face_areas, 0.0)
    area_sum = float(face_areas.sum())
    if area_sum <= 0.0:
        raise ValueError(f"Mesh has non-positive total area: {mesh_path}")

    probs = face_areas / area_sum
    chosen = rng.choice(face_areas.shape[0], size=n, replace=True, p=probs)
    tri = vertices[faces[chosen]]  # (n, 3, 3)

    u = rng.random(n).astype(np.float32, copy=False)
    v = rng.random(n).astype(np.float32, copy=False)
    sqrt_u = np.sqrt(u).astype(np.float32, copy=False)
    w0 = 1.0 - sqrt_u
    w1 = sqrt_u * (1.0 - v)
    w2 = sqrt_u * v
    points = (
        tri[:, 0, :] * w0[:, None]
        + tri[:, 1, :] * w1[:, None]
        + tri[:, 2, :] * w2[:, None]
    )
    return np.asarray(points, dtype=np.float32)


def _apply_visual_transform(points: np.ndarray, visual: MeshVisual) -> np.ndarray:
    # scale -> rotate -> translate
    pts = points * visual.scale[None, :]
    R = _rpy_to_rot(visual.origin_rpy)
    pts = (R @ pts.T).T
    pts = pts + visual.origin_xyz[None, :]
    return pts


def _mesh_area(mesh_path: Path, visual: MeshVisual, mesh_cache: dict[str, Any] | None = None) -> float:
    _require_trimesh()
    mesh = _load_mesh(mesh_path, mesh_cache)
    if mesh.is_empty:
        return 0.0
    # apply scale to area (scale^2)
    scale = visual.scale
    scale_area = float(np.mean(scale) ** 2) if np.all(scale > 0) else 1.0
    return float(mesh.area) * scale_area


def _allocate_points_with_bounds(
    weights: np.ndarray,
    total_points: int,
    *,
    min_per_visual: int,
    max_per_visual: int,
) -> List[int]:
    n = int(weights.shape[0])
    if n <= 0:
        return []
    if min_per_visual > max_per_visual:
        raise ValueError(f"Invalid bounds: min_per_visual ({min_per_visual}) > max_per_visual ({max_per_visual}).")

    min_total = int(min_per_visual * n)
    max_total = int(max_per_visual * n)
    if total_points < min_total or total_points > max_total:
        raise ValueError(
            f"Cannot allocate {total_points} points across {n} visuals with "
            f"bounds [{min_per_visual}, {max_per_visual}] (feasible range: [{min_total}, {max_total}])."
        )

    w = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    w_sum = float(w.sum())
    if w_sum <= 0.0:
        w = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        w = w / w_sum

    counts = np.full(n, int(min_per_visual), dtype=np.int64)
    caps = np.full(n, int(max_per_visual - min_per_visual), dtype=np.int64)
    remaining = int(total_points - counts.sum())
    if remaining == 0:
        return counts.astype(int).tolist()

    desired = w * remaining
    extra = np.floor(desired).astype(np.int64)
    extra = np.minimum(extra, caps)
    counts += extra
    caps -= extra
    remaining -= int(extra.sum())
    if remaining == 0:
        return counts.astype(int).tolist()

    frac = desired - np.floor(desired)
    for idx in np.argsort(-frac):
        if remaining == 0:
            break
        if caps[idx] <= 0:
            continue
        counts[idx] += 1
        caps[idx] -= 1
        remaining -= 1
    if remaining == 0:
        return counts.astype(int).tolist()

    while remaining > 0:
        active = np.where(caps > 0)[0]
        if active.size == 0:
            raise RuntimeError("Internal allocation error: no capacity left while points remain.")
        aw = w[active]
        aw_sum = float(aw.sum())
        if aw_sum <= 0.0:
            aw = np.full(active.shape[0], 1.0 / active.shape[0], dtype=np.float64)
        else:
            aw = aw / aw_sum
        chunk = np.floor(aw * remaining).astype(np.int64)
        chunk = np.minimum(chunk, caps[active])
        if int(chunk.sum()) == 0:
            chunk[0] = 1
        counts[active] += chunk
        caps[active] -= chunk
        remaining -= int(chunk.sum())

    if int(counts.sum()) != total_points:
        raise RuntimeError("Internal allocation error: allocated points do not match requested total.")
    return counts.astype(int).tolist()


def _allocate_points(
    visuals: List[MeshVisual],
    total_points: int,
    *,
    mode: str = "uniform",
    min_per_visual: int = 200,
    max_per_visual: int = 20000,
    area_exp: float = 1.0,
    mesh_cache: dict[str, Any] | None = None,
) -> List[int]:
    n = len(visuals)
    if n == 0:
        return []

    if mode == "uniform":
        weights = np.ones((n,), dtype=np.float64)
        return _allocate_points_with_bounds(
            weights,
            total_points,
            min_per_visual=min_per_visual,
            max_per_visual=max_per_visual,
        )

    areas = np.array([_mesh_area(v.mesh_path, v, mesh_cache) for v in visuals], dtype=np.float64)
    areas = np.maximum(areas, 1e-6)
    weights = np.power(areas, float(area_exp))
    return _allocate_points_with_bounds(
        weights,
        total_points,
        min_per_visual=min_per_visual,
        max_per_visual=max_per_visual,
    )


def generate_gaussians(
    urdf_path: Path,
    package_root: Path,
    total_points: int,
    sigma_mm: float,
    include_regex: str | None,
    exclude_regex: str | None,
    *,
    visuals: List[MeshVisual] | None = None,
    verbose: bool = False,
    alloc_mode: str = "uniform",
    min_per_visual: int = 200,
    max_per_visual: int = 20000,
    area_exp: float = 1.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    if visuals is None:
        visuals = _parse_urdf_meshes(urdf_path, package_root)
        visuals = _filter_visuals(visuals, include_regex, exclude_regex)
    if not visuals:
        raise ValueError("No mesh visuals found in URDF.")

    mesh_cache: dict[str, Any] = {}
    rng = np.random.default_rng(seed)
    counts = _allocate_points(
        visuals,
        total_points=total_points,
        mode=alloc_mode,
        min_per_visual=min_per_visual,
        max_per_visual=max_per_visual,
        area_exp=area_exp,
        mesh_cache=mesh_cache,
    )
    if verbose:
        print(
            f"[gs_init] sample_visuals={len(visuals)} total_points={total_points} "
            f"alloc_mode={alloc_mode} min_per_visual={min_per_visual} max_per_visual={max_per_visual} "
            f"area_exp={area_exp}"
        )
        for v, n in zip(visuals, counts):
            print(f"  - {v.link}: {n} pts | {v.mesh_path.name}")
    all_points: List[np.ndarray] = []
    all_link_ids: List[np.ndarray] = []
    link_names: List[str] = []
    link_index: Dict[str, int] = {}

    for visual, n in zip(visuals, counts):
        if n <= 0:
            continue
        pts = _sample_points_on_mesh(visual.mesh_path, n, rng=rng, mesh_cache=mesh_cache)
        pts = _apply_visual_transform(pts, visual)  # in link-local frame

        if visual.link not in link_index:
            link_index[visual.link] = len(link_names)
            link_names.append(visual.link)
        lid = link_index[visual.link]
        all_points.append(pts)
        all_link_ids.append(np.full((pts.shape[0],), lid, dtype=np.int32))

    points = np.concatenate(all_points, axis=0).astype(np.float32)
    link_ids = np.concatenate(all_link_ids, axis=0)
    sigma = (sigma_mm / 1000.0)  # mm -> meters
    sigmas = np.full((points.shape[0], 3), sigma, dtype=np.float32)

    meta = {
        "urdf": str(urdf_path),
        "package_root": str(package_root),
        "total_points": int(points.shape[0]),
        "sigma_mm": float(sigma_mm),
        "link_names": link_names,
        "include_regex": include_regex,
        "exclude_regex": exclude_regex,
        "alloc_mode": alloc_mode,
        "min_per_visual": int(min_per_visual),
        "max_per_visual": int(max_per_visual),
        "area_exp": float(area_exp),
        "seed": seed,
    }
    return points, sigmas, link_ids, link_names, meta


def _transform_points_by_link(
    points: np.ndarray,
    link_ids: np.ndarray,
    link_names: List[str],
    link_T_world: dict[str, np.ndarray],
) -> np.ndarray:
    points_world = points.copy()
    for lid, name in enumerate(link_names):
        mask = link_ids == lid
        if not np.any(mask):
            continue
        T_link = link_T_world.get(name)
        if T_link is None:
            continue
        P = points_world[mask]
        P_h = np.concatenate([P, np.ones((P.shape[0], 1), dtype=np.float32)], axis=1)
        P_w = (T_link @ P_h.T).T[:, :3]
        points_world[mask] = P_w
    return points_world


def _plot_mesh_and_points(
    visuals: List[MeshVisual],
    points: np.ndarray,
    link_ids: np.ndarray,
    link_names: List[str],
    link_T_world: dict[str, np.ndarray],
    sample_points: int = 5000,
    mesh_alpha: float = 0.35,
) -> None:
    _require_trimesh()
    # Subsample points for display
    if points.shape[0] > sample_points:
        idx = np.random.choice(points.shape[0], size=sample_points, replace=False)
        pts = points[idx]
        link_ids = link_ids[idx] if link_ids is not None else link_ids
    else:
        pts = points

    scene = trimesh.Scene()

    # Add meshes (semi-transparent)
    for vis in visuals:
        if not vis.mesh_path.exists():
            continue
        mesh = trimesh.load_mesh(str(vis.mesh_path), force="mesh")
        if mesh.is_empty:
            continue
        # apply visual transform: scale -> rotate -> translate
        mesh.apply_scale(vis.scale)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = _rpy_to_rot(vis.origin_rpy)
        T[:3, 3] = vis.origin_xyz
        T_link = link_T_world.get(vis.link, np.eye(4, dtype=np.float32))
        mesh.apply_transform(T_link @ T)

        # set color with alpha
        if mesh.visual.kind != "face":
            mesh.visual = trimesh.visual.ColorVisuals(mesh, face_colors=[200, 200, 200, int(255 * mesh_alpha)])
        else:
            colors = mesh.visual.face_colors
            colors[:, 3] = int(255 * mesh_alpha)
            mesh.visual.face_colors = colors
        scene.add_geometry(mesh)

    # Add points
    pts_world = _transform_points_by_link(pts, link_ids, link_names, link_T_world)

    pc = trimesh.points.PointCloud(pts_world, colors=[255, 0, 0, 255])
    scene.add_geometry(pc)

    # Base origin (green) + axes
    if "base" in link_T_world:
        base_origin = link_T_world["base"][:3, 3]
        base_sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.015)
        base_sphere.visual = trimesh.visual.ColorVisuals(base_sphere, vertex_colors=[0, 255, 0, 255])
        T_base = np.eye(4, dtype=np.float32)
        T_base[:3, 3] = base_origin
        base_sphere.apply_transform(T_base)
        scene.add_geometry(base_sphere)
        base_axes = trimesh.creation.axis(origin_size=0.01, axis_length=0.15)
        base_axes.apply_transform(T_base)
        scene.add_geometry(base_axes)

    # Head camera frame
    head_link = "camera_head_link"
    if head_link in link_T_world:
        T_head = link_T_world[head_link].copy()
        head_axes = trimesh.creation.axis(origin_size=0.008, axis_length=0.12)
        head_axes.apply_transform(T_head)
        scene.add_geometry(head_axes)

    scene.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Init 3DGS means from URDF meshes")
    parser.add_argument("--urdf", default=DEFAULT_URDF, help="URDF path")
    parser.add_argument("--package-root", default=DEFAULT_PACKAGE_ROOT, help="URDF package root")
    parser.add_argument("--total-points", type=int, default=400000, help="Total sampled points")
    parser.add_argument("--sigma-mm", type=float, default=2.0, help="Sigma (mm) for isotropic gaussians")
    parser.add_argument("--out", default="/tmp/gs_init.npz", help="Output .npz file")
    parser.add_argument(
        "--output-frame",
        default="base",
        choices=["base", "link_local"],
        help="Frame for exported points",
    )
    parser.add_argument(
        "--alloc-mode",
        default="uniform",
        choices=["uniform", "area"],
        help="Point allocation mode: uniform per link or area-weighted",
    )
    parser.add_argument("--min-per-visual", type=int, default=50, help="Min points per visual (area mode)")
    parser.add_argument("--max-per-visual", type=int, default=50000, help="Max points per visual (area mode)")
    parser.add_argument("--area-exp", type=float, default=0.7, help="Area exponent for allocation (area mode)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling (set for reproducibility)")
    parser.add_argument(
        "--include-regex",
        default=DEFAULT_INCLUDE_REGEX,
        help="Regex to include link names (default: arms + hands)",
    )
    parser.add_argument("--exclude-regex", default=None, help="Regex to exclude link names")
    parser.add_argument("--all-links", action="store_true", help="Disable link filtering (use all links)")
    parser.add_argument("--plot", action="store_true", help="Plot sampled points")
    parser.add_argument("--plot-sample", type=int, default=5000, help="Max points to plot")
    parser.add_argument("--plot-mesh", action="store_true", help="Plot mesh + points in trimesh Scene")
    parser.add_argument("--q", default=None, help="Joint vector for URDF placement (32D). If not set, zeros.")
    parser.add_argument("--verbose", action="store_true", help="Print progress info")
    args = parser.parse_args()

    include_regex = None if args.all_links else args.include_regex
    exclude_regex = args.exclude_regex

    t0 = time.perf_counter()
    all_visuals = _parse_urdf_meshes(Path(args.urdf), Path(args.package_root))
    filtered_visuals = _filter_visuals(all_visuals, include_regex, exclude_regex)

    points, sigmas, link_ids, link_names, meta = generate_gaussians(
        Path(args.urdf),
        Path(args.package_root),
        total_points=args.total_points,
        sigma_mm=args.sigma_mm,
        include_regex=include_regex,
        exclude_regex=exclude_regex,
        visuals=filtered_visuals,
        verbose=args.verbose,
        alloc_mode=args.alloc_mode,
        min_per_visual=args.min_per_visual,
        max_per_visual=args.max_per_visual,
        area_exp=args.area_exp,
        seed=args.seed,
    )
    if args.verbose:
        dt = time.perf_counter() - t0
        print(f"[gs_init] done in {dt:.1f}s, points={points.shape[0]}")

    link_T_world: dict[str, np.ndarray] | None = None
    output_frame = args.output_frame

    if output_frame == "base" or args.plot_mesh:
        q = _load_q(args.q) if args.q is not None else None
        q_map = _build_q_map(q)
        link_T_world = _compute_link_transforms(Path(args.urdf), q_map=q_map)

    points_out = points
    if output_frame == "base":
        if link_T_world is None:
            raise RuntimeError("Internal error: missing link transforms for base export")
        points_out = _transform_points_by_link(points, link_ids, link_names, link_T_world)

    meta["output_frame"] = output_frame

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, points=points_out, sigmas=sigmas, link_ids=link_ids, link_names=link_names)
    (out_path.with_suffix(".json")).write_text(json.dumps(meta, indent=2))

    if args.plot:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("matplotlib is required for --plot") from e

        # random subsample for display
        n = min(points.shape[0], args.plot_sample)
        idx = np.random.choice(points.shape[0], size=n, replace=False)
        pts = points[idx]
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, alpha=0.6)
        ax.set_title("Sampled GS means (arms + hands)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.tight_layout()
        plt.show()

    if args.plot_mesh:
        if link_T_world is None:
            q = _load_q(args.q) if args.q is not None else None
            q_map = _build_q_map(q)
            link_T_world = _compute_link_transforms(Path(args.urdf), q_map=q_map)
        _plot_mesh_and_points(
            visuals=all_visuals,
            points=points,
            link_ids=link_ids,
            link_names=link_names,
            link_T_world=link_T_world,
            sample_points=args.plot_sample,
        )


if __name__ == "__main__":
    main()
