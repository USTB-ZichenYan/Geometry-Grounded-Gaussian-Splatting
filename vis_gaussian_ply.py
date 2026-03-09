#!/usr/bin/env python3
"""Quick viewer for Gaussian-Splatting point clouds saved as PLY."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData

SH_C0 = 0.28209479177387814


def _parse_views(spec: str) -> list[tuple[float, float]]:
    views: list[tuple[float, float]] = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        e, a = part.split(",")
        views.append((float(e), float(a)))
    if not views:
        raise ValueError("No valid views parsed from --views.")
    return views


def _norm01(x: np.ndarray, lo_q: float = 1.0, hi_q: float = 99.0) -> np.ndarray:
    lo, hi = np.percentile(x, [lo_q, hi_q])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def _color_values(vertices, xyz: np.ndarray, mode: str, seed: int) -> tuple[np.ndarray, bool]:
    names = set(vertices.dtype.names or [])

    if mode == "opacity":
        if "opacity" in names:
            op_raw = np.asarray(vertices["opacity"], dtype=np.float32)
            return (1.0 / (1.0 + np.exp(-op_raw))).astype(np.float32), False
        return np.ones((xyz.shape[0],), dtype=np.float32), False

    if mode == "scale":
        s_names = [f"scale_{i}" for i in range(3)]
        if all(n in names for n in s_names):
            s = np.stack([np.asarray(vertices[n], dtype=np.float32) for n in s_names], axis=1)
            s = np.exp(np.clip(s, -20.0, 20.0))
            return np.linalg.norm(s, axis=1).astype(np.float32), False
        return xyz[:, 2].astype(np.float32), False

    if mode == "z":
        return xyz[:, 2].astype(np.float32), False

    if mode == "rgb":
        dc_names = ["f_dc_0", "f_dc_1", "f_dc_2"]
        if all(n in names for n in dc_names):
            dc = np.stack([np.asarray(vertices[n], dtype=np.float32) for n in dc_names], axis=1)
            rgb = np.clip(dc * SH_C0 + 0.5, 0.0, 1.0).astype(np.float32)
            return rgb, True
        return np.full((xyz.shape[0], 3), 0.7, dtype=np.float32), True

    rng = np.random.default_rng(seed)
    return rng.random((xyz.shape[0],), dtype=np.float32), False


def _set_axes_equal(ax, xyz: np.ndarray) -> None:
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    ctr = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    radius = max(radius, 1e-6)
    ax.set_xlim(ctr[0] - radius, ctr[0] + radius)
    ax.set_ylim(ctr[1] - radius, ctr[1] + radius)
    ax.set_zlim(ctr[2] - radius, ctr[2] + radius)


def _fig_to_rgb_array(fig) -> np.ndarray:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return buf.reshape(h, w, 3)


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize 3DGS PLY points.")
    p.add_argument("--ply", required=True, help="Path to point_cloud.ply")
    p.add_argument("--out", default=None, help="Output image path (.png).")
    p.add_argument("--mode", choices=["opacity", "scale", "z", "rgb", "random"], default="opacity")
    p.add_argument("--sample", type=int, default=100000, help="Max points to draw.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--point-size", type=float, default=0.25)
    p.add_argument("--cmap", default="turbo")
    p.add_argument("--views", default="20,30;20,120;75,-90")
    p.add_argument("--title", default="")
    p.add_argument("--video-out", default=None, help="Optional mp4/gif output path.")
    p.add_argument("--video-frames", type=int, default=120)
    p.add_argument("--video-fps", type=int, default=24)
    p.add_argument("--video-elev", type=float, default=20.0)
    args = p.parse_args()

    ply_path = Path(args.ply)
    if not ply_path.exists():
        raise FileNotFoundError(ply_path)
    out_img = Path(args.out) if args.out else ply_path.with_name(ply_path.stem + "_vis.png")

    ply = PlyData.read(str(ply_path))
    vertices = ply["vertex"].data
    xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32)
    n_all = xyz.shape[0]
    if n_all == 0:
        raise RuntimeError("No points in PLY.")

    rng = np.random.default_rng(args.seed)
    n = min(args.sample, n_all)
    idx = rng.choice(n_all, size=n, replace=False) if n < n_all else np.arange(n_all)
    xyz = xyz[idx]

    color_vals, is_rgb = _color_values(vertices, np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1), args.mode, args.seed)
    color_vals = color_vals[idx]

    views = _parse_views(args.views)
    fig = plt.figure(figsize=(4.8 * len(views), 4.2))
    for i, (elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(1, len(views), i, projection="3d")
        if is_rgb:
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=color_vals, s=args.point_size, linewidths=0)
        else:
            c = _norm01(color_vals)
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=c, cmap=args.cmap, s=args.point_size, linewidths=0)
        _set_axes_equal(ax, xyz)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(f"e{elev:.0f}, a{azim:.0f}")

    if args.title:
        fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig(out_img, dpi=220)
    plt.close(fig)
    print(f"[saved] image: {out_img} (points={n}/{n_all}, mode={args.mode})")

    if args.video_out:
        try:
            import imageio.v2 as imageio
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("imageio is required for --video-out") from exc

        out_video = Path(args.video_out)
        fig = plt.figure(figsize=(6.2, 5.6))
        ax = fig.add_subplot(111, projection="3d")
        if is_rgb:
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=color_vals, s=args.point_size, linewidths=0)
        else:
            c = _norm01(color_vals)
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=c, cmap=args.cmap, s=args.point_size, linewidths=0)
        _set_axes_equal(ax, xyz)
        ax.set_axis_off()
        if args.title:
            ax.set_title(args.title)

        frames = max(1, int(args.video_frames))
        with imageio.get_writer(out_video, fps=int(args.video_fps)) as wr:
            for i in range(frames):
                az = 360.0 * i / frames
                ax.view_init(elev=args.video_elev, azim=az)
                wr.append_data(_fig_to_rgb_array(fig))
        plt.close(fig)
        print(f"[saved] video: {out_video} ({frames} frames @ {args.video_fps} fps)")


if __name__ == "__main__":
    main()
