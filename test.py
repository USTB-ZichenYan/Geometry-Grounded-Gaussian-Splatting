#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _fmt(v: float) -> str:
    return f"{v:.6f}"


def _load_meta(npz_path: Path, meta_path_arg: str | None) -> tuple[dict, Path | None]:
    if meta_path_arg is not None:
        meta_path = Path(meta_path_arg)
        if not meta_path.exists():
            raise FileNotFoundError(f"meta json not found: {meta_path}")
        return json.loads(meta_path.read_text()), meta_path

    auto_meta = npz_path.with_suffix(".json")
    if auto_meta.exists():
        return json.loads(auto_meta.read_text()), auto_meta
    return {}, None


def _extract_link_names(raw: np.ndarray) -> list[str]:
    # npz string arrays may come in unicode/object dtypes depending on writer.
    return [str(x) for x in raw.tolist()]


def _print_npz_contents(blob: np.lib.npyio.NpzFile, *, sample_rows: int = 3, sample_items_1d: int = 10) -> None:
    print("npz contents:")
    print(f"  keys: {list(blob.files)}")
    for key in blob.files:
        arr = np.asarray(blob[key])
        print(f"  [{key}]")
        print(f"    shape: {arr.shape}")
        print(f"    dtype: {arr.dtype}")
        if arr.size == 0:
            print("    sample: <empty>")
            continue
        if arr.ndim == 1:
            n = min(sample_items_1d, arr.shape[0])
            print(f"    sample[:{n}]: {arr[:n]}")
        else:
            n = min(sample_rows, arr.shape[0])
            print(f"    sample[:{n}]:")
            print(arr[:n])


def main() -> None:
    parser = argparse.ArgumentParser(description="Check 3DGS init point statistics and z-mean range.")
    parser.add_argument("--npz", default="./gs_init.npz", help="Path to gs_init.npz")
    parser.add_argument("--meta", default=None, help="Optional path to meta json (default: same stem as npz)")
    parser.add_argument("--z-min", type=float, default=1.0, help="Lower bound of acceptable z mean (meters)")
    parser.add_argument("--z-max", type=float, default=2.0, help="Upper bound of acceptable z mean (meters)")
    parser.add_argument("--top-links", type=int, default=10, help="Show top-K links by point count")
    args = parser.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(f"npz not found: {npz_path}")
    if args.z_min > args.z_max:
        raise ValueError(f"invalid z range: z_min ({args.z_min}) > z_max ({args.z_max})")

    blob = np.load(npz_path, allow_pickle=True)
    if "points" not in blob:
        raise KeyError(f"'points' not found in {npz_path}")
    points = np.asarray(blob["points"], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"expected points shape (N,3), got {points.shape}")
    if points.shape[0] == 0:
        raise ValueError("points is empty")

    meta, meta_path = _load_meta(npz_path, args.meta)
    output_frame = str(meta.get("output_frame", "unknown"))

    _print_npz_contents(blob)
    print()

    mean_xyz = points.mean(axis=0)
    std_xyz = points.std(axis=0)
    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)
    p01_xyz = np.percentile(points, 1.0, axis=0)
    p50_xyz = np.percentile(points, 50.0, axis=0)
    p99_xyz = np.percentile(points, 99.0, axis=0)

    print(f"npz: {npz_path}")
    print(f"meta: {meta_path if meta_path is not None else '(not found)'}")
    print(f"frame: {output_frame}")
    print(f"num_points: {points.shape[0]}")
    print()
    print("xyz stats (meters):")
    print(f"  mean: [{_fmt(mean_xyz[0])}, {_fmt(mean_xyz[1])}, {_fmt(mean_xyz[2])}]")
    print(f"  std:  [{_fmt(std_xyz[0])}, {_fmt(std_xyz[1])}, {_fmt(std_xyz[2])}]")
    print(f"  min:  [{_fmt(min_xyz[0])}, {_fmt(min_xyz[1])}, {_fmt(min_xyz[2])}]")
    print(f"  p01:  [{_fmt(p01_xyz[0])}, {_fmt(p01_xyz[1])}, {_fmt(p01_xyz[2])}]")
    print(f"  p50:  [{_fmt(p50_xyz[0])}, {_fmt(p50_xyz[1])}, {_fmt(p50_xyz[2])}]")
    print(f"  p99:  [{_fmt(p99_xyz[0])}, {_fmt(p99_xyz[1])}, {_fmt(p99_xyz[2])}]")
    print(f"  max:  [{_fmt(max_xyz[0])}, {_fmt(max_xyz[1])}, {_fmt(max_xyz[2])}]")

    if "link_ids" in blob and "link_names" in blob:
        link_ids = np.asarray(blob["link_ids"], dtype=np.int64)
        link_names = _extract_link_names(np.asarray(blob["link_names"]))
        print()
        print(f"per-link centroid (top {args.top_links} by count):")
        unique_ids, counts = np.unique(link_ids, return_counts=True)
        order = np.argsort(-counts)
        shown = 0
        for idx in order:
            lid = int(unique_ids[idx])
            count = int(counts[idx])
            if lid < 0 or lid >= len(link_names):
                name = f"<invalid:{lid}>"
            else:
                name = link_names[lid]
            mask = link_ids == lid
            centroid = points[mask].mean(axis=0)
            print(
                f"  {name:<24s} count={count:6d} "
                f"mean=[{_fmt(centroid[0])}, {_fmt(centroid[1])}, {_fmt(centroid[2])}]"
            )
            shown += 1
            if shown >= args.top_links:
                break

    z_mean = float(mean_xyz[2])
    ok = args.z_min <= z_mean <= args.z_max
    print()
    print(f"z-mean check: {_fmt(z_mean)} in [{args.z_min:.3f}, {args.z_max:.3f}] -> {'PASS' if ok else 'FAIL'}")
    if output_frame != "base":
        print("warning: output_frame is not base; z-range expectation may not apply.")

    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
