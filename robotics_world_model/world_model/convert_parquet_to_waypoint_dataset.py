"""
Convert one parquet episode to waypoint-style dataset:
  - warm_start.npy (N, 32)
  - waypoint_1.png ... waypoint_N.png

This matches the lightweight format used by:
  robotics_world_model/dual_arm_grab_data_orange

Example:
  ROOT=/home/SENSETIME/yanzichen/data/file/Geometry-Grounded-Gaussian-Splatting
  PY=/home/SENSETIME/yanzichen/anaconda3/envs/gggs/bin/python

  "$PY" -m world_model.convert_parquet_to_waypoint_dataset \
    --in-parquet "$ROOT/robotics_world_model/parquet_data/episode_000000_clean.parquet" \
    --out-dir "$ROOT/robotics_world_model/dual_arm_grab_data_train" \
    --q-source state \
    --image-key observation.images.cam_high
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert parquet episode to waypoint dataset format.")
    p.add_argument("--in-parquet", required=True, help="Input parquet file path")
    p.add_argument("--out-dir", required=True, help="Output folder path")
    p.add_argument(
        "--q-source",
        choices=["state", "action"],
        default="state",
        help="Which column to export as warm_start.npy",
    )
    p.add_argument(
        "--image-key",
        default="observation.images.cam_high",
        help="Parquet struct column containing {'bytes','path'}",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="Optional frame cap; -1 means all",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    in_parquet = Path(args.in_parquet)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_parquet.is_file():
        raise FileNotFoundError(in_parquet)

    q_col = "observation.state" if args.q_source == "state" else "action"
    img_col = args.image_key

    table = pq.read_table(str(in_parquet), columns=[q_col, img_col])
    n = table.num_rows
    if args.max_frames > 0:
        n = min(n, int(args.max_frames))

    q_arr = np.asarray(table[q_col].to_pylist()[:n], dtype=np.float32)
    if q_arr.ndim != 2 or q_arr.shape[1] < 32:
        raise ValueError(f"Unexpected q shape from {q_col}: {q_arr.shape}")
    q_arr = q_arr[:, :32]
    np.save(out_dir / "warm_start.npy", q_arr)

    img_structs = table[img_col].to_pylist()[:n]
    written = 0
    for i, item in enumerate(img_structs, start=1):
        if not isinstance(item, dict) or "bytes" not in item:
            continue
        raw = item["bytes"]
        if raw is None:
            continue
        (out_dir / f"waypoint_{i}.png").write_bytes(raw)
        written += 1

    meta = {
        "source_parquet": str(in_parquet),
        "q_source": args.q_source,
        "image_key": img_col,
        "frames_total": int(table.num_rows),
        "frames_exported": int(n),
        "images_written": int(written),
        "warm_start_shape": list(q_arr.shape),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"[DONE] {in_parquet}")
    print(f"  out_dir: {out_dir}")
    print(f"  warm_start: {out_dir / 'warm_start.npy'} shape={q_arr.shape}")
    print(f"  images: {written}")


if __name__ == "__main__":
    main()

