"""
Prepare split dataset (train/test) into warm_start format expected by conversion tools.

Input split directory example:
  split_dir/
    actual_positions.npy   # [N,14]
    target_positions.npy   # [N,14]
    images/
      waypoint_1.png ...

Output (in-place by default):
  split_dir/
    warm_start.npy         # [N,32]
    images/waypoint_*.png
    split_meta.json

Joint mapping (14 -> q32):
  left arm 7 dims  -> q32[2:9]
  right arm 7 dims -> q32[15:22]
  others are zeros.

Example:
  ROOT=/home/SENSETIME/yanzichen/data/file/Geometry-Grounded-Gaussian-Splatting
  PY=/home/SENSETIME/yanzichen/anaconda3/envs/gggs/bin/python
  "$PY" -m world_model.prepare_split_warmstart_dataset \
    --split-dir "$ROOT/robotics_world_model/train" \
    --q-source actual
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def _natural_key(p: Path) -> tuple[int, str]:
    m = re.search(r"(\d+)", p.stem)
    if m:
        return (int(m.group(1)), p.name)
    return (10**9, p.name)


def _to_q32(q14: np.ndarray) -> np.ndarray:
    if q14.ndim != 2 or q14.shape[1] != 14:
        raise ValueError(f"Expected q14 shape [N,14], got {q14.shape}")
    out = np.zeros((q14.shape[0], 32), dtype=np.float32)
    out[:, 2:9] = q14[:, 0:7]       # left arm
    out[:, 15:22] = q14[:, 7:14]    # right arm
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare split data into warm_start.npy (q32)")
    ap.add_argument("--split-dir", required=True, help="Path to train/ or test/ directory")
    ap.add_argument("--q-source", choices=["actual", "target"], default="actual")
    ap.add_argument("--images-subdir", default="images")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    split_dir = Path(args.split_dir)
    if not split_dir.is_dir():
        raise FileNotFoundError(split_dir)

    q_path = split_dir / ("actual_positions.npy" if args.q_source == "actual" else "target_positions.npy")
    if not q_path.is_file():
        raise FileNotFoundError(q_path)
    q14 = np.load(q_path)
    q32 = _to_q32(np.asarray(q14, dtype=np.float32))

    img_dir = split_dir / args.images_subdir
    if not img_dir.is_dir():
        raise FileNotFoundError(img_dir)
    imgs = sorted(img_dir.glob("waypoint_*.png"), key=_natural_key)
    if not imgs:
        raise FileNotFoundError(f"No waypoint images in {img_dir}")

    n = min(len(imgs), q32.shape[0])
    if n <= 0:
        raise RuntimeError("No valid frames to export")
    q32 = q32[:n]

    warm_path = split_dir / "warm_start.npy"
    meta_path = split_dir / "split_meta.json"
    if not args.dry_run:
        np.save(warm_path, q32)
        meta = {
            "split_dir": str(split_dir),
            "q_source": args.q_source,
            "q14_path": str(q_path),
            "images_dir": str(img_dir),
            "frames_q": int(np.asarray(q14).shape[0]),
            "frames_images": int(len(imgs)),
            "frames_used": int(n),
            "warm_start_shape": list(q32.shape),
            "mapping": {"left": [2, 9], "right": [15, 22]},
        }
        meta_path.write_text(json.dumps(meta, indent=2))

    print(f"[DONE] split={split_dir.name} q_source={args.q_source}")
    print(f"  warm_start: {warm_path} shape={q32.shape}")
    print(f"  images: {len(imgs)} used={n}")
    print(f"  meta: {meta_path}")


if __name__ == "__main__":
    main()

