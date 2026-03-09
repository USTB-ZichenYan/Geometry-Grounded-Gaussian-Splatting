"""
Recolor robot foreground in green-screen PNG images.

Foreground rule:
  pixel is foreground (robot) if NOT green-screen.
Green-screen test:
  g >= green_g_min and (g-r)>=green_margin and (g-b)>=green_margin

Input:
  - directory with waypoint_*.png
Output:
  - recolored PNGs to out-dir (same filenames)
  - non-image files are copied as-is (e.g., warm_start.npy)

Example:
  ROOT=/home/SENSETIME/yanzichen/data/file/Geometry-Grounded-Gaussian-Splatting
  PY=/home/SENSETIME/yanzichen/anaconda3/envs/gggs/bin/python
  "$PY" "$ROOT/robotics_world_model/world_model/recolor_greenscreen_robot.py" \
    --in-dir "$ROOT/robotics_world_model/dual_arm_grab_data" \
    --out-dir "$ROOT/robotics_world_model/dual_arm_grab_data_orange" \
    --robot-rgb 220,120,40
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def _parse_rgb(s: str) -> tuple[int, int, int]:
    vals = [int(x.strip()) for x in s.split(",")]
    if len(vals) != 3:
        raise ValueError(f"--robot-rgb expects 'R,G,B', got: {s}")
    for v in vals:
        if v < 0 or v > 255:
            raise ValueError(f"RGB value out of range [0,255]: {v}")
    return vals[0], vals[1], vals[2]


def _recolor_one(
    src: Path,
    dst: Path,
    robot_rgb: tuple[int, int, int],
    green_g_min: int,
    green_margin: int,
) -> tuple[int, int]:
    im = np.array(Image.open(src).convert("RGB"), dtype=np.uint8)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    is_green = (g >= green_g_min) & ((g.astype(np.int16) - r.astype(np.int16)) >= green_margin) & (
        (g.astype(np.int16) - b.astype(np.int16)) >= green_margin
    )
    fg = ~is_green
    out = im.copy()
    out[fg] = np.array(robot_rgb, dtype=np.uint8)
    Image.fromarray(out, mode="RGB").save(dst)
    return int(fg.sum()), int(fg.size)


def main() -> None:
    ap = argparse.ArgumentParser(description="Recolor robot foreground in green-screen images")
    ap.add_argument("--in-dir", required=True, help="Input directory (contains waypoint_*.png)")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--robot-rgb", default="220,120,40", help="Target robot color, e.g. 220,120,40")
    ap.add_argument("--green-g-min", type=int, default=100)
    ap.add_argument("--green-margin", type=int, default=30)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    robot_rgb = _parse_rgb(args.robot_rgb)

    pngs = sorted(in_dir.glob("waypoint_*.png"))
    if not pngs:
        raise FileNotFoundError(f"No waypoint_*.png found under: {in_dir}")

    fg_total = 0
    pix_total = 0
    for p in pngs:
        fg, tot = _recolor_one(
            p,
            out_dir / p.name,
            robot_rgb=robot_rgb,
            green_g_min=int(args.green_g_min),
            green_margin=int(args.green_margin),
        )
        fg_total += fg
        pix_total += tot

    # copy sidecar files
    for f in in_dir.iterdir():
        if f.suffix.lower() == ".png":
            continue
        if f.is_file():
            shutil.copy2(f, out_dir / f.name)

    ratio = float(fg_total) / float(max(pix_total, 1))
    print(f"[recolor] images={len(pngs)} out={out_dir}")
    print(f"[recolor] robot_rgb={robot_rgb}, fg_ratio={ratio:.4f}")


if __name__ == "__main__":
    main()

