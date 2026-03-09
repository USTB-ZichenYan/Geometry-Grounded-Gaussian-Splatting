"""
Create left-arm-only masks from green-screen images.

Output convention:
  - White (255): left arm foreground
  - Black (0): background / non-left-arm region
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

from PIL import Image


def _largest_component(binary, w_lim: int, h: int):
    visited = [[False] * w_lim for _ in range(h)]
    best = []
    dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for y in range(h):
        for x in range(w_lim):
            if binary[y][x] == 0 or visited[y][x]:
                continue
            q = deque([(x, y)])
            visited[y][x] = True
            comp = []
            while q:
                cx, cy = q.popleft()
                comp.append((cx, cy))
                for dx, dy in dirs:
                    nx, ny = cx + dx, cy + dy
                    if nx < 0 or nx >= w_lim or ny < 0 or ny >= h:
                        continue
                    if visited[ny][nx] or binary[ny][nx] == 0:
                        continue
                    visited[ny][nx] = True
                    q.append((nx, ny))
            if len(comp) > len(best):
                best = comp
    return best


def _build_left_mask(
    img_path: Path,
    green_g_min: int,
    green_margin: int,
    left_ratio: float,
    largest_component: bool,
) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    split_x = int(max(1, min(w, round(w * float(left_ratio)))))
    px = list(img.getdata())

    # 1) Green-screen removal + left-half filter
    fg = [[0] * split_x for _ in range(h)]
    for y in range(h):
        off = y * w
        for x in range(split_x):
            r, g, b = px[off + x]
            is_green = (g >= green_g_min) and ((g - r) >= green_margin) and ((g - b) >= green_margin)
            fg[y][x] = 0 if is_green else 1

    # 2) Keep only largest connected component to avoid torso/noise bleeding
    keep = _largest_component(fg, split_x, h) if largest_component else [
        (x, y) for y in range(h) for x in range(split_x) if fg[y][x] == 1
    ]

    out = Image.new("L", (w, h), 0)
    out_px = out.load()
    for x, y in keep:
        out_px[x, y] = 255
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Create left-arm-only masks from green-screen images")
    p.add_argument("--dataset-dir", required=True, help="GS dataset dir (contains images/<cam-id>)")
    p.add_argument("--cam-id", default="head", help="Camera id under images/")
    p.add_argument("--green-g-min", type=int, default=100, help="Green threshold: minimum G")
    p.add_argument("--green-margin", type=int, default=30, help="Green threshold: require G-R and G-B >= margin")
    p.add_argument("--left-ratio", type=float, default=0.5, help="Keep x in [0, left_ratio*W)")
    p.add_argument("--no-largest-component", action="store_true", help="Disable largest-component filtering")
    args = p.parse_args()

    ds = Path(args.dataset_dir)
    img_dir = ds / "images" / args.cam_id
    out_dir = ds / "robot_masks" / args.cam_id
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(img_dir.glob("frame_*.png"))
    if not imgs:
        raise FileNotFoundError(f"No frames found under: {img_dir}")

    n = 0
    for pth in imgs:
        out = _build_left_mask(
            pth,
            green_g_min=int(args.green_g_min),
            green_margin=int(args.green_margin),
            left_ratio=float(args.left_ratio),
            largest_component=not args.no_largest_component,
        )
        out.save(out_dir / pth.name)
        n += 1

    print(f"[left-mask] saved {n} masks -> {out_dir}")
    print(
        f"[left-mask] params: g_min={args.green_g_min}, margin={args.green_margin}, "
        f"left_ratio={args.left_ratio}, largest_component={not args.no_largest_component}"
    )


if __name__ == "__main__":
    main()

