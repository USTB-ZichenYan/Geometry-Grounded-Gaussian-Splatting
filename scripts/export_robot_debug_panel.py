#!/usr/bin/env python3
"""
Export a 2x2 debug panel for robot training diagnostics.

Panel layout:
  [0,0] Prior points (camera view)
  [0,1] Mask (white=bg keep, black=robot)
  [1,0] GT robot region only (mask-applied)
  [1,1] Render result
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def _load_rgb(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def _load_l(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    return Image.open(path).convert("L")


def _label(im: Image.Image, text: str) -> Image.Image:
    out = im.copy()
    d = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    d.rectangle((0, 0, min(out.width, 220), 18), fill=(0, 0, 0))
    d.text((4, 4), text, fill=(255, 255, 255), font=font)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Export 2x2 robot debug panel")
    p.add_argument("--orig", required=True, help="Original GT image path")
    p.add_argument("--prior", required=True, help="Prior points projection image path")
    p.add_argument("--mask", required=True, help="Mask path (white=bg, black=robot)")
    p.add_argument("--render", required=True, help="Render image path")
    p.add_argument("--out", required=True, help="Output panel image path")
    args = p.parse_args()

    orig = _load_rgb(Path(args.orig))
    prior = _load_rgb(Path(args.prior))
    mask = _load_l(Path(args.mask))
    render = _load_rgb(Path(args.render))

    # robot mask: invert keep-mask (white bg -> black; black robot -> white)
    robot_m = ImageOps.invert(mask)
    black = Image.new("RGB", orig.size, (0, 0, 0))
    gt_robot = Image.composite(orig, black, robot_m)

    mask_rgb = Image.merge("RGB", (mask, mask, mask))

    a = _label(prior, "Prior Points")
    b = _label(mask_rgb, "Mask (white=bg)")
    c = _label(gt_robot, "GT Robot Region")
    d = _label(render, "Render")

    w, h = orig.size
    panel = Image.new("RGB", (w * 2, h * 2), (0, 0, 0))
    panel.paste(a, (0, 0))
    panel.paste(b, (w, 0))
    panel.paste(c, (0, h))
    panel.paste(d, (w, h))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
