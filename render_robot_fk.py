"""
Render FK-driven robot Gaussians.

Usage example:
  ROOT=/home/SENSETIME/yanzichen/data/file/Geometry-Grounded-Gaussian-Splatting
  PY=/home/SENSETIME/yanzichen/anaconda3/envs/gggs/bin/python
  EXT_PYTHONPATH="$ROOT/submodules/diff-gaussian-rasterization:$ROOT/submodules/simple-knn:$ROOT/submodules/warp-patch-ncc:$ROOT/fused-ssim"
  WM_ROOT="$ROOT/robotics_world_model"
  PYTHONPATH="$EXT_PYTHONPATH:$WM_ROOT:${PYTHONPATH:-}" \
  "$PY" "$ROOT/render_robot_fk.py" \
    -s "$ROOT/gggs_run/gs_colmap_left_1f" \
    --poses-json "$ROOT/gggs_run/gs_dataset_raw_1f/poses.json" \
    --robot-model-path "$ROOT/gggs_run/robot_fk_out_left_1f_min" \
    --out-dir "$ROOT/gggs_run/robot_fk_render_left_1f_min" \
    --split all --resolution 1 --data_device cpu \
    --save-gt --save-components --robot-opacity-bias 4
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from gaussian_renderer import render
from scene import GaussianModel
from scene.dataset_readers import readColmapSceneInfo
from utils.camera_utils import cameraList_from_camInfos

try:
    from robotics_world_model.world_model.kinematics_common import (
        build_q_map_from_q32,
        compute_link_transforms,
    )
except Exception:
    from world_model.kinematics_common import (  # type: ignore
        build_q_map_from_q32,
        compute_link_transforms,
    )


def _load_q_lookup(poses_json: Path) -> Tuple[Dict[str, np.ndarray], List[str]]:
    payload = json.loads(poses_json.read_text())
    frames = payload.get("frames", [])
    q_by_name: Dict[str, np.ndarray] = {}
    ordered_names: List[str] = []
    for fr in frames:
        q = fr.get("q")
        if q is None:
            raise ValueError(f"poses.json frame {fr.get('frame_id')} missing q")
        name = f"{fr['frame_id']}_{fr['cam_id']}"
        q_by_name[name] = np.asarray(q, dtype=np.float32)
        ordered_names.append(name)
    return q_by_name, ordered_names


def _resolve_model_ply(model_or_ply: Path, iteration: int) -> Path:
    if model_or_ply.is_file() and model_or_ply.suffix.lower() == ".ply":
        return model_or_ply
    point_cloud_root = model_or_ply / "point_cloud"
    if not point_cloud_root.is_dir():
        raise FileNotFoundError(f"Expected .ply or model dir with point_cloud/: {model_or_ply}")
    if iteration >= 0:
        p = point_cloud_root / f"iteration_{iteration}" / "point_cloud.ply"
        if not p.is_file():
            raise FileNotFoundError(f"Point cloud checkpoint not found: {p}")
        return p
    cands: List[Tuple[int, Path]] = []
    for p in point_cloud_root.glob("iteration_*/point_cloud.ply"):
        m = re.match(r"iteration_(\d+)", p.parent.name)
        if m is not None:
            cands.append((int(m.group(1)), p))
    if not cands:
        raise FileNotFoundError(f"No point_cloud.ply under: {point_cloud_root}")
    cands.sort(key=lambda x: x[0])
    return cands[-1][1]


def _resolve_robot_state_path(model_or_ply: Path, explicit_path: Optional[str]) -> Path:
    if explicit_path:
        p = Path(explicit_path)
        if not p.is_file():
            raise FileNotFoundError(f"--robot-state path not found: {p}")
        return p

    roots: List[Path] = []
    if model_or_ply.is_dir():
        roots.append(model_or_ply)
    elif model_or_ply.is_file():
        roots.extend(list(model_or_ply.parents))

    for root in roots:
        p = root / "robot_fk_state_final.pth"
        if p.is_file():
            return p

    for root in roots:
        cands = sorted(root.glob("robot_fk_state_*.pth"))
        if cands:
            return cands[-1]

    raise FileNotFoundError("Cannot locate robot FK state (.pth). Pass --robot-state explicitly.")


def _load_views(source_path: str, split: str, resolution: int, data_device: str):
    if split == "all":
        scene_info = readColmapSceneInfo(source_path, images=None, eval=False)
        cam_infos = scene_info.train_cameras
    else:
        scene_info = readColmapSceneInfo(source_path, images=None, eval=True)
        cam_infos = scene_info.train_cameras if split == "train" else scene_info.test_cameras
    cam_args = SimpleNamespace(resolution=resolution, data_device=data_device)
    return cameraList_from_camInfos(cam_infos, 1.0, cam_args)


def _try_build_video(frames_dir: Path, video_path: Path, fps: int) -> bool:
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "%05d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(video_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception:
        return False


class FKDeformer:
    def __init__(
        self,
        local_xyz: torch.Tensor,
        link_ids: np.ndarray,
        link_names: List[str],
        urdf_path: Path,
    ) -> None:
        self.urdf_path = urdf_path
        self.device = local_xyz.device
        self.local_xyz = local_xyz
        self.link_names = list(link_names)

        if local_xyz.shape[0] != int(link_ids.shape[0]):
            raise ValueError(f"Point count mismatch: xyz={local_xyz.shape[0]} link_ids={link_ids.shape[0]}")

        self.idx_by_link: Dict[int, torch.Tensor] = {}
        link_ids_t = torch.tensor(link_ids.astype(np.int64).tolist(), dtype=torch.int64)
        for lid in np.unique(link_ids):
            idx = torch.where(link_ids_t == int(lid))[0]
            if idx.numel() > 0:
                self.idx_by_link[int(lid)] = idx.to(self.device)

    @torch.no_grad()
    def apply(self, q: np.ndarray, out_xyz: torch.Tensor) -> None:
        q_map = build_q_map_from_q32(np.asarray(q, dtype=np.float32))
        T_cur = compute_link_transforms(self.urdf_path, q_map=q_map)
        out_xyz.zero_()
        for lid, link_name in enumerate(self.link_names):
            idx = self.idx_by_link.get(lid)
            if idx is None:
                continue
            T = T_cur.get(link_name)
            if T is None:
                continue
            R = torch.tensor(T[:3, :3], dtype=torch.float32, device=self.device)
            t = torch.tensor(T[:3, 3], dtype=torch.float32, device=self.device)
            out_xyz[idx] = self.local_xyz[idx] @ R.T + t[None, :]


def _build_parser() -> ArgumentParser:
    p = ArgumentParser(description="Render FK robot sequence")
    p.add_argument("-s", "--source-path", required=True, help="COLMAP source dir")
    p.add_argument("--poses-json", required=True, help="poses.json with q")
    p.add_argument("--robot-model-path", required=True, help="Robot model dir or point_cloud.ply")
    p.add_argument("--robot-iteration", type=int, default=-1, help="Robot checkpoint iteration; -1=latest")
    p.add_argument("--robot-state", default=None, help="robot_fk_state_*.pth path (optional)")
    p.add_argument("--urdf", default=None, help="Override URDF path")
    p.add_argument("--out-dir", required=True, help="Output dir")
    p.add_argument("--split", choices=["all", "train", "test"], default="all")
    p.add_argument("--resolution", type=int, default=1, help="Image downscale factor")
    p.add_argument("--data_device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--kernel-size", type=float, default=0.0)
    p.add_argument("--sh-degree", type=int, default=3)
    p.add_argument("--sg-degree", type=int, default=0)
    p.add_argument("--save-gt", action="store_true")
    p.add_argument("--save-components", action="store_true")
    p.add_argument("--robot-opacity-bias", type=float, default=0.0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--make-video", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p


def main() -> None:
    args = _build_parser().parse_args(sys.argv[1:])
    torch.set_grad_enabled(False)

    q_by_name, ordered_pose_names = _load_q_lookup(Path(args.poses_json))
    views = _load_views(args.source_path, args.split, args.resolution, args.data_device)
    if not views:
        raise RuntimeError("No cameras/views loaded from source.")

    view_by_name = {v.image_name: v for v in views}
    render_names: List[str] = []
    seen = set()
    for name in ordered_pose_names:
        if name in view_by_name and name in q_by_name and name not in seen:
            render_names.append(name)
            seen.add(name)
    for name in sorted(view_by_name.keys()):
        if name in q_by_name and name not in seen:
            render_names.append(name)
            seen.add(name)
    if not render_names:
        raise RuntimeError("No overlap between source cameras and poses.json names")

    out_dir = Path(args.out_dir)
    render_dir = out_dir / "renders"
    gt_dir = out_dir / "gt"
    robot_dir = out_dir / "robot"
    alpha_dir = out_dir / "robot_alpha"
    render_dir.mkdir(parents=True, exist_ok=True)
    if args.save_gt:
        gt_dir.mkdir(parents=True, exist_ok=True)
    if args.save_components:
        robot_dir.mkdir(parents=True, exist_ok=True)
        alpha_dir.mkdir(parents=True, exist_ok=True)

    robot_ply = _resolve_model_ply(Path(args.robot_model_path), int(args.robot_iteration))
    state_path = _resolve_robot_state_path(Path(args.robot_model_path), args.robot_state)
    state = torch.load(str(state_path), map_location="cpu")

    if "link_ids" not in state or "link_names" not in state:
        raise KeyError(f"Invalid robot state file: {state_path}")
    link_ids = np.asarray(state["link_ids"], dtype=np.int64)
    link_names = [str(x) for x in np.asarray(state["link_names"]).tolist()]
    urdf = args.urdf if args.urdf else state.get("urdf")
    if urdf is None:
        raise ValueError("URDF path missing. Pass --urdf or provide it in robot state.")

    if not args.quiet:
        print(f"[robot] {robot_ply}")
        print(f"[state] {state_path}")
        print(f"[views] {len(render_names)} ({args.split})")

    robot_gaussians = GaussianModel(int(args.sh_degree), int(args.sg_degree))
    robot_gaussians.load_ply(str(robot_ply))
    if float(args.robot_opacity_bias) != 0.0:
        robot_gaussians._opacity.data.add_(float(args.robot_opacity_bias))
        if not args.quiet:
            print(f"[robot] applied opacity bias: {float(args.robot_opacity_bias):.3f}")

    # local link coordinates come from canonical npz in training;
    # after training, state["gaussians"] is saved at q_ref/world. Recover local from q_ref if available.
    q_ref = np.asarray(state.get("q_ref", q_by_name[render_names[0]]), dtype=np.float32)
    canonical_is_link_local = bool(state.get("canonical_is_link_local", False))
    xyz_now = robot_gaussians.get_xyz.detach().clone()
    if canonical_is_link_local:
        local_xyz = xyz_now
    else:
        # convert world(q_ref) -> local using q_ref transforms
        q_map_ref = build_q_map_from_q32(q_ref)
        T_ref = compute_link_transforms(Path(urdf), q_map=q_map_ref)
        local_xyz = xyz_now.clone()
        link_ids_t = torch.tensor(link_ids.astype(np.int64).tolist(), dtype=torch.int64, device=xyz_now.device)
        for lid, name in enumerate(link_names):
            idx = torch.where(link_ids_t == int(lid))[0]
            if idx.numel() == 0:
                continue
            T = T_ref.get(name)
            if T is None:
                continue
            R = torch.tensor(T[:3, :3], dtype=torch.float32, device=xyz_now.device)
            t = torch.tensor(T[:3, 3], dtype=torch.float32, device=xyz_now.device)
            local_xyz[idx] = (xyz_now[idx] - t[None, :]) @ R

    fk = FKDeformer(local_xyz=local_xyz, link_ids=link_ids, link_names=link_names, urdf_path=Path(urdf))

    bg_device = robot_gaussians.get_xyz.device
    black_bg = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=bg_device)
    pipe = Namespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)

    for idx, name in enumerate(tqdm(render_names, desc="Render Robot")):
        v = view_by_name[name]
        q = q_by_name[name]
        fk.apply(q, robot_gaussians._xyz.data)
        pkg = render(v, robot_gaussians, pipe, black_bg, args.kernel_size, require_depth=False)
        rgb = pkg["render"]
        alpha = pkg["mask"].clamp(0.0, 1.0)
        f = f"{idx:05d}.png"
        torchvision.utils.save_image(rgb, str(render_dir / f))
        if args.save_gt:
            torchvision.utils.save_image(v.original_image[0:3], str(gt_dir / f))
        if args.save_components:
            torchvision.utils.save_image(rgb, str(robot_dir / f))
            torchvision.utils.save_image(alpha, str(alpha_dir / f))

    if args.make_video:
        video_path = out_dir / "video.mp4"
        if _try_build_video(render_dir, video_path, int(args.fps)):
            print(f"[video] {video_path}")
        else:
            print("[WARN] ffmpeg failed or not found; frames are still available.")

    print(f"[DONE] renders: {render_dir}")


if __name__ == "__main__":
    main()

