"""
Minimal FK-driven robot training.

Design goals:
  - Keep geometry deterministic: xyz is always driven by FK(q), not learned by optimizer.
  - Use dataset mask supervision only:
      * foreground photometric loss on robot region
      * alpha leak penalty on background region
      * alpha coverage penalty on robot region
  - Avoid extra branches/regularizers to simplify debugging.
"""

from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene

try:
    from robotics_world_model.world_model.kinematics_common import (
        build_q_map_from_q32,
        compute_link_transforms,
        load_q_input,
    )
except Exception:
    from world_model.kinematics_common import (  # type: ignore
        build_q_map_from_q32,
        compute_link_transforms,
        load_q_input,
    )


def _prepare_output_dir(model_path: str, args_dict: dict) -> None:
    os.makedirs(model_path, exist_ok=True)
    with open(os.path.join(model_path, "cfg_args"), "w", encoding="utf-8") as f:
        f.write(str(args_dict))


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


def _masked_l1(pred: torch.Tensor, gt: torch.Tensor, mask3: torch.Tensor) -> torch.Tensor:
    denom = mask3.sum()
    if float(denom.item()) < 1.0:
        return torch.zeros([], dtype=pred.dtype, device=pred.device)
    return (torch.abs(pred - gt) * mask3).sum() / denom


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum()
    if float(denom.item()) < 1.0:
        return torch.zeros([], dtype=x.dtype, device=x.device)
    return (x * mask).sum() / denom


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
        self.link_names = list(link_names)
        self.local_xyz = local_xyz

        if local_xyz.shape[0] != int(link_ids.shape[0]):
            raise ValueError(
                f"Point count mismatch: xyz={local_xyz.shape[0]} link_ids={link_ids.shape[0]}"
            )

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


def main() -> None:
    parser = ArgumentParser(description="Minimal FK-driven robot training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--poses-json", required=True)
    parser.add_argument("--init-npz", required=True, help="Canonical link_local npz")
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--q-ref", default=None, help="Optional q for final save pose")

    parser.add_argument("--robot-mask-thr", type=float, default=0.5)
    parser.add_argument("--lambda-robot", type=float, default=3.0)
    parser.add_argument("--lambda-alpha-bg", type=float, default=0.05)
    parser.add_argument("--lambda-alpha-fg", type=float, default=0.05)
    parser.add_argument("--freeze-scale", type=int, default=0, help="1: freeze gaussian scale; 0: allow scale learning")
    parser.add_argument("--freeze-rotation", type=int, default=0, help="1: freeze gaussian rotation; 0: allow rotation learning")
    parser.add_argument("--lock-opacity-one", type=int, default=1, help="1: lock opacity to ~1 (no opacity learning)")
    parser.add_argument("--scale-init-mul", type=float, default=1.0, help="Multiply initial gaussian scale before training")
    parser.add_argument("--frame-order", choices=["sequential", "random"], default="sequential")
    parser.add_argument("--steps-per-frame", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    # force a minimal stable setup
    args.eval = False
    args.disable_filter3D = True
    args.densify_until_iter = 0
    args.position_lr_init = 0.0
    args.position_lr_final = 0.0
    args.scaling_lr = 0.0 if int(args.freeze_scale) == 1 else max(0.0, float(args.scaling_lr))
    args.rotation_lr = 0.0 if int(args.freeze_rotation) == 1 else max(0.0, float(args.rotation_lr))
    if int(args.lock_opacity_one) == 1:
        args.opacity_lr = 0.0

    q_by_image, ordered_pose_names = _load_q_lookup(Path(args.poses_json))

    print(f"Optimizing robot FK MIN model: {args.model_path}")
    _prepare_output_dir(args.model_path, vars(args))

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    gaussians = GaussianModel(dataset.sh_degree, dataset.sg_degree)
    scene = Scene(dataset, gaussians, shuffle=False)
    gaussians.training_setup(opt)
    gaussians.reset_3D_filter()
    if int(args.lock_opacity_one) == 1:
        # Sigmoid(logit) cannot be exactly 1.0; use a numerically stable near-opaque target.
        target_alpha = 1.0 - 1e-6
        target_logit = float(np.log(target_alpha / (1.0 - target_alpha)))
        gaussians._opacity.data.fill_(target_logit)
    if float(args.scale_init_mul) > 0.0 and abs(float(args.scale_init_mul) - 1.0) > 1e-6:
        gaussians._scaling.data.add_(float(np.log(float(args.scale_init_mul))))

    data = np.load(args.init_npz, allow_pickle=False)
    link_ids = np.asarray(data["link_ids"], dtype=np.int64)
    link_names = [str(x) for x in np.asarray(data["link_names"]).tolist()]
    local_xyz = torch.tensor(np.asarray(data["points"], dtype=np.float32).tolist(), dtype=torch.float32, device=gaussians.get_xyz.device)
    if gaussians.get_xyz.shape[0] != local_xyz.shape[0]:
        raise ValueError(
            f"xyz count ({gaussians.get_xyz.shape[0]}) != init local points ({local_xyz.shape[0]}). "
            "Rebuild colmap from matching init npz."
        )
    fk = FKDeformer(local_xyz=local_xyz, link_ids=link_ids, link_names=link_names, urdf_path=Path(args.urdf))

    cam_by_name = {cam.image_name: cam for cam in scene.getTrainCameras()}
    train_entries = []
    for name in ordered_pose_names:
        cam = cam_by_name.get(name)
        q = q_by_image.get(name)
        if cam is None or q is None:
            continue
        if cam.gt_mask is None:
            raise RuntimeError("Camera has no gt_mask; rebuild colmap with masks first.")
        train_entries.append((cam, q, name))
    if not train_entries:
        raise RuntimeError("No train entries found.")
    print(f"[data] train cameras={len(train_entries)}")

    bg_black = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=gaussians.get_xyz.device)
    gaussians.optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(range(1, int(opt.iterations) + 1), desc="Robot FK MIN training")
    steps_per_frame = max(1, int(args.steps_per_frame))

    for it in pbar:
        if args.frame_order == "random":
            idx = int(np.random.randint(0, len(train_entries)))
        else:
            idx = ((it - 1) // steps_per_frame) % len(train_entries)
        cam, q, frame_name = train_entries[idx]

        fk.apply(q, gaussians._xyz.data)
        pkg = render(cam, gaussians, pipe, bg_black, dataset.kernel_size, require_depth=False)
        pred_rgb = pkg["render"]
        pred_alpha = pkg["mask"].clamp(0.0, 1.0)

        gt = cam.original_image.to(pred_rgb.device, non_blocking=True)
        bg_keep = (cam.gt_mask > float(args.robot_mask_thr)).float().to(gt.device)
        robot_keep = (1.0 - bg_keep).clamp(0.0, 1.0)
        robot_keep3 = robot_keep.expand_as(gt)

        robot_photo = _masked_l1(pred_rgb, gt, robot_keep3)
        alpha_bg = _masked_mean(pred_alpha, bg_keep)
        alpha_fg = _masked_mean(1.0 - pred_alpha, robot_keep)

        loss = (
            float(args.lambda_robot) * robot_photo
            + float(args.lambda_alpha_bg) * alpha_bg
            + float(args.lambda_alpha_fg) * alpha_fg
        )
        loss.backward()
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)

        if it % 10 == 0:
            pbar.set_postfix(
                {
                    "frame": frame_name,
                    "loss": f"{loss.item():.4f}",
                    "robot": f"{robot_photo.item():.4f}",
                    "alpha_bg": f"{alpha_bg.item():.4f}",
                    "alpha_fg": f"{alpha_fg.item():.4f}",
                }
            )

        if it % int(args.save_every) == 0:
            save_q = q_by_image[ordered_pose_names[0]]
            if args.q_ref is not None:
                save_q = np.asarray(load_q_input(args.q_ref), dtype=np.float32)
            fk.apply(save_q, gaussians._xyz.data)
            scene.save(it)

    save_q = q_by_image[ordered_pose_names[0]]
    if args.q_ref is not None:
        save_q = np.asarray(load_q_input(args.q_ref), dtype=np.float32)
    fk.apply(save_q, gaussians._xyz.data)
    scene.save(int(opt.iterations))
    torch.save(
        {
            "gaussians": gaussians.capture(),
            "link_names": link_names,
            "link_ids": link_ids,
            "q_ref": save_q,
            "urdf": str(args.urdf),
            "canonical_is_link_local": False,
        },
        os.path.join(args.model_path, "robot_fk_state_final.pth"),
    )
    print("Done.")


if __name__ == "__main__":
    main()
