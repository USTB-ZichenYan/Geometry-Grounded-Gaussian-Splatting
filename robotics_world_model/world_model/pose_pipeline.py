"""
Compute camera poses from robot joint vector q.

Output:
  JSON list with 3 records per frame: head / left_wrist / right_wrist.
  Each record contains pose (T_base_cam) and K.

Copy-paste run:
  ROOT=/home/SENSETIME/yanzichen/data/file/Geometry-Grounded-Gaussian-Splatting
  WM_ROOT="$ROOT/robotics_world_model"
  PYTHONPATH="$WM_ROOT" python3 -m world_model.pose_pipeline \
    --q "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0" \
    --frame-id frame_000000 \
    --out "$ROOT/poses.json"

Optional:
  - --q can also be a .npy/.json/.txt file path.
  - --left-tcp-cam / --right-tcp-cam accepts 4x4 tcp->camera extrinsic.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
try:
    from .kinematics_common import (
        DEFAULT_URDF,
        HEAD_LINK,
        LEFT_EE_LINK,
        RIGHT_EE_LINK,
        build_q_map_from_q32,
        compute_link_transforms,
        load_q_input,
    )
except ImportError:  # pragma: no cover
    from kinematics_common import (  # type: ignore
        DEFAULT_URDF,
        HEAD_LINK,
        LEFT_EE_LINK,
        RIGHT_EE_LINK,
        build_q_map_from_q32,
        compute_link_transforms,
        load_q_input,
    )


def _eye4() -> np.ndarray:
    return np.eye(4, dtype=np.float32)


def _default_T_head_link_cam() -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    # Fixed conversion from URDF head_link frame to optical camera frame
    # (x right, y down, z forward; COLMAP-compatible after inversion).
    T[:3, :3] = np.array(
        [
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    return T


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def K(self) -> list[list[float]]:
        return [
            [float(self.fx), 0.0, float(self.cx)],
            [0.0, float(self.fy), float(self.cy)],
            [0.0, 0.0, 1.0],
        ]


@dataclass(frozen=True)
class PoseConfig:
    urdf_path: str = DEFAULT_URDF
    head_link: str = HEAD_LINK
    left_ee_link: str = LEFT_EE_LINK
    right_ee_link: str = RIGHT_EE_LINK

    # Head camera extrinsic (head_link -> head_camera_frame)
    T_head_link_cam: np.ndarray = dataclasses.field(default_factory=_default_T_head_link_cam)

    # Hand-eye extrinsics (TCP -> camera)
    T_left_tcp_cam: np.ndarray = dataclasses.field(default_factory=_eye4)
    T_right_tcp_cam: np.ndarray = dataclasses.field(default_factory=_eye4)


@dataclass(frozen=True)
class PoseRecord:
    frame_id: str
    cam_id: str
    pose: list[list[float]]
    K: list[list[float]]


class PosePipeline:
    def __init__(self, cfg: PoseConfig, head_K: CameraIntrinsics, wrist_K: CameraIntrinsics):
        self.cfg = cfg
        self.head_K = head_K
        self.wrist_K = wrist_K

    def compute_poses(self, q: np.ndarray) -> dict[str, np.ndarray]:
        q_map = build_q_map_from_q32(np.asarray(q, dtype=np.float32))
        link_T_world = compute_link_transforms(
            Path(self.cfg.urdf_path),
            q_map=q_map,
        )
        if self.cfg.head_link not in link_T_world:
            raise KeyError(f"Head link '{self.cfg.head_link}' not found in URDF FK transforms.")
        if self.cfg.left_ee_link not in link_T_world:
            raise KeyError(f"Left EE link '{self.cfg.left_ee_link}' not found in URDF FK transforms.")
        if self.cfg.right_ee_link not in link_T_world:
            raise KeyError(f"Right EE link '{self.cfg.right_ee_link}' not found in URDF FK transforms.")

        T_base_head_link = link_T_world[self.cfg.head_link]
        T_base_left_tcp = link_T_world[self.cfg.left_ee_link]
        T_base_right_tcp = link_T_world[self.cfg.right_ee_link]

        T_base_head = T_base_head_link @ self.cfg.T_head_link_cam
        T_base_left_cam = T_base_left_tcp @ self.cfg.T_left_tcp_cam
        T_base_right_cam = T_base_right_tcp @ self.cfg.T_right_tcp_cam

        return {
            "head": T_base_head,
            "left_wrist": T_base_left_cam,
            "right_wrist": T_base_right_cam,
        }

    def to_records(self, frame_id: str, poses: dict[str, np.ndarray]) -> list[PoseRecord]:
        return [
            PoseRecord(frame_id, "head", poses["head"].tolist(), self.head_K.K()),
            PoseRecord(frame_id, "left_wrist", poses["left_wrist"].tolist(), self.wrist_K.K()),
            PoseRecord(frame_id, "right_wrist", poses["right_wrist"].tolist(), self.wrist_K.K()),
        ]


def _load_q(arg: str) -> np.ndarray:
    return load_q_input(arg)


def _load_T(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix == ".npy":
        T = np.load(p)
    else:
        T = np.asarray(json.loads(p.read_text()), dtype=np.float32)
    if T.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix, got {T.shape}")
    return T.astype(np.float32)


def save_pose_json(records: Iterable[PoseRecord], out_path: Path) -> None:
    payload = [dataclasses.asdict(r) for r in records]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))


def build_default_intrinsics() -> tuple[CameraIntrinsics, CameraIntrinsics]:
    # Default intrinsics for 1280x720 captures used in this project.
    head_K = CameraIntrinsics(fx=1200.0, fy=1200.0, cx=640.0, cy=360.0, width=1280, height=720)
    wrist_K = CameraIntrinsics(fx=1200.0, fy=1200.0, cx=640.0, cy=360.0, width=1280, height=720)
    return head_K, wrist_K


def main() -> None:
    parser = argparse.ArgumentParser(description="Pose pipeline: q -> three camera poses")
    parser.add_argument("--q", required=True, help="Comma/space-separated q or a path to .npy/.json/.txt")
    parser.add_argument("--frame-id", default="frame_000000", help="Frame id for output json")
    parser.add_argument("--out", default="poses.json", help="Output json path")
    parser.add_argument("--head-link-cam", default=None, help="4x4 extrinsic (head_link->head cam) .json/.npy")
    parser.add_argument("--left-tcp-cam", default=None, help="4x4 extrinsic (tcp->left cam) .json/.npy")
    parser.add_argument("--right-tcp-cam", default=None, help="4x4 extrinsic (tcp->right cam) .json/.npy")

    args = parser.parse_args()

    q = _load_q(args.q)
    cfg = PoseConfig()
    if args.head_link_cam:
        cfg = dataclasses.replace(cfg, T_head_link_cam=_load_T(args.head_link_cam))
    if args.left_tcp_cam:
        cfg = dataclasses.replace(cfg, T_left_tcp_cam=_load_T(args.left_tcp_cam))
    if args.right_tcp_cam:
        cfg = dataclasses.replace(cfg, T_right_tcp_cam=_load_T(args.right_tcp_cam))

    head_K, wrist_K = build_default_intrinsics()
    pipeline = PosePipeline(cfg, head_K, wrist_K)

    poses = pipeline.compute_poses(q)
    records = pipeline.to_records(args.frame_id, poses)
    save_pose_json(records, Path(args.out))


if __name__ == "__main__":
    main()
