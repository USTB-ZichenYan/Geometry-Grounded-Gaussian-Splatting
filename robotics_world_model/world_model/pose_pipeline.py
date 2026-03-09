"""
Compute camera poses from robot joint vector q.

Mainline only:
  p_base -> T_head_base_to_cam (w2c) -> perspective divide -> K -> pixel

Head camera pose is a fixed 4x4 matrix in projection semantics (base -> camera, w2c).
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


_CALIB_PATH = Path(__file__).with_name("head_camera_calib.json")


def _load_head_calib() -> dict:
    if not _CALIB_PATH.exists():
        raise FileNotFoundError(f"Camera calibration file not found: {_CALIB_PATH}")
    obj = json.loads(_CALIB_PATH.read_text())
    if "head" not in obj:
        raise KeyError(f"Invalid calibration file (missing 'head'): {_CALIB_PATH}")
    return obj


def _default_T_head_base_to_cam() -> np.ndarray:
    calib = _load_head_calib()
    w2c = np.asarray(calib["head"]["w2c"], dtype=np.float32)
    if w2c.shape != (4, 4):
        raise ValueError(f"head.w2c must be 4x4 in {_CALIB_PATH}, got {w2c.shape}")
    return w2c


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

    # Fixed direct head pose in projection semantics: base -> camera (w2c).
    T_head_base_to_cam: np.ndarray = dataclasses.field(default_factory=_default_T_head_base_to_cam)

    # Hand-eye extrinsics (TCP -> camera) used by wrist cameras.
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
        link_T_world = compute_link_transforms(Path(self.cfg.urdf_path), q_map=q_map)

        if self.cfg.head_link not in link_T_world:
            raise KeyError(f"Head link '{self.cfg.head_link}' not found in URDF FK transforms.")
        if self.cfg.left_ee_link not in link_T_world:
            raise KeyError(f"Left EE link '{self.cfg.left_ee_link}' not found in URDF FK transforms.")
        if self.cfg.right_ee_link not in link_T_world:
            raise KeyError(f"Right EE link '{self.cfg.right_ee_link}' not found in URDF FK transforms.")

        T_base_left_tcp = link_T_world[self.cfg.left_ee_link]
        T_base_right_tcp = link_T_world[self.cfg.right_ee_link]

        T_head_w2c = np.asarray(self.cfg.T_head_base_to_cam, dtype=np.float32)
        T_left_w2c = (T_base_left_tcp @ self.cfg.T_left_tcp_cam).astype(np.float32)
        T_right_w2c = (T_base_right_tcp @ self.cfg.T_right_tcp_cam).astype(np.float32)

        return {
            "head": T_head_w2c,
            "left_wrist": T_left_w2c,
            "right_wrist": T_right_w2c,
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
    calib = _load_head_calib()
    h = calib["head"]
    w = calib.get("wrist", h)
    head_K = CameraIntrinsics(
        fx=float(h["fx"]),
        fy=float(h["fy"]),
        cx=float(h["cx"]),
        cy=float(h["cy"]),
        width=int(h["width"]),
        height=int(h["height"]),
    )
    wrist_K = CameraIntrinsics(
        fx=float(w["fx"]),
        fy=float(w["fy"]),
        cx=float(w["cx"]),
        cy=float(w["cy"]),
        width=int(w.get("width", h["width"])),
        height=int(w.get("height", h["height"])),
    )
    return head_K, wrist_K


def main() -> None:
    parser = argparse.ArgumentParser(description="Pose pipeline: q -> camera w2c poses")
    parser.add_argument("--q", required=True, help="Comma/space-separated q or a path to .npy/.json/.txt")
    parser.add_argument("--frame-id", default="frame_000000", help="Frame id for output json")
    parser.add_argument("--out", default="poses.json", help="Output json path")

    parser.add_argument("--left-tcp-cam", default=None, help="4x4 extrinsic (tcp->left cam) .json/.npy")
    parser.add_argument("--right-tcp-cam", default=None, help="4x4 extrinsic (tcp->right cam) .json/.npy")

    args = parser.parse_args()

    q = _load_q(args.q)
    cfg = PoseConfig()

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
