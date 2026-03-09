#!/usr/bin/env bash
set -euo pipefail

# Camera-fixed q-driven warm_start interpolation render test (batched).
#
# Goal:
#   Render arbitrary number of sampled joint states (NUM_SAMPLES) under one fixed
#   head camera from head_camera_calib.json, independent of source frame count.
#   Uses batch rendering to avoid OOM when NUM_SAMPLES is large.
#
# Usage:
#   NUM_SAMPLES=500 BATCH_SIZE=20 bash scripts/test_warmstart_interp_render.sh
#
# Optional env overrides:
#   ROOT=/path/to/Geometry-Grounded-Gaussian-Splatting
#   PY=/path/to/python
#   WM_ROOT=$ROOT/robotics_world_model
#   EXT_PYTHONPATH=...
#   WARM_NPY=$ROOT/robotics_world_model/dual_arm_grab_data_orange/warm_start.npy
#   SOURCE_COLMAP=$ROOT/gggs_run/gs_colmap_orange_left_mf
#   ROBOT_MODEL=$ROOT/gggs_run/robot_fk_out_orange_left_mf
#   OUT_DIR=$ROOT/gggs_run/robot_fk_render_warm_interp
#   NUM_SAMPLES=500
#   BATCH_SIZE=20
#   CAM_ID=head
#   HEAD_CALIB_JSON=$WM_ROOT/world_model/head_camera_calib.json

ROOT="${ROOT:-/home/SENSETIME/yanzichen/data/file/Geometry-Grounded-Gaussian-Splatting}"
PY="${PY:-/home/SENSETIME/yanzichen/anaconda3/envs/gggs/bin/python}"
WM_ROOT="${WM_ROOT:-$ROOT/robotics_world_model}"
EXT_PYTHONPATH="${EXT_PYTHONPATH:-$ROOT/submodules/diff-gaussian-rasterization:$ROOT/submodules/simple-knn:$ROOT/submodules/warp-patch-ncc:$ROOT/fused-ssim}"

WARM_NPY="${WARM_NPY:-$ROOT/robotics_world_model/dual_arm_grab_data_orange/warm_start.npy}"
SOURCE_COLMAP="${SOURCE_COLMAP:-$ROOT/gggs_run/gs_colmap_orange_left_mf}"
ROBOT_MODEL="${ROBOT_MODEL:-$ROOT/gggs_run/robot_fk_out_orange_left_mf}"
OUT_DIR="${OUT_DIR:-$ROOT/gggs_run/robot_fk_render_warm_interp}"
NUM_SAMPLES="${NUM_SAMPLES:-120}"
BATCH_SIZE="${BATCH_SIZE:-20}"
CAM_ID="${CAM_ID:-head}"
HEAD_CALIB_JSON="${HEAD_CALIB_JSON:-$WM_ROOT/world_model/head_camera_calib.json}"

WORK_DIR="$ROOT/gggs_run/warm_interp_test"
INTERP_Q_NPY="$WORK_DIR/interp_q.npy"
FIXED_CAM_JSON="$WORK_DIR/fixed_cam.json"
BATCH_ROOT="$WORK_DIR/batches"

export WARM_NPY SOURCE_COLMAP ROBOT_MODEL OUT_DIR NUM_SAMPLES BATCH_SIZE CAM_ID HEAD_CALIB_JSON \
  WORK_DIR INTERP_Q_NPY FIXED_CAM_JSON BATCH_ROOT

mkdir -p "$WORK_DIR"

echo "[cfg] WARM_NPY=$WARM_NPY"
echo "[cfg] SOURCE_COLMAP=$SOURCE_COLMAP"
echo "[cfg] ROBOT_MODEL=$ROBOT_MODEL"
echo "[cfg] OUT_DIR=$OUT_DIR"
echo "[cfg] NUM_SAMPLES=$NUM_SAMPLES BATCH_SIZE=$BATCH_SIZE CAM_ID=$CAM_ID"
echo "[cfg] HEAD_CALIB_JSON=$HEAD_CALIB_JSON"

"$PY" - <<'PY'
import json
import os
from pathlib import Path

import numpy as np

warm_npy = Path(os.environ["WARM_NPY"])
head_calib_json = Path(os.environ["HEAD_CALIB_JSON"])
interp_q_npy = Path(os.environ["INTERP_Q_NPY"])
fixed_cam_json = Path(os.environ["FIXED_CAM_JSON"])
num_samples = int(os.environ["NUM_SAMPLES"])

if not warm_npy.exists():
    raise FileNotFoundError(warm_npy)
if not head_calib_json.exists():
    raise FileNotFoundError(head_calib_json)
if num_samples <= 0:
    raise ValueError("NUM_SAMPLES must be > 0")


def pick_q_array(obj):
    if isinstance(obj, np.ndarray) and obj.ndim == 2 and obj.shape[1] >= 6:
        return obj.astype(np.float32)
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        if obj.shape == ():
            return pick_q_array(obj.item())
        if len(obj) > 0:
            return pick_q_array(obj[0])
    if isinstance(obj, dict):
        for k in ("q", "qs", "state", "states", "joint", "joints"):
            if k in obj:
                return pick_q_array(np.asarray(obj[k], dtype=np.float32))
        for v in obj.values():
            try:
                got = pick_q_array(v)
                if got is not None:
                    return got
            except Exception:
                pass
    if isinstance(obj, (list, tuple)):
        arr = np.asarray(obj, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] >= 6:
            return arr
        for v in obj:
            try:
                got = pick_q_array(v)
                if got is not None:
                    return got
            except Exception:
                pass
    return None

raw = np.load(warm_npy, allow_pickle=True)
q_seq = pick_q_array(raw)
if q_seq is None:
    raise RuntimeError(f"Cannot parse q sequence from: {warm_npy}")
if q_seq.shape[1] > 32:
    q_seq = q_seq[:, :32]
if q_seq.shape[1] < 32:
    raise ValueError(f"q dim too small: {q_seq.shape}")

T = q_seq.shape[0]
ts = np.linspace(0.0, max(T - 1, 0), num_samples, dtype=np.float32)
interp = np.zeros((num_samples, q_seq.shape[1]), dtype=np.float32)
for i, t in enumerate(ts):
    a = int(np.floor(t))
    b = min(a + 1, T - 1)
    w = float(t - a)
    interp[i] = (1.0 - w) * q_seq[a] + w * q_seq[b]

interp_q_npy.parent.mkdir(parents=True, exist_ok=True)
np.save(interp_q_npy, interp)

calib = json.loads(head_calib_json.read_text())["head"]
fixed_cam_json.write_text(
    json.dumps(
        {
            "width": int(calib["width"]),
            "height": int(calib["height"]),
            "fx": float(calib["fx"]),
            "fy": float(calib["fy"]),
            "cx": float(calib["cx"]),
            "cy": float(calib["cy"]),
            "w2c": calib["w2c"],
        },
        indent=2,
    )
)
print(f"[saved] {interp_q_npy} shape={interp.shape}")
print(f"[saved] {fixed_cam_json}")
PY

TEMPLATE_IMG="$(find "$SOURCE_COLMAP/images" -maxdepth 1 -type f -name '*.png' | sort | head -n 1 || true)"
if [[ -z "$TEMPLATE_IMG" ]]; then
  echo "[ERR] no template image found in $SOURCE_COLMAP/images"
  exit 1
fi

echo "[cfg] TEMPLATE_IMG=$TEMPLATE_IMG"

rm -rf "$BATCH_ROOT" "$OUT_DIR"
mkdir -p "$BATCH_ROOT" "$OUT_DIR/renders" "$OUT_DIR/gt" "$OUT_DIR/robot" "$OUT_DIR/robot_alpha"

TOTAL="$NUM_SAMPLES"
STEP="$BATCH_SIZE"
GLOBAL_IDX=0

for ((START=0; START<TOTAL; START+=STEP)); do
  END=$((START + STEP))
  if (( END > TOTAL )); then END=$TOTAL; fi

  BATCH_DIR="$BATCH_ROOT/b$(printf '%06d' "$START")_$(printf '%06d' "$END")"
  BATCH_SOURCE="$BATCH_DIR/source"
  BATCH_POSES="$BATCH_DIR/poses.json"
  BATCH_OUT="$BATCH_DIR/out"

  export START END TEMPLATE_IMG BATCH_SOURCE BATCH_POSES FIXED_CAM_JSON INTERP_Q_NPY CAM_ID SOURCE_COLMAP

  "$PY" - <<'PY'
import json
import os
import shutil
from pathlib import Path

import numpy as np

start = int(os.environ["START"])
end = int(os.environ["END"])
template_img = Path(os.environ["TEMPLATE_IMG"])
batch_source = Path(os.environ["BATCH_SOURCE"])
batch_poses = Path(os.environ["BATCH_POSES"])
fixed_cam_json = Path(os.environ["FIXED_CAM_JSON"])
interp_q_npy = Path(os.environ["INTERP_Q_NPY"])
cam_id = str(os.environ["CAM_ID"])
source_colmap = Path(os.environ["SOURCE_COLMAP"])

if batch_source.exists():
    shutil.rmtree(batch_source)
(batch_source / "images").mkdir(parents=True, exist_ok=True)
(batch_source / "sparse/0").mkdir(parents=True, exist_ok=True)

interp = np.load(interp_q_npy)
cam = json.loads(fixed_cam_json.read_text())
pose = np.asarray(cam["w2c"], dtype=np.float64)


def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    Kq = np.array(
        [
            [Rxx - Ryy - Rzz, 0.0, 0.0, 0.0],
            [Ryx + Rxy, Ryy - Rxx - Rzz, 0.0, 0.0],
            [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0.0],
            [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
        ],
        dtype=np.float64,
    ) / 3.0
    eigvals, eigvecs = np.linalg.eigh(Kq)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1.0
    return qvec

qvec = rotmat2qvec(pose[:3, :3])
tvec = pose[:3, 3]

images_txt_lines = [
    "# Image list with two lines of data per image:",
    "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
    "#   POINTS2D[] as (X, Y, POINT3D_ID)",
    f"# Number of images: {max(0, end-start)}",
]

frames = []
for i, gidx in enumerate(range(start, end), start=1):
    frame_id = f"interp_{gidx:06d}"
    image_name = f"{frame_id}_{cam_id}.png"
    shutil.copy2(template_img, batch_source / "images" / image_name)
    images_txt_lines.append(
        f"{i} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {tvec[0]} {tvec[1]} {tvec[2]} 1 {image_name}"
    )
    images_txt_lines.append("")
    frames.append(
        {
            "frame_id": frame_id,
            "cam_id": cam_id,
            "pose": pose.astype(float).tolist(),
            "q": interp[gidx].astype(float).tolist(),
            "image_path": f"images/{cam_id}/{image_name}",
        }
    )

batch_poses.write_text(json.dumps({"frames": frames}, indent=2))

(batch_source / "sparse/0/cameras.txt").write_text(
    "\n".join(
        [
            "# Camera list with one line of data per camera:",
            "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
            "# Number of cameras: 1",
            f"1 PINHOLE {int(cam['width'])} {int(cam['height'])} {float(cam['fx'])} {float(cam['fy'])} {float(cam['cx'])} {float(cam['cy'])}",
            "",
        ]
    )
)
(batch_source / "sparse/0/images.txt").write_text("\n".join(images_txt_lines) + "\n")

src_pts = source_colmap / "sparse/0/points3D.txt"
dst_pts = batch_source / "sparse/0/points3D.txt"
if src_pts.exists():
    shutil.copy2(src_pts, dst_pts)
else:
    dst_pts.write_text(
        "# 3D point list with one line of data per point:\n"
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n"
        "# Number of points: 1\n"
        "1 0.0 0.0 1.0 128 128 128 1.0\n"
    )

print(f"[batch] prepared {start}:{end} -> {batch_source}")
PY

  PYTHONPATH="$EXT_PYTHONPATH:$WM_ROOT:${PYTHONPATH:-}" \
  "$PY" "$ROOT/render_robot_fk.py" \
    -s "$BATCH_SOURCE" \
    --poses-json "$BATCH_POSES" \
    --robot-model-path "$ROBOT_MODEL" \
    --out-dir "$BATCH_OUT" \
    --split all \
    --resolution 1 \
    --data_device cpu \
    --kernel-size 0 \
    --robot-opacity-bias 0 \
    --save-components \
    --save-gt

  COUNT=$((END - START))
  for ((i=0; i<COUNT; i++)); do
    SRC_NAME=$(printf '%05d.png' "$i")
    DST_NAME=$(printf '%05d.png' "$GLOBAL_IDX")

    if [[ -f "$BATCH_OUT/renders/$SRC_NAME" ]]; then cp "$BATCH_OUT/renders/$SRC_NAME" "$OUT_DIR/renders/$DST_NAME"; fi
    if [[ -f "$BATCH_OUT/gt/$SRC_NAME" ]]; then cp "$BATCH_OUT/gt/$SRC_NAME" "$OUT_DIR/gt/$DST_NAME"; fi
    if [[ -f "$BATCH_OUT/robot/$SRC_NAME" ]]; then cp "$BATCH_OUT/robot/$SRC_NAME" "$OUT_DIR/robot/$DST_NAME"; fi
    if [[ -f "$BATCH_OUT/robot_alpha/$SRC_NAME" ]]; then cp "$BATCH_OUT/robot_alpha/$SRC_NAME" "$OUT_DIR/robot_alpha/$DST_NAME"; fi

    GLOBAL_IDX=$((GLOBAL_IDX + 1))
  done

  echo "[batch] done ${START}:${END}, merged_total=$GLOBAL_IDX"
done

if command -v ffmpeg >/dev/null 2>&1; then
  ffmpeg -y -framerate 30 -i "$OUT_DIR/renders/%05d.png" -c:v libx264 -pix_fmt yuv420p "$OUT_DIR/video.mp4" >/dev/null 2>&1 || true
fi

echo "[DONE]"
echo "renders: $OUT_DIR/renders"
echo "robot:   $OUT_DIR/robot"
echo "alpha:   $OUT_DIR/robot_alpha"
echo "gt:      $OUT_DIR/gt"
echo "video:   $OUT_DIR/video.mp4"
