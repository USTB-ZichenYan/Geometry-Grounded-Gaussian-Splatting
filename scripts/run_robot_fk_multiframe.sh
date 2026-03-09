#!/usr/bin/env bash
set -euo pipefail

# Multi-frame FK training/render pipeline (left arm).
#
# Usage:
#   bash scripts/run_robot_fk_multiframe.sh
#
# Optional env overrides:
#   ROOT=/path/to/Geometry-Grounded-Gaussian-Splatting
#   PY=/path/to/python
#   WM_ROOT=$ROOT/robotics_world_model
#   DATA_ROOT=$WM_ROOT/dual_arm_grab_data
#   DATASET_DIR=$ROOT/gggs_run/gs_dataset_orange_mf
#   COLMAP_DIR=$ROOT/gggs_run/gs_colmap_orange_left_mf
#   INIT_NPZ=$ROOT/gggs_run/gs_init_left_orange_200k.npz
#   MODEL_DIR=$ROOT/gggs_run/robot_fk_out_orange_left_mf
#   RENDER_DIR=$ROOT/gggs_run/robot_fk_render_orange_left_mf
#   STRIDE=1
#   MAX_FRAMES=120
#   ITERATIONS=12000

ROOT="${ROOT:-/home/SENSETIME/yanzichen/data/file/Geometry-Grounded-Gaussian-Splatting}"
PY="${PY:-/home/SENSETIME/yanzichen/anaconda3/envs/gggs/bin/python}"
WM_ROOT="${WM_ROOT:-$ROOT/robotics_world_model}"
DATA_ROOT="${DATA_ROOT:-$WM_ROOT/dual_arm_grab_data_orange}"
URDF="${URDF:-$WM_ROOT/tianyi2_urdf-tianyi2.0/urdf/tianyi2.0_urdf_with_hands.urdf}"
EXT_PYTHONPATH="${EXT_PYTHONPATH:-$ROOT/submodules/diff-gaussian-rasterization:$ROOT/submodules/simple-knn:$ROOT/submodules/warp-patch-ncc:$ROOT/fused-ssim}"

DATASET_DIR="${DATASET_DIR:-$ROOT/gggs_run/gs_dataset_orange_mf}"
COLMAP_DIR="${COLMAP_DIR:-$ROOT/gggs_run/gs_colmap_orange_left_mf}"
INIT_NPZ="${INIT_NPZ:-$ROOT/gggs_run/gs_init_left_orange_200k.npz}"
MODEL_DIR="${MODEL_DIR:-$ROOT/gggs_run/robot_fk_out_orange_left_mf}"
RENDER_DIR="${RENDER_DIR:-$ROOT/gggs_run/robot_fk_render_orange_left_mf}"

STRIDE="${STRIDE:-1}"
MAX_FRAMES="${MAX_FRAMES:-120}"
ITERATIONS="${ITERATIONS:-12000}"

echo "[cfg] ROOT=$ROOT"
echo "[cfg] DATA_ROOT=$DATA_ROOT"
echo "[cfg] DATASET_DIR=$DATASET_DIR"
echo "[cfg] COLMAP_DIR=$COLMAP_DIR"
echo "[cfg] INIT_NPZ=$INIT_NPZ"
echo "[cfg] MODEL_DIR=$MODEL_DIR"
echo "[cfg] RENDER_DIR=$RENDER_DIR"
echo "[cfg] STRIDE=$STRIDE MAX_FRAMES=$MAX_FRAMES ITERATIONS=$ITERATIONS"

cd "$ROOT"

if [[ ! -f "$INIT_NPZ" ]]; then
  echo "[1/6] INIT_NPZ not found, generate left-arm 200k prior"
  "$PY" "$WM_ROOT/world_model/gs_init_from_urdf.py" \
    --urdf "$URDF" \
    --out "$INIT_NPZ" \
    --total-points 200000 \
    --include-regex "(left|_l_|_l_link$)" \
    --output-frame link_local \
    --seed 0
else
  echo "[1/6] Reuse INIT_NPZ: $INIT_NPZ"
fi

echo "[2/6] Export multi-frame dataset (head-only)"
rm -rf "$DATASET_DIR"
PYTHONPATH="$WM_ROOT" "$PY" -m world_model.gs_dataset \
  --data-root "$DATA_ROOT" \
  --out-dir "$DATASET_DIR" \
  --head-only \
  --q-source state \
  --stride "$STRIDE" \
  --max-frames "$MAX_FRAMES" \
  --black-mean-thr 1.0 \
  --black-var-thr 1.0

echo "[3/6] Build left-arm mask from green screen"
PYTHONPATH="$WM_ROOT" "$PY" -m world_model.create_left_arm_mask \
  --dataset-dir "$DATASET_DIR" \
  --cam-id head

echo "[4/6] Convert gs_dataset -> COLMAP (w2c, robot-only init)"
rm -rf "$COLMAP_DIR"
PYTHONPATH="$WM_ROOT" "$PY" -m world_model.gs_dataset_to_colmap \
  --in-dir "$DATASET_DIR" \
  --out-dir "$COLMAP_DIR" \
  --init-npz "$INIT_NPZ" \
  --init-max-points 200000 \
  --bg-num-points 0 \
  --pose-convention w2c

echo "[5/6] Train multi-frame FK model (sequential frames)"
rm -rf "$MODEL_DIR"
PYTHONPATH="$EXT_PYTHONPATH:$WM_ROOT:${PYTHONPATH:-}" \
"$PY" "$ROOT/train_robot_fk_min.py" \
  -s "$COLMAP_DIR" \
  -m "$MODEL_DIR" \
  --poses-json "$DATASET_DIR/poses.json" \
  --init-npz "$INIT_NPZ" \
  --urdf "$URDF" \
  --iterations "$ITERATIONS" \
  --frame-order sequential \
  --steps-per-frame 1 \
  --lambda-robot 5.0 \
  --lambda-alpha-bg 0.5 \
  --lambda-alpha-fg 0.25 \
  --freeze-scale 1 \
  --freeze-rotation 0 \
  --resolution 1 \
  --data_device cpu

echo "[6/6] Render multi-frame result"
rm -rf "$RENDER_DIR"
PYTHONPATH="$EXT_PYTHONPATH:$WM_ROOT:${PYTHONPATH:-}" \
"$PY" "$ROOT/render_robot_fk.py" \
  -s "$COLMAP_DIR" \
  --poses-json "$DATASET_DIR/poses.json" \
  --robot-model-path "$MODEL_DIR" \
  --out-dir "$RENDER_DIR" \
  --split all \
  --resolution 1 \
  --data_device cpu \
  --kernel-size 0 \
  --robot-opacity-bias 0 \
  --save-components \
  --save-gt

echo "[DONE]"
echo "model : $MODEL_DIR"
echo "render: $RENDER_DIR/renders"
echo "alpha : $RENDER_DIR/robot_alpha"
