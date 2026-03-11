#!/usr/bin/env bash
set -euo pipefail

# Adapt robotics_world_model/{train,test} and run train/test rendering.
#
# Input expected:
#   robotics_world_model/train/
#     actual_positions.npy or target_positions.npy (N,14)
#     images/waypoint_*.png
#   robotics_world_model/test/
#     actual_positions.npy or target_positions.npy (N,14)
#     images/waypoint_*.png
#
# Notes:
# - q14 will be mapped to q32:
#     left arm  -> q32[2:9]
#     right arm -> q32[15:22]
# - This script trains on train split and renders both train/test splits.
#
# Example:
#   bash scripts/run_train_test_split_robot_fk.sh

ROOT="${ROOT:-/home/SENSETIME/yanzichen/data/file/Geometry-Grounded-Gaussian-Splatting}"
PY="${PY:-/home/SENSETIME/yanzichen/anaconda3/envs/gggs/bin/python}"
WM_ROOT="${WM_ROOT:-$ROOT/robotics_world_model}"
EXT_PYTHONPATH="${EXT_PYTHONPATH:-$ROOT/submodules/diff-gaussian-rasterization:$ROOT/submodules/simple-knn:$ROOT/submodules/warp-patch-ncc:$ROOT/fused-ssim}"

URDF="${URDF:-$WM_ROOT/tianyi2_urdf-tianyi2.0/urdf/tianyi2.0_urdf_with_hands.urdf}"
PKG_ROOT="${PKG_ROOT:-$WM_ROOT/tianyi2_urdf-tianyi2.0}"

Q_SOURCE="${Q_SOURCE:-target}" # actual|target
INIT_NPZ="${INIT_NPZ:-$ROOT/gggs_run/gs_init_left_orange_200k_plus_base.npz}"
INIT_MAX_POINTS="${INIT_MAX_POINTS:-200000}"
TRAIN_ITERS="${TRAIN_ITERS:-10000}"

DATA_TRAIN="$WM_ROOT/train"
DATA_TEST="$WM_ROOT/test"
DS_TRAIN="$ROOT/gggs_run/gs_dataset_train"
DS_TEST="$ROOT/gggs_run/gs_dataset_test"
CM_TRAIN="$ROOT/gggs_run/gs_colmap_train"
CM_TEST="$ROOT/gggs_run/gs_colmap_test"
MODEL_OUT="$ROOT/gggs_run/robot_fk_out_train_v1"
RENDER_TRAIN="$ROOT/gggs_run/robot_fk_render_train_v1"
RENDER_TEST="$ROOT/gggs_run/robot_fk_render_test_v1"

echo "[cfg] ROOT=$ROOT"
echo "[cfg] Q_SOURCE=$Q_SOURCE INIT_NPZ=$INIT_NPZ INIT_MAX_POINTS=$INIT_MAX_POINTS"
echo "[cfg] TRAIN_ITERS=$TRAIN_ITERS"

echo "[1/6] prepare split warm_start.npy (q14 -> q32)"
PYTHONPATH="$WM_ROOT" "$PY" -m world_model.prepare_split_warmstart_dataset --split-dir "$DATA_TRAIN" --q-source "$Q_SOURCE"
PYTHONPATH="$WM_ROOT" "$PY" -m world_model.prepare_split_warmstart_dataset --split-dir "$DATA_TEST" --q-source "$Q_SOURCE"

echo "[2/6] convert warm_start/images -> gs_dataset train/test"
rm -rf "$DS_TRAIN" "$DS_TEST"
PYTHONPATH="$WM_ROOT" "$PY" -m world_model.convert_warmstart_images_to_gs_dataset \
  --in-dir "$DATA_TRAIN" --out-dir "$DS_TRAIN" \
  --warm-start warm_start.npy --image-glob "images/waypoint_*.png"
PYTHONPATH="$WM_ROOT" "$PY" -m world_model.convert_warmstart_images_to_gs_dataset \
  --in-dir "$DATA_TEST" --out-dir "$DS_TEST" \
  --warm-start warm_start.npy --image-glob "images/waypoint_*.png"

echo "[3/6] gs_dataset -> colmap train/test (with FK masks)"
rm -rf "$CM_TRAIN" "$CM_TEST"
PYTHONPATH="$WM_ROOT" "$PY" -m world_model.gs_dataset_to_colmap \
  --in-dir "$DS_TRAIN" --out-dir "$CM_TRAIN" \
  --init-npz "$INIT_NPZ" --init-max-points "$INIT_MAX_POINTS" \
  --bg-num-points 0 --pose-convention w2c \
  --robot-mask-urdf "$URDF" --robot-mask-package-root "$PKG_ROOT" \
  --robot-mask-points 30000 --robot-mask-radius-px 2 \
  --robot-mask-include-regex "(left|_l_|_l_link$|L_base_link)"
PYTHONPATH="$WM_ROOT" "$PY" -m world_model.gs_dataset_to_colmap \
  --in-dir "$DS_TEST" --out-dir "$CM_TEST" \
  --init-npz "$INIT_NPZ" --init-max-points "$INIT_MAX_POINTS" \
  --bg-num-points 0 --pose-convention w2c \
  --robot-mask-urdf "$URDF" --robot-mask-package-root "$PKG_ROOT" \
  --robot-mask-points 30000 --robot-mask-radius-px 2 \
  --robot-mask-include-regex "(left|_l_|_l_link$|L_base_link)"

echo "[4/6] train on train split"
rm -rf "$MODEL_OUT"
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
PYTHONPATH="$EXT_PYTHONPATH:$WM_ROOT:${PYTHONPATH:-}" \
"$PY" "$ROOT/train_robot_fk_min.py" \
  -s "$CM_TRAIN" -m "$MODEL_OUT" \
  --poses-json "$DS_TRAIN/poses.json" \
  --init-npz "$INIT_NPZ" \
  --urdf "$URDF" \
  --iterations "$TRAIN_ITERS" \
  --resolution 1 --data_device cpu \
  --frame-order sequential --steps-per-frame 1 \
  --freeze-scale 1 --freeze-rotation 0 \
  --stage-a-iters 1000 \
  --preflight-check 1 --preflight-strict 1 \
  --save-every 1000

echo "[5/6] render on train split"
rm -rf "$RENDER_TRAIN"
PYTHONPATH="$EXT_PYTHONPATH:$WM_ROOT:${PYTHONPATH:-}" \
"$PY" "$ROOT/render_robot_fk.py" \
  -s "$CM_TRAIN" \
  --poses-json "$DS_TRAIN/poses.json" \
  --robot-model-path "$MODEL_OUT" \
  --out-dir "$RENDER_TRAIN" \
  --split all --resolution 1 --data_device cpu \
  --kernel-size 0 --robot-opacity-bias 0 \
  --save-components --save-gt

echo "[6/6] render on test split"
rm -rf "$RENDER_TEST"
PYTHONPATH="$EXT_PYTHONPATH:$WM_ROOT:${PYTHONPATH:-}" \
"$PY" "$ROOT/render_robot_fk.py" \
  -s "$CM_TEST" \
  --poses-json "$DS_TEST/poses.json" \
  --robot-model-path "$MODEL_OUT" \
  --out-dir "$RENDER_TEST" \
  --split all --resolution 1 --data_device cpu \
  --kernel-size 0 --robot-opacity-bias 0 \
  --save-components --save-gt

echo "[extra] export train/test 2x2 debug panels (prior/mask/gt/render)"
DEBUG_DIR="$ROOT/gggs_run/debug_split_v1"
mkdir -p "$DEBUG_DIR"

PYTHONPATH="$EXT_PYTHONPATH:$WM_ROOT:${PYTHONPATH:-}" \
"$PY" "$ROOT/scripts/vis_prior_points_frame.py" \
  --init-npz "$INIT_NPZ" \
  --poses-json "$DS_TRAIN/poses.json" \
  --cameras-json "$DS_TRAIN/cameras.json" \
  --urdf "$URDF" \
  --frame-id frame_000000 \
  --cam-id head \
  --out "$DEBUG_DIR/prior_train_00000.png" \
  --point-radius 1 \
  --sample-points 20000

"$PY" "$ROOT/scripts/export_robot_debug_panel.py" \
  --orig "$CM_TRAIN/images/frame_000000_head.png" \
  --prior "$DEBUG_DIR/prior_train_00000.png" \
  --mask "$CM_TRAIN/masks/frame_000000_head.png" \
  --render "$RENDER_TRAIN/renders/00000.png" \
  --out "$DEBUG_DIR/panel_train_00000.png"

PYTHONPATH="$EXT_PYTHONPATH:$WM_ROOT:${PYTHONPATH:-}" \
"$PY" "$ROOT/scripts/vis_prior_points_frame.py" \
  --init-npz "$INIT_NPZ" \
  --poses-json "$DS_TEST/poses.json" \
  --cameras-json "$DS_TEST/cameras.json" \
  --urdf "$URDF" \
  --frame-id frame_000000 \
  --cam-id head \
  --out "$DEBUG_DIR/prior_test_00000.png" \
  --point-radius 1 \
  --sample-points 20000

"$PY" "$ROOT/scripts/export_robot_debug_panel.py" \
  --orig "$CM_TEST/images/frame_000000_head.png" \
  --prior "$DEBUG_DIR/prior_test_00000.png" \
  --mask "$CM_TEST/masks/frame_000000_head.png" \
  --render "$RENDER_TEST/renders/00000.png" \
  --out "$DEBUG_DIR/panel_test_00000.png"

echo "[DONE]"
echo "model:        $MODEL_OUT"
echo "train render: $RENDER_TRAIN"
echo "test  render: $RENDER_TEST"
echo "debug panel:  $DEBUG_DIR/panel_train_00000.png"
echo "debug panel:  $DEBUG_DIR/panel_test_00000.png"
