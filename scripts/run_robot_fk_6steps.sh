#!/usr/bin/env bash
set -euo pipefail

# One-click 6-step pipeline for left-arm FK reconstruction (orange dataset).
#
# Usage:
#   bash scripts/run_robot_fk_6steps.sh
#
# Optional env overrides:
#   ROOT=/path/to/Geometry-Grounded-Gaussian-Splatting
#   PY=/path/to/python
#   WM_ROOT=$ROOT/robotics_world_model
#   URDF=$WM_ROOT/tianyi2_urdf-tianyi2.0/urdf/tianyi2.0_urdf_with_hands.urdf

ROOT="${ROOT:-/home/SENSETIME/yanzichen/data/file/Geometry-Grounded-Gaussian-Splatting}"
PY="${PY:-/home/SENSETIME/yanzichen/anaconda3/envs/gggs/bin/python}"
WM_ROOT="${WM_ROOT:-$ROOT/robotics_world_model}"
URDF="${URDF:-$WM_ROOT/tianyi2_urdf-tianyi2.0/urdf/tianyi2.0_urdf_with_hands.urdf}"
EXT_PYTHONPATH="${EXT_PYTHONPATH:-$ROOT/submodules/diff-gaussian-rasterization:$ROOT/submodules/simple-knn:$ROOT/submodules/warp-patch-ncc:$ROOT/fused-ssim}"

echo "[cfg] ROOT=$ROOT"
echo "[cfg] PY=$PY"
echo "[cfg] WM_ROOT=$WM_ROOT"
echo "[cfg] URDF=$URDF"

cd "$ROOT"

echo "[1/6] Generate left-arm mask"
PYTHONPATH="$WM_ROOT" "$PY" -m world_model.create_left_arm_mask \
  --dataset-dir "$ROOT/gggs_run/gs_dataset_orange_1f" \
  --cam-id head

echo "[2/6] URDF -> Gaussian prior (200k)"
"$PY" "$WM_ROOT/world_model/gs_init_from_urdf.py" \
  --urdf "$URDF" \
  --out "$ROOT/gggs_run/gs_init_left_orange_200k.npz" \
  --total-points 200000 \
  --include-regex "(left|_l_|_l_link$)" \
  --output-frame link_local \
  --seed 0

echo "[3/6] gs_dataset -> COLMAP (w2c)"
rm -rf "$ROOT/gggs_run/gs_colmap_orange_left_200k"
PYTHONPATH="$WM_ROOT" "$PY" -m world_model.gs_dataset_to_colmap \
  --in-dir "$ROOT/gggs_run/gs_dataset_orange_1f" \
  --out-dir "$ROOT/gggs_run/gs_colmap_orange_left_200k" \
  --init-npz "$ROOT/gggs_run/gs_init_left_orange_200k.npz" \
  --init-max-points 200000 \
  --bg-num-points 0 \
  --pose-convention w2c

echo "[4/6] Train FK model"
rm -rf "$ROOT/gggs_run/robot_fk_out_orange_left_200k"
PYTHONPATH="$EXT_PYTHONPATH:$WM_ROOT:${PYTHONPATH:-}" \
"$PY" "$ROOT/train_robot_fk_min.py" \
  -s "$ROOT/gggs_run/gs_colmap_orange_left_200k" \
  -m "$ROOT/gggs_run/robot_fk_out_orange_left_200k" \
  --poses-json "$ROOT/gggs_run/gs_dataset_orange_1f/poses.json" \
  --init-npz "$ROOT/gggs_run/gs_init_left_orange_200k.npz" \
  --urdf "$URDF" \
  --iterations 4000 \
  --lambda-robot 5.0 \
  --lambda-alpha-bg 0.5 \
  --lambda-alpha-fg 0.25 \
  --freeze-scale 1 \
  --freeze-rotation 0 \
  --resolution 1 \
  --data_device cpu

echo "[5/6] Render"
rm -rf "$ROOT/gggs_run/robot_fk_render_orange_left_200k"
PYTHONPATH="$EXT_PYTHONPATH:$WM_ROOT:${PYTHONPATH:-}" \
"$PY" "$ROOT/render_robot_fk.py" \
  -s "$ROOT/gggs_run/gs_colmap_orange_left_200k" \
  --poses-json "$ROOT/gggs_run/gs_dataset_orange_1f/poses.json" \
  --robot-model-path "$ROOT/gggs_run/robot_fk_out_orange_left_200k" \
  --out-dir "$ROOT/gggs_run/robot_fk_render_orange_left_200k" \
  --split all \
  --resolution 1 \
  --data_device cpu \
  --kernel-size 0 \
  --robot-opacity-bias 0 \
  --save-components \
  --save-gt

echo "[6/6] Export debug panel"
"$PY" "$ROOT/scripts/export_robot_debug_panel.py" \
  --orig "$ROOT/gggs_run/gs_colmap_orange_left_200k/images/frame_000000_head.png" \
  --mask "$ROOT/gggs_run/gs_colmap_orange_left_200k/masks/frame_000000_head.png" \
  --render "$ROOT/gggs_run/robot_fk_render_orange_left_200k/renders/00000.png" \
  --out "$ROOT/gggs_run/debug/panel_00000.png"

echo "[DONE]"
echo "renders: $ROOT/gggs_run/robot_fk_render_orange_left_200k/renders"
echo "alpha  : $ROOT/gggs_run/robot_fk_render_orange_left_200k/robot_alpha"
echo "panel  : $ROOT/gggs_run/debug/panel_00000.png"
