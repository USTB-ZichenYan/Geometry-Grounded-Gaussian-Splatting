#!/usr/bin/env bash
# Usage:
#   bash scripts/run_left_arm_mask.sh
#
# Optional env overrides:
#   ROOT=/path/to/Geometry-Grounded-Gaussian-Splatting
#   PY=/path/to/python
#   DATASET_DIR=$ROOT/gggs_run/gs_dataset_raw_1f
#   CAM_ID=head
#   GREEN_G_MIN=100
#   GREEN_MARGIN=30
#   LEFT_RATIO=0.5
#   LOG_DIR=$ROOT/gggs_run/logs
#   LOG_FILE=$LOG_DIR/left_arm_mask_custom.log
#
# Output:
#   - robot masks: <DATASET_DIR>/robot_masks/<CAM_ID>/frame_*.png
#   - run log:     <LOG_FILE> (or auto-generated under LOG_DIR)
#
# Example:
#   cd /home/SENSETIME/yanzichen/data/file/Geometry-Grounded-Gaussian-Splatting
#   ROOT=$PWD \
#   PY=/home/SENSETIME/yanzichen/anaconda3/envs/gggs/bin/python \
#   DATASET_DIR=$ROOT/gggs_run/gs_dataset_raw_1f \
#   CAM_ID=head \
#   GREEN_G_MIN=100 \
#   GREEN_MARGIN=30 \
#   LEFT_RATIO=0.5 \
#   bash scripts/run_left_arm_mask.sh
set -euo pipefail

ROOT="${ROOT:-/home/SENSETIME/yanzichen/data/file/Geometry-Grounded-Gaussian-Splatting}"
PY="${PY:-/home/SENSETIME/yanzichen/anaconda3/envs/gggs/bin/python}"
DATASET_DIR="${DATASET_DIR:-$ROOT/gggs_run/gs_dataset_raw_1f}"
CAM_ID="${CAM_ID:-head}"

"$PY" "$ROOT/robotics_world_model/world_model/create_left_arm_mask.py" \
  --dataset-dir "$DATASET_DIR" \
  --cam-id "$CAM_ID" \
  --green-g-min "${GREEN_G_MIN:-100}" \
  --green-margin "${GREEN_MARGIN:-30}" \
  --left-ratio "${LEFT_RATIO:-0.5}"
