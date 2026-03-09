#!/usr/bin/env bash
set -euo pipefail

# One-click pipeline: URDF init -> dataset -> robot COLMAP -> FK train -> robot render

usage() {
  cat <<'USAGE'
Usage:
  bash gggs_pipeline.sh

Optional env overrides:
  GGGS_ROOT=/path/to/Geometry-Grounded-Gaussian-Splatting
  WORLD_MODEL_ROOT=/path/to/Geometry-Grounded-Gaussian-Splatting/robotics_world_model
  DATA_ROOT=/path/to/dual_arm_grab_data
  URDF_PATH=/path/to/tianyi2.0_urdf_with_hands.urdf
  OUT_ROOT=./gggs_run

  TOTAL_POINTS=20000
  SIGMA_MM=2.0
  ALLOC_MODE=area
  AREA_EXP=0.7
  SEED=0
  Q_VECTOR="comma,separated,32d,q"
  DATA_STRIDE=3
  DATA_MAX_FRAMES=240
  Q_DELTA_MAX=
  Q_SOURCE=state
  ROBOT_LINK_INCLUDE_REGEX=(shoulder|elbow|wrist|tcp|thumb|index|middle|ring|little)
  ROBOT_LINK_EXCLUDE_REGEX=(base|waist|torso|pelvis|head|camera|neck|chest)
  CPU_THREADS=8
  HEAD_ONLY=0
  ROBOT_MASK_POINTS=60000
  ROBOT_MASK_RADIUS_PX=3
  ROBOT_FK_ITERATIONS=20000
  ROBOT_CANONICAL_NPZ=
  ROBOT_FK_DYNAMIC=1
  ROBOT_FK_FRAME_ORDER=sequential
  ROBOT_FK_STEPS_PER_FRAME=1
  ROBOT_FK_MIN_ROBOT_PIXELS=64
  ROBOT_FK_LAMBDA_ROBOT=1.0
  ROBOT_FK_LAMBDA_ALPHA_BG=0.1
  ROBOT_FK_LAMBDA_ALPHA_FG=0.2
  ROBOT_FK_LAMBDA_DSSIM=0.0
  ROBOT_FK_SCALING_LR=0.005
  ROBOT_FK_ROTATION_LR=0.001
  ROBOT_FK_SAVE_EVERY=2000
  FREEZE_ROBOT_XYZ=0
  FREEZE_ROBOT_RGB=128,128,128
  FREEZE_ROBOT_TOL=3.0
  BLACK_MEAN_THR=1.0
  BLACK_VAR_THR=1.0
  SINGLE_IMAGE_TRAIN=0
  CLEAN_RUN=0

  TRAIN_RESOLUTION=2
Examples:
  bash gggs_pipeline.sh
  HEAD_ONLY=1 CLEAN_RUN=1 DATA_STRIDE=1 DATA_MAX_FRAMES=240 TOTAL_POINTS=20000 ROBOT_FK_ITERATIONS=6000 bash gggs_pipeline.sh
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

GGGS_ROOT="${GGGS_ROOT:-/home/SENSETIME/yanzichen/data/file/Geometry-Grounded-Gaussian-Splatting}"
WORLD_MODEL_ROOT="${WORLD_MODEL_ROOT:-$GGGS_ROOT/robotics_world_model}"
DATA_ROOT="${DATA_ROOT:-/home/SENSETIME/yanzichen/data/file/Geometry-Grounded-Gaussian-Splatting/robotics_world_model/dual_arm_grab_data}"
URDF_PATH="${URDF_PATH:-$WORLD_MODEL_ROOT/tianyi2_urdf-tianyi2.0/urdf/tianyi2.0_urdf_with_hands.urdf}"
EXT_PYTHONPATH="${GGGS_ROOT}/submodules/diff-gaussian-rasterization:${GGGS_ROOT}/submodules/simple-knn:${GGGS_ROOT}/submodules/warp-patch-ncc:${GGGS_ROOT}/fused-ssim"
PYTHON_BIN="${PYTHON_BIN:-python3}"

OUT_ROOT="${OUT_ROOT:-$GGGS_ROOT/gggs_run}"
GS_INIT_NPZ="${GS_INIT_NPZ:-$OUT_ROOT/gs_init.npz}"
GS_DATASET="${GS_DATASET:-$OUT_ROOT/gs_dataset}"
GS_COLMAP="${GS_COLMAP:-$OUT_ROOT/gs_colmap}"

TOTAL_POINTS="${TOTAL_POINTS:-20000}"
SIGMA_MM="${SIGMA_MM:-2.0}"
ALLOC_MODE="${ALLOC_MODE:-area}"
AREA_EXP="${AREA_EXP:-0.7}"
SEED="${SEED:-0}"
Q_VECTOR="${Q_VECTOR:-}"
DATA_STRIDE="${DATA_STRIDE:-3}"
DATA_MAX_FRAMES="${DATA_MAX_FRAMES-240}"
Q_DELTA_MAX="${Q_DELTA_MAX:-}"
Q_SOURCE="${Q_SOURCE:-state}"
ROBOT_LINK_INCLUDE_REGEX="${ROBOT_LINK_INCLUDE_REGEX:-(shoulder|elbow|wrist|tcp|thumb|index|middle|ring|little)}"
ROBOT_LINK_EXCLUDE_REGEX="${ROBOT_LINK_EXCLUDE_REGEX:-(base|waist|torso|pelvis|head|camera|neck|chest)}"
CPU_THREADS="${CPU_THREADS:-8}"
HEAD_ONLY="${HEAD_ONLY:-0}"
ROBOT_MASK_POINTS="${ROBOT_MASK_POINTS:-60000}"
ROBOT_MASK_RADIUS_PX="${ROBOT_MASK_RADIUS_PX:-3}"
ROBOT_FK_ITERATIONS="${ROBOT_FK_ITERATIONS:-20000}"
ROBOT_CANONICAL_NPZ="${ROBOT_CANONICAL_NPZ:-}"
ROBOT_FK_DYNAMIC="${ROBOT_FK_DYNAMIC:-1}"
ROBOT_FK_FRAME_ORDER="${ROBOT_FK_FRAME_ORDER:-sequential}"
ROBOT_FK_STEPS_PER_FRAME="${ROBOT_FK_STEPS_PER_FRAME:-1}"
ROBOT_FK_MIN_ROBOT_PIXELS="${ROBOT_FK_MIN_ROBOT_PIXELS:-64}"
ROBOT_FK_LAMBDA_ROBOT="${ROBOT_FK_LAMBDA_ROBOT:-1.0}"
ROBOT_FK_LAMBDA_ALPHA_BG="${ROBOT_FK_LAMBDA_ALPHA_BG:-0.0}"
ROBOT_FK_LAMBDA_ALPHA_FG="${ROBOT_FK_LAMBDA_ALPHA_FG:-0.0}"
ROBOT_FK_LAMBDA_DSSIM="${ROBOT_FK_LAMBDA_DSSIM:-0.0}"
ROBOT_FK_SCALING_LR="${ROBOT_FK_SCALING_LR:-0.0}"
ROBOT_FK_ROTATION_LR="${ROBOT_FK_ROTATION_LR:-0.0}"
ROBOT_FK_SAVE_EVERY="${ROBOT_FK_SAVE_EVERY:-2000}"
FREEZE_ROBOT_XYZ="${FREEZE_ROBOT_XYZ:-0}"
FREEZE_ROBOT_RGB="${FREEZE_ROBOT_RGB:-128,128,128}"
FREEZE_ROBOT_TOL="${FREEZE_ROBOT_TOL:-3.0}"
BLACK_MEAN_THR="${BLACK_MEAN_THR:-1.0}"
BLACK_VAR_THR="${BLACK_VAR_THR:-1.0}"
SINGLE_IMAGE_TRAIN="${SINGLE_IMAGE_TRAIN:-0}"
CLEAN_RUN="${CLEAN_RUN:-0}"

DATA_DEVICE="${DATA_DEVICE:-cpu}"
TRAIN_RESOLUTION="${TRAIN_RESOLUTION:-2}"
PYTORCH_CUDA_ALLOC_CONF_VALUE="${PYTORCH_CUDA_ALLOC_CONF_VALUE:-max_split_size_mb:128}"
ROBOT_FK_OUT="${ROBOT_FK_OUT:-$OUT_ROOT/robot_fk_out}"
ROBOT_FK_COLMAP="${ROBOT_FK_COLMAP:-$GS_COLMAP}"
ROBOT_FK_RESOLUTION="${ROBOT_FK_RESOLUTION:-$TRAIN_RESOLUTION}"
ROBOT_FK_DATA_DEVICE="${ROBOT_FK_DATA_DEVICE:-$DATA_DEVICE}"
ROBOT_FK_Q_REF="${ROBOT_FK_Q_REF:-}"

echo "[CFG] FK training is core logic -> always enabled"
echo "[CFG] FK masks are always enabled"

if [[ ! -d "$GGGS_ROOT" ]]; then
  echo "[ERROR] GGGS_ROOT not found: $GGGS_ROOT"
  exit 1
fi
if [[ ! -d "$WORLD_MODEL_ROOT" ]]; then
  echo "[ERROR] WORLD_MODEL_ROOT not found: $WORLD_MODEL_ROOT"
  exit 1
fi
if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[ERROR] DATA_ROOT not found: $DATA_ROOT"
  exit 1
fi
if [[ ! -f "$URDF_PATH" ]]; then
  echo "[ERROR] URDF_PATH not found: $URDF_PATH"
  exit 1
fi

mkdir -p "$OUT_ROOT"

echo "[CFG] GGGS_ROOT=$GGGS_ROOT"
echo "[CFG] WORLD_MODEL_ROOT=$WORLD_MODEL_ROOT"
echo "[CFG] DATA_ROOT=$DATA_ROOT"
echo "[CFG] URDF_PATH=$URDF_PATH"
echo "[CFG] OUT_ROOT=$OUT_ROOT"
echo "[CFG] PYTHON_BIN=$PYTHON_BIN"
echo "[CFG] TOTAL_POINTS=$TOTAL_POINTS SIGMA_MM=$SIGMA_MM ALLOC_MODE=$ALLOC_MODE AREA_EXP=$AREA_EXP SEED=$SEED"
echo "[CFG] EXT_PYTHONPATH=$EXT_PYTHONPATH"
echo "[CFG] DATA_DEVICE=$DATA_DEVICE TRAIN_RESOLUTION=$TRAIN_RESOLUTION PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF_VALUE"
echo "[CFG] DATA_STRIDE=$DATA_STRIDE DATA_MAX_FRAMES=$DATA_MAX_FRAMES Q_DELTA_MAX=${Q_DELTA_MAX:-<none>} CPU_THREADS=$CPU_THREADS"
echo "[CFG] Q_SOURCE=$Q_SOURCE"
echo "[CFG] ROBOT_LINK_INCLUDE_REGEX=$ROBOT_LINK_INCLUDE_REGEX"
echo "[CFG] ROBOT_LINK_EXCLUDE_REGEX=$ROBOT_LINK_EXCLUDE_REGEX"
echo "[CFG] HEAD_ONLY=$HEAD_ONLY ROBOT_MASK_POINTS=$ROBOT_MASK_POINTS ROBOT_MASK_RADIUS_PX=$ROBOT_MASK_RADIUS_PX"
echo "[CFG] ROBOT_FK_ITERATIONS=$ROBOT_FK_ITERATIONS ROBOT_FK_OUT=$ROBOT_FK_OUT"
echo "[CFG] ROBOT_CANONICAL_NPZ=${ROBOT_CANONICAL_NPZ:-<init-npz>} ROBOT_FK_DYNAMIC=$ROBOT_FK_DYNAMIC"
echo "[CFG] ROBOT_FK_FRAME_ORDER=$ROBOT_FK_FRAME_ORDER ROBOT_FK_STEPS_PER_FRAME=$ROBOT_FK_STEPS_PER_FRAME"
echo "[CFG] ROBOT_FK_LAMBDA_DSSIM=$ROBOT_FK_LAMBDA_DSSIM"
echo "[CFG] ROBOT_FK_LAMBDA_ALPHA_FG=$ROBOT_FK_LAMBDA_ALPHA_FG"
echo "[CFG] ROBOT_FK_SCALING_LR=$ROBOT_FK_SCALING_LR ROBOT_FK_ROTATION_LR=$ROBOT_FK_ROTATION_LR"
echo "[CFG] FREEZE_ROBOT_XYZ=$FREEZE_ROBOT_XYZ FREEZE_ROBOT_RGB=$FREEZE_ROBOT_RGB FREEZE_ROBOT_TOL=$FREEZE_ROBOT_TOL"
echo "[CFG] BLACK_MEAN_THR=$BLACK_MEAN_THR BLACK_VAR_THR=$BLACK_VAR_THR"
echo "[CFG] SINGLE_IMAGE_TRAIN=$SINGLE_IMAGE_TRAIN"
echo "[CFG] CLEAN_RUN=$CLEAN_RUN"
echo "[CFG] Robot-only pipeline"

export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"

if [[ "$CLEAN_RUN" == "1" ]]; then
  echo "[CLEAN] Removing previous outputs"
  rm -rf "$GS_DATASET" "$GS_COLMAP" "$ROBOT_FK_OUT" "$ROBOT_FK_COLMAP" \
    "$OUT_ROOT/gggs_out" "$OUT_ROOT/gggs_out_head" "$OUT_ROOT/robot_fk_render" "$OUT_ROOT/robot_fk_render_debug"
fi

FREEZE_XYZ_FLAGS=""
if [[ "$FREEZE_ROBOT_XYZ" == "1" ]]; then
  FREEZE_XYZ_FLAGS="--freeze_robot_xyz --freeze_robot_rgb $FREEZE_ROBOT_RGB --freeze_robot_tol $FREEZE_ROBOT_TOL"
fi

if [[ "$SINGLE_IMAGE_TRAIN" == "1" ]]; then
  echo "[CFG] SINGLE_IMAGE_TRAIN=1 -> force DATA_MAX_FRAMES=1, DATA_STRIDE=1"
  DATA_MAX_FRAMES="1"
  DATA_STRIDE="1"
fi

echo "[1/5] URDF -> robot canonical GS init"
PYTHONPATH="$WORLD_MODEL_ROOT" \
"$PYTHON_BIN" "$WORLD_MODEL_ROOT/world_model/gs_init_from_urdf.py" \
  --urdf "$URDF_PATH" \
  --out "$GS_INIT_NPZ" \
  --total-points "$TOTAL_POINTS" \
  --sigma-mm "$SIGMA_MM" \
  --alloc-mode "$ALLOC_MODE" \
  --area-exp "$AREA_EXP" \
  --include-regex "$ROBOT_LINK_INCLUDE_REGEX" \
  --exclude-regex "$ROBOT_LINK_EXCLUDE_REGEX" \
  --seed "$SEED" \
  --output-frame link_local \
  ${Q_VECTOR:+--q "$Q_VECTOR"} \
  --verbose

echo "[2/5] Dataset -> images + poses"
DATASET_ARGS=(
  --data-root "$DATA_ROOT"
  --out-dir "$GS_DATASET"
  --stride "$DATA_STRIDE"
  --q-source "$Q_SOURCE"
  --black-mean-thr "$BLACK_MEAN_THR"
  --black-var-thr "$BLACK_VAR_THR"
)
if [[ -n "$DATA_MAX_FRAMES" ]]; then
  DATASET_ARGS+=(--max-frames "$DATA_MAX_FRAMES")
fi
if [[ -n "$Q_DELTA_MAX" ]]; then
  DATASET_ARGS+=(--q-delta-max "$Q_DELTA_MAX")
fi
if [[ "$HEAD_ONLY" == "1" ]]; then
  DATASET_ARGS+=(--head-only)
fi
PYTHONPATH="$WORLD_MODEL_ROOT" \
"$PYTHON_BIN" "$WORLD_MODEL_ROOT/world_model/gs_dataset.py" \
  "${DATASET_ARGS[@]}"

echo "[3/5] gs_dataset -> robot COLMAP"
COLMAP_ARGS=(
  --in-dir "$GS_DATASET"
  --out-dir "$ROBOT_FK_COLMAP"
  --pose-convention w2c
  --init-npz "$GS_INIT_NPZ"
  --init-max-points "$TOTAL_POINTS"
  --init-color 128,128,128
  --bg-num-points 0
  --robot-mask-urdf "$URDF_PATH"
  --robot-mask-points "$ROBOT_MASK_POINTS"
  --robot-mask-radius-px "$ROBOT_MASK_RADIUS_PX"
  --robot-mask-include-regex "$ROBOT_LINK_INCLUDE_REGEX"
  --robot-mask-exclude-regex "$ROBOT_LINK_EXCLUDE_REGEX"
)
PYTHONPATH="$WORLD_MODEL_ROOT" \
"$PYTHON_BIN" "$WORLD_MODEL_ROOT/world_model/gs_dataset_to_colmap.py" \
  "${COLMAP_ARGS[@]}"
ROBOT_TRAIN_SOURCE="$ROBOT_FK_COLMAP"

echo "[4/5] Train FK robot model"
ROBOT_Q_REF_ARGS=()
if [[ -n "$ROBOT_FK_Q_REF" ]]; then
  ROBOT_Q_REF_ARGS=(--q-ref "$ROBOT_FK_Q_REF")
fi
ROBOT_CANONICAL_ARGS=()
if [[ -n "$ROBOT_CANONICAL_NPZ" ]]; then
  ROBOT_CANONICAL_ARGS=(--robot-canonical-npz "$ROBOT_CANONICAL_NPZ")
fi
PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF_VALUE" \
PYTHONPATH="$EXT_PYTHONPATH:$WORLD_MODEL_ROOT:${PYTHONPATH:-}" \
"$PYTHON_BIN" train_robot_fk.py \
  -s "$ROBOT_TRAIN_SOURCE" \
  -m "$ROBOT_FK_OUT" \
  --poses-json "$GS_DATASET/poses.json" \
  --init-npz "$GS_INIT_NPZ" \
  --urdf "$URDF_PATH" \
  --dynamic-fk "$ROBOT_FK_DYNAMIC" \
  --frame-order "$ROBOT_FK_FRAME_ORDER" \
  --steps-per-frame "$ROBOT_FK_STEPS_PER_FRAME" \
  --iterations "$ROBOT_FK_ITERATIONS" \
  --resolution "$ROBOT_FK_RESOLUTION" \
  --data_device "$ROBOT_FK_DATA_DEVICE" \
  --scaling_lr "$ROBOT_FK_SCALING_LR" \
  --rotation_lr "$ROBOT_FK_ROTATION_LR" \
  --min-robot-pixels "$ROBOT_FK_MIN_ROBOT_PIXELS" \
  --lambda-robot "$ROBOT_FK_LAMBDA_ROBOT" \
  --lambda-alpha-bg "$ROBOT_FK_LAMBDA_ALPHA_BG" \
  --lambda-alpha-fg "$ROBOT_FK_LAMBDA_ALPHA_FG" \
  --lambda-dssim "$ROBOT_FK_LAMBDA_DSSIM" \
  --save-every "$ROBOT_FK_SAVE_EVERY" \
  "${ROBOT_CANONICAL_ARGS[@]}" \
  "${ROBOT_Q_REF_ARGS[@]}"

echo "[5/5] Render robot views"
PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF_VALUE" \
PYTHONPATH="$EXT_PYTHONPATH:$WORLD_MODEL_ROOT:${PYTHONPATH:-}" \
"$PYTHON_BIN" render_robot_fk.py \
  -s "$ROBOT_TRAIN_SOURCE" \
  --poses-json "$GS_DATASET/poses.json" \
  --robot-model-path "$ROBOT_FK_OUT" \
  --out-dir "$OUT_ROOT/robot_fk_render" \
  --split all \
  --resolution "$ROBOT_FK_RESOLUTION" \
  --data_device "$ROBOT_FK_DATA_DEVICE"

echo "[DONE] Output:"
echo "  GS init:    $GS_INIT_NPZ"
echo "  Dataset:    $GS_DATASET"
echo "  Robot COLMAP: $ROBOT_TRAIN_SOURCE"
echo "  Robot out:    $ROBOT_FK_OUT"
echo "  Robot render: $OUT_ROOT/robot_fk_render"
