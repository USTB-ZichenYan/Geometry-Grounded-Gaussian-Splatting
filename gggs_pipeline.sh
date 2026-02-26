#!/usr/bin/env bash
set -euo pipefail

# One-click pipeline: URDF init -> dataset -> COLMAP -> train -> render

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

  TOTAL_POINTS=200000
  SIGMA_MM=2.0
  ALLOC_MODE=area
  AREA_EXP=0.7
  SEED=0
  Q_VECTOR="comma,separated,32d,q"
  DATA_STRIDE=3
  DATA_MAX_FRAMES=240
  Q_DELTA_MAX=
  Q_SOURCE=state
  HEAD_LINK_CAM=
  ROBOT_LINK_INCLUDE_REGEX=(shoulder|elbow|wrist|tcp|thumb|index|middle|ring|little)
  ROBOT_LINK_EXCLUDE_REGEX=(base|waist|torso|pelvis|head|camera|neck|chest)
  CPU_THREADS=8
  HEAD_ONLY=0
  BACKGROUND_ONLY=0
  ROBOT_FK_MASK=0
  ROBOT_MASK_POINTS=60000
  ROBOT_MASK_RADIUS_PX=3
  ROBOT_FK_TRAIN=0
  ROBOT_FK_ITERATIONS=20000
  ROBOT_FK_BG_ITERATION=-1
  ROBOT_FK_MIN_ROBOT_PIXELS=64
  ROBOT_FK_LAMBDA_ROBOT=1.0
  ROBOT_FK_LAMBDA_BG=1.0
  ROBOT_FK_LAMBDA_ALPHA_BG=0.1
  ROBOT_FK_SAVE_EVERY=2000
  BLACK_MEAN_THR=1.0
  BLACK_VAR_THR=1.0
  CLEAN_RUN=0

  ITERATIONS=10000
  USE_DECOUPLED_APPEARANCE=0
  TRAIN_RESOLUTION=2
  AUTO_BG_MASK=0
  BG_MASK_STD_THR=0.03
  BG_MASK_DILATE=5
  DISABLE_FILTER3D=1
  DENSIFY_UNTIL_ITER=0

Examples:
  bash gggs_pipeline.sh

  OUT_ROOT=./gggs_run ITERATIONS=10000 USE_DECOUPLED_APPEARANCE=1 bash gggs_pipeline.sh

  Q_VECTOR="0,0,..." DATA_STRIDE=2 DATA_MAX_FRAMES=400 TRAIN_RESOLUTION=2 DISABLE_FILTER3D=1 DENSIFY_UNTIL_ITER=0 bash gggs_pipeline.sh

  HEAD_ONLY=1 CLEAN_RUN=1 DATA_STRIDE=1 DATA_MAX_FRAMES= bash gggs_pipeline.sh

  HEAD_ONLY=1 CLEAN_RUN=1 DATA_STRIDE=1 Q_DELTA_MAX=0.03 TRAIN_RESOLUTION=1 DENSIFY_UNTIL_ITER=3000 bash gggs_pipeline.sh

  HEAD_ONLY=1 CLEAN_RUN=1 DATA_STRIDE=1 TRAIN_RESOLUTION=1 DENSIFY_UNTIL_ITER=3000 AUTO_BG_MASK=1 BG_MASK_STD_THR=0.03 BG_MASK_DILATE=5 bash gggs_pipeline.sh

  HEAD_ONLY=1 BACKGROUND_ONLY=1 CLEAN_RUN=1 DATA_STRIDE=1 TRAIN_RESOLUTION=1 DENSIFY_UNTIL_ITER=0 AUTO_BG_MASK=1 bash gggs_pipeline.sh

  HEAD_ONLY=1 BACKGROUND_ONLY=1 ROBOT_FK_MASK=1 CLEAN_RUN=1 DATA_STRIDE=1 TRAIN_RESOLUTION=1 DENSIFY_UNTIL_ITER=0 AUTO_BG_MASK=0 bash gggs_pipeline.sh

  HEAD_ONLY=1 BACKGROUND_ONLY=1 ROBOT_FK_TRAIN=1 CLEAN_RUN=1 DATA_STRIDE=1 TRAIN_RESOLUTION=1 DENSIFY_UNTIL_ITER=2000 bash gggs_pipeline.sh
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

OUT_ROOT="${OUT_ROOT:-$GGGS_ROOT/gggs_run}"
GS_INIT_NPZ="${GS_INIT_NPZ:-$OUT_ROOT/gs_init.npz}"
GS_DATASET="${GS_DATASET:-$OUT_ROOT/gs_dataset}"
GS_COLMAP="${GS_COLMAP:-$OUT_ROOT/gs_colmap}"

TOTAL_POINTS="${TOTAL_POINTS:-200000}"
SIGMA_MM="${SIGMA_MM:-2.0}"
ALLOC_MODE="${ALLOC_MODE:-area}"
AREA_EXP="${AREA_EXP:-0.7}"
SEED="${SEED:-0}"
Q_VECTOR="${Q_VECTOR:-}"
DATA_STRIDE="${DATA_STRIDE:-3}"
DATA_MAX_FRAMES="${DATA_MAX_FRAMES-240}"
Q_DELTA_MAX="${Q_DELTA_MAX:-}"
Q_SOURCE="${Q_SOURCE:-state}"
HEAD_LINK_CAM="${HEAD_LINK_CAM:-}"
ROBOT_LINK_INCLUDE_REGEX="${ROBOT_LINK_INCLUDE_REGEX:-(shoulder|elbow|wrist|tcp|thumb|index|middle|ring|little)}"
ROBOT_LINK_EXCLUDE_REGEX="${ROBOT_LINK_EXCLUDE_REGEX:-(base|waist|torso|pelvis|head|camera|neck|chest)}"
CPU_THREADS="${CPU_THREADS:-8}"
HEAD_ONLY="${HEAD_ONLY:-0}"
BACKGROUND_ONLY="${BACKGROUND_ONLY:-0}"
ROBOT_FK_MASK="${ROBOT_FK_MASK:-0}"
ROBOT_MASK_POINTS="${ROBOT_MASK_POINTS:-60000}"
ROBOT_MASK_RADIUS_PX="${ROBOT_MASK_RADIUS_PX:-3}"
ROBOT_FK_TRAIN="${ROBOT_FK_TRAIN:-0}"
ROBOT_FK_ITERATIONS="${ROBOT_FK_ITERATIONS:-20000}"
ROBOT_FK_BG_ITERATION="${ROBOT_FK_BG_ITERATION:--1}"
ROBOT_FK_MIN_ROBOT_PIXELS="${ROBOT_FK_MIN_ROBOT_PIXELS:-64}"
ROBOT_FK_LAMBDA_ROBOT="${ROBOT_FK_LAMBDA_ROBOT:-1.0}"
ROBOT_FK_LAMBDA_BG="${ROBOT_FK_LAMBDA_BG:-1.0}"
ROBOT_FK_LAMBDA_ALPHA_BG="${ROBOT_FK_LAMBDA_ALPHA_BG:-0.1}"
ROBOT_FK_SAVE_EVERY="${ROBOT_FK_SAVE_EVERY:-2000}"
BLACK_MEAN_THR="${BLACK_MEAN_THR:-1.0}"
BLACK_VAR_THR="${BLACK_VAR_THR:-1.0}"
CLEAN_RUN="${CLEAN_RUN:-0}"

if [[ -z "${GGGS_OUT:-}" ]]; then
  if [[ "$HEAD_ONLY" == "1" ]]; then
    GGGS_OUT="$OUT_ROOT/gggs_out_head"
  else
    GGGS_OUT="$OUT_ROOT/gggs_out"
  fi
fi

ITERATIONS="${ITERATIONS:-10000}"
USE_DECOUPLED_APPEARANCE="${USE_DECOUPLED_APPEARANCE:-0}"
DATA_DEVICE="${DATA_DEVICE:-cpu}"
TRAIN_RESOLUTION="${TRAIN_RESOLUTION:-2}"
AUTO_BG_MASK="${AUTO_BG_MASK:-0}"
BG_MASK_STD_THR="${BG_MASK_STD_THR:-0.03}"
BG_MASK_DILATE="${BG_MASK_DILATE:-5}"
DISABLE_FILTER3D="${DISABLE_FILTER3D:-1}"
DENSIFY_UNTIL_ITER="${DENSIFY_UNTIL_ITER:-0}"
PYTORCH_CUDA_ALLOC_CONF_VALUE="${PYTORCH_CUDA_ALLOC_CONF_VALUE:-max_split_size_mb:128}"
ROBOT_FK_OUT="${ROBOT_FK_OUT:-$OUT_ROOT/robot_fk_out}"
ROBOT_FK_COLMAP="${ROBOT_FK_COLMAP:-$OUT_ROOT/gs_colmap_robot}"
ROBOT_FK_BG_MODEL_PATH="${ROBOT_FK_BG_MODEL_PATH:-$GGGS_OUT}"
ROBOT_FK_RESOLUTION="${ROBOT_FK_RESOLUTION:-$TRAIN_RESOLUTION}"
ROBOT_FK_DATA_DEVICE="${ROBOT_FK_DATA_DEVICE:-$DATA_DEVICE}"
ROBOT_FK_Q_REF="${ROBOT_FK_Q_REF:-}"

if [[ "$ROBOT_FK_TRAIN" == "1" && "$ROBOT_FK_MASK" != "1" ]]; then
  echo "[CFG] ROBOT_FK_TRAIN=1 -> force ROBOT_FK_MASK=1 for mask-dependent robot loss"
  ROBOT_FK_MASK=1
fi

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
echo "[CFG] TOTAL_POINTS=$TOTAL_POINTS SIGMA_MM=$SIGMA_MM ALLOC_MODE=$ALLOC_MODE AREA_EXP=$AREA_EXP SEED=$SEED"
echo "[CFG] EXT_PYTHONPATH=$EXT_PYTHONPATH"
echo "[CFG] DATA_DEVICE=$DATA_DEVICE TRAIN_RESOLUTION=$TRAIN_RESOLUTION PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF_VALUE"
echo "[CFG] AUTO_BG_MASK=$AUTO_BG_MASK BG_MASK_STD_THR=$BG_MASK_STD_THR BG_MASK_DILATE=$BG_MASK_DILATE"
echo "[CFG] DATA_STRIDE=$DATA_STRIDE DATA_MAX_FRAMES=$DATA_MAX_FRAMES Q_DELTA_MAX=${Q_DELTA_MAX:-<none>} CPU_THREADS=$CPU_THREADS"
echo "[CFG] Q_SOURCE=$Q_SOURCE"
echo "[CFG] HEAD_LINK_CAM=${HEAD_LINK_CAM:-<default>}"
echo "[CFG] ROBOT_LINK_INCLUDE_REGEX=$ROBOT_LINK_INCLUDE_REGEX"
echo "[CFG] ROBOT_LINK_EXCLUDE_REGEX=$ROBOT_LINK_EXCLUDE_REGEX"
echo "[CFG] HEAD_ONLY=$HEAD_ONLY BACKGROUND_ONLY=$BACKGROUND_ONLY ROBOT_FK_MASK=$ROBOT_FK_MASK ROBOT_MASK_POINTS=$ROBOT_MASK_POINTS ROBOT_MASK_RADIUS_PX=$ROBOT_MASK_RADIUS_PX"
echo "[CFG] ROBOT_FK_TRAIN=$ROBOT_FK_TRAIN ROBOT_FK_ITERATIONS=$ROBOT_FK_ITERATIONS ROBOT_FK_BG_MODEL_PATH=$ROBOT_FK_BG_MODEL_PATH ROBOT_FK_OUT=$ROBOT_FK_OUT"
echo "[CFG] BLACK_MEAN_THR=$BLACK_MEAN_THR BLACK_VAR_THR=$BLACK_VAR_THR"
echo "[CFG] CLEAN_RUN=$CLEAN_RUN"
echo "[CFG] DISABLE_FILTER3D=$DISABLE_FILTER3D"
echo "[CFG] DENSIFY_UNTIL_ITER=$DENSIFY_UNTIL_ITER"

export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"

if [[ "$CLEAN_RUN" == "1" ]]; then
  echo "[CLEAN] Removing previous outputs"
  rm -rf "$GS_DATASET" "$GS_COLMAP" "$GGGS_OUT" "$ROBOT_FK_OUT" "$ROBOT_FK_COLMAP"
fi

FILTER3D_FLAG=""
if [[ "$DISABLE_FILTER3D" == "1" ]]; then
  FILTER3D_FLAG="--disable_filter3D"
fi

BG_MASK_FLAGS=""
if [[ "$AUTO_BG_MASK" == "1" ]]; then
  BG_MASK_FLAGS="--auto_bg_mask --bg_mask_std_thr $BG_MASK_STD_THR --bg_mask_dilate $BG_MASK_DILATE"
fi

echo "[1/5] URDF -> GS init"
PYTHONPATH="$WORLD_MODEL_ROOT" \
python3 -m world_model.gs_init_from_urdf \
  --urdf "$URDF_PATH" \
  --out "$GS_INIT_NPZ" \
  --total-points "$TOTAL_POINTS" \
  --sigma-mm "$SIGMA_MM" \
  --alloc-mode "$ALLOC_MODE" \
  --area-exp "$AREA_EXP" \
  --include-regex "$ROBOT_LINK_INCLUDE_REGEX" \
  --exclude-regex "$ROBOT_LINK_EXCLUDE_REGEX" \
  --seed "$SEED" \
  --output-frame base \
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
if [[ -n "$HEAD_LINK_CAM" ]]; then
  DATASET_ARGS+=(--head-link-cam "$HEAD_LINK_CAM")
fi
PYTHONPATH="$WORLD_MODEL_ROOT" \
python3 -m world_model.gs_dataset \
  "${DATASET_ARGS[@]}"

echo "[3/5] gs_dataset -> COLMAP"
COLMAP_ARGS=(
  --in-dir "$GS_DATASET"
  --out-dir "$GS_COLMAP"
)
if [[ "$BACKGROUND_ONLY" != "1" ]]; then
  COLMAP_ARGS+=(
    --init-npz "$GS_INIT_NPZ"
    --init-max-points "$TOTAL_POINTS"
    --init-color 128,128,128
  )
else
  echo "[3/5] BACKGROUND_ONLY=1 -> skip robot init points"
fi
if [[ "$ROBOT_FK_MASK" == "1" ]]; then
  COLMAP_ARGS+=(
    --robot-mask-urdf "$URDF_PATH"
    --robot-mask-points "$ROBOT_MASK_POINTS"
    --robot-mask-radius-px "$ROBOT_MASK_RADIUS_PX"
    --robot-mask-include-regex "$ROBOT_LINK_INCLUDE_REGEX"
    --robot-mask-exclude-regex "$ROBOT_LINK_EXCLUDE_REGEX"
  )
fi
PYTHONPATH="$WORLD_MODEL_ROOT" \
python3 -m world_model.gs_dataset_to_colmap \
  "${COLMAP_ARGS[@]}"

echo "[4/5] Train Geometry-Grounded GS"
cd "$GGGS_ROOT"
if [[ "$USE_DECOUPLED_APPEARANCE" == "1" ]]; then
  PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF_VALUE" \
  PYTHONPATH="$EXT_PYTHONPATH:${PYTHONPATH:-}" \
  python3 train.py -s "$GS_COLMAP" -m "$GGGS_OUT" --iterations "$ITERATIONS" --eval --use_decoupled_appearance 3 --data_device "$DATA_DEVICE" --resolution "$TRAIN_RESOLUTION" --densify_until_iter "$DENSIFY_UNTIL_ITER" $FILTER3D_FLAG $BG_MASK_FLAGS
else
  PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF_VALUE" \
  PYTHONPATH="$EXT_PYTHONPATH:${PYTHONPATH:-}" \
  python3 train.py -s "$GS_COLMAP" -m "$GGGS_OUT" --iterations "$ITERATIONS" --eval --data_device "$DATA_DEVICE" --resolution "$TRAIN_RESOLUTION" --densify_until_iter "$DENSIFY_UNTIL_ITER" $FILTER3D_FLAG $BG_MASK_FLAGS
fi

echo "[5/5] Render novel views"
PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF_VALUE" \
PYTHONPATH="$EXT_PYTHONPATH:${PYTHONPATH:-}" \
python3 render.py -m "$GGGS_OUT" --data_device "$DATA_DEVICE" --resolution "$TRAIN_RESOLUTION"

if [[ "$ROBOT_FK_TRAIN" == "1" ]]; then
  ROBOT_TRAIN_SOURCE="$GS_COLMAP"
  if [[ "$BACKGROUND_ONLY" == "1" ]]; then
    echo "[6/7] Build robot COLMAP (with init points) for FK training"
    PYTHONPATH="$WORLD_MODEL_ROOT" \
    python3 -m world_model.gs_dataset_to_colmap \
      --in-dir "$GS_DATASET" \
      --out-dir "$ROBOT_FK_COLMAP" \
      --init-npz "$GS_INIT_NPZ" \
      --init-max-points "$TOTAL_POINTS" \
      --init-color 128,128,128 \
      --robot-mask-urdf "$URDF_PATH" \
      --robot-mask-points "$ROBOT_MASK_POINTS" \
      --robot-mask-radius-px "$ROBOT_MASK_RADIUS_PX" \
      --robot-mask-include-regex "$ROBOT_LINK_INCLUDE_REGEX" \
      --robot-mask-exclude-regex "$ROBOT_LINK_EXCLUDE_REGEX"
    ROBOT_TRAIN_SOURCE="$ROBOT_FK_COLMAP"
  elif [[ ! -d "$GS_COLMAP/masks" ]]; then
    echo "[ERROR] ROBOT_FK_TRAIN=1 requires masks in $GS_COLMAP/masks. Set ROBOT_FK_MASK=1 and rerun."
    exit 1
  fi

  echo "[7/7] Train FK robot model with frozen background"
  ROBOT_Q_REF_ARGS=()
  if [[ -n "$ROBOT_FK_Q_REF" ]]; then
    ROBOT_Q_REF_ARGS=(--q-ref "$ROBOT_FK_Q_REF")
  fi
  PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF_VALUE" \
  PYTHONPATH="$EXT_PYTHONPATH:$WORLD_MODEL_ROOT:${PYTHONPATH:-}" \
  python3 train_robot_fk.py \
    -s "$ROBOT_TRAIN_SOURCE" \
    -m "$ROBOT_FK_OUT" \
    --poses-json "$GS_DATASET/poses.json" \
    --init-npz "$GS_INIT_NPZ" \
    --urdf "$URDF_PATH" \
    --bg-model-path "$ROBOT_FK_BG_MODEL_PATH" \
    --bg-iteration "$ROBOT_FK_BG_ITERATION" \
    --iterations "$ROBOT_FK_ITERATIONS" \
    --resolution "$ROBOT_FK_RESOLUTION" \
    --data_device "$ROBOT_FK_DATA_DEVICE" \
    --min-robot-pixels "$ROBOT_FK_MIN_ROBOT_PIXELS" \
    --lambda-robot "$ROBOT_FK_LAMBDA_ROBOT" \
    --lambda-bg "$ROBOT_FK_LAMBDA_BG" \
    --lambda-alpha-bg "$ROBOT_FK_LAMBDA_ALPHA_BG" \
    --save-every "$ROBOT_FK_SAVE_EVERY" \
    "${ROBOT_Q_REF_ARGS[@]}"
fi

echo "[DONE] Output:"
echo "  GS init:    $GS_INIT_NPZ"
echo "  Dataset:    $GS_DATASET"
echo "  COLMAP:     $GS_COLMAP"
echo "  Model out:  $GGGS_OUT"
if [[ "$ROBOT_FK_TRAIN" == "1" ]]; then
  echo "  Robot COLMAP: $ROBOT_TRAIN_SOURCE"
  echo "  Robot out:    $ROBOT_FK_OUT"
fi
