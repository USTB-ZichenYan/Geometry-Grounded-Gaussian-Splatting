# Robot FK Minimal Pipeline (Left Arm, Orange Dataset)

This README provides a directly runnable 6-step flow:

1. Generate left-arm mask
2. Generate URDF Gaussian prior
3. Convert dataset to COLMAP (w2c)
4. Train FK model
5. Render
6. Export debug panel

## 0) Environment

```bash
cd /home/SENSETIME/yanzichen/data/file/Geometry-Grounded-Gaussian-Splatting
ROOT=$PWD
PY=/home/SENSETIME/yanzichen/anaconda3/envs/gggs/bin/python
WM_ROOT="$ROOT/robotics_world_model"
URDF="$WM_ROOT/tianyi2_urdf-tianyi2.0/urdf/tianyi2.0_urdf_with_hands.urdf"
EXT_PYTHONPATH="$ROOT/submodules/diff-gaussian-rasterization:$ROOT/submodules/simple-knn:$ROOT/submodules/warp-patch-ncc:$ROOT/fused-ssim"
```

## 1) Generate Left-Arm Mask

Skip if already generated.

```bash
PYTHONPATH="$WM_ROOT" "$PY" -m world_model.create_left_arm_mask \
  --in-dir "$ROOT/gggs_run/gs_dataset_orange_1f/images/head" \
  --out-dir "$ROOT/gggs_run/gs_dataset_orange_1f/robot_masks/head"
```

## 2) Generate Left-Arm URDF Gaussian Prior (200k)

```bash
"$PY" "$WM_ROOT/world_model/gs_init_from_urdf.py" \
  --urdf "$URDF" \
  --out "$ROOT/gggs_run/gs_init_left_orange_200k.npz" \
  --total-points 200000 \
  --include-regex "(left|_l_|_l_link$)" \
  --output-frame link_local \
  --seed 0
```

## 3) Convert gs_dataset -> COLMAP (w2c fixed)

```bash
rm -rf "$ROOT/gggs_run/gs_colmap_orange_left_200k"
PYTHONPATH="$WM_ROOT" "$PY" -m world_model.gs_dataset_to_colmap \
  --in-dir "$ROOT/gggs_run/gs_dataset_orange_1f" \
  --out-dir "$ROOT/gggs_run/gs_colmap_orange_left_200k" \
  --init-npz "$ROOT/gggs_run/gs_init_left_orange_200k.npz" \
  --init-max-points 200000 \
  --bg-num-points 0 \
  --pose-convention w2c
```

## 4) Train FK Minimal Model

Notes:
- `train_robot_fk_min.py` defaults to `--lock-opacity-one 1`
- This run uses fixed scale to reduce diffusion

```bash
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
```

## 5) Render

```bash
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
```

## 6) Export 2x2 Debug Panel

```bash
"$PY" "$ROOT/scripts/export_robot_debug_panel.py" \
  --orig "$ROOT/gggs_run/gs_colmap_orange_left_200k/images/frame_000000_head.png" \
  --mask "$ROOT/gggs_run/gs_colmap_orange_left_200k/masks/frame_000000_head.png" \
  --render "$ROOT/gggs_run/robot_fk_render_orange_left_200k/renders/00000.png" \
  --out "$ROOT/gggs_run/debug/panel_00000.png"
```

## Outputs

- Model: `gggs_run/robot_fk_out_orange_left_200k`
- Renders: `gggs_run/robot_fk_render_orange_left_200k/renders`
- Alpha: `gggs_run/robot_fk_render_orange_left_200k/robot_alpha`
- GT: `gggs_run/robot_fk_render_orange_left_200k/gt`
- Panel: `gggs_run/debug/panel_00000.png`
