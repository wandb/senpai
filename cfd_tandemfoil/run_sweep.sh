#!/bin/bash
# Phase 6 adaptive-norm-v2 sweep: 8 GPUs
set -e
cd /workspace/senpai/cfd_tandemfoil

BASE="python train.py --agent fern \
  --wandb_group phase6/adaptive-norm-v2 \
  --field_decoder --adaln_output --use_lion --lr 2e-4 \
  --aug aoa_perturb --aug_full_dsdf_rot --high_p_clamp \
  --n_layers 3 --slice_num 96 --tandem_ramp \
  --domain_layernorm --domain_velhead --ema_decay 0.999 \
  --weight_decay 5e-5 --cosine_T_max 160 --disable_pcgrad \
  --pressure_first --pressure_deep --residual_prediction \
  --surface_refine --surface_refine_hidden 192 --surface_refine_layers 3"

# GPU 0: Baseline s42
CUDA_VISIBLE_DEVICES=0 $BASE --seed 42 \
  --wandb_name "fern/baseline-s42" &

# GPU 1: Baseline s43
CUDA_VISIBLE_DEVICES=1 $BASE --seed 43 \
  --wandb_name "fern/baseline-s43" &

# GPU 2: Asinh scale=1.0 s42 (confirmed from #2042)
CUDA_VISIBLE_DEVICES=2 $BASE --asinh_pressure --asinh_scale 1.0 --seed 42 \
  --wandb_name "fern/asinh-s1.0-s42" &

# GPU 3: Asinh scale=1.0 s43
CUDA_VISIBLE_DEVICES=3 $BASE --asinh_pressure --asinh_scale 1.0 --seed 43 \
  --wandb_name "fern/asinh-s1.0-s43" &

# GPU 4: Asinh scale=0.75
CUDA_VISIBLE_DEVICES=4 $BASE --asinh_pressure --asinh_scale 0.75 --seed 42 \
  --wandb_name "fern/asinh-s0.75-s42" &

# GPU 5: Asinh scale=0.5
CUDA_VISIBLE_DEVICES=5 $BASE --asinh_pressure --asinh_scale 0.5 --seed 42 \
  --wandb_name "fern/asinh-s0.5-s42" &

# GPU 6: Asinh s1.0 + adaptive norm
CUDA_VISIBLE_DEVICES=6 $BASE --asinh_pressure --asinh_scale 1.0 --adaptive_norm --seed 42 \
  --wandb_name "fern/asinh-s1.0-adaptive-s42" &

# GPU 7: Adaptive norm only (no asinh)
CUDA_VISIBLE_DEVICES=7 $BASE --adaptive_norm --seed 42 \
  --wandb_name "fern/adaptive-only-s42" &

echo "All 8 runs launched. Waiting..."
wait
echo "All 8 runs completed."
