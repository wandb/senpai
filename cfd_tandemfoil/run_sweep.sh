#!/bin/bash
# Phase 6 SAM sweep: baseline vs SAM rho=0.02/0.05/0.1, 2 seeds each
# SAM active from epoch 120 (last ~25% of training, matching original design intent)
# Bug fix: old code used MAX_EPOCHS*0.75=375 which was unreachable; now uses sam_start_epoch
set -e
cd /workspace/senpai/cfd_tandemfoil

BASE="python train.py --agent fern \
  --wandb_group phase6/sam-v4 \
  --asinh_pressure --asinh_scale 0.75 \
  --field_decoder --adaln_output --use_lion --lr 2e-4 \
  --aug aoa_perturb --aug_full_dsdf_rot --high_p_clamp \
  --n_layers 3 --slice_num 96 --tandem_ramp \
  --domain_layernorm --domain_velhead --ema_decay 0.999 \
  --weight_decay 5e-5 --cosine_T_max 160 --disable_pcgrad \
  --pressure_first --pressure_deep --residual_prediction \
  --surface_refine --surface_refine_hidden 192 --surface_refine_layers 3"

# GPU 0-1: Baseline (no SAM), seeds 42/43
CUDA_VISIBLE_DEVICES=0 $BASE --seed 42 --wandb_name "fern/sam4-baseline-s42" &
CUDA_VISIBLE_DEVICES=1 $BASE --seed 43 --wandb_name "fern/sam4-baseline-s43" &

# GPU 2-3: SAM rho=0.05, seeds 42/43
CUDA_VISIBLE_DEVICES=2 $BASE --adaln_sam --sam_rho 0.05 --sam_start_epoch 120 --seed 42 --wandb_name "fern/sam4-rho005-s42" &
CUDA_VISIBLE_DEVICES=3 $BASE --adaln_sam --sam_rho 0.05 --sam_start_epoch 120 --seed 43 --wandb_name "fern/sam4-rho005-s43" &

# GPU 4-5: SAM rho=0.1, seeds 42/43
CUDA_VISIBLE_DEVICES=4 $BASE --adaln_sam --sam_rho 0.1 --sam_start_epoch 120 --seed 42 --wandb_name "fern/sam4-rho01-s42" &
CUDA_VISIBLE_DEVICES=5 $BASE --adaln_sam --sam_rho 0.1 --sam_start_epoch 120 --seed 43 --wandb_name "fern/sam4-rho01-s43" &

# GPU 6-7: SAM rho=0.02, seeds 42/43
CUDA_VISIBLE_DEVICES=6 $BASE --adaln_sam --sam_rho 0.02 --sam_start_epoch 120 --seed 42 --wandb_name "fern/sam4-rho002-s42" &
CUDA_VISIBLE_DEVICES=7 $BASE --adaln_sam --sam_rho 0.02 --sam_start_epoch 120 --seed 43 --wandb_name "fern/sam4-rho002-s43" &

echo "All 8 runs launched. Waiting..."
wait
echo "All 8 runs completed."
