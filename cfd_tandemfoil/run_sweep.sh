#!/bin/bash
# Phase 6 inviscid-cp sweep: 8 GPUs
set -e
cd /workspace/senpai/cfd_tandemfoil

BASE="python train.py --agent fern \
  --wandb_group phase6/inviscid-cp \
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

# GPU 2: +Cp input s42
CUDA_VISIBLE_DEVICES=2 $BASE --inviscid_cp --seed 42 \
  --wandb_name "fern/cp-input-s42" &

# GPU 3: +Cp input s43
CUDA_VISIBLE_DEVICES=3 $BASE --inviscid_cp --seed 43 \
  --wandb_name "fern/cp-input-s43" &

# GPU 4: +Cp + residual s42
CUDA_VISIBLE_DEVICES=4 $BASE --inviscid_cp --cp_residual --seed 42 \
  --wandb_name "fern/cp-residual-s42" &

# GPU 5: +Cp + residual s43
CUDA_VISIBLE_DEVICES=5 $BASE --inviscid_cp --cp_residual --seed 43 \
  --wandb_name "fern/cp-residual-s43" &

# GPU 6: +Cp + surf_weight 25 s42  (note: surf_weight is adaptive so this is same as cp-input)
CUDA_VISIBLE_DEVICES=6 $BASE --inviscid_cp --surf_weight 25 --seed 42 \
  --wandb_name "fern/cp-surfweight25-s42" &

# GPU 7: +Cp + residual + wider refine 256
CUDA_VISIBLE_DEVICES=7 $BASE --inviscid_cp --cp_residual --surface_refine_hidden 256 --seed 42 \
  --wandb_name "fern/cp-residual-wider-s42" &

echo "All 8 runs launched. Waiting..."
wait
echo "All 8 runs completed."
