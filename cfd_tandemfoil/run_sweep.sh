#!/bin/bash
# Phase 6 asinh s=0.75 multi-seed validation: 8 GPUs, seeds 42-49
set -e
cd /workspace/senpai/cfd_tandemfoil

BASE="python train.py --agent fern \
  --wandb_group phase6/asinh-075-multiseed \
  --field_decoder --adaln_output --use_lion --lr 2e-4 \
  --aug aoa_perturb --aug_full_dsdf_rot --high_p_clamp \
  --n_layers 3 --slice_num 96 --tandem_ramp \
  --domain_layernorm --domain_velhead --ema_decay 0.999 \
  --weight_decay 5e-5 --cosine_T_max 160 --disable_pcgrad \
  --pressure_first --pressure_deep --residual_prediction \
  --surface_refine --surface_refine_hidden 192 --surface_refine_layers 3 \
  --asinh_pressure --asinh_scale 0.75"

for i in $(seq 0 7); do
  seed=$((42 + i))
  CUDA_VISIBLE_DEVICES=$i $BASE --seed $seed \
    --wandb_name "fern/asinh-075-s${seed}" &
done

echo "All 8 runs launched. Waiting..."
wait
echo "All 8 runs completed."
