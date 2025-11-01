#!/bin/bash
#PBS -j oe
#PBS -q GPU-1
#PBS -l select=1:ngpus=1
#PBS -M s2516027@jaist.ac.jp
#PBS -m be

cd "$PBS_O_WORKDIR"

source ~/.bashrc
module purge
module load cuda/12.1
conda activate kan_mammote
mkdir -p sh_scripts/kmm_renew/sh_logs/dygmamba

# Run experiments for DyGMamba with kan_mammote_dual_kmote time encoder
# Dataset: mooc

timestamp=$(date +%Y%m%d_%H%M%S)
python experiment_unified2.py \
    --single_encoder kan_mammote_dual_kmote \
    --models DyGMamba \
    --datasets mooc \
    --num_epochs 200 \
    --data_ratio 1.0 \
    --test_interval_epochs 100 \
    --learning_rate 2e-5 \
    --weight_decay 1e-4 \
    --optimizer AdamW \
    --disable_progress_bar \
    --num_runs 1 \
    --max_grad_norm 1.0 \
    > sh_scripts/kmm_renew/sh_logs/dygmamba/dygmamba_mooc_kmote_${timestamp}.log 2>&1