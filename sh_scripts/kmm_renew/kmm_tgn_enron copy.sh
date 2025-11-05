#!/bin/bash
#PBS -j oe
#PBS -q GPU-1A
#PBS -l select=1:ngpus=1
#PBS -M s2516027@jaist.ac.jp
#PBS -m be

cd "$PBS_O_WORKDIR"

source ~/.bashrc
module purge
module load cuda/12.1
conda activate kan_mammote
mkdir -p sh_scripts/kmm_renew/sh_logs/tgn

# Run experiments for TGN with kan_mammote_dual_kmote time encoder
# Dataset: USLegis

timestamp=$(date +%Y%m%d_%H%M%S)
python experiment_unified2.py \
    --single_encoder kan_mammote_dual_kmote \
    --models TGN \
    --datasets uci \
    --num_epochs 200 \
    --data_ratio 1.0 \
    --test_interval_epochs 100 \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --optimizer AdamW \
    --disable_progress_bar \
    --num_runs 1 \
    --max_grad_norm 1.0 \
    --encoder_dropout 0.1 \
    --expert_dim 64 \
    --mamba_d_state 128 \
    --mamba_headdim 64 \
    > sh_scripts/kmm_renew/sh_logs/tgn/tgn_uci_kmote_${timestamp}.log 2>&1