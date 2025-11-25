#!/bin/bash
#PBS -j oe
#PBS -q GPU-S
#PBS -l select=1:ngpus=1
#PBS -M s2516027@jaist.ac.jp
#PBS -m be

cd "$PBS_O_WORKDIR"

source ~/.bashrc
module purge
module load cuda/12.1
conda activate kan_mammote
mkdir -p sh_scripts/kmm_211168/sh_logs/tcl

# Run experiments for TCL with kan_mammote_dual_kmote time encoder
# Dataset: UNvote
# Best config from Optuna trial 25: AP=0.5256

timestamp=$(date +%Y%m%d_%H%M%S)
python experiment_unified2.py \
    --single_encoder kan_mammote_dual_kmote \
    --models TCL \
    --datasets UNvote \
    --num_epochs 200 \
    --data_ratio 1.0 \
    --dropout 0.1 \
    --encoder_dropout 0.1 \
    --expert_dim 256 \
    --mamba_expand 4 \
    --mamba_d_state 256 \
    --mamba_headdim 32 \
    --test_interval_epochs 100 \
    --learning_rate 3.2767383266601276e-05 \
    --weight_decay 1.544708694797184e-08 \
    --batch_size 200 \
    --optimizer AdamW \
    --disable_progress_bar \
    --num_runs 1 \
    --max_grad_norm 1.0 \
    > sh_scripts/kmm_211168/sh_logs/tcl/tcl_unvote_kmote_${timestamp}.log 2>&1