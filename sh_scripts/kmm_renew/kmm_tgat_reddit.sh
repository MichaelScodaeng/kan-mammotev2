#!/bin/bash
#PBS -j oe
#PBS -q GPU-LA
#PBS -l select=1:ngpus=1
#PBS -M s2516027@jaist.ac.jp
#PBS -m be

cd "$PBS_O_WORKDIR"

source ~/.bashrc
module purge
module load cuda/12.1
conda activate kan_mammote
mkdir -p sh_scripts/kmm_renew/sh_logs/tgat

# Run experiments for TGAT with kan_mammote_dual_kmote time encoder
# Dataset: wikipedia and reddit

timestamp=$(date +%Y%m%d_%H%M%S)
CUDA_VISIBLE_DEVICES=0 \
python experiment_unified2.py \
    --single_encoder kan_mammote_dual_kmote \
    --models TGAT \
    --datasets wikipedia \
    --num_epochs 200 \
    --data_ratio 1.0 \
    --test_interval_epochs 100 \
    --learning_rate 2e-5 \
    --weight_decay 1e-4 \
    --optimizer AdamW \
    --disable_progress_bar \
    --num_runs 1 \
    --max_grad_norm 1.0 \
    > sh_scripts/kmm_renew/sh_logs/tgat/tgat_wikipedia_kmote_${timestamp}.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 \
python experiment_unified2.py \
    --single_encoder kan_mammote_dual_kmote \
    --models TGAT \
    --datasets reddit \
    --num_epochs 200 \
    --data_ratio 1.0 \
    --test_interval_epochs 100 \
    --learning_rate 2e-5 \
    --weight_decay 1e-4 \
    --optimizer AdamW \
    --disable_progress_bar \
    --num_runs 1 \
    --max_grad_norm 1.0 \
    > sh_scripts/kmm_renew/sh_logs/tgat/tgat_reddit_kmote_${timestamp}.log 2>&1 &
wait