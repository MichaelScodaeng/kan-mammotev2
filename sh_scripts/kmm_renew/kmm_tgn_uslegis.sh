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

python experiment_unified2.py \
    --single_encoder kan_mammote_dual_kmote \
    --models TGN \
    --datasets USLegis \
    --num_epochs 200 \
    --data_ratio 1.0 \
    --test_interval_epochs 100 \
    --learning_rate 5e-5 \
    --weight_decay 1e-2 \
    --optimizer AdamW8bit \
    --disable_progress_bar \
    --num_runs 1 \
    > sh_scripts/kmm_renew/sh_logs/tgn/tgn_uslegis_kmote.log 2>&1