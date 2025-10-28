#!/bin/bash
#PBS -j oe
#PBS -q GPU-1A
#PBS -l select=1:ngpus=1
#PBS -M s2516027@jaist.ac.jp
#PBS -m be

cd "$PBS_O_WORKDIR"

source ~/.bashrc
module purge
module load cuda/12.8u1
conda activate kan_mammote
mkdir -p sh_scripts/kmm_aftertune/sh_logs/dygmamba

# Run experiments for DyGMamba with kan_mammote_dual_kmote time encoder
# Fourth batch (GPU-1): UNvote, Contacts

# UNvote (no best config for DyGMamba, using defaults: expert_dim=128, mamba_d_state=256, mamba_expand=4, encoder_dropout=0.1)
python experiment_unified2.py --single_encoder kan_mammote_dual_kmote --models DyGMamba \
 --datasets UNtrade --disable_progress_bar --num_runs 1 \
 --expert_dim 128 --mamba_d_state 64 --mamba_expand 4 --encoder_dropout 0.1 \
 > sh_scripts/kmm_aftertune/sh_logs/dygmamba/dygmamba_unvete_kmote.log 2>&1

# Contacts (no best config for DyGMamba, using defaults: expert_dim=128, mamba_d_state=256, mamba_expand=4, encoder_dropout=0.1)
python experiment_unified2.py --single_encoder kan_mammote_dual_kmote --models DyGMamba \
 --datasets enron --disable_progress_bar --num_runs 1 \
 --expert_dim 128 --mamba_d_state 64 --mamba_expand 4 --encoder_dropout 0.1 \
 > sh_scripts/kmm_aftertune/sh_logs/dygmamba/dygmamba_contacts_kmote.log 2>&1