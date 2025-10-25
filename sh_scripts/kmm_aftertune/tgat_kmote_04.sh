#!/bin/bash
#PBS -j oe
#PBS -q GPU-LA
#PBS -l select=1:ngpus=1
#PBS -M s2516027@jaist.ac.jp
#PBS -m be

cd "$PBS_O_WORKDIR"

source ~/.bashrc
module purge
module load cuda/12.8u1
conda activate kan_mammote
mkdir -p sh_scripts/kmm_aftertune/sh_logs/tgat

# Run experiments for TGAT with kan_mammote_dual_kmote time encoder
# Fourth batch (GPU-LA): UNvote, Contacts

# UNvote (has best config: expert_dim=256, mamba_d_state=512, mamba_expand=2, encoder_dropout=0.1)
python experiment_unified.py --single_encoder kan_mammote_dual_kmote --models TGAT \
 --datasets UNvote --disable_progress_bar --num_runs 1 \
 --expert_dim 64 --mamba_d_state 256 --mamba_expand 2 --encoder_dropout 0.1 \
 > sh_scripts/kmm_aftertune/sh_logs/tgat/tgat_unvete_kmote.log 2>&1

# Contacts (no best config for TGAT, using defaults: expert_dim=128, mamba_d_state=256, mamba_expand=4, encoder_dropout=0.1)
python experiment_unified.py --single_encoder kan_mammote_dual_kmote --models TGAT \
 --datasets Contacts --disable_progress_bar --num_runs 1 \
 --expert_dim 64 --mamba_d_state 128 --mamba_expand 4 --encoder_dropout 0.1 \
 > sh_scripts/kmm_aftertune/sh_logs/tgat/tgat_contacts_kmote.log 2>&1