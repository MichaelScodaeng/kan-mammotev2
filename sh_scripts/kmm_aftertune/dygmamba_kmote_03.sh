#!/bin/bash
#PBS -j oe
#PBS -q GPU-1
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
# Third batch (GPU-1): SocialEvo, Flights

# SocialEvo (no best config for DyGMamba, using defaults: expert_dim=128, mamba_d_state=256, mamba_expand=4, encoder_dropout=0.1)
python experiment_unified2.py --single_encoder kan_mammote_dual_kmote --models DyGMamba \
 --datasets SocialEvo --disable_progress_bar --num_runs 1 \
 --expert_dim 64 --mamba_d_state 64 --mamba_expand 4 --mamba_headdim 32 --encoder_dropout 0.1 \
 > sh_scripts/kmm_aftertune/sh_logs/dygmamba/dygmamba_socialevo_kmote.log 2>&1

# Flights (no best config for DyGMamba, using defaults: expert_dim=128, mamba_d_state=256, mamba_expand=4, encoder_dropout=0.1)
python experiment_unified2.py --single_encoder kan_mammote_dual_kmote --models DyGMamba \
 --datasets Flights --disable_progress_bar --num_runs 1 \
 --expert_dim 64 --mamba_d_state 64 --mamba_expand 4 --mamba_headdim 32 --encoder_dropout 0.1 \
 > sh_scripts/kmm_aftertune/sh_logs/dygmamba/dygmamba_flights_kmote.log 2>&1