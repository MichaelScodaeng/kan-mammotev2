#!/bin/bash
#PBS -j oe
#PBS -q GPU-L
#PBS -l select=1:ngpus=1
#PBS -M s2516027@jaist.ac.jp
#PBS -m be

cd "$PBS_O_WORKDIR"

source ~/.bashrc
module purge
module load cuda/12.8u1
conda activate kan_mammote
mkdir -p sh_scripts/kmm_aftertune/sh_logs/dygformer

# Run experiments for DyGFormer with kan_mammote_dual_kmote time encoder
# Third batch (GPU-L): SocialEvo, Flights

# SocialEvo (no best config for DyGFormer, using defaults: expert_dim=128, mamba_d_state=256, mamba_expand=4, encoder_dropout=0.1)
python experiment_unified2.py --single_encoder kan_mammote_dual_kmote --models DyGFormer \
 --datasets SocialEvo --disable_progress_bar --num_runs 1 \
 --expert_dim 64 --mamba_d_state 64 --mamba_expand 4 --mamba_headdim 32 --encoder_dropout 0.1 \
 > sh_scripts/kmm_aftertune/sh_logs/dygformer/dygformer_socialevo_kmote.log 2>&1

# Flights (no best config for DyGFormer, using defaults: expert_dim=128, mamba_d_state=256, mamba_expand=4, encoder_dropout=0.1)
python experiment_unified2.py --single_encoder kan_mammote_dual_kmote --models DyGFormer \
 --datasets Flights --disable_progress_bar --num_runs 1 \
 --expert_dim 64 --mamba_d_state 64 --mamba_expand 4 --mamba_headdim 32 --encoder_dropout 0.1 \
 > sh_scripts/kmm_aftertune/sh_logs/dygformer/dygformer_flights_kmote.log 2>&1