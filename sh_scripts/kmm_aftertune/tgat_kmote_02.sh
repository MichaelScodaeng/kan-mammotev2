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
mkdir -p sh_scripts/kmm_aftertune/sh_logs/tgat

# Run experiments for TGAT with kan_mammote_dual_kmote time encoder
# Second batch (GPU-1): mooc, lastfm

# MOOC (has best config: expert_dim=128, mamba_d_state=512, mamba_expand=4, encoder_dropout=0.1)
python experiment_unified.py --single_encoder kan_mammote_dual_kmote --models TGAT \
 --datasets mooc --disable_progress_bar --num_runs 1 \
 --expert_dim 64 --mamba_d_state 256 --mamba_expand 4 --encoder_dropout 0.1 \
 > sh_scripts/kmm_aftertune/sh_logs/tgat/tgat_mooc_kmote.log 2>&1

# LastFM (has best config: expert_dim=128, mamba_d_state=256, mamba_expand=4, encoder_dropout=0.0)
python experiment_unified.py --single_encoder kan_mammote_dual_kmote --models TGAT \
 --datasets lastfm --disable_progress_bar --num_runs 1 \
 --expert_dim 64 --mamba_d_state 128 --mamba_expand 4 --encoder_dropout 0.0 \
 > sh_scripts/kmm_aftertune/sh_logs/tgat/tgat_lastfm_kmote.log 2>&1