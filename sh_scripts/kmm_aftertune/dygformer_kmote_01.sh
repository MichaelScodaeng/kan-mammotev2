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
mkdir -p sh_scripts/kmm_aftertune/sh_logs/dygformer

# Run experiments for DyGFormer with kan_mammote_dual_kmote time encoder
# First batch (GPU-1A): wikipedia, reddit

# Wikipedia (has best config: expert_dim=256, mamba_d_state=256, mamba_expand=4, encoder_dropout=0.1)
python experiment_unified.py --single_encoder kan_mammote_dual_kmote --models DyGFormer \
 --datasets wikipedia --disable_progress_bar --num_runs 1 \
 --expert_dim 256 --mamba_d_state 256 --mamba_expand 4 --encoder_dropout 0.1 \
 > sh_scripts/kmm_aftertune/sh_logs/dygformer/dygformer_wikipedia_kmote.log 2>&1

# Reddit (no best config for DyGFormer, using defaults: expert_dim=128, mamba_d_state=256, mamba_expand=4, encoder_dropout=0.1)
python experiment_unified.py --single_encoder kan_mammote_dual_kmote --models DyGFormer \
 --datasets reddit --disable_progress_bar --num_runs 1 \
 --expert_dim 128 --mamba_d_state 256 --mamba_expand 4 --encoder_dropout 0.1 \
 > sh_scripts/kmm_aftertune/sh_logs/dygformer/dygformer_reddit_kmote.log 2>&1