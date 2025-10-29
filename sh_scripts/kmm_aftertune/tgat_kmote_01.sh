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
mkdir -p sh_scripts/kmm_aftertune/sh_logs/tgat

# Run experiments for TGAT with kan_mammote_dual_kmote time encoder
# First batch (GPU-1): wikipedia, reddit

# Wikipedia (has best config: expert_dim=256, mamba_d_state=128, mamba_expand=4, encoder_dropout=0.2)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

python experiment_unified2.py --single_encoder kan_mammote_dual_kmote --models TGAT \
 --datasets wikipedia --disable_progress_bar --num_runs 1 \
 --expert_dim 64 --mamba_d_state 64 --mamba_expand 4 --mamba_headdim 32 --encoder_dropout 0.2 \
 > sh_scripts/kmm_aftertune/sh_logs/tgat/tgat_wikipedia_kmote.log 2>&1

# Reddit (no best config for TGAT, using defaults: expert_dim=128, mamba_d_state=256, mamba_expand=4, encoder_dropout=0.1)
python experiment_unified2.py --single_encoder kan_mammote_dual_kmote --models TGAT \
 --datasets reddit --disable_progress_bar --num_runs 1 \
 --expert_dim 64 --mamba_d_state 64 --mamba_expand 4 --mamba_headdim 32 --encoder_dropout 0.1 \
 > sh_scripts/kmm_aftertune/sh_logs/tgat/tgat_reddit_kmote.log 2>&1