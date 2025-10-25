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
mkdir -p sh_scripts/kmm_aftertune/sh_logs/dygformer

# Run experiments for DyGFormer with kan_mammote_dual_kmote time encoder
# Spare script (GPU-1): Remaining datasets (enron, uci, CanParl, UNtrade, USLegis)

# Enron (no best config for DyGFormer, using defaults: expert_dim=128, mamba_d_state=256, mamba_expand=4, encoder_dropout=0.1)
python experiment_unified.py --single_encoder kan_mammote_dual_kmote --models DyGFormer \
 --datasets enron --disable_progress_bar --num_runs 1 \
 --expert_dim 128 --mamba_d_state 256 --mamba_expand 4 --encoder_dropout 0.1 \
 > sh_scripts/kmm_aftertune/sh_logs/dygformer/dygformer_enron_kmote.log 2>&1

# UCI (no best config for DyGFormer, using defaults: expert_dim=128, mamba_d_state=256, mamba_expand=4, encoder_dropout=0.1)
python experiment_unified.py --single_encoder kan_mammote_dual_kmote --models DyGFormer \
 --datasets uci --disable_progress_bar --num_runs 1 \
 --expert_dim 128 --mamba_d_state 256 --mamba_expand 4 --encoder_dropout 0.1 \
 > sh_scripts/kmm_aftertune/sh_logs/dygformer/dygformer_uci_kmote.log 2>&1

# CanParl (has best config: expert_dim=256, mamba_d_state=512, mamba_expand=4, encoder_dropout=0.0)
python experiment_unified.py --single_encoder kan_mammote_dual_kmote --models DyGFormer \
 --datasets CanParl --disable_progress_bar --num_runs 1 \
 --expert_dim 256 --mamba_d_state 512 --mamba_expand 4 --encoder_dropout 0.0 \
 > sh_scripts/kmm_aftertune/sh_logs/dygformer/dygformer_canparl_kmote.log 2>&1

# UNtrade (has best config: expert_dim=256, mamba_d_state=256, mamba_expand=4, encoder_dropout=0.3)
python experiment_unified.py --single_encoder kan_mammote_dual_kmote --models DyGFormer \
 --datasets UNtrade --disable_progress_bar --num_runs 1 \
 --expert_dim 256 --mamba_d_state 256 --mamba_expand 4 --encoder_dropout 0.3 \
 > sh_scripts/kmm_aftertune/sh_logs/dygformer/dygformer_untrade_kmote.log 2>&1

# USLegis (no best config for DyGFormer, using defaults: expert_dim=128, mamba_d_state=256, mamba_expand=4, encoder_dropout=0.1)
python experiment_unified.py --single_encoder kan_mammote_dual_kmote --models DyGFormer \
 --datasets USLegis --disable_progress_bar --num_runs 1 \
 --expert_dim 128 --mamba_d_state 256 --mamba_expand 4 --encoder_dropout 0.1 \
 > sh_scripts/kmm_aftertune/sh_logs/dygformer/dygformer_uslegis_kmote.log 2>&1