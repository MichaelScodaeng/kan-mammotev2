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
# Spare script: Remaining datasets (enron, uci, CanParl, UNtrade, USLegis)

# Enron (has best config: expert_dim=128, mamba_d_state=256, mamba_expand=4, encoder_dropout=0.1)
python experiment_unified.py --single_encoder kan_mammote_dual_kmote --models TGAT \
 --datasets enron --disable_progress_bar --num_runs 1 \
 --expert_dim 128 --mamba_d_state 256 --mamba_expand 4 --encoder_dropout 0.1 \
 > sh_scripts/kmm_aftertune/sh_logs/tgat/tgat_enron_kmote.log 2>&1

# UCI (has best config: expert_dim=256, mamba_d_state=128, mamba_expand=2, encoder_dropout=0.2)
python experiment_unified.py --single_encoder kan_mammote_dual_kmote --models TGAT \
 --datasets uci --disable_progress_bar --num_runs 1 \
 --expert_dim 256 --mamba_d_state 128 --mamba_expand 2 --encoder_dropout 0.2 \
 > sh_scripts/kmm_aftertune/sh_logs/tgat/tgat_uci_kmote.log 2>&1

# CanParl (has best config: expert_dim=256, mamba_d_state=256, mamba_expand=2, encoder_dropout=0.2)
python experiment_unified.py --single_encoder kan_mammote_dual_kmote --models TGAT \
 --datasets CanParl --disable_progress_bar --num_runs 1 \
 --expert_dim 256 --mamba_d_state 256 --mamba_expand 2 --encoder_dropout 0.2 \
 > sh_scripts/kmm_aftertune/sh_logs/tgat/tgat_canparl_kmote.log 2>&1

# UNtrade (has best config: expert_dim=256, mamba_d_state=512, mamba_expand=4, encoder_dropout=0.3)
python experiment_unified.py --single_encoder kan_mammote_dual_kmote --models TGAT \
 --datasets UNtrade --disable_progress_bar --num_runs 1 \
 --expert_dim 256 --mamba_d_state 512 --mamba_expand 4 --encoder_dropout 0.3 \
 > sh_scripts/kmm_aftertune/sh_logs/tgat/tgat_untrade_kmote.log 2>&1

# USLegis (has best config: expert_dim=256, mamba_d_state=256, mamba_expand=2, encoder_dropout=0.2)
python experiment_unified.py --single_encoder kan_mammote_dual_kmote --models TGAT \
 --datasets USLegis --disable_progress_bar --num_runs 1 \
 --expert_dim 256 --mamba_d_state 256 --mamba_expand 2 --encoder_dropout 0.2 \
 > sh_scripts/kmm_aftertune/sh_logs/tgat/tgat_uslegis_kmote.log 2>&1