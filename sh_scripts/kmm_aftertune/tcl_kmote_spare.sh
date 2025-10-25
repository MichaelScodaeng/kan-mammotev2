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
mkdir -p sh_scripts/kmm_aftertune/sh_logs/tcl

# Run experiments for TCL with kan_mammote_dual_kmote time encoder
# Spare script: Remaining datasets (enron, uci, CanParl, UNtrade, USLegis)

# Enron (has best config: expert_dim=256, mamba_d_state=256, mamba_expand=4, encoder_encoder_dropout=0.1)
python experiment_unified.py --single_encoder kan_mammote_dual_kmote --models TCL \
 --datasets enron --disable_progress_bar --num_runs 1 \
 --expert_dim 256 --mamba_d_state 256 --mamba_expand 4 --encoder_dropout 0.1 \
 > sh_scripts/kmm_aftertune/sh_logs/tcl/tcl_enron_kmote.log 2>&1

# UCI (has best config: expert_dim=256, mamba_d_state=128, mamba_expand=2, encoder_encoder_dropout=0.0)
python experiment_unified.py --single_encoder kan_mammote_dual_kmote --models TCL \
 --datasets uci --disable_progress_bar --num_runs 1 \
 --expert_dim 256 --mamba_d_state 128 --mamba_expand 2 --encoder_dropout 0.0 \
 > sh_scripts/kmm_aftertune/sh_logs/tcl/tcl_uci_kmote.log 2>&1

# CanParl (no best config for TCL, using defaults: expert_dim=128, mamba_d_state=256, mamba_expand=4, encoder_encoder_dropout=0.1)
python experiment_unified.py --single_encoder kan_mammote_dual_kmote --models TCL \
 --datasets CanParl --disable_progress_bar --num_runs 1 \
 --expert_dim 128 --mamba_d_state 256 --mamba_expand 4 --encoder_dropout 0.1 \
 > sh_scripts/kmm_aftertune/sh_logs/tcl/tcl_canparl_kmote.log 2>&1

# UNtrade (no best config for TCL, using defaults: expert_dim=128, mamba_d_state=256, mamba_expand=4, encoder_encoder_dropout=0.1)
python experiment_unified.py --single_encoder kan_mammote_dual_kmote --models TCL \
 --datasets UNtrade --disable_progress_bar --num_runs 1 \
 --expert_dim 128 --mamba_d_state 256 --mamba_expand 4 --encoder_dropout 0.1 \
 > sh_scripts/kmm_aftertune/sh_logs/tcl/tcl_untrade_kmote.log 2>&1

# USLegis (has best config: expert_dim=256, mamba_d_state=128, mamba_expand=4, encoder_encoder_dropout=0.0)
python experiment_unified.py --single_encoder kan_mammote_dual_kmote --models TCL \
 --datasets USLegis --disable_progress_bar --num_runs 1 \
 --expert_dim 256 --mamba_d_state 128 --mamba_expand 4 --encoder_dropout 0.0 \
 > sh_scripts/kmm_aftertune/sh_logs/tcl/tcl_uslegis_kmote.log 2>&1