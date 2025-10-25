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
# First batch: wikipedia, reddit, mooc, lastfm

# Wikipedia (has best config: expert_dim=128, mamba_d_state=512, mamba_expand=4, encoder_encoder_dropout=0.3)
python experiment_unified.py --single_encoder kan_mammote_dual_kmote --models TCL \
 --datasets wikipedia --disable_progress_bar --num_runs 1 \
 --expert_dim 128 --mamba_d_state 512 --mamba_expand 4 --encoder_dropout 0.3 \
 > sh_scripts/kmm_aftertune/sh_logs/tcl/tcl_wikipedia_kmote.log 2>&1

# Reddit (has best config: expert_dim=256, mamba_d_state=512, mamba_expand=2, encoder_encoder_dropout=0.1)
python experiment_unified.py --single_encoder kan_mammote_dual_kmote --models TCL \
 --datasets reddit --disable_progress_bar --num_runs 1 \
 --expert_dim 256 --mamba_d_state 512 --mamba_expand 2 --encoder_dropout 0.1 \
 > sh_scripts/kmm_aftertune/sh_logs/tcl/tcl_reddit_kmote.log 2>&1

# MOOC (has best config: expert_dim=256, mamba_d_state=512, mamba_expand=4, encoder_encoder_dropout=0.0)
python experiment_unified.py --single_encoder kan_mammote_dual_kmote --models TCL \
 --datasets mooc --disable_progress_bar --num_runs 1 \
 --expert_dim 256 --mamba_d_state 512 --mamba_expand 4 --encoder_dropout 0.0 \
 > sh_scripts/kmm_aftertune/sh_logs/tcl/tcl_mooc_kmote.log 2>&1

# LastFM (no best config for TCL, using defaults: expert_dim=128, mamba_d_state=256, mamba_expand=4, encoder_encoder_dropout=0.1)
python experiment_unified.py --single_encoder kan_mammote_dual_kmote --models TCL \
 --datasets lastfm --disable_progress_bar --num_runs 1 \
 --expert_dim 128 --mamba_d_state 256 --mamba_expand 4 --encoder_dropout 0.1 \
 > sh_scripts/kmm_aftertune/sh_logs/tcl/tcl_lastfm_kmote.log 2>&1