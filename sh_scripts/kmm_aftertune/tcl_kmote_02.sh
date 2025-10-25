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
# Second batch: SocialEvo, Flights, UNvote, Contacts

# SocialEvo (no best config for TCL, using defaults: expert_dim=128, mamba_d_state=256, mamba_expand=4, encoder_encoder_dropout=0.1)
python experiment_unified.py --single_encoder kan_mammote_dual_kmote --models TCL \
 --datasets SocialEvo --disable_progress_bar --num_runs 1 \
 --expert_dim 128 --mamba_d_state 256 --mamba_expand 4 --encoder_dropout 0.1 \
 > sh_scripts/kmm_aftertune/sh_logs/tcl/tcl_socialevo_kmote.log 2>&1

# Flights (no best config for TCL, using defaults: expert_dim=128, mamba_d_state=256, mamba_expand=4, encoder_encoder_dropout=0.1)
python experiment_unified.py --single_encoder kan_mammote_dual_kmote --models TCL \
 --datasets Flights --disable_progress_bar --num_runs 1 \
 --expert_dim 128 --mamba_d_state 256 --mamba_expand 4 --encoder_dropout 0.1 \
 > sh_scripts/kmm_aftertune/sh_logs/tcl/tcl_flights_kmote.log 2>&1

# UNvote (no best config for TCL, using defaults: expert_dim=128, mamba_d_state=256, mamba_expand=4, encoder_encoder_dropout=0.1)
python experiment_unified.py --single_encoder kan_mammote_dual_kmote --models TCL \
 --datasets UNvote --disable_progress_bar --num_runs 1 \
 --expert_dim 128 --mamba_d_state 256 --mamba_expand 4 --encoder_dropout 0.1 \
 > sh_scripts/kmm_aftertune/sh_logs/tcl/tcl_unvete_kmote.log 2>&1

# Contacts (no best config for TCL, using defaults: expert_dim=128, mamba_d_state=256, mamba_expand=4, encoder_encoder_dropout=0.1)
python experiment_unified.py --single_encoder kan_mammote_dual_kmote --models TCL \
 --datasets Contacts --disable_progress_bar --num_runs 1 \
 --expert_dim 128 --mamba_d_state 256 --mamba_expand 4 --encoder_dropout 0.1 \
 > sh_scripts/kmm_aftertune/sh_logs/tcl/tcl_contacts_kmote.log 2>&1