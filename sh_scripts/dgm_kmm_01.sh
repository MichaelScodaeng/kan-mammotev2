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
mkdir -p sh_scripts/sh_logs/kmm
# Run your experiment
python experiment_unified.py --models "DyGMamba" --single_encoder "kan_mammote_dual_kmote" --datasets Contacts mooc UNtrade --disable_progress_bar --num_runs 1 > sh_scripts/sh_logs/kmm/kmm_DyGMamba_01.log 2>&1
