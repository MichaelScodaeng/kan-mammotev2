#!/bin/bash
#PBS -j oe
#PBS -q GPU-1
#PBS -l select=1:ngpus=1
#PBS -M s2516027@jaist.ac.jp
#PBS -m be

cd "$PBS_O_WORKDIR"

source ~/.bashrc
conda activate kan_mammote

# Run your experiment
python experiment_unified.py --models "DyGFormer" --single_encoder "lete" --datasets lastfm --disable_progress_bar --num_runs 1 > current_thin/sh_logs/DyGFormer_lete_04_lastfm.log 2>&1