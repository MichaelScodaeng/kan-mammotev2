#!/bin/bash
#PBS -j oe
#PBS -q GPU-1A
#PBS -l select=1:ngpus=1
#PBS -M s2516027@jaist.ac.jp
#PBS -m be

cd "$PBS_O_WORKDIR"

source ~/.bashrc
conda activate kan_mammote

# Run your experiment
python experiment_unified.py --models "TGAT" --single_encoder "lete" --datasets UNtrade UNvote --disable_progress_bar --num_runs 1 > current_thin/sh_logs/TGAT_lete_02_untrade_unvete.log 2>&1