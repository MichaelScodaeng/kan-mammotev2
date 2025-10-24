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
mkdir -p sh_scripts/thin/231068_sh/sh_logs/mercer/tgat
# Run your experiment - TGAT remaining datasets
python experiment_unified.py --models "TGAT" --single_encoder "mercer" --datasets enron uci \
 --disable_progress_bar --num_runs 1 > sh_scripts/thin/231068_sh/sh_logs/mercer/tgat/mercer_tgat_03.log 2>&1