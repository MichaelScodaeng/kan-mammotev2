#!/bin/bash
#PBS -j oe
#PBS -q GPU-1
#PBS -l select=1:ngpus=1
#PBS -M s2516027@jaist.ac.jp
#PBS -m be

cd "$PBS_O_WORKDIR"

source ~/.bashrc
module purge
module load cuda/12.1
conda activate kan_mammote
mkdir -p optuna_scripts/epochs/sh_logs
# Run your experiment
python -u run_full_benchmark.py \
    > optuna_scripts/epochs/sh_logs/running_time01.log 2>&1
