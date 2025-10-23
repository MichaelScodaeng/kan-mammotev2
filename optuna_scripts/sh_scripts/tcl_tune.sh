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
mkdir -p optuna_scripts/sh_scripts/sh_logs
# Run your experiment
python -u tune_kan_mammote_optuna.py \
    --multi_dataset \
    --datasets uci USLegis enron reddit mooc UNtrade UNvote SocialEvo Contacts CanParl \
    --models TCL \
    --trials_per_combo 30 \
    > optuna_scripts/sh_scripts/sh_logs/tcl_tune01.log 2>&1
