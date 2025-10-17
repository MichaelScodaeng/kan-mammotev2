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

# Run your experiment
python experiment_unified.py --models "DyGFormer" --single_encoder "lete" --datasets Contacts --disable_progress_bar --num_runs 1 > current_thin/sh_logs/DyGFormer_lete_01_contacts.log 2>&1