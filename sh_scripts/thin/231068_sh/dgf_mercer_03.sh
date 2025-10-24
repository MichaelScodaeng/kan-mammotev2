#!/bin/bash
#PBS -j oe
#PBS -q GPU-S
#PBS -l select=1:ngpus=2
#PBS -M s2516027@jaist.ac.jp
#PBS -m be

cd "$PBS_O_WORKDIR"

source ~/.bashrc
module purge
module load cuda/12.8u1
conda activate kan_mammote
mkdir -p sh_scripts/thin/231068_sh/sh_logs/mercer/dgf
# Task assignments - DyGFormer medium datasets
# Task 1 on GPU 0
CUDA_VISIBLE_DEVICES=0 \
python experiment_unified.py --models "DyGFormer" --single_encoder "mercer" --datasets enron \
 --disable_progress_bar --num_runs 1 > sh_scripts/thin/231068_sh/sh_logs/mercer/dgf/mercer_dgf_03-01.log 2>&1 &

# Task 2 on GPU 1
CUDA_VISIBLE_DEVICES=1 \
python experiment_unified.py --models "DyGFormer" --single_encoder "mercer" --datasets UNtrade \
 --disable_progress_bar --num_runs 1 > sh_scripts/thin/231068_sh/sh_logs/mercer/dgf/mercer_dgf_03-02.log 2>&1 &

wait