#!/bin/bash
#PBS -j oe
#PBS -q GPU-LA
#PBS -l select=1:ngpus=2
#PBS -M s2516027@jaist.ac.jp
#PBS -m be

cd "$PBS_O_WORKDIR"

source ~/.bashrc
module purge
module load cuda/12.8u1
conda activate kan_mammote
mkdir -p sh_scripts/thiti/sh_logs/mercer/dygmamba
# Task assignments - Heaviest combinations (DyGMamba + heaviest datasets)
# Task 1 on GPU 0
CUDA_VISIBLE_DEVICES=0 \
python experiment_unified.py --single_encoder lete --models DyGMamba \
 --datasets Flights --disable_progress_bar --num_runs 1 \
 > sh_scripts/thiti/sh_logs/mercer/dygmamba/dgm_lete_05a.log 2>&1 &


# Task 2 on GPU 1
CUDA_VISIBLE_DEVICES=1 \
python experiment_unified.py --single_encoder lete --models DyGMamba \
 --datasets lastfm --disable_progress_bar --num_runs 1 \
 > sh_scripts/thiti/sh_logs/mercer/dygmamba/dgm_lete_05b.log 2>&1 &


wait