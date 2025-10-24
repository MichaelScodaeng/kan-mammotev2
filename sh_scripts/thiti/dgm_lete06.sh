#!/bin/bash
#PBS -j oe
#PBS -q GPU-S
#PBS -l select=1:ngpus=1
#PBS -M s2516027@jaist.ac.jp
#PBS -m be

cd "$PBS_O_WORKDIR"

source ~/.bashrc
module purge
module load cuda/12.8u1
conda activate kan_mammotev2
mkdir -p sh_scripts/thiti/sh_logs/mercer/dygmamba
# Run your experiment - DyGMamba with lete encoder
python experiment_unified.py --single_encoder lete --models DyGMamba \
 --datasets SocialEvo --disable_progress_bar --num_runs 1 \
 > sh_scripts/thiti/sh_logs/mercer/dygmamba/dgm_lete_06.log 2>&1
