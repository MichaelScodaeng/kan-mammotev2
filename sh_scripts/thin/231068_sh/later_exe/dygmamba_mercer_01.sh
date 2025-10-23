#!/bin/bash
#PBS -j oe
#PBS -q GPU-I
#PBS -l select=1:ngpus=1
#PBS -M s2516027@jaist.ac.jp
#PBS -m be

cd "$PBS_O_WORKDIR"

source ~/.bashrc
module purge
module load cuda/12.8u1
conda activate kan_mammotev2
mkdir -p sh_scripts/thin/231068_sh/sh_logs/mercer/dygmamba
# Run your experiment - DyGMamba datasets
python experiment_unified.py --models "DyGMamba" --single_encoder "mercer" --datasets UNvote USLegis \
 --disable_progress_bar --num_runs 1 > sh_scripts/thin/231068_sh/sh_logs/mercer/dygmamba/mercer_dygmamba_01.log 2>&1