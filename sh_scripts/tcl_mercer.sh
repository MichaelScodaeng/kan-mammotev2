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
mkdir -p sh_scripts/sh_logs/mercer
# Run your experiment
python experiment_unified.py --models "TCL" --single_encoder "mercer" --datasets Flights --disable_progress_bar --num_runs 1 > sh_scripts/sh_logs/mercer/mercer_TCL_01.log 2>&1
