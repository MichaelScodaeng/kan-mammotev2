#!/bin/bash
#PBS -j oe
#PBS -q GPU-1A
#PBS -l select=1:ngpus=1
#PBS -M s2516027@jaist.ac.jp
#PBS -m be

cd "$PBS_O_WORKDIR"

source ~/.bashrc
module purge
module load cuda/12.8u1
conda activate kan_mammote

# Run your experiment
python experiment_unified.py --models "TGAT" --single_encoder "lete" --datasets Flights USLegis --disable_progress_bar --num_runs 1 > current_thiti/sh_logs/TGAT_lete_04_flights_uslegis.log 2>&1