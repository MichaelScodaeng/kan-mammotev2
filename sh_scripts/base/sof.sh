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
mkdir -p sh_scripts/base/kmm
# Run your experiment
python experiments/stackoverflow_badge_prediction.py --encoders k_mote_rel k_mote_abs > sh_scripts/base/kmm/sof03.log 2>&1 

echo "All experiments finished."
