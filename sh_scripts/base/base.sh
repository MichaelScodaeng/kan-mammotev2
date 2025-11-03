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
python experiments/stackoverflow_badge_prediction.py > sh_scripts/base/kmm/sof.log 2>&1 
python experiments/event_based_mnist_experiment.py > sh_scripts/base/kmm/ebm.log 2>&1 

echo "All experiments finished."
