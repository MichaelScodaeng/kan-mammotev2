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
mkdir -p tune_kmm/jodie
# Run your experiment
python tune_kan_mammote_direct.py --models JODIE > tune_kmm/jodie/tune_jodie.log 2>&1
