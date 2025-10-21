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
mkdir -p tune_kmm/dgm
# Run your experiment
python -u tune_kan_mammote_direct.py --models DyGMamba > tune_kmm/dgm/tune_dgm.log 2>&1
