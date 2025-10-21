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
mkdir -p tune_kmm/tcl
# Run your experiment
python tune_kan_mammote_direct.py --models TCL > tune_kmm/tcl/tune_tcl.log 2>&1
