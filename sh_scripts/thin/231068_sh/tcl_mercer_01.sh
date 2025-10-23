#!/bin/bash
#PBS -j oe
#PBS -q GPU-1
#PBS -l select=1:ngpus=1
#PBS -M s2516027@jaist.ac.jp
#PBS -m be

cd "$PBS_O_WORKDIR"

source ~/.bashrc
module purge
module load cuda/12.8u1
conda activate kan_mammotev2
mkdir -p sh_scripts/thin/231068_sh/sh_logs/mercer/tcl
# Run your experiment - TCL heavier datasets
python experiment_unified.py --models "TCL" --single_encoder "mercer" --datasets Contacts Flights \
 --disable_progress_bar --num_runs 1 > sh_scripts/thin/231068_sh/sh_logs/mercer/tcl/mercer_tcl_01.log 2>&1