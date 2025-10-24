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
mkdir -p sh_scripts/thin/231068_sh/sh_logs/mercer/tcl
mkdir -p sh_scripts/thin/231068_sh/sh_logs/mercer/jodie
# Run your experiment - TCL light dataset + JODIE
python experiment_unified.py --models "TCL" --single_encoder "mercer" --datasets USLegis \
 --disable_progress_bar --num_runs 1 > sh_scripts/thin/231068_sh/sh_logs/mercer/tcl/mercer_tcl_03.log 2>&1

python experiment_unified.py --models "JODIE" --single_encoder "mercer" --datasets wikipedia \
 --disable_progress_bar --num_runs 1 > sh_scripts/thin/231068_sh/sh_logs/mercer/jodie/mercer_jodie_01.log 2>&1