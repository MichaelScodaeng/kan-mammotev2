#!/bin/bash
#PBS -j oe
#PBS -q GPU-1A
#PBS -l select=1:ngpus=1
#PBS -M s2516027@jaist.ac.jp
#PBS -m be

source /etc/profile.d/modules.sh
module purge
module load cuda/12.1
module load singularity/3.9.5   # only if needed

cd ${PBS_O_WORKDIR}
source ~/.bashrc
conda activate kan_mammote


nvidia-smi -a > nvidia-smi.log
# Run your experiment
python experiment_unified.py --single_encoder "kan_mammote" --disable_progress_bar > experiment_kan_mammote_a100_02.log 2>&1