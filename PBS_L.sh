#!/bin/bash
#PBS -j oe
#PBS -q GPU-L
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
python experiment_bochner.py --single_encoder "bochner" --num_epochs 1000 > experiment_bochn_L_01.log 2>&1