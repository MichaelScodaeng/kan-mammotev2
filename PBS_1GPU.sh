#!/bin/csh

#################################################################
# A40 1GPU Job Script for HPC System "KAGAYAKI" 
#                                       2022.3.3 k-miya
#################################################################
#PBS -N gpu
#PBS -j oe
#PBS -q GPU-1
#PBS -l select=1:ngpus=1
#PBS -M s2516027@jaist.ac.jp -m be
#PBS -m be

# Load module system
source /etc/profile.d/modules.sh
module purge
module load cuda/12.1
module load singularity/3.9.5   # only if needed

cd ${PBS_O_WORKDIR}

source ~/.bashrc
conda activate kan_mammote

nvidia-smi -a > nvidia-smi.log
# Run your experiment
python experiment_no.py > experiment_no_1.log 2>&1