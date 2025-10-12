#!/bin/bash
#PBS -j oe
#PBS -q GPU-1A
#PBS -l select=1:ngpus=1
#PBS -M s2516027@jaist.ac.jp
#PBS -m be

# Load module system
source /etc/profile.d/modules.sh
module purge
module load cuda/12.1
module load singularity/3.9.5   # only if needed

# Change to the directory from which the job was submitted
cd "$PBS_O_WORKDIR"

# Activate conda environment (Bash)
# Make sure you have this in your ~/.bashrc:
#   eval "$(conda shell.bash hook)"
source ~/.bashrc
conda activate kan_mammote

# Check environment for logging
which python
python --version
nvidia-smi -a > nvidia-smi.log

# Debug checkpoint existence before running experiment  
echo "=== CHECKPOINT DEBUGGING ==="
echo "Working directory: $(pwd)"
echo "Looking for existing checkpoints..."
find ./saved_models -name "*kan_mammote_dual_kmote*" -name "*checkpoint*" -type f | head -5
echo "=============================="

# Run your experiment
python experiment_unified.py --models "TCL" --single_encoder "kan_mammote_dual_kmote" --disable_progress_bar --num_runs 1 > kanmammote_TCL.log 2>&1
