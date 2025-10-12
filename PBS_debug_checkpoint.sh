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

# Change to the directory from which the job was submitted
cd "$PBS_O_WORKDIR"

# Activate conda environment
source ~/.bashrc
conda activate kan_mammote

# Check environment for logging
which python
python --version
nvidia-smi -a > nvidia-smi.log

# Debug checkpoint paths before running experiment
echo "=== CHECKPOINT PATH DEBUGGING ==="
echo "Working directory: $(pwd)"
echo "Checking for TCL mercer checkpoints:"
find ./saved_models -name "*mercer*" -name "*checkpoint*" -type f 2>/dev/null || echo "No checkpoints found"
echo "Checking specific path:"
ls -la "./saved_models/TCL/lastfm/TCL_mercer_seed0/" 2>/dev/null || echo "Directory not found"
echo "=================================="

# Run experiment with mercer encoder and only specific model/dataset for testing
python experiment_unified.py \
  --models "TCL" \
  --datasets "lastfm" \
  --single_encoder "mercer" \
  --disable_progress_bar \
  --num_runs 1 \
  --resume_only > debug_mercer_checkpoint.log 2>&1

echo "=== EXPERIMENT COMPLETED ==="
echo "Log file contents:"
tail -20 debug_mercer_checkpoint.log