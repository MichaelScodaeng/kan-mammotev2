#!/bin/bash
#PBS -j oe
#PBS -q GPU-1A
#PBS -l select=1:ngpus=1
#PBS -M s2516027@jaist.ac.jp
#PBS -m be
#PBS -l walltime=48:00:00

# HPC Job Script for Time Encoder Experiments
# Usage: qsub -v TIME_ENCODER=kan_mammote run_time_encoder_experiment.sh

# Load module system
source /etc/profile.d/modules.sh
module purge
module load cuda/12.1
module load singularity/3.9.5   # only if needed

# Change to the directory from which the job was submitted
cd "$PBS_O_WORKDIR"

# Activate conda environment
source ~/.bashrc
conda activate kan_mammote

# Set default time encoder if not provided
if [ -z "$TIME_ENCODER" ]; then
    echo "Warning: TIME_ENCODER not set. Using 'original' as default."
    TIME_ENCODER="original"
fi

echo "=== Time Encoder Experiment ==="
echo "Time Encoder: $TIME_ENCODER"
echo "Job ID: $PBS_JOBID"
echo "Node: $(hostname)"
echo "Working Directory: $(pwd)"
echo "Start Time: $(date)"
echo "================================"

# Check environment for logging
which python
python --version
nvidia-smi -a > nvidia-smi_${TIME_ENCODER}_${PBS_JOBID}.log

# Run the experiment for the specific time encoder
python experiment_kanmammote.py \
    --single_encoder "$TIME_ENCODER" \
    --timeout_hours 24 \
    --num_runs 5 \
    > experiment_${TIME_ENCODER}_${PBS_JOBID}.log 2>&1

# Check exit status
if [ $? -eq 0 ]; then
    echo "Experiment completed successfully for $TIME_ENCODER"
else
    echo "Experiment failed for $TIME_ENCODER"
fi

echo "End Time: $(date)"

# Generate a final report for this encoder
python experiment_kanmammote.py \
    --single_encoder "$TIME_ENCODER" \
    --generate_report \
    >> experiment_${TIME_ENCODER}_${PBS_JOBID}.log 2>&1

echo "Job completed for TIME_ENCODER=$TIME_ENCODER"
