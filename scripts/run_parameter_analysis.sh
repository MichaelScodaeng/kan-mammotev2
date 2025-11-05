#!/bin/bash
#PBS -q GPU-1
#PBS -l select=1:ngpus=1
#PBS -l walltime=24:00:00
#PBS -N kan_mammote_param_analysis
#PBS -o parameter_analysis.o
#PBS -e parameter_analysis.e

# Load necessary modules
module load anaconda3/personal

# Navigate to project directory
cd /home/s2516027/kan-mammotev3/kan-mammotev2

# Activate conda environment if needed
# conda activate your_environment

# Run parameter analysis
echo "Starting KAN-MAMMOTE Parameter Analysis..."
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU Info:"
nvidia-smi

# Run the analysis
python experiments/kan_mammote_parameter_analysis_v2.py \
    --output_dir "results/kan_mammote_parameter_analysis_$(date +%Y%m%d_%H%M%S)"

echo "Analysis completed at: $(date)"