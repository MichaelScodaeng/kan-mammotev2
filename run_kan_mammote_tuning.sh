#!/bin/bash
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000
#PBS -l walltime=48:00:00
#PBS -q gpu
#PBS -o tuning_kan_mammote.out
#PBS -e tuning_kan_mammote.err

"""
KAN-MAMMOTE Full Hyperparameter Tuning Script
==============================================

This script runs comprehensive hyperparameter tuning for KAN-MAMMOTE Full encoder
with different tuning modes and configurations.

Usage:
  bash run_kan_mammote_tuning.sh [mode] [epochs] [split]
  
Modes:
  - quick: Fast exploration (few parameter combinations)
  - comprehensive: Exhaustive search (many parameter combinations)  
  - efficiency_focused: Focus on smaller models
  - training_focused: Focus on training hyperparameters
  
Examples:
  bash run_kan_mammote_tuning.sh quick 30 1
  bash run_kan_mammote_tuning.sh comprehensive 50 1
  bash run_kan_mammote_tuning.sh training_focused 40 1
"""

# Change to the correct directory
cd $PBS_O_WORKDIR
cd /home/s2516027/kan-mammotev3/kan-mammotev2

# Activate environment
module load anaconda3/personal
source activate mammote_exp

# Set default parameters
MODE=${1:-comprehensive}
EPOCHS=${2:-50}
SPLIT=${3:-1}

echo "🔧 Starting KAN-MAMMOTE Full Hyperparameter Tuning"
echo "📊 Mode: $MODE"
echo "⏰ Epochs per configuration: $EPOCHS"
echo "📁 Using data split: $SPLIT"
echo "📅 Start time: $(date)"

# Ensure data directory exists
DATA_DIR="NeuralPointProcess-master/data/real/so"
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Data directory not found: $DATA_DIR"
    exit 1
fi

# Run the tuning experiment
python experiments/tune_kan_mammote_full.py \
    --tuning_mode $MODE \
    --epochs $EPOCHS \
    --split $SPLIT \
    --data_dir $DATA_DIR \
    --experiment_dir "kan_mammote_tuning_${MODE}_split${SPLIT}" \
    --use_amp \
    --max_sequence_length 100 \
    --min_sequence_length 3 \
    --normalize_time \
    --use_proper_split \
    --val_ratio 0.3

echo "📅 End time: $(date)"
echo "✅ Tuning completed!"

# Show quick summary if results exist
RESULTS_DIR="kan_mammote_tuning_${MODE}_split${SPLIT}"
if [ -f "${RESULTS_DIR}/tuning_summary.txt" ]; then
    echo ""
    echo "📊 Quick Results Summary:"
    echo "========================"
    head -20 "${RESULTS_DIR}/tuning_summary.txt"
fi