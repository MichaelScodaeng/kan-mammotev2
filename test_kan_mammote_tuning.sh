#!/bin/bash
"""
Quick Test Script for KAN-MAMMOTE Full Hyperparameter Tuning
=============================================================

This script runs a quick test of the hyperparameter tuning with minimal configurations
to verify everything works before running the full tuning.
"""

cd /home/s2516027/kan-mammotev3/kan-mammotev2

echo "🧪 Testing KAN-MAMMOTE Full Hyperparameter Tuning"
echo "📊 Running quick test with minimal configurations..."

# Test with very limited parameters for quick verification
python experiments/tune_kan_mammote_full.py \
    --tuning_mode quick \
    --epochs 5 \
    --split 1 \
    --data_dir "NeuralPointProcess-master/data/real/so" \
    --experiment_dir "test_kan_mammote_tuning" \
    --max_sequence_length 50 \
    --min_sequence_length 3 \
    --normalize_time \
    --use_proper_split \
    --val_ratio 0.3

echo "✅ Test completed!"

# Show results if available
if [ -f "test_kan_mammote_tuning/tuning_summary.txt" ]; then
    echo ""
    echo "📊 Test Results:"
    echo "==============="
    cat "test_kan_mammote_tuning/tuning_summary.txt"
fi