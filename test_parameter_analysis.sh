#!/bin/bash

# Quick test of KAN-MAMMOTE Parameter Analysis
echo "=== KAN-MAMMOTE Parameter Analysis Quick Test ==="

# Set up environment
source ~/.bashrc
conda activate kan_mammote

# Create test output directory
mkdir -p parameter_analysis_test

# Run quick test with only 3 configurations
python experiments/kan_mammote_parameter_analysis.py \
    --output_dir parameter_analysis_test \
    --quick_test \
    2>&1 | tee parameter_analysis_test/quick_test_log.txt

echo ""
echo "=== Quick Test Complete ==="
echo "Check results in: parameter_analysis_test/"