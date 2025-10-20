#!/bin/bash
# Quick script to generate and submit hyperparameter tuning jobs

set -e  # Exit on error

echo "=========================================="
echo "KAN-MAMMOTE Fast Hyperparameter Tuning"
echo "=========================================="
echo ""

# Default: Generate jobs for all datasets and models
python tune_kan_mammote_fast.py \
    --output_dir ./hptune_jobs \
    "$@"

echo ""
echo "=========================================="
echo "Jobs generated successfully!"
echo "=========================================="
