#!/bin/bash
# Test hyperparameter tuning on a single dataset/model before full run

echo "=========================================="
echo "Testing KAN-MAMMOTE Hyperparameter Tuning"
echo "Testing on: wikipedia / TGAT"
echo "=========================================="

python tune_kan_mammote_fast.py \
    --datasets wikipedia \
    --models TGAT \
    --max_configs 2 \
    --output_dir ./hptune_test \
    --dry_run

echo ""
echo "=========================================="
echo "Test completed! Check ./hptune_test/"
echo "=========================================="
echo ""
echo "To submit test jobs:"
echo "  bash ./hptune_test/submit_all_jobs.sh"
