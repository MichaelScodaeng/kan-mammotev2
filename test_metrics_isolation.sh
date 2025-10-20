#!/bin/bash
#
# Quick test to verify metrics isolation fix works
#
set -e

echo "=========================================="
echo "TESTING METRICS ISOLATION FIX"
echo "=========================================="
echo ""
echo "This will run a 2-epoch test to verify that:"
echo "  1. saved_models uses suffix ✓"
echo "  2. saved_results uses suffix ✓"
echo "  3. saved_metrics uses suffix ✓ (NEWLY FIXED)"
echo ""
echo "Expected behavior:"
echo "  Baseline: saved_metrics/JODIE/lastfm/JODIE_kan_mammote_dual_kmote_seed0/"
echo "  Test:     saved_metrics/JODIE/lastfm/JODIE_kan_mammote_dual_kmote_seed0_test_isolation/"
echo ""
echo "Press Enter to continue, Ctrl+C to abort..."
read

echo "Running 2-epoch test experiment..."
python experiments/train_link_prediction.py \
  --model_name "JODIE" \
  --dataset_name "lastfm" \
  --time_encoder_type "kan_mammote_dual_kmote" \
  --num_runs 1 \
  --num_epochs 2 \
  --save_model_name_suffix "_test_isolation" \
  --disable_progress_bar

echo ""
echo "=========================================="
echo "VERIFICATION"
echo "=========================================="
echo ""
echo "Checking saved_models isolation:"
ls -d saved_models/JODIE/lastfm/*_test_isolation* 2>/dev/null && echo "✅ PASS" || echo "❌ FAIL"
echo ""
echo "Checking saved_results isolation:"
ls saved_results/JODIE/lastfm/*_test_isolation* 2>/dev/null && echo "✅ PASS" || echo "❌ FAIL"
echo ""
echo "Checking saved_metrics isolation (THE FIX):"
ls -d saved_metrics/JODIE/lastfm/*_test_isolation* 2>/dev/null && echo "✅ PASS - Fix works!" || echo "❌ FAIL - Fix didn't work"
echo ""
echo "Full directory listing:"
echo "saved_models/JODIE/lastfm/:"
ls -la saved_models/JODIE/lastfm/ | tail -5
echo ""
echo "saved_results/JODIE/lastfm/:"
ls -la saved_results/JODIE/lastfm/ | tail -5
echo ""
echo "saved_metrics/JODIE/lastfm/:"
ls -la saved_metrics/JODIE/lastfm/ | tail -5
echo ""
echo "=========================================="
echo "If all three show ✅ PASS, the fix is working!"
echo "=========================================="
