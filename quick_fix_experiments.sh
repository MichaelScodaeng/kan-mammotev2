#!/bin/bash
#
# Quick Fix Experiments for KAN-MAMMOTE
# Run ONLY the 3 most critical tests to see if hyperparameter fixes work
#
# If these 3 work, KMM is salvageable. If not, stick with LeTE.
#

echo "=========================================="
echo "KAN-MAMMOTE Quick Fix Validation"
echo "=========================================="
echo ""
echo "Testing 3 critical datasets with optimized hyperparameters:"
echo "1. lastfm (worst failure: -10.17%)"
echo "2. uci (best success: +1.57%)"  
echo "3. mooc (moderate failure: -5.20%)"
echo ""
echo "Expected runtime: ~6-8 hours total"
echo "=========================================="
echo ""

# Experiment 1: Fix the WORST case (lastfm)
echo "[1/3] Testing lastfm with aggressive fixes..."
python experiment_unified.py \
  --models "JODIE" \
  --single_encoder "kan_mammote_dual_kmote" \
  --datasets lastfm \
  --learning_rate 0.00001 \
  --weight_decay 0.001 \
  --patience 40 \
  --num_runs 1 \
  --prefix "quickfix_lastfm"

echo ""
echo "lastfm complete. Check if Test AP improved from 0.6519 to ~0.72+"
echo ""

# Experiment 2: Validate the SUCCESS case still works (uci)
echo "[2/3] Testing uci to ensure fixes don't break what works..."
python experiment_unified.py \
  --models "TCL" \
  --single_encoder "kan_mammote_dual_kmote" \
  --datasets uci \
  --learning_rate 0.00005 \
  --weight_decay 0.0005 \
  --patience 30 \
  --num_runs 1 \
  --prefix "quickfix_uci"

echo ""
echo "uci complete. Check if Test AP maintained ~0.93+"
echo ""

# Experiment 3: Fix MODERATE case (mooc)
echo "[3/3] Testing mooc with moderate fixes..."
python experiment_unified.py \
  --models "JODIE" \
  --single_encoder "kan_mammote_dual_kmote" \
  --datasets mooc \
  --learning_rate 0.00003 \
  --weight_decay 0.0005 \
  --patience 30 \
  --num_runs 1 \
  --prefix "quickfix_mooc"

echo ""
echo "=========================================="
echo "Quick Fix Experiments Complete!"
echo "=========================================="
echo ""
echo "DECISION CRITERIA:"
echo "-------------------"
echo "✅ If lastfm improved by +5% → KMM is salvageable, continue tuning"
echo "✅ If mooc improved by +3% → Fixes are working"
echo "✅ If uci maintained performance → No regression"
echo ""
echo "❌ If no improvement → Use LeTE, don't waste more time"
echo ""
echo "Check results in: results/ or saved_results/"
echo "=========================================="
