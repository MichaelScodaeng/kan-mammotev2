#!/bin/bash
# Submit all optimized KAN-MAMMOTE training jobs
# Based on best Optuna hyperparameters from November 2025

echo "🚀 Submitting KAN-MAMMOTE Training Jobs with Optimal Hyperparameters"
echo "============================================================="

# TCL Models (3 datasets)
echo "📊 Submitting TCL jobs..."
echo "  → Reddit TCL (AP=0.9738, trial 21)"
qsub sh_scripts/kmm_211168/kmm_tcl_reddit.sh

echo "  → UNvote TCL (AP=0.5256, trial 25)" 
qsub sh_scripts/kmm_211168/kmm_tcl_unvote.sh

echo "  → Wikipedia TCL (AP=0.9699, trial 3)"
qsub sh_scripts/kmm_211168/kmm_tcl_wikipedia.sh

# TGN Models (1 dataset - needs investigation)
echo "🔍 Submitting TGN jobs..."
echo "  → Wikipedia TGN (AP=0.0, trial 0) - NEEDS DEBUGGING"
qsub sh_scripts/kmm_211168/kmm_tgn_wikipedia.sh

echo ""
echo "✅ All jobs submitted!"
echo "📝 Monitor logs in: sh_scripts/kmm_211168/sh_logs/"
echo "🔧 Note: Wikipedia TGN had validation AP=0.0 - investigate if needed"
echo ""
echo "📊 Expected performance:"
echo "  - Reddit TCL: ~97.38% AP"
echo "  - Wikipedia TCL: ~96.99% AP" 
echo "  - UNvote TCL: ~52.56% AP"
echo "  - Wikipedia TGN: Needs debugging (0% AP)"