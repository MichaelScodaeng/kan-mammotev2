#!/bin/bash

# Quick KAN-MAMMOTE Hyperparameter Test
# Test the optimization on a single dataset-model combination

echo "🧪 Testing KAN-MAMMOTE Hyperparameter Optimization..."
echo "=================================================="

# Install Optuna if needed
echo "📦 Installing/checking Optuna..."
pip install optuna plotly kaleido

# Create directories
mkdir -p hyperparameter_results

# Test configuration
DATASET="wikipedia"  # Start with wikipedia (good performance baseline)
MODEL="TGAT"        # Start with TGAT (simpler than DyGMamba)
N_TRIALS=20         # Small number for testing
TIMEOUT=1800        # 30 minutes

echo ""
echo "🔍 Testing optimization:"
echo "   Dataset: $DATASET"
echo "   Model: $MODEL"
echo "   Trials: $N_TRIALS"
echo "   Timeout: ${TIMEOUT}s (30 min)"
echo ""

# Run single optimization
python hyperparameter_search_kanmammote.py \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --n_trials $N_TRIALS \
    --timeout $TIMEOUT \
    --seed 42

echo ""
echo "🎯 Test optimization completed!"
echo "Check hyperparameter_results/ for results"
