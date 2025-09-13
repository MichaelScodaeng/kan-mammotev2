#!/bin/bash

# Debug KANMAMMOTE Performance Script
# This runs training with extensive debugging to identify performance issues

echo "🔍 Starting KANMAMMOTE Debugging Session..."
echo "=================================================="

# Set dataset and model
DATASET="wikipedia"
MODEL="JODIE"
TIME_ENCODER="KANMAMMOTE"

# Create debug log directory
mkdir -p experiment_logs_debug
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="experiment_logs_debug/debug_${MODEL}_${TIME_ENCODER}_${TIMESTAMP}.log"

echo "📝 Logging to: $LOG_FILE"

# Run training with logging
python -m DyGMamba.train_link_prediction \
    --dataset_name "$DATASET" \
    --model_name "$MODEL" \
    --time_encoder_type "$TIME_ENCODER" \
    --num_runs "5"\
    --load_best_configs \
    --seed 0 \
    --use_validation_and_test_neighbor_co_occurrence_constraint \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "🔍 Debugging Complete!"
echo "📊 Quick Analysis:"
echo "=================================="

# Extract key metrics from log
echo "📈 Final Validation ROC-AUC:"
grep "validate roc_auc" "$LOG_FILE" | tail -1

echo ""
echo "📉 Regularization Losses:"
grep -E "validate.*(sobolev|variation|balance)" "$LOG_FILE" | tail -3

echo ""
echo "🎯 KANMAMMOTE Summary Lines:"
grep "🎯 KANMAMMOTE" "$LOG_FILE" | tail -5

echo ""
echo "📝 Full log available at: $LOG_FILE"
echo "🔧 To analyze further:"
echo "   grep '🔍 KANMAMMOTE' \"$LOG_FILE\""
echo "   grep '🎯 KANMAMMOTE' \"$LOG_FILE\""
