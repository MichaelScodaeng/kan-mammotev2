#!/bin/bash
# Simple test script to verify checkpoint resuming works

echo "=== CHECKPOINT RESUMING TEST ==="
echo "Working directory: $(pwd)"

# Check if checkpoint exists
CHECKPOINT_PATH="./saved_models/TCL/lastfm/TCL_mercer_seed0/checkpoint_epoch_50.pth"
echo "Checking checkpoint: $CHECKPOINT_PATH"

if [ -f "$CHECKPOINT_PATH" ]; then
    echo "✅ Checkpoint file exists"
    echo "File size: $(ls -lh "$CHECKPOINT_PATH" | awk '{print $5}')"
    
    # Test the training script with checkpoint resuming
    echo ""
    echo "🧪 Testing checkpoint resuming..."
    python experiments/train_link_prediction.py \
        --model_name TCL \
        --dataset_name lastfm \
        --time_encoder_type mercer \
        --num_runs 1 \
        --num_epochs 55 \
        --load_best_configs \
        --resume_from_checkpoint "$CHECKPOINT_PATH" \
        --save_checkpoints \
        --checkpoint_strategy smart \
        --max_checkpoints_to_keep 3 \
        --validate_checkpoints \
        --disable_progress_bar &
    
    # Let it run for a few seconds then check if checkpoint still exists
    sleep 10
    
    if [ -f "$CHECKPOINT_PATH" ]; then
        echo "✅ SUCCESS: Checkpoint preserved during resuming!"
    else
        echo "❌ FAILED: Checkpoint was deleted during resuming!"
    fi
    
    # Kill the training process
    pkill -f "train_link_prediction.py"
    
else
    echo "❌ Checkpoint file not found - test cannot proceed"
    echo "Available checkpoints:"
    find ./saved_models -name "*checkpoint*.pth" | head -5
fi

echo "=== TEST COMPLETED ==="