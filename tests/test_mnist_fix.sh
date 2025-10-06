#!/bin/bash
# Quick test to verify the Event-Based MNIST fixes work

echo "🧪 Testing Event-Based MNIST Experiment Fixes"
echo "=============================================="
echo ""
echo "This will run a quick 10-epoch test on sm_kernel_only encoder"
echo "to verify the learning issue is fixed."
echo ""

python event_based_mnist_experiment.py \
    --encoders sm_kernel_only \
    --epochs 10 \
    --batch_size 128 \
    --max_events 50 \
    --threshold 0.9 \
    --save_results test_sm_kernel_results.json

echo ""
echo "=============================================="
echo "✅ Test complete! Check the output above."
echo "If validation accuracy improves beyond 11.35%, the fix works!"
