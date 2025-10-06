#!/bin/bash
# Quick test script to verify KAN-MAMMOTE warm-up implementation

echo "=========================================="
echo "Testing KAN-MAMMOTE Warm-up Implementation"
echo "=========================================="
echo ""

echo "1. Testing basic warm-up effect..."
python test_warmup_effect.py 2>&1 | tail -30
echo ""

echo "=========================================="
echo "2. Testing KAN-MAMMOTE warm-up..."
python test_kan_mammote_warmup.py
echo ""

echo "=========================================="
echo "✅ All tests complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run your training script - warm-up will happen automatically"
echo "2. Check logs for the warm-up message before training starts"
echo "3. Enjoy faster training! 🚀"
