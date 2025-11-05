#!/bin/bash

echo "=== KAN-MAMMOTE Parameter Analysis Quick Test (Tables + Plots) ==="
echo "Testing with 3 configurations to verify both table and plot generation..."

# Set up environment
source ~/.bashrc
conda activate kan_mammote

# Create test output directory
mkdir -p parameter_test_combined

# Run quick test with both tables and plots
python experiments/kan_mammote_parameter_analysis_v2.py \
    --test \
    --output_dir parameter_test_combined \
    2>&1 | tee parameter_test_combined/test_log.txt

echo ""
echo "=== Quick Test Results ==="
echo "📋 Tables generated:"
if [ -d parameter_test_combined/analysis ]; then
    ls -la parameter_test_combined/analysis/
else
    echo "No tables directory found"
fi

echo ""
echo "📈 Plots generated:"
if [ -d parameter_test_combined/plots ]; then
    ls -la parameter_test_combined/plots/
else
    echo "No plots directory found"
fi

echo ""
echo "=== Test Summary ==="
if [ -f parameter_test_combined/analysis/summary_insights.txt ]; then
    echo "✅ Tables successfully generated"
    head -10 parameter_test_combined/analysis/summary_insights.txt
else
    echo "❌ Tables generation failed"
fi

if [ -f parameter_test_combined/plots/efficiency_analysis_*.png ]; then
    echo "✅ Plots successfully generated"
    echo "Generated plot files:"
    ls parameter_test_combined/plots/*.png
else
    echo "❌ Plot generation failed"
fi

echo ""
echo "=== Full Test Complete ==="
echo "Check results in: parameter_test_combined/"
echo "  📋 analysis/ - CSV tables and LaTeX files"
echo "  📈 plots/ - PNG visualization files"