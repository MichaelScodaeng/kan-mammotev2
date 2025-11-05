#!/bin/bash
#PBS -j oe
#PBS -q GPU-1
#PBS -l select=1:ngpus=1
#PBS -M s2516027@jaist.ac.jp
#PBS -m be

cd "$PBS_O_WORKDIR"

source ~/.bashrc
module purge
module load cuda/12.8u1
conda activate kan_mammote

# Create output directory
mkdir -p kan_mammote_parameter_analysis

echo "=============================================="
echo "KAN-MAMMOTE Parameter Analysis (Tables + Plots)"
echo "=============================================="
echo "Start time: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Run the enhanced parameter analysis with both tables and plots
python experiments/kan_mammote_parameter_analysis_v2.py \
    --output_dir kan_mammote_parameter_analysis \
    2>&1 | tee kan_mammote_parameter_analysis/full_experiment_log.txt

echo ""
echo "=============================================="
echo "Analysis Complete!"
echo "End time: $(date)"
echo "=============================================="

# Show results summary
echo ""
echo "=== GENERATED OUTPUTS ==="
echo "📋 Tables:"
if [ -d kan_mammote_parameter_analysis/analysis ]; then
    ls -la kan_mammote_parameter_analysis/analysis/
fi

echo ""
echo "📈 Plots:"
if [ -d kan_mammote_parameter_analysis/plots ]; then
    ls -la kan_mammote_parameter_analysis/plots/
fi

echo ""
echo "=== QUICK INSIGHTS ==="
if [ -f kan_mammote_parameter_analysis/analysis/summary_insights.txt ]; then
    echo "Top findings:"
    head -20 kan_mammote_parameter_analysis/analysis/summary_insights.txt
fi

echo ""
echo "=============================================="
echo "✅ Complete Results Available At:"
echo "   📋 Tables: kan_mammote_parameter_analysis/analysis/"
echo "   📈 Plots: kan_mammote_parameter_analysis/plots/"
echo "   📄 Full log: kan_mammote_parameter_analysis/full_experiment_log.txt"
echo "=============================================="