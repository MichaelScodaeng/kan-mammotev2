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
mkdir -p parameter_analysis_results

echo "=============================================="
echo "KAN-MAMMOTE Parameter Analysis Experiment"
echo "=============================================="
echo "Start time: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Run the parameter analysis
python experiments/kan_mammote_parameter_analysis.py \
    --output_dir parameter_analysis_results \
    2>&1 | tee parameter_analysis_results/experiment_log.txt

echo ""
echo "=============================================="
echo "Analysis Complete!"
echo "End time: $(date)"
echo "=============================================="

# Show results summary
if [ -f parameter_analysis_results/efficiency_report_*.txt ]; then
    echo ""
    echo "=== EFFICIENCY REPORT SUMMARY ==="
    tail -20 parameter_analysis_results/efficiency_report_*.txt
fi

echo ""
echo "Results saved in: parameter_analysis_results/"
echo "Check the following files:"
echo "  - parameter_analysis_*.json (complete results)"
echo "  - parameter_analysis_*.csv (analysis data)"
echo "  - parameter_analysis_plots_*.png (visualizations)"
echo "  - efficiency_report_*.txt (detailed analysis)"