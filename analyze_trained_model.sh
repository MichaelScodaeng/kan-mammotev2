#!/bin/bash
# Quick K-MOTE Gating Analysis for Your Trained Model
# Run this after your DyGMamba + UCI training is complete

echo "🔍 Analyzing K-MOTE gating weights for DyGMamba + UCI (already trained)"
echo "=================================================================="

# Make sure we're in the right environment
source ~/.bashrc
conda activate kan_mammote

# Run the K-MOTE gating analysis on your trained model
python kmote_gating_visualization.py \
    --model_name DyGMamba \
    --dataset_name uci \
    --time_encoder_type kan_mammote_dual_kmote \
    --sample_size 1000 \
    --seed 0 \
    --output_dir ./kmote_analysis_dygmamba_uci

echo ""
echo "✅ Analysis complete! Results saved to: ./kmote_analysis_dygmamba_uci/"
echo ""
echo "📊 Generated files:"
echo "   • PNG visualization: Shows gating weight patterns for both absolute and relative K-MOTE"
echo "   • Text report: Detailed statistics on expert utilization"
echo "   • PKL file: Complete analysis data"
echo ""
echo "🔍 What you'll see:"
echo "   • How much each expert (Spline, Fourier, Wavelet) is used"
echo "   • Temporal patterns in expert selection"
echo "   • Specialization of absolute vs relative K-MOTE components"