#!/bin/bash
# K-MOTE Gating Analysis Runner
# 
# This script provides easy commands to analyze K-MOTE gating weights
# in KAN-MAMMOTE for your professor's visualization request.

echo "🔍 K-MOTE Gating Weight Analysis for KAN-MAMMOTE"
echo "=================================================="

# Function to run overall analysis
run_overall_analysis() {
    echo "Running overall K-MOTE gating analysis..."
    python kmote_gating_visualization.py \
        --model_name TCL \
        --dataset_name uci \
        --time_encoder_type kan_mammote_dual_kmote \
        --sample_size 1000 \
        --output_dir ./kmote_analysis_results
}

# Function to run node-level analysis
run_node_analysis() {
    echo "Running node-level K-MOTE analysis..."
    python node_level_kmote_analysis.py \
        --model_name TCL \
        --dataset_name uci \
        --time_encoder_type kan_mammote_dual_kmote \
        --auto_find_nodes \
        --min_interactions 15 \
        --max_nodes 3 \
        --output_dir ./node_analysis_results
}

# Function to analyze specific node
analyze_specific_node() {
    local node_id=$1
    echo "Analyzing specific node: $node_id"
    python node_level_kmote_analysis.py \
        --model_name TCL \
        --dataset_name uci \
        --time_encoder_type kan_mammote_dual_kmote \
        --node_id $node_id \
        --output_dir ./node_analysis_results
}

# Main menu
echo ""
echo "Choose analysis type:"
echo "1) Overall K-MOTE gating patterns (population-level)"
echo "2) Node-level K-MOTE analysis (individual nodes)"
echo "3) Analyze specific node (provide node ID)"
echo "4) Run both overall and node-level analysis"
echo ""

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        run_overall_analysis
        ;;
    2)
        run_node_analysis
        ;;
    3)
        read -p "Enter node ID to analyze: " node_id
        analyze_specific_node $node_id
        ;;
    4)
        echo "Running complete analysis..."
        run_overall_analysis
        echo ""
        run_node_analysis
        ;;
    *)
        echo "Invalid choice. Please run again with choice 1-4."
        exit 1
        ;;
esac

echo ""
echo "✅ Analysis complete! Check the output directories for results:"
echo "   📊 Overall analysis: ./kmote_analysis_results/"
echo "   🔍 Node analysis: ./node_analysis_results/"
echo ""
echo "Key outputs:"
echo "   • PNG visualizations showing gating weight patterns"
echo "   • Text reports with expert utilization statistics" 
echo "   • Pickle files with complete analysis data"
echo ""
echo "📖 Interpretation guide:"
echo "   • Spline expert: Smooth temporal trends"
echo "   • Fourier expert: Periodic/cyclical patterns"
echo "   • Wavelet expert: Sharp changes/anomalies"