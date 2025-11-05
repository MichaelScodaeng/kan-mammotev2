#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_model=a100
#PBS -N kan_mammote_tuning
#PBS -q gpu

# KAN-MAMMOTE Full Hyperparameter Tuning Job Script
# This script runs comprehensive hyperparameter tuning for kan_mammote_full

# Load modules and activate environment
module load lang/miniconda3/4.12.0
source activate /home/s2516027/miniconda3/envs/experimental

# Change to project directory
cd /home/s2516027/kan-mammotev3/kan-mammotev2

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=0

# Print system info
echo "🖥️ Starting KAN-MAMMOTE Full hyperparameter tuning"
echo "📅 Date: $(date)"
echo "🔧 GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
echo "💾 Available GPU memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)"
echo ""

# Function to run tuning with different modes
run_tuning() {
    local mode=$1
    local epochs=$2
    local description=$3
    
    echo "🚀 Running $description"
    echo "   Mode: $mode"
    echo "   Epochs per config: $epochs"
    echo ""
    
    python experiments/tune_kan_mammote_full.py \
        --tuning_mode $mode \
        --epochs $epochs \
        --batch_size 128 \
        --split 1 \
        --use_amp \
        --experiment_dir "kan_mammote_tuning_${mode}" \
        --max_sequence_length 100 \
        --normalize_time \
        --use_proper_split \
        --val_ratio 0.3
    
    echo "✅ Completed $description"
    echo ""
}

# Choose tuning strategy based on argument
TUNING_STRATEGY=${1:-"comprehensive"}

case $TUNING_STRATEGY in
    "quick")
        echo "🏃 Running QUICK tuning (fast exploration)"
        run_tuning "quick" 30 "Quick Tuning (3x2x2x2x2 = 48 configs)"
        ;;
    
    "efficiency")
        echo "⚡ Running EFFICIENCY-FOCUSED tuning (small models)"
        run_tuning "efficiency_focused" 50 "Efficiency-Focused Tuning (smaller models)"
        ;;
    
    "comprehensive")
        echo "🔬 Running COMPREHENSIVE tuning (thorough exploration)"
        run_tuning "comprehensive" 50 "Comprehensive Tuning (4x4x3x3x3 = ~320 configs)"
        ;;
    
    "all")
        echo "🌟 Running ALL tuning modes sequentially"
        run_tuning "quick" 30 "Quick Tuning Phase"
        run_tuning "efficiency_focused" 40 "Efficiency-Focused Phase"
        run_tuning "comprehensive" 50 "Comprehensive Tuning Phase"
        ;;
    
    *)
        echo "❌ Unknown tuning strategy: $TUNING_STRATEGY"
        echo "Available strategies: quick, efficiency, comprehensive, all"
        exit 1
        ;;
esac

echo "🎉 All tuning completed!"
echo "📊 Check the results in the respective tuning directories"

# Display summary of results
echo ""
echo "📋 RESULTS SUMMARY:"
for dir in kan_mammote_tuning_*/; do
    if [ -d "$dir" ]; then
        echo "📁 $dir"
        if [ -f "${dir}/tuning_summary.txt" ]; then
            echo "   $(head -n 10 "${dir}/tuning_summary.txt" | grep "Best Val MRR" || echo "   Results processing...")"
        fi
    fi
done