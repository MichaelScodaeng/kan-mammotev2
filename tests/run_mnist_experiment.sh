#!/bin/bash
# Quick start guide for Event-Based MNIST experiments

echo "================================"
echo "Event-Based MNIST Experiment"
echo "Quick Start Guide"
echo "================================"
echo ""

echo "📋 Available Encoders:"
echo "  - lstm_only          : Plain LSTM baseline (no time encoding)"
echo "  - lete               : LeTE (Fourier time encoding)"
echo "  - mercer             : Mercer kernel expansion"
echo "  - bochner            : Bochner (Gaussian Fourier features)"
echo "  - sm_kernel_only     : SM-Kernel only (ablation)"
echo "  - kmote_abs_only     : K-MOTE absolute only (ablation)"
echo "  - kmote_rel_only     : K-MOTE relative only (ablation)"
echo "  - dual_stream_baseline : K-MOTE + SM-Kernel (ablation)"
echo "  - kan_mammote_lite   : KAN-MAMMOTE Lite (production)"
echo "  - kan_mammote_full   : Full KAN-MAMMOTE with Mamba"
echo ""

echo "🚀 Example Commands:"
echo ""

echo "1. Quick test (2 encoders, 10 epochs):"
echo "   python event_based_mnist_experiment.py --encoders lstm_only lete --epochs 10 --batch_size 256"
echo ""

echo "2. Paper comparison (LSTM vs LeTE vs KAN-MAMMOTE):"
echo "   python event_based_mnist_experiment.py --encoders lstm_only lete kan_mammote_full --epochs 50"
echo ""

echo "3. Full ablation study (all encoders):"
echo "   python event_based_mnist_experiment.py --epochs 50"
echo ""

echo "4. Custom configuration:"
echo "   python event_based_mnist_experiment.py \\"
echo "       --encoders lstm_only lete \\"
echo "       --epochs 100 \\"
echo "       --batch_size 512 \\"
echo "       --embedding_dim 64 \\"
echo "       --hidden_dim 256 \\"
echo "       --threshold 0.9 \\"
echo "       --max_events None"
echo ""

echo "📊 Output Files (timestamped):"
echo "  - results_TIMESTAMP.json         : Full experiment data"
echo "  - results_TIMESTAMP.csv          : Summary table"
echo "  - results_TIMESTAMP_curves.png   : Training curves plot"
echo "  - results_TIMESTAMP_epoch_history/ : Detailed epoch CSVs"
echo ""

echo "✅ All set! Run your experiments."
