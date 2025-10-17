#!/bin/bash

# Event-Based MNIST Time Encoder Experiment Runner
# =================================================

echo "🧪 Event-Based MNIST Time Encoder Comparison"
echo "============================================="

# Default configuration
EPOCHS=200
BATCH_SIZE=512
EMBEDDING_DIM=32
ENCODERS="lete lete_relative kan_mammote_full"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --encoders)
            shift
            ENCODERS=""
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                ENCODERS="$ENCODERS $1"
                shift
            done
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --embedding_dim)
            EMBEDDING_DIM="$2"
            shift 2
            ;;
        --resume)
            RESUME_EXP="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --epochs NUM          Number of training epochs (default: 200)"
            echo "  --encoders LIST       Space-separated list of encoders to test"
            echo "  --batch_size NUM      Batch size (default: 512)"
            echo "  --embedding_dim NUM   Embedding dimension (default: 32)"
            echo "  --resume PATH         Resume from experiment directory"
            echo "  --help, -h           Show this help message"
            echo ""
            echo "Available encoders:"
            echo "  lete                 LeTE (absolute time)"
            echo "  lete_relative        LeTE (relative time differences)"
            echo "  lstm_only            LSTM baseline (no time encoding)"
            echo "  kan_mammote_full     Full KAN-MAMMOTE"
            echo "  kan_mammote_lite     Lite KAN-MAMMOTE"
            echo "  sm_kernel_only       SM-Kernel only (ablation)"
            echo "  kmote_abs_only       K-MOTE absolute only (ablation)"
            echo "  kmote_rel_only       K-MOTE relative only (ablation)"
            echo ""
            echo "Examples:"
            echo "  $0 --epochs 100 --encoders lete lete_relative"
            echo "  $0 --resume mnist_experiments/run_20251017_143022"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if Python script exists
if [[ ! -f "event_based_mnist_experiment.py" ]]; then
    echo "❌ Error: event_based_mnist_experiment.py not found in current directory"
    echo "Please run this script from the experiments/ directory"
    exit 1
fi

# Resume mode
if [[ -n "$RESUME_EXP" ]]; then
    echo "🔄 Resume mode: $RESUME_EXP"
    if [[ ! -d "$RESUME_EXP" ]]; then
        echo "❌ Error: Experiment directory not found: $RESUME_EXP"
        exit 1
    fi
    
    echo "📁 Checking available checkpoints..."
    python event_based_mnist_experiment.py --resume_experiment "$RESUME_EXP"
    exit 0
fi

# Normal training mode
echo "⚙️ Configuration:"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Embedding Dim: $EMBEDDING_DIM"
echo "  Encoders: $ENCODERS"
echo ""

# Check if CUDA is available
echo "🔍 Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "✅ CUDA is available - training will use GPU acceleration"
else
    echo "⚠️  CUDA not available - training will use CPU (slower)"
fi

echo ""
echo "🚀 Starting experiment..."
echo "💾 Checkpoints will be saved automatically every 10 epochs"
echo "🔄 If interrupted, resume with: $0 --resume <experiment_dir>"
echo ""

# Run the experiment
python event_based_mnist_experiment.py \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --embedding_dim "$EMBEDDING_DIM" \
    --encoders $ENCODERS

# Check if experiment succeeded
if [[ $? -eq 0 ]]; then
    echo ""
    echo "✅ Experiment completed successfully!"
    echo "📊 Results saved in mnist_experiments/run_<timestamp>/"
    echo ""
    echo "📁 Output files:"
    echo "  - models/: Best trained models"
    echo "  - checkpoints/: Training checkpoints for resuming"
    echo "  - epoch_history/: Detailed training logs"
    echo "  - results.json: Complete experiment results"
    echo "  - results.csv: Summary table"
    echo "  - curves.png: Training curves plot"
else
    echo ""
    echo "❌ Experiment failed with exit code: $?"
    echo "💡 Check error messages above"
    echo "🔄 If partially completed, you can resume with:"
    echo "   $0 --resume mnist_experiments/run_<timestamp>"
fi