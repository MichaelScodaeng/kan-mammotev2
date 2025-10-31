#!/bin/bash
#PBS -j oe
#PBS -q GPU-1
#PBS -l select=1:ngpus=1
#PBS -l walltime=24:00:00
#PBS -M s2516027@jaist.ac.jp
#PBS -m be

cd "$PBS_O_WORKDIR"

source ~/.bashrc
module purge
module load cuda/12.1
conda activate kan_mammote

# Consistent experiment dirs
EXP_ROOT="event_based_mnist_experiments"
mkdir -p "$EXP_ROOT/run_logs"

echo "🧪 Event-Based MNIST Time Encoder Comparison"
echo "============================================="

# Defaults
EPOCHS=200
BATCH_SIZE=512
EMBEDDING_DIM=32
ENCODERS="kan_mammote_dual_kmote kmote_abs_only kmote_rel_only"
RESUME_EXP=""

# Parse CLI
while [[ $# -gt 0 ]]; do
  case "$1" in
    --epochs)        EPOCHS="$2"; shift 2 ;;
    --encoders)
      shift
      ENCODERS=""
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        ENCODERS+="${ENCODERS:+ }$1"
        shift
      done
      ;;
    --batch_size)    BATCH_SIZE="$2"; shift 2 ;;
    --embedding_dim) EMBEDDING_DIM="$2"; shift 2 ;;
    --resume)        RESUME_EXP="$2"; shift 2 ;;
    --help|-h)
      echo "Usage: $0 [OPTIONS]"
      echo "  --epochs NUM"
      echo "  --encoders LIST            (space-separated)"
      echo "  --batch_size NUM"
      echo "  --embedding_dim NUM"
      echo "  --resume PATH              (experiment directory)"
      echo ""
      echo "Encoders:"
      echo "  lete | lete_relative | lstm_only | kan_mammote_full | kan_mammote_lite"
      echo "  sm_kernel_only | kmote_abs_only | kmote_rel_only | kan_mammote_dual_kmote"
      echo ""
      echo "Examples:"
      echo "  $0 --epochs 100 --encoders lete lete_relative"
      echo "  $0 --resume event_based_mnist_experiments/run_20251017_143022"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1 ;;
  esac
done

# Sanity: script presence
if [[ ! -f "event_based_mnist_experiment.py" ]]; then
  echo "❌ Error: event_based_mnist_experiment.py not found in current directory"
  echo "Please run this script from the experiments/ directory"
  exit 1
fi

# CUDA check (single call)
echo "🔍 Checking CUDA availability..."
python - <<'PY'
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
PY

# Resume mode
if [[ -n "$RESUME_EXP" ]]; then
  echo "🔄 Resume mode: $RESUME_EXP"
  if [[ ! -d "$RESUME_EXP" ]]; then
    echo "❌ Error: Experiment directory not found: $RESUME_EXP"
    exit 1
  fi
  echo "📁 Checking available checkpoints..."
  python event_based_mnist_experiment.py --resume_experiment "$RESUME_EXP"
  exit $?
fi

# Normal training
echo "⚙️ Configuration:"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Embedding Dim: $EMBEDDING_DIM"
echo "  Encoders: $ENCODERS"
echo ""
echo "🚀 Starting experiment..."
echo "💾 Checkpoints will be saved automatically every 10 epochs"
echo "🔄 Resume with: $0 --resume ${EXP_ROOT}/run_<timestamp>"
echo ""

python event_based_mnist_experiment.py \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --embedding_dim "$EMBEDDING_DIM" \
  --encoders $ENCODERS
exit_code=$?

if [[ $exit_code -eq 0 ]]; then
  echo ""
  echo "✅ Experiment completed successfully!"
  echo "📊 Results saved in ${EXP_ROOT}/run_<timestamp>/"
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
  echo "❌ Experiment failed with exit code: $exit_code"
  echo "💡 Check error messages above"
  echo "🔄 If partially completed, resume with:"
  echo "   $0 --resume ${EXP_ROOT}/run_<timestamp>"
fi

exit $exit_code
