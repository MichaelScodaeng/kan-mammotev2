# Event-Based MNIST Time Encoder Experiment Guide

## Overview

This experiment compares different time encoders on Event-Based MNIST classification, including new variants for **absolute vs. relative time** encoding.

## Available Time Encoders

### Baseline Encoders
- `lstm_only` - Plain LSTM with embedding lookup (no time encoding)

### LeTE Variants
- `lete` - Original LeTE with **absolute** pixel positions
- `lete_relative` - LeTE with **relative** position differences

### Time2Vec Variants  
- `time2vec` - Time2Vec with **absolute** pixel positions
- `time2vec_relative` - Time2Vec with **relative** position differences

### Mercer Variants
- `mercer` - Mercer kernel with **absolute** pixel positions  
- `mercer_relative` - Mercer kernel with **relative** position differences

### Bochner (if available)
- `bochner` - Bochner features with **absolute** pixel positions

### KAN-MAMMOTE Variants
- `kan_mammote_lite` - Lightweight KAN-MAMMOTE (uses both abs + rel time)
- `kan_mammote_full` - Full KAN-MAMMOTE architecture
- `kan_mammote_dual_kmote` - Dual K-MOTE variant

### Ablation Studies  
- `sm_kernel_only` - SM-Kernel component only
- `kmote_abs_only` - K-MOTE with absolute time only
- `kmote_rel_only` - K-MOTE with relative time only
- `dual_stream_baseline` - Simple dual-stream baseline

## Quick Start

### 1. Run All Available Encoders (Full Comparison)
```bash
# Run comprehensive comparison with default settings
python event_based_mnist_experiment.py

# Or use the runner script:
./run_mnist_experiment.sh
```

### 2. Test Specific Encoders
```bash
# Test absolute vs relative variants
python event_based_mnist_experiment.py --encoders lete lete_relative time2vec time2vec_relative mercer mercer_relative

# Test KAN-MAMMOTE variants only
python event_based_mnist_experiment.py --encoders kan_mammote_lite kan_mammote_full kan_mammote_dual_kmote

# Quick test with fewer epochs
python event_based_mnist_experiment.py --epochs 50 --encoders lete kan_mammote_lite
```

### 3. Custom Configuration
```bash
python event_based_mnist_experiment.py \
    --epochs 200 \
    --batch_size 512 \
    --embedding_dim 32 \
    --encoders lete lete_relative time2vec time2vec_relative \
    --experiment_dir my_experiments
```

## Checkpoint System

### Resume Training from Latest Checkpoint
```bash
# Resume all encoders from latest experiment
python event_based_mnist_experiment.py --resume_training

# Resume from specific experiment directory
python event_based_mnist_experiment.py --resume_experiment mnist_experiments/run_20251018_143022

# Resume specific encoder only
python event_based_mnist_experiment.py \
    --resume_experiment mnist_experiments/run_20251018_143022 \
    --resume_encoder kan_mammote_full \
    --additional_epochs 50
```

### Manual Checkpoint Management
```bash
# Disable automatic checkpointing (not recommended for long training)
python event_based_mnist_experiment.py --no_checkpoints

# Change checkpoint frequency (default: every 10 epochs)
python event_based_mnist_experiment.py --checkpoint_every 5
```

## Output Structure

Each experiment creates a timestamped directory:
```
mnist_experiments/
├── run_20251018_143022/
│   ├── mnist_time_encoder_results.json     # Complete results + metadata
│   ├── mnist_time_encoder_results.csv      # Summary table
│   ├── mnist_time_encoder_results_curves.png # Training curves plot
│   ├── models/                             # Saved model checkpoints
│   │   ├── best_lete_model.pth
│   │   ├── best_kan_mammote_full_model.pth
│   │   └── ...
│   ├── epoch_history/                      # Per-encoder training history
│   │   ├── lete_epoch_history.csv
│   │   ├── kan_mammote_full_epoch_history.csv
│   │   └── ...
│   └── checkpoints/                        # Training checkpoints for resume
│       ├── checkpoint_lete_latest.pth
│       ├── checkpoint_kan_mammote_full_epoch_100.pth
│       └── ...
```

## Key Differences: Absolute vs. Relative Time

### **Absolute Time Encoders** (`lete`, `time2vec`, `mercer`)
- Input: Raw pixel positions [45, 123, 200, 456, 783]
- Learn patterns from absolute spatial positions

### **Relative Time Encoders** (`lete_relative`, `time2vec_relative`, `mercer_relative`)  
- Input: Position differences [45, 78, 77, 256, 327] (first position stays absolute)
- Learn patterns from relative spatial jumps/distances

### **Dual-Time Encoders** (KAN-MAMMOTE variants)
- Input: **Both** absolute AND relative time simultaneously
- Can learn from both absolute positions and relative patterns

## HPC/Cluster Usage

For running on clusters like JAIST's system:

```bash
#!/bin/bash
#PBS -j oe
#PBS -q GPU-1
#PBS -l select=1:ngpus=1
#PBS -M your_email@jaist.ac.jp
#PBS -m be

cd "$PBS_O_WORKDIR"

source ~/.bashrc
module purge
module load cuda/12.8u1
conda activate kan_mammote

# Run experiment
python event_based_mnist_experiment.py \
    --epochs 200 \
    --encoders lete lete_relative time2vec time2vec_relative mercer mercer_relative \
    --experiment_dir /path/to/results > experiment.log 2>&1
```

## Expected Results

The experiment will output:
1. **Best validation accuracy** for each encoder (primary comparison metric)
2. **Training curves** showing learning progression  
3. **Statistical comparison** of absolute vs. relative time variants
4. **Model checkpoints** for the best performing models

## Troubleshooting

### Import Errors
If you see `ImportError` for specific encoders:
```bash
# Check which encoders are available
python -c "
import sys; sys.path.insert(0, '.')
from experiments.event_based_mnist_experiment import get_available_encoders
print('Available encoders:', get_available_encoders())
"
```

### Memory Issues
```bash
# Reduce batch size for memory-constrained systems
python event_based_mnist_experiment.py --batch_size 256

# Test with fewer encoders
python event_based_mnist_experiment.py --encoders lstm_only lete
```

### Resume Failed Training
```bash
# Find latest experiment
ls -la mnist_experiments/

# Resume from specific checkpoint
python event_based_mnist_experiment.py \
    --resume_experiment mnist_experiments/run_20251018_143022 \
    --resume_encoder kan_mammote_full
```

This setup now provides comprehensive comparison between absolute and relative time encoding approaches across multiple encoder architectures!