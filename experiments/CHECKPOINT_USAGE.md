# Checkpoint System Usage Guide

## Overview
The Event-Based MNIST experiment now includes a comprehensive checkpoint system that allows you to:
- Resume interrupted training sessions
- Continue training specific encoders for more epochs  
- Save training progress automatically

## Features Added

### 1. LeTE Relative Time Variant
- New encoder: `lete_relative` - Uses relative time differences instead of absolute positions
- Comparison baseline to standard `lete` encoder

### 2. Automatic Checkpoint Saving
- Saves checkpoints every 10 epochs (configurable)
- Saves best model checkpoints immediately when validation accuracy improves
- Stores complete training state (model, optimizer, history, metrics)

### 3. Resume Training Capability  
- Resume from latest checkpoint for any encoder
- Resume entire experiments or specific encoders
- Continue training for additional epochs

## Command Examples

### Basic Training (with checkpoints)
```bash
# Train all available encoders with checkpoint saving
python event_based_mnist_experiment.py --epochs 200

# Train specific encoders
python event_based_mnist_experiment.py --encoders lete lete_relative kan_mammote_full --epochs 100

# Train with custom checkpoint frequency
python event_based_mnist_experiment.py --checkpoint_every 5 --epochs 50
```

### Resume Training
```bash
# Resume training from latest checkpoints (if training was interrupted)
python event_based_mnist_experiment.py --resume_training --epochs 200

# Resume from specific experiment directory
python event_based_mnist_experiment.py --resume_experiment mnist_experiments/run_20251017_143022

# Resume specific encoder for more epochs
python event_based_mnist_experiment.py --resume_experiment mnist_experiments/run_20251017_143022 --resume_encoder kan_mammote_full --additional_epochs 50
```

### Disable Checkpoints (not recommended)
```bash
# Training without checkpoints (saves disk space but risky for long training)
python event_based_mnist_experiment.py --no_checkpoints --epochs 200
```

## File Structure

After running experiments, you'll get this structure:
```
mnist_experiments/
└── run_20251017_143022/
    ├── models/                          # Best models
    │   ├── best_model_lete.pth
    │   ├── best_model_lete_relative.pth
    │   └── best_model_kan_mammote_full.pth
    ├── checkpoints/                     # Training checkpoints
    │   ├── checkpoint_lete_latest.pth
    │   ├── checkpoint_lete_epoch_50.pth
    │   ├── checkpoint_lete_relative_latest.pth
    │   └── checkpoint_kan_mammote_full_latest.pth
    ├── epoch_history/                   # Per-epoch CSV logs
    │   ├── lete_history.csv
    │   └── lete_relative_history.csv
    ├── mnist_time_encoder_results.json  # Complete results
    ├── mnist_time_encoder_results.csv   # Summary table
    └── mnist_time_encoder_results_curves.png  # Training plots
```

## Checkpoint Management

### Check Available Checkpoints
```bash
# List all checkpoints in an experiment
python event_based_mnist_experiment.py --resume_experiment mnist_experiments/run_20251017_143022
```

### Resume Specific Encoder
```bash
# Resume kan_mammote_full for 50 more epochs
python event_based_mnist_experiment.py \
    --resume_experiment mnist_experiments/run_20251017_143022 \
    --resume_encoder kan_mammote_full \
    --additional_epochs 50
```

### Manual Checkpoint Loading (in Python)
```python
import torch
from experiments.event_based_mnist_experiment import TimeEncoderClassifier, load_checkpoint

# Load model and checkpoint
model = TimeEncoderClassifier(encoder_type='kan_mammote_full', embedding_dim=32)
checkpoint_data = load_checkpoint('path/to/checkpoint_kan_mammote_full_latest.pth', model)

print(f"Loaded epoch: {checkpoint_data['epoch']}")
print(f"Best val acc: {checkpoint_data['best_val_acc']:.2f}%")
```

## Performance Tips

1. **Long Training**: Always use checkpoints for training >50 epochs
2. **Cluster/HPC**: Essential for job scheduling systems that may kill long jobs
3. **Disk Space**: Checkpoints are ~50-100MB each, plan accordingly
4. **Resume Strategy**: Resume from `latest` checkpoints for most recent state

## Troubleshooting

### Training Interrupted
```bash
# Find your experiment directory
ls mnist_experiments/

# Resume from latest checkpoints
python event_based_mnist_experiment.py --resume_training --experiment_dir mnist_experiments/run_YYYYMMDD_HHMMSS
```

### Specific Encoder Failed
```bash
# Check what encoders have checkpoints
python event_based_mnist_experiment.py --resume_experiment mnist_experiments/run_YYYYMMDD_HHMMSS

# Resume just the failed encoder
python event_based_mnist_experiment.py \
    --resume_experiment mnist_experiments/run_YYYYMMDD_HHMMSS \
    --resume_encoder failed_encoder_name \
    --additional_epochs 100
```

### Out of Disk Space
```bash
# Clean old checkpoints (keep only latest and best models)
find mnist_experiments/ -name "checkpoint_*_epoch_*.pth" -delete

# Or disable checkpoints for short runs
python event_based_mnist_experiment.py --no_checkpoints --epochs 20
```