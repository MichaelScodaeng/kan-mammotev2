# KAN-MAMMOTE Project Structure

This document describes the reorganized file structure for the KAN-MAMMOTE project.

## Directory Structure

```
kan-mammotev2/
├── models/                              # Main model implementations
│   ├── time_encoders/                   # Time encoding modules
│   │   ├── __init__.py                  # Time encoder package
│   │   ├── factory.py                   # Factory for creating encoders
│   │   ├── base_encoder.py              # Base time encoder interface
│   │   ├── original_encoder.py          # Traditional cosine encoding
│   │   ├── lete_baseline.py             # LeTE implementation
│   │   ├── lete_encoder.py              # Enhanced LeTE (new)
│   │   ├── k_mote.py                    # K-MOTE implementation
│   │   ├── sm_kernel.py                 # SM-Kernel implementation
│   │   ├── controllable_mamba2.py       # Controllable Mamba2
│   │   └── kan_mammote.py               # Main KAN-MAMMOTE class
│   └── gnn_backbones/                   # GNN backbone models
│       ├── DyGMamba.py                  # DyGMamba model
│       ├── TGAT.py                      # TGAT model
│       ├── MemoryModel.py               # Memory-based models
│       ├── modules.py                   # Common modules
│       └── ...                          # Other GNN models
├── experiments/                         # Experiment scripts
│   ├── train_link_prediction.py        # Main training script
│   └── evaluate_models_utils.py        # Evaluation utilities
├── utils/                               # Utility modules
│   ├── __init__.py
│   ├── DataLoader.py                    # Data loading utilities
│   ├── metrics.py                       # Evaluation metrics
│   ├── utils.py                         # General utilities
│   └── ...
├── config/                              # Configuration files
│   └── lstm_comparison_config.json
├── imported_lib/                        # External libraries
│   ├── fast-kan/                        # KAN implementations
│   ├── faster_kan/                      # Faster KAN implementations
│   ├── mamba/                           # Mamba SSM library
│   ├── LeTE/                            # LeTE reference
│   └── ...
├── run_time_encoder_comparison.sh       # Main experiment runner
├── analyze_results.py                   # Results analysis script
├── experiment_*.py                      # Individual experiment scripts
└── ...                                  # Other project files
```

## Usage

### Running Time Encoder Comparison

To compare all three time encoders (Original, LeTE, KAN-MAMMOTE):

```bash
./run_time_encoder_comparison.sh
```

### Individual Experiments

For Original encoder:
```bash
python experiments/train_link_prediction.py --time_encoder_type original --dataset_name wikipedia
```

For LeTE:
```bash
python experiments/train_link_prediction.py --time_encoder_type lete --dataset_name wikipedia
```

For KAN-MAMMOTE:
```bash
python experiments/train_link_prediction.py --time_encoder_type kan_mammote --dataset_name wikipedia --expert_dim 64 --num_mixtures 8
```

### Analyzing Results

After running experiments:
```bash
python analyze_results.py --result_dir results/time_encoder_comparison/
```

## Key Features

1. **Modular Design**: Time encoders are now in their own module for easy comparison
2. **Factory Pattern**: Easy creation of different encoder types
3. **Preserved Logic**: Your original implementations are maintained without changes
4. **Experiment Framework**: Structured approach to running and comparing encoders
5. **Results Analysis**: Automated analysis and visualization of results

## Time Encoder Types

- **original**: Traditional cosine-based encoding (baseline)
- **lete**: Learnable Time Encoding with Fourier and spline components
- **kan_mammote**: Your novel dual-stream KAN-MAMMOTE architecture

The factory in `models/time_encoders/factory.py` handles creating the appropriate encoder based on the type specified in experiments.
