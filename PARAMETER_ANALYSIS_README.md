# KAN-MAMMOTE Parameter Analysis

## Overview

This experiment conducts a comprehensive parameter analysis to demonstrate that KAN-MAMMOTE's superiority isn't solely due to higher parameter count. We systematically vary key architectural parameters and measure the relationship between model complexity and performance.

## Chain of Thought Analysis

### Objective
**Demonstrate efficiency**: Show that KAN-MAMMOTE achieves superior performance through architectural innovation, not just parameter scaling.

### Methodology
1. **Fixed Training Setup**: Keep all training hyperparameters constant (learning rate, batch size, etc.)
2. **Vary Architecture**: Systematically change key model parameters
3. **Measure Efficiency**: Plot AUC-ROC vs FLOPs and parameter count
4. **Analyze Patterns**: Identify efficient configurations and diminishing returns

### Key Parameters Analyzed

| Parameter | Range | Description |
|-----------|-------|-------------|
| `expert_dim` | [32, 64, 128, 256] | Controls K-MOTE expert capacity |
| `mamba_d_state` | [64, 128, 256, 512] | Controls Mamba state dimension |
| `mamba_expand` | [2, 4, 8] | Controls Mamba expansion factor |
| `mamba_headdim` | [16, 32, 64] | Controls Mamba head dimension |

### Experimental Setup

- **Model**: TGN (reliable baseline)
- **Dataset**: UCI (good balance of complexity and training time)
- **Time Encoder**: kan_mammote_dual_kmote
- **Training**: 200 epochs, early stopping patience=30
- **Evaluation**: Test AUC-ROC (primary metric)

## Usage

### Full Analysis
```bash
# Submit to job queue
qsub run_parameter_analysis.sh

# Or run directly
python experiments/kan_mammote_parameter_analysis.py --output_dir parameter_analysis_results
```

### Quick Test (3 configurations)
```bash
bash test_parameter_analysis.sh
```

### Plot Only (from existing results)
```bash
python experiments/kan_mammote_parameter_analysis.py \
    --plot_only parameter_analysis_results/parameter_analysis_20241104_*.json \
    --output_dir parameter_analysis_plots
```

## Output Files

### Primary Results
- `parameter_analysis_TIMESTAMP.json` - Complete experimental data
- `parameter_analysis_TIMESTAMP.csv` - Flattened data for analysis
- `parameter_analysis_plots_TIMESTAMP.png` - Comprehensive visualizations

### Analysis Reports
- `efficiency_report_TIMESTAMP.txt` - Detailed efficiency analysis
- `experiment_log.txt` - Full execution log

## Key Visualizations

1. **Performance vs Parameter Count** - Shows diminishing returns of parameter scaling
2. **Performance vs FLOPs** - Demonstrates computational efficiency
3. **Efficiency Frontier** - Identifies optimal configurations (Pareto frontier)
4. **Parameter Impact Analysis** - Shows individual parameter effects
5. **Memory Efficiency** - Memory usage vs performance
6. **Training Efficiency** - Training time vs performance

## Expected Insights

### If KAN-MAMMOTE is Truly Efficient:
- **Weak correlation** between parameter count and performance (< 0.7)
- **Clear efficiency leaders** achieving high AUC with low parameter count
- **Diminishing returns** for very large configurations
- **Architectural sweet spots** where specific parameter combinations excel

### If It's Just Parameter Scaling:
- **Strong correlation** between parameter count and performance (> 0.8)
- **Linear scaling** where bigger always = better
- **No clear efficiency leaders**

## Strategic Parameter Combinations

### Efficiency-Focused (Low Parameters, High Performance)
```python
{'expert_dim': 32, 'mamba_d_state': 64, 'mamba_expand': 2, 'mamba_headdim': 16}    # Ultra-light
{'expert_dim': 64, 'mamba_d_state': 128, 'mamba_expand': 2, 'mamba_headdim': 32}   # Light
```

### Baseline Configurations
```python
{'expert_dim': 64, 'mamba_d_state': 128, 'mamba_expand': 4, 'mamba_headdim': 32}   # Small
{'expert_dim': 128, 'mamba_d_state': 256, 'mamba_expand': 4, 'mamba_headdim': 32}  # Medium  
{'expert_dim': 256, 'mamba_d_state': 512, 'mamba_expand': 4, 'mamba_headdim': 64}  # Large
```

### High-Performance Configurations
```python
{'expert_dim': 256, 'mamba_d_state': 256, 'mamba_expand': 8, 'mamba_headdim': 32}
{'expert_dim': 128, 'mamba_d_state': 512, 'mamba_expand': 8, 'mamba_headdim': 64}
```

## Computational Complexity Analysis

### FLOPs Estimation
The script estimates FLOPs for each component:
- **K-MOTE Absolute/Relative**: Expert MLPs + gating mechanisms
- **Mamba2**: Input projections + state space operations + output projections
- **Modulator Head**: Temporal control signal generation
- **Output Projection**: Final embedding transformation

### Parameter Counting
- **Total Parameters**: All model weights
- **Trainable Parameters**: Learnable weights (excluding frozen)
- **Memory Estimate**: Approximate memory usage (float32)

## Success Criteria

### For Demonstrating Efficiency:
1. **Find configurations** with <50K parameters achieving >90% of max performance
2. **Show diminishing returns** for parameter scaling beyond certain thresholds
3. **Identify architectural patterns** that favor efficiency
4. **Demonstrate FLOPs efficiency** where computation correlates better with performance than raw parameter count

### For Publication:
- **Efficiency plot** showing clear Pareto frontier
- **Parameter impact analysis** showing which components matter most
- **Comparison table** between efficient and large configurations
- **Quantitative evidence** that performance gains aren't just from "bigger models"

## Troubleshooting

### Common Issues:
1. **CUDA OOM**: Reduce batch_size in fixed_config
2. **Slow execution**: Use --quick_test for development
3. **Import errors**: Ensure kan_mammote environment is activated
4. **Missing results**: Check saved_results/ directory patterns

### Debug Mode:
```bash
# Run single configuration manually
python experiments/train_link_prediction.py \
    --model_name TGN \
    --dataset_name uci \
    --time_encoder kan_mammote_dual_kmote \
    --expert_dim 64 \
    --mamba_d_state 128 \
    --mamba_expand 4 \
    --mamba_headdim 32 \
    --num_runs 1 \
    --seed 42
```

## Next Steps

1. **Run full analysis** (~4-6 hours for all configurations)
2. **Analyze efficiency patterns** in generated reports
3. **Create publication figures** from best visualizations
4. **Compare against baseline methods** (LeTE, Mercer, etc.)
5. **Extend to other datasets** if patterns are consistent