# KAN-MAMMOTE Parameter Analysis Framework

This framework conducts comprehensive parameter analysis to demonstrate that KAN-MAMMOTE's superiority isn't solely due to higher parameter count. Instead of simple plots, we generate detailed **tables** that provide clearer insights into parameter efficiency.

## Why Tables Instead of Plots?

### 1. **Multi-dimensional Analysis**
- Different parameters (`expert_dim`, `mamba_d_state`, `head_dim`, `n_layers`) affect performance differently
- Tables allow us to see exact values and relationships that might be obscured in plots
- Clear comparison of efficiency metrics across configurations

### 2. **Parameter Efficiency Focus**
- **AUC per GFLOP**: Performance efficiency relative to computational cost
- **AUC per Million Parameters**: Performance efficiency relative to model size
- **Direct comparison**: Easy to identify which configurations achieve best efficiency

### 3. **Scientific Rigor**
- Precise numerical values for reproducibility
- Rankings and statistical summaries
- LaTeX tables ready for publication

## Generated Analysis Tables

### 1. Main Results Table
```
| Rank | Config    | Expert_Dim | Mamba_D_State | Head_Dim | N_Layers | Params(M) | FLOPs(G) | AUC-ROC | Time(min) |
|------|-----------|------------|---------------|----------|----------|-----------|----------|---------|-----------|
| 1    | config_05 | 128        | 128           | 32       | 2        | 2.45      | 15.2     | 0.8542  | 12.3      |
| 2    | config_03 | 64         | 128           | 32       | 2        | 1.89      | 11.8     | 0.8498  | 9.7       |
| ...  | ...       | ...        | ...           | ...      | ...      | ...       | ...      | ...     | ...       |
```

### 2. Efficiency Analysis Table
```
| Config    | Expert_Dim | Mamba_D_State | Head_Dim | AUC-ROC | AUC_per_GFLOP | AUC_per_MParam | FLOP_Eff_Rank |
|-----------|------------|---------------|----------|---------|---------------|----------------|---------------|
| config_03 | 64         | 128           | 32       | 0.8498  | 0.0720        | 0.4496         | 1             |
| config_05 | 128        | 128           | 32       | 0.8542  | 0.0562        | 0.3486         | 2             |
| ...       | ...        | ...           | ...      | ...     | ...           | ...            | ...           |
```

### 3. Parameter Impact Analysis
- Statistical analysis of how each parameter affects performance
- Correlation coefficients between parameters and AUC-ROC
- Mean performance for different parameter values

### 4. Summary Insights
- Best overall configuration
- Most FLOP-efficient configuration  
- Most parameter-efficient configuration
- Key findings and recommendations

## Usage

### Quick Test Run
```bash
python experiments/kan_mammote_parameter_analysis_v2.py --test
```

### Full Analysis
```bash
python experiments/kan_mammote_parameter_analysis_v2.py --output_dir results/parameter_analysis
```

### Batch Job Submission
```bash
qsub scripts/run_parameter_analysis.sh
```

## Strategic Parameter Combinations

Instead of full factorial design (which would be computationally expensive), we use strategic combinations:

1. **Baseline Configurations**: Small, Medium, Large
2. **Parameter Scaling**: Vary one parameter at a time from baseline
3. **Efficiency-Focused**: Configurations optimized for efficiency

This approach provides comprehensive insights while keeping computational cost manageable.

## Key Insights from Table Analysis

The table format allows us to clearly demonstrate:

1. **Efficiency Leaders**: Configurations that achieve high AUC with fewer parameters/FLOPs
2. **Diminishing Returns**: Where adding parameters doesn't improve efficiency
3. **Sweet Spots**: Optimal parameter combinations for different efficiency metrics
4. **Trade-offs**: Performance vs computational cost relationships

## Output Files

```
parameter_analysis_results/
├── intermediate_results.json          # Raw experiment results
├── analysis/
│   ├── main_results.csv              # Main results table
│   ├── main_results.tex              # LaTeX table for paper
│   ├── efficiency_analysis.csv       # Efficiency metrics
│   ├── efficiency_analysis.tex       # LaTeX efficiency table
│   ├── parameter_impact_analysis.csv # Statistical parameter analysis
│   └── summary_insights.txt          # Key findings and recommendations
```

## Benefits for Your Research

1. **Clear Evidence**: Tables provide concrete evidence that performance isn't just about parameter count
2. **Publication Ready**: LaTeX tables ready for your paper
3. **Efficiency Focus**: Highlights configurations that achieve best efficiency ratios
4. **Reproducible**: Exact numerical values for reproducibility
5. **Comprehensive**: Multiple perspectives on parameter efficiency

This table-based approach provides much clearer insights than plots when dealing with multi-dimensional parameter analysis!