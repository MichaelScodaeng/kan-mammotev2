# KAN-MAMMOTE Parameter Analysis (Tables + Plots)

## Overview

This comprehensive analysis demonstrates that KAN-MAMMOTE's performance gains aren't solely due to higher parameter count. We provide **both detailed tables and visualizations** to show the relationship between model parameters, computational cost (FLOPs), and performance.

## Key Outputs

### 📋 Analysis Tables
1. **Main Results Table** (`main_results.csv`, `main_results.tex`)
   - Performance ranking of all configurations
   - Parameter counts and FLOPs for each configuration
   - Training times and AUC-ROC scores

2. **Efficiency Analysis Table** (`efficiency_analysis.csv`, `efficiency_analysis.tex`)
   - AUC per GFLOP rankings (computational efficiency)
   - AUC per Million Parameters rankings (parameter efficiency)
   - Best efficiency configurations highlighted

3. **Parameter Impact Analysis** (`parameter_impact_analysis.csv`)
   - Statistical analysis of how each parameter affects performance
   - Mean, std, min, max for each parameter value
   - Correlation coefficients with performance

4. **Summary Insights** (`summary_insights.txt`)
   - Top performing configurations
   - Most efficient configurations
   - Key findings and recommendations

### 📈 Visualization Plots

1. **Main Efficiency Analysis** (`efficiency_analysis_TIMESTAMP.png`)
   - 6-subplot comprehensive figure showing:
     - Performance vs Parameter Count (with efficiency frontier)
     - Performance vs FLOPs (main efficiency plot)
     - Computational Efficiency ratios
     - Model Complexity relationships
     - Training time efficiency
     - Top 5 configuration comparison

2. **Parameter Impact Plots** (`parameter_impact_TIMESTAMP.png`)
   - Individual impact analysis for each parameter:
     - Expert Dimension effects
     - Mamba D State effects  
     - Head Dimension effects
     - Number of Layers effects

3. **Efficiency Frontier Analysis** (`efficiency_frontier_TIMESTAMP.png`)
   - Detailed Pareto frontier analysis:
     - FLOPs efficiency frontier
     - Parameter efficiency frontier
     - Optimal configurations highlighted

## Usage

### Full Analysis (Both Tables and Plots)
```bash
# Run complete analysis
python experiments/kan_mammote_parameter_analysis_v2.py --output_dir results

# Or submit to job queue
qsub run_parameter_analysis_v2.sh
```

### Quick Test (3 configurations)
```bash
python experiments/kan_mammote_parameter_analysis_v2.py --test --output_dir test_results
```

## Strategic Parameter Combinations

We test these carefully selected configurations:

### 🎯 Baseline Configurations
- **Small**: `{expert_dim: 64, mamba_d_state: 64, head_dim: 16, n_layers: 1}`
- **Medium**: `{expert_dim: 128, mamba_d_state: 128, head_dim: 32, n_layers: 2}`
- **Large**: `{expert_dim: 256, mamba_d_state: 256, head_dim: 64, n_layers: 3}`

### ⚡ Efficiency-Focused Configurations
- **Low Params, High State**: `{expert_dim: 64, mamba_d_state: 128, head_dim: 32, n_layers: 2}`
- **High Expert, Low State**: `{expert_dim: 128, mamba_d_state: 64, head_dim: 32, n_layers: 2}`
- **Balanced Efficient**: `{expert_dim: 128, mamba_d_state: 128, head_dim: 16, n_layers: 2}`

### 🔬 Systematic Variations
Each parameter is varied individually while keeping others at baseline values.

## Key Visualizations Explained

### 1. Performance vs FLOPs Plot
- **X-axis**: Computational cost (GFLOPs)
- **Y-axis**: AUC-ROC performance
- **Colors**: Different model parameters
- **Red dashed line**: Efficiency frontier (optimal configurations)

**What to look for:**
- Configurations above the frontier line are most efficient
- Steep drops indicate diminishing returns
- Color patterns show which parameters drive efficiency

### 2. Efficiency Frontier
- **Purpose**: Identify Pareto-optimal configurations
- **Interpretation**: Points on the frontier achieve maximum performance for given computational cost
- **Value**: Shows configurations that balance performance and efficiency

### 3. Parameter Impact Bars
- **Shows**: How each parameter individually affects performance
- **Error bars**: Variation across different configurations
- **Insight**: Which parameters have strongest impact on performance

## Expected Insights

### 🎯 If KAN-MAMMOTE is Truly Efficient:
1. **Weak correlation** between parameter count and performance (< 0.7)
2. **Clear efficiency leaders** in tables and plots
3. **Steep efficiency frontier** showing optimal configurations
4. **Diminishing returns** visible in parameter scaling plots

### ⚠️ If It's Just Parameter Scaling:
1. **Strong correlation** between parameters and performance (> 0.8)
2. **Linear scaling** in plots (bigger = better)
3. **No clear efficiency frontier**
4. **Monotonic parameter impact**

## File Organization

```
results/
├── analysis/                          # 📋 Tables and Text Analysis
│   ├── main_results.csv              # Primary results table
│   ├── main_results.tex              # LaTeX table for papers
│   ├── efficiency_analysis.csv       # Efficiency rankings
│   ├── efficiency_analysis.tex       # LaTeX efficiency table
│   ├── parameter_impact_analysis.csv # Statistical parameter analysis
│   └── summary_insights.txt          # Key findings and recommendations
├── plots/                             # 📈 Visualizations
│   ├── efficiency_analysis_TIMESTAMP.png     # Main 6-subplot figure
│   ├── parameter_impact_TIMESTAMP.png        # Parameter effect plots
│   └── efficiency_frontier_TIMESTAMP.png     # Pareto frontier analysis
└── intermediate_results.json         # Raw experimental data
```

## Interpreting Results

### Table Analysis
1. **Sort by Performance Rank** to see best configurations
2. **Sort by AUC_per_GFLOP** to find most computationally efficient
3. **Sort by AUC_per_MParam** to find most parameter efficient
4. **Compare configurations** with similar parameter counts

### Plot Analysis
1. **Efficiency Frontier**: Look for configurations on or near the red line
2. **Color Patterns**: Identify which parameter values cluster near high performance
3. **Parameter Impact**: See which parameters have steepest slopes (strongest effect)
4. **Training Efficiency**: Balance performance gains vs training time

## Success Criteria

### ✅ Demonstrating Architectural Efficiency:
- Multiple configurations achieve >90% of max performance with <50% of max parameters
- Clear efficiency frontier with distinct optimal points
- Parameter impact analysis shows architectural sweet spots
- Diminishing returns visible for very large configurations

### 📊 For Publication Use:
- **Table 1**: Main results showing performance vs complexity
- **Table 2**: Efficiency analysis highlighting optimal configurations  
- **Figure 1**: Efficiency frontier plot showing Pareto-optimal points
- **Figure 2**: Parameter impact analysis showing architectural insights

## Computational Complexity

### FLOPs Estimation Components:
- **Embedding Layer**: Input feature transformation
- **KAN Layers**: B-spline based transformations (n_layers dependent)
- **Mamba Layers**: State space computation (expand factor dependent)
- **Attention Mechanism**: Head-wise attention computation
- **Output Layers**: Final classification

### Parameter Counting:
- **Total Parameters**: All model weights
- **Trainable Parameters**: Learnable weights
- **Memory Estimate**: Float32 storage requirements

## Next Steps

1. **Analyze Results**: Examine tables and plots for efficiency patterns
2. **Identify Winners**: Find configurations that achieve efficiency goals
3. **Publication Prep**: Use LaTeX tables and PNG plots directly in papers
4. **Extended Analysis**: Apply insights to optimize other datasets
5. **Comparison Study**: Compare efficiency against baseline methods

This comprehensive approach provides both the **numerical precision of tables** and the **visual insights of plots** to make a compelling case for KAN-MAMMOTE's architectural efficiency.