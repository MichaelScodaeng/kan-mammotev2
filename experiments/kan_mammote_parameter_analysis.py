#!/usr/bin/env python3
"""
KAN-MAMMOTE Parameter Analysis Experiment
==========================================

This experiment conducts a comprehensive parameter analysis to demonstrate that 
KAN-MAMMOTE's superiority isn't solely due to higher parameter count.

Methodology:
1. Vary key architectural parameters (expert_dim, mamba_d_state, mamba_expand, mamba_headdim)
2. Keep training hyperparameters constant
3. Measure AUC-ROC vs FLOPs for each configuration
4. Use TGN + UCI dataset for controlled experiments
5. Generate efficiency plot showing performance vs computational cost

Key Parameters to Analyze:
- expert_dim: [32, 64, 128, 256] - Controls K-MOTE expert capacity
- mamba_d_state: [64, 128, 256, 512] - Controls Mamba state dimension  
- mamba_expand: [2, 4, 8] - Controls Mamba expansion factor
- mamba_headdim: [16, 32, 64] - Controls Mamba head dimension

Output:
- Parameter count vs AUC-ROC scatter plot
- FLOPs vs AUC-ROC efficiency plot
- Detailed analysis table with all configurations
- Model complexity breakdown by component

Usage:
    python experiments/kan_mammote_parameter_analysis.py --output_dir parameter_analysis_results
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
from datetime import datetime
from tqdm import tqdm
import itertools
from typing import Dict, List, Tuple, Any
import warnings
import subprocess
import glob

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import necessary modules
from experiments.train_link_prediction import run_link_prediction_experiment
from models.time_encoders.kan_mammote import KAN_MAMMOTE

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class FLOPsCounter:
    """
    FLOPs counter for neural network operations.
    Estimates computational complexity in Floating Point Operations.
    """
    
    @staticmethod
    def count_linear_flops(input_size: int, output_size: int, batch_size: int) -> int:
        """Count FLOPs for Linear layer: batch_size * input_size * output_size * 2"""
        return batch_size * input_size * output_size * 2
    
    @staticmethod
    def count_bmm_flops(batch_size: int, n: int, m: int, k: int) -> int:
        """Count FLOPs for batch matrix multiplication: batch_size * n * m * k * 2"""
        return batch_size * n * m * k * 2
    
    @staticmethod
    def count_elementwise_flops(num_elements: int) -> int:
        """Count FLOPs for elementwise operations (activation, etc.)"""
        return num_elements
    
    @staticmethod
    def estimate_kan_mammote_flops(
        batch_size: int,
        seq_len: int,
        expert_dim: int,
        embedding_dim: int,
        mamba_d_state: int,
        mamba_d_conv: int,
        mamba_expand: int,
        mamba_headdim: int
    ) -> Dict[str, int]:
        """
        Estimate FLOPs for KAN-MAMMOTE forward pass.
        
        Returns breakdown by component for analysis.
        """
        flops = {}
        
        # K-MOTE Absolute (assume ~4 experts with small MLPs)
        kmote_abs_flops = (
            FLOPsCounter.count_linear_flops(1, expert_dim//4, batch_size * seq_len) * 4 +  # Expert MLPs
            FLOPsCounter.count_linear_flops(expert_dim, expert_dim, batch_size * seq_len) +  # Gating
            FLOPsCounter.count_elementwise_flops(batch_size * seq_len * expert_dim * 8)  # Nonlinearities
        )
        flops['kmote_abs'] = kmote_abs_flops
        
        # K-MOTE Relative (similar to absolute)
        flops['kmote_rel'] = kmote_abs_flops
        
        # Mamba2 Operations (simplified estimate)
        mamba_inner_dim = expert_dim * mamba_expand
        mamba_dt_rank = max(16, expert_dim // 16)
        nheads = expert_dim // mamba_headdim
        
        # Input projection (u, v gates + dt)
        mamba_input_proj = FLOPsCounter.count_linear_flops(
            expert_dim, mamba_inner_dim * 2 + mamba_dt_rank, batch_size * seq_len
        )
        
        # State space operations (simplified)
        mamba_ssm_flops = (
            batch_size * seq_len * mamba_d_state * nheads * 10  # Approximation for SSM ops
        )
        
        # Output projection
        mamba_output_proj = FLOPsCounter.count_linear_flops(
            mamba_inner_dim, expert_dim, batch_size * seq_len
        )
        
        mamba_total = mamba_input_proj + mamba_ssm_flops + mamba_output_proj
        flops['mamba2'] = mamba_total
        
        # Modulator head (for ControllableMamba2)
        modulator_flops = (
            FLOPsCounter.count_linear_flops(expert_dim, expert_dim//2, batch_size * seq_len) +
            FLOPsCounter.count_linear_flops(expert_dim//2, nheads * 2, batch_size * seq_len)
        )
        flops['modulator'] = modulator_flops
        
        # Output projection
        output_proj_flops = FLOPsCounter.count_linear_flops(
            expert_dim, embedding_dim, batch_size * seq_len
        )
        flops['output_proj'] = output_proj_flops
        
        # Elementwise operations (activations, norms, etc.)
        elementwise_flops = FLOPsCounter.count_elementwise_flops(
            batch_size * seq_len * expert_dim * 5  # Rough estimate
        )
        flops['elementwise'] = elementwise_flops
        
        # Total FLOPs
        flops['total'] = sum(flops.values())
        
        return flops

class KANMAMMOTEParameterAnalyzer:
    """
    Comprehensive parameter analysis for KAN-MAMMOTE.
    
    This class orchestrates experiments across different parameter configurations
    to analyze the relationship between model complexity and performance.
    """
    
    def __init__(self, output_dir: str = 'parameter_analysis_results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Fixed experimental settings for fair comparison
        self.fixed_config = {
            'model_name': 'TGN',
            'dataset_name': 'uci',  # Good balance of complexity and training time
            'time_encoder': 'kan_mammote_dual_kmote',
            'num_epochs': 200,
            'num_runs': 1,  # Single run for faster analysis
            'learning_rate': 0.0001,
            'patience': 30,
            'batch_size': 200,
            'seed': 42,
            'disable_progress_bar': True
        }
        
        # Parameter ranges to explore
        self.param_ranges = {
            'expert_dim': [32, 64, 128, 256],
            'mamba_d_state': [64, 128, 256, 512], 
            'mamba_expand': [2, 4, 8],
            'mamba_headdim': [16, 32, 64]
        }
        
        # Generate all parameter combinations
        self.param_combinations = self._generate_param_combinations()
        
        print(f"📊 KAN-MAMMOTE Parameter Analysis")
        print(f"   Output directory: {output_dir}")
        print(f"   Fixed config: {self.fixed_config}")
        print(f"   Parameter ranges: {self.param_ranges}")
        print(f"   Total combinations: {len(self.param_combinations)}")
    
    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for analysis."""
        combinations = []
        
        # Strategy 1: Full factorial (might be too many - 4*4*3*3 = 144 combinations)
        # Let's use a more strategic approach
        
        # Strategy 2: Strategic sampling
        # 1. Baseline configurations
        baseline_configs = [
            {'expert_dim': 64, 'mamba_d_state': 128, 'mamba_expand': 4, 'mamba_headdim': 32},  # Small
            {'expert_dim': 128, 'mamba_d_state': 256, 'mamba_expand': 4, 'mamba_headdim': 32}, # Medium  
            {'expert_dim': 256, 'mamba_d_state': 512, 'mamba_expand': 4, 'mamba_headdim': 64}, # Large
        ]
        
        # 2. Systematic variations (vary one parameter at a time from medium baseline)
        medium_base = {'expert_dim': 128, 'mamba_d_state': 256, 'mamba_expand': 4, 'mamba_headdim': 32}
        
        systematic_configs = []
        for param_name, param_values in self.param_ranges.items():
            for value in param_values:
                if value != medium_base[param_name]:  # Skip the baseline value
                    config = medium_base.copy()
                    config[param_name] = value
                    systematic_configs.append(config)
        
        # 3. Efficiency-focused configurations (low parameters, potentially high performance)
        efficiency_configs = [
            {'expert_dim': 32, 'mamba_d_state': 64, 'mamba_expand': 2, 'mamba_headdim': 16},   # Ultra-light
            {'expert_dim': 64, 'mamba_d_state': 128, 'mamba_expand': 2, 'mamba_headdim': 32},  # Light
            {'expert_dim': 32, 'mamba_d_state': 128, 'mamba_expand': 4, 'mamba_headdim': 16},  # Low expert, high state
            {'expert_dim': 128, 'mamba_d_state': 64, 'mamba_expand': 2, 'mamba_headdim': 64},  # High expert, low state
        ]
        
        # 4. High-performance configurations
        high_perf_configs = [
            {'expert_dim': 256, 'mamba_d_state': 256, 'mamba_expand': 8, 'mamba_headdim': 32},
            {'expert_dim': 128, 'mamba_d_state': 512, 'mamba_expand': 8, 'mamba_headdim': 64},
            {'expert_dim': 256, 'mamba_d_state': 128, 'mamba_expand': 4, 'mamba_headdim': 64},
        ]
        
        # Combine all configurations
        all_configs = baseline_configs + systematic_configs + efficiency_configs + high_perf_configs
        
        # Remove duplicates
        seen = set()
        unique_configs = []
        for config in all_configs:
            config_tuple = tuple(sorted(config.items()))
            if config_tuple not in seen:
                seen.add(config_tuple)
                unique_configs.append(config)
        
        return unique_configs
    
    def _estimate_model_complexity(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate model complexity metrics for a configuration.
        
        Returns parameter count, FLOPs, and memory estimates.
        """
        # Create a dummy model to count parameters
        dummy_model = KAN_MAMMOTE(
            embedding_dim=128,  # Fixed for TGN
            expert_dim=config['expert_dim'],
            mamba_d_state=config['mamba_d_state'],
            mamba_d_conv=4,  # Fixed
            mamba_expand=config['mamba_expand'],
            mamba_headdim=config['mamba_headdim'],
            use_controllable_mamba=True,
            dropout=0.1
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in dummy_model.parameters())
        trainable_params = sum(p.numel() for p in dummy_model.parameters() if p.requires_grad)
        
        # Estimate FLOPs (assume typical batch size and sequence length)
        batch_size = 200  # From fixed config
        seq_len = 1      # TGN typically processes single events
        
        flops_breakdown = FLOPsCounter.estimate_kan_mammote_flops(
            batch_size=batch_size,
            seq_len=seq_len,
            expert_dim=config['expert_dim'],
            embedding_dim=128,
            mamba_d_state=config['mamba_d_state'],
            mamba_d_conv=4,
            mamba_expand=config['mamba_expand'],
            mamba_headdim=config['mamba_headdim']
        )
        
        # Memory estimate (rough)
        memory_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'memory_mb': memory_mb,
            'flops_breakdown': flops_breakdown,
            'total_flops': flops_breakdown['total'],
            'config': config
        }
    
    def run_single_experiment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single experiment with given parameter configuration.
        
        Returns performance metrics and complexity analysis.
        """
        print(f"🔧 Running experiment with config: {config}")
        
        try:
            # Build command to run experiment using subprocess
            cmd = [
                'python', 'experiments/train_link_prediction.py',
                '--model_name', self.fixed_config['model_name'],
                '--dataset_name', self.fixed_config['dataset_name'],
                '--time_encoder', self.fixed_config['time_encoder'],
                '--num_epochs', str(self.fixed_config['num_epochs']),
                '--num_runs', str(self.fixed_config['num_runs']),
                '--learning_rate', str(self.fixed_config['learning_rate']),
                '--patience', str(self.fixed_config['patience']),
                '--batch_size', str(self.fixed_config['batch_size']),
                '--seed', str(self.fixed_config['seed']),
                '--expert_dim', str(config['expert_dim']),
                '--mamba_d_state', str(config['mamba_d_state']),
                '--mamba_expand', str(config['mamba_expand']),
                '--mamba_headdim', str(config['mamba_headdim']),
                '--gpu', '0'
            ]
            
            if self.fixed_config['disable_progress_bar']:
                cmd.append('--disable_progress_bar')
            
            # Run experiment
            print(f"   Running command: {' '.join(cmd)}")
            result_output = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
            
            if result_output.returncode != 0:
                raise Exception(f"Experiment failed with return code {result_output.returncode}: {result_output.stderr}")
            
            # Parse results from output or saved files
            # Look for saved results file
            results_pattern = f"saved_results/{self.fixed_config['model_name']}_{self.fixed_config['dataset_name']}_*{self.fixed_config['time_encoder']}*.json"
            result_files = glob.glob(results_pattern)
            
            if result_files:
                # Load most recent result file
                latest_file = max(result_files, key=os.path.getctime)
                with open(latest_file, 'r') as f:
                    results = json.load(f)
                
                # Extract performance metrics - handle list format
                if isinstance(results, list) and len(results) > 0:
                    results = results[0]  # Take first run
                
                test_ap = results.get('test_ap', results.get('test_average_precision', 0.0))
                test_auc = results.get('test_auc', results.get('test_roc_auc', 0.0)) 
                val_ap = results.get('val_ap', results.get('val_average_precision', 0.0))
                val_auc = results.get('val_auc', results.get('val_roc_auc', 0.0))
                training_time = results.get('total_train_time', results.get('train_time', 0.0))
            else:
                # Parse from stdout if no file found
                output_lines = result_output.stdout.split('\n')
                test_auc = 0.0
                test_ap = 0.0
                val_auc = 0.0
                val_ap = 0.0
                training_time = 0.0
                
                for line in output_lines:
                    if 'Test AUC:' in line or 'test_auc' in line.lower():
                        try:
                            # Try multiple patterns
                            if 'Test AUC:' in line:
                                test_auc = float(line.split('Test AUC:')[1].split(',')[0].strip())
                            elif 'test_auc' in line.lower():
                                # Look for test_auc: value pattern
                                parts = line.lower().split('test_auc')
                                if len(parts) > 1:
                                    val_part = parts[1].strip().lstrip(':').strip()
                                    test_auc = float(val_part.split()[0].replace(',', ''))
                        except:
                            pass
                    elif 'Test AP:' in line or 'test_ap' in line.lower():
                        try:
                            if 'Test AP:' in line:
                                test_ap = float(line.split('Test AP:')[1].split(',')[0].strip())
                            elif 'test_ap' in line.lower():
                                parts = line.lower().split('test_ap')
                                if len(parts) > 1:
                                    val_part = parts[1].strip().lstrip(':').strip()
                                    test_ap = float(val_part.split()[0].replace(',', ''))
                        except:
                            pass
            
            # Get complexity metrics
            complexity = self._estimate_model_complexity(config)
            
            # Combine results
            result = {
                'config': config,
                'performance': {
                    'test_ap': test_ap,
                    'test_auc': test_auc,
                    'val_ap': val_ap,
                    'val_auc': val_auc,
                    'training_time': training_time
                },
                'complexity': complexity,
                'success': True
            }
            
            print(f"   ✅ Success - Test AUC: {test_auc:.4f}, Params: {complexity['total_params']:,}, FLOPs: {complexity['total_flops']:,}")
            
        except Exception as e:
            print(f"   ❌ Failed - Error: {str(e)}")
            complexity = self._estimate_model_complexity(config)
            result = {
                'config': config,
                'performance': {
                    'test_ap': 0.0,
                    'test_auc': 0.0,
                    'val_ap': 0.0,
                    'val_auc': 0.0,
                    'training_time': 0.0
                },
                'complexity': complexity,
                'success': False,
                'error': str(e)
            }
        
        return result
    
    def run_full_analysis(self) -> List[Dict[str, Any]]:
        """
        Run complete parameter analysis across all configurations.
        
        Returns list of results for all experiments.
        """
        print(f"\n🚀 Starting KAN-MAMMOTE Parameter Analysis")
        print(f"   Total experiments: {len(self.param_combinations)}")
        print(f"   Estimated time: ~{len(self.param_combinations) * 10:.0f} minutes")
        
        all_results = []
        
        for i, config in enumerate(tqdm(self.param_combinations, desc="Running experiments")):
            print(f"\n[{i+1}/{len(self.param_combinations)}] Configuration: {config}")
            
            result = self.run_single_experiment(config)
            all_results.append(result)
            
            # Save intermediate results every 5 experiments
            if (i + 1) % 5 == 0:
                self._save_intermediate_results(all_results, f"intermediate_{i+1}.json")
        
        # Save final results
        self._save_results(all_results)
        
        return all_results
    
    def _save_intermediate_results(self, results: List[Dict[str, Any]], filename: str):
        """Save intermediate results to prevent data loss."""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def _save_results(self, results: List[Dict[str, Any]]):
        """Save complete results to multiple formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. JSON format (complete data)
        json_file = os.path.join(self.output_dir, f"parameter_analysis_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # 2. CSV format (flattened for analysis)
        csv_data = []
        for result in results:
            if result['success']:
                row = {
                    'expert_dim': result['config']['expert_dim'],
                    'mamba_d_state': result['config']['mamba_d_state'],
                    'mamba_expand': result['config']['mamba_expand'],
                    'mamba_headdim': result['config']['mamba_headdim'],
                    'test_ap': result['performance']['test_ap'],
                    'test_auc': result['performance']['test_auc'],
                    'val_ap': result['performance']['val_ap'],
                    'val_auc': result['performance']['val_auc'],
                    'training_time': result['performance']['training_time'],
                    'total_params': result['complexity']['total_params'],
                    'total_flops': result['complexity']['total_flops'],
                    'memory_mb': result['complexity']['memory_mb']
                }
                csv_data.append(row)
        
        csv_file = os.path.join(self.output_dir, f"parameter_analysis_{timestamp}.csv")
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False)
        
        print(f"📊 Results saved:")
        print(f"   JSON: {json_file}")
        print(f"   CSV: {csv_file}")
        
        return json_file, csv_file
    
    def generate_analysis_plots(self, results: List[Dict[str, Any]]):
        """
        Generate comprehensive analysis plots.
        
        Creates multiple visualizations to analyze efficiency and performance.
        """
        # Filter successful results
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            print("❌ No successful results to plot")
            return
        
        # Extract data for plotting
        data = []
        for result in successful_results:
            data.append({
                'expert_dim': result['config']['expert_dim'],
                'mamba_d_state': result['config']['mamba_d_state'],
                'mamba_expand': result['config']['mamba_expand'],
                'mamba_headdim': result['config']['mamba_headdim'],
                'test_auc': result['performance']['test_auc'],
                'test_ap': result['performance']['test_ap'],
                'total_params': result['complexity']['total_params'],
                'total_flops': result['complexity']['total_flops'],
                'memory_mb': result['complexity']['memory_mb'],
                'training_time': result['performance']['training_time']
            })
        
        df = pd.DataFrame(data)
        
        # Create comprehensive figure
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('KAN-MAMMOTE Parameter Analysis: Performance vs Complexity', fontsize=16)
        
        # Plot 1: AUC vs Parameters
        ax1 = axes[0, 0]
        scatter1 = ax1.scatter(df['total_params'], df['test_auc'], 
                              c=df['expert_dim'], cmap='viridis', 
                              s=60, alpha=0.7)
        ax1.set_xlabel('Total Parameters')
        ax1.set_ylabel('Test AUC')
        ax1.set_title('Performance vs Parameter Count')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label='Expert Dim')
        
        # Plot 2: AUC vs FLOPs
        ax2 = axes[0, 1]
        scatter2 = ax2.scatter(df['total_flops'], df['test_auc'], 
                              c=df['mamba_d_state'], cmap='plasma', 
                              s=60, alpha=0.7)
        ax2.set_xlabel('Total FLOPs')
        ax2.set_ylabel('Test AUC')
        ax2.set_title('Performance vs Computational Cost')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='Mamba D State')
        
        # Plot 3: Efficiency Frontier (Pareto plot)
        ax3 = axes[0, 2]
        # Color by configuration type
        colors = []
        for _, row in df.iterrows():
            if row['total_params'] < 50000:
                colors.append('green')  # Efficient
            elif row['total_params'] < 200000:
                colors.append('orange')  # Medium
            else:
                colors.append('red')    # Large
        
        ax3.scatter(df['total_params'], df['test_auc'], c=colors, s=60, alpha=0.7)
        ax3.set_xlabel('Total Parameters')
        ax3.set_ylabel('Test AUC')
        ax3.set_title('Efficiency Analysis')
        ax3.grid(True, alpha=0.3)
        
        # Add legend for efficiency
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor='green', label='Efficient (<50K params)'),
            plt.Rectangle((0,0),1,1, facecolor='orange', label='Medium (50K-200K params)'),
            plt.Rectangle((0,0),1,1, facecolor='red', label='Large (>200K params)')
        ]
        ax3.legend(handles=legend_elements)
        
        # Plot 4: Parameter Impact Analysis
        ax4 = axes[1, 0]
        param_impact = df.groupby('expert_dim')['test_auc'].mean()
        ax4.bar(param_impact.index, param_impact.values, alpha=0.7)
        ax4.set_xlabel('Expert Dimension')
        ax4.set_ylabel('Mean Test AUC')
        ax4.set_title('Impact of Expert Dimension')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Memory vs Performance
        ax5 = axes[1, 1]
        scatter5 = ax5.scatter(df['memory_mb'], df['test_auc'], 
                              c=df['mamba_expand'], cmap='coolwarm', 
                              s=60, alpha=0.7)
        ax5.set_xlabel('Memory (MB)')
        ax5.set_ylabel('Test AUC')
        ax5.set_title('Memory Efficiency')
        ax5.grid(True, alpha=0.3)
        plt.colorbar(scatter5, ax=ax5, label='Mamba Expand')
        
        # Plot 6: Training Time vs Performance
        ax6 = axes[1, 2]
        scatter6 = ax6.scatter(df['training_time'], df['test_auc'], 
                              c=df['mamba_headdim'], cmap='rainbow', 
                              s=60, alpha=0.7)
        ax6.set_xlabel('Training Time (seconds)')
        ax6.set_ylabel('Test AUC')
        ax6.set_title('Training Efficiency')
        ax6.grid(True, alpha=0.3)
        plt.colorbar(scatter6, ax=ax6, label='Mamba Head Dim')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = os.path.join(self.output_dir, f"parameter_analysis_plots_{timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Analysis plots saved: {plot_file}")
        
        # Generate efficiency report
        self._generate_efficiency_report(df)
    
    def _generate_efficiency_report(self, df: pd.DataFrame):
        """Generate text-based efficiency analysis report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_dir, f"efficiency_report_{timestamp}.txt")
        
        with open(report_file, 'w') as f:
            f.write("KAN-MAMMOTE Parameter Analysis Report\n")
            f.write("="*50 + "\n\n")
            
            # Top performers by AUC
            f.write("TOP 5 CONFIGURATIONS BY TEST AUC:\n")
            f.write("-" * 40 + "\n")
            top_auc = df.nlargest(5, 'test_auc')
            for i, (_, row) in enumerate(top_auc.iterrows(), 1):
                f.write(f"{i}. AUC: {row['test_auc']:.4f} | ")
                f.write(f"Params: {row['total_params']:,} | ")
                f.write(f"expert_dim={row['expert_dim']}, ")
                f.write(f"mamba_d_state={row['mamba_d_state']}, ")
                f.write(f"mamba_expand={row['mamba_expand']}, ")
                f.write(f"mamba_headdim={row['mamba_headdim']}\n")
            
            # Most efficient configurations (high AUC per parameter)
            f.write("\nTOP 5 MOST EFFICIENT CONFIGURATIONS (AUC/Param):\n")
            f.write("-" * 50 + "\n")
            df['efficiency'] = df['test_auc'] / (df['total_params'] / 1000)  # AUC per 1K params
            top_efficient = df.nlargest(5, 'efficiency')
            for i, (_, row) in enumerate(top_efficient.iterrows(), 1):
                f.write(f"{i}. Efficiency: {row['efficiency']:.6f} | ")
                f.write(f"AUC: {row['test_auc']:.4f} | ")
                f.write(f"Params: {row['total_params']:,} | ")
                f.write(f"expert_dim={row['expert_dim']}, ")
                f.write(f"mamba_d_state={row['mamba_d_state']}, ")
                f.write(f"mamba_expand={row['mamba_expand']}, ")
                f.write(f"mamba_headdim={row['mamba_headdim']}\n")
            
            # Parameter impact analysis
            f.write("\nPARAMETER IMPACT ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            
            for param in ['expert_dim', 'mamba_d_state', 'mamba_expand', 'mamba_headdim']:
                impact = df.groupby(param)['test_auc'].agg(['mean', 'std', 'count'])
                f.write(f"\n{param.upper()}:\n")
                for value, stats in impact.iterrows():
                    f.write(f"  {value}: Mean AUC = {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})\n")
            
            # Key insights
            f.write("\nKEY INSIGHTS:\n")
            f.write("-" * 15 + "\n")
            
            # Best efficiency point
            best_efficient = df.loc[df['efficiency'].idxmax()]
            f.write(f"• Most efficient configuration achieves {best_efficient['test_auc']:.4f} AUC ")
            f.write(f"with only {best_efficient['total_params']:,} parameters\n")
            
            # Diminishing returns analysis
            param_corr = df['total_params'].corr(df['test_auc'])
            f.write(f"• Parameter-Performance correlation: {param_corr:.3f} ")
            if param_corr < 0.7:
                f.write("(Weak correlation suggests diminishing returns)\n")
            else:
                f.write("(Strong correlation suggests parameter scaling is important)\n")
            
            # FLOPs efficiency
            flops_corr = df['total_flops'].corr(df['test_auc'])
            f.write(f"• FLOPs-Performance correlation: {flops_corr:.3f}\n")
            
            f.write(f"\nReport generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"📋 Efficiency report saved: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='KAN-MAMMOTE Parameter Analysis')
    parser.add_argument('--output_dir', type=str, default='parameter_analysis_results',
                        help='Output directory for results')
    parser.add_argument('--quick_test', action='store_true',
                        help='Run only a few configurations for testing')
    parser.add_argument('--plot_only', type=str, default=None,
                        help='Path to existing results JSON file to generate plots only')
    
    args = parser.parse_args()
    
    if args.plot_only:
        # Load existing results and generate plots only
        with open(args.plot_only, 'r') as f:
            results = json.load(f)
        
        analyzer = KANMAMMOTEParameterAnalyzer(args.output_dir)
        analyzer.generate_analysis_plots(results)
        return
    
    # Create analyzer
    analyzer = KANMAMMOTEParameterAnalyzer(args.output_dir)
    
    if args.quick_test:
        # Use only first 3 configurations for testing
        analyzer.param_combinations = analyzer.param_combinations[:3]
        print(f"🧪 Quick test mode: running {len(analyzer.param_combinations)} configurations")
    
    # Run full analysis
    results = analyzer.run_full_analysis()
    
    # Generate plots and reports
    analyzer.generate_analysis_plots(results)
    
    print(f"\n✅ KAN-MAMMOTE Parameter Analysis Complete!")
    print(f"   Successful experiments: {sum(1 for r in results if r['success'])}/{len(results)}")
    print(f"   Results saved in: {analyzer.output_dir}")
    
    # Show top 3 configurations
    successful_results = [r for r in results if r['success']]
    if successful_results:
        successful_results.sort(key=lambda x: x['performance']['test_auc'], reverse=True)
        print(f"\n🏆 Top 3 Configurations by Test AUC:")
        for i, result in enumerate(successful_results[:3], 1):
            config = result['config']
            perf = result['performance']
            complexity = result['complexity']
            print(f"   {i}. AUC: {perf['test_auc']:.4f} | Params: {complexity['total_params']:,} | {config}")

if __name__ == '__main__':
    main()