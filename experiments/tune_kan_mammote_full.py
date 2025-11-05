#!/usr/bin/env python3
"""
KAN-MAMMOTE Full Hyperparameter Tuning for Stack Overflow Badge Prediction
===========================================================================

This script performs comprehensive hyperparameter tuning for the kan_mammote_full encoder
using grid search to find the optimal configuration for Stack Overflow badge prediction.

The tuning includes both architecture and training hyperparameters:

Architecture Parameters:
- expert_dim: KAN expert dimension (32, 64, 128, 256)
- mamba_d_state: Mamba state space dimension (64, 128, 256, 512)  
- mamba_expand: Mamba expansion factor (2, 4, 8)
- mamba_headdim: Mamba head dimension (16, 32, 64)
- embedding_dim: Overall embedding dimension (64, 128, 256)

Training Hyperparameters:
- learning_rate: Learning rate (1e-5 to 2e-3)
- batch_size: Batch size (16, 32, 64, 128, 256, 512)
- weight_decay: L2 regularization (0.0, 1e-6, 1e-5, 1e-4, 1e-3)

Tuning Modes:
- quick: Fast exploration with fewer parameter combinations
- comprehensive: Exhaustive search across all parameter combinations
- efficiency_focused: Focus on smaller, more efficient models
- training_focused: Focus on training hyperparameters with fixed architecture

Usage:
    python experiments/tune_kan_mammote_full.py --epochs 50 --batch_size 128 --split 1
    
Example with training-focused tuning:
    python experiments/tune_kan_mammote_full.py --tuning_mode training_focused --epochs 30 --split 1
    
Example with efficiency-focused tuning:
    python experiments/tune_kan_mammote_full.py --tuning_mode efficiency_focused --epochs 50 --split 1
"""

import os
import sys
import json
import time
import itertools
import argparse
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import the main experiment function
from experiments.stackoverflow_badge_prediction import run_experiment, get_available_encoders

class KANMAMMOTETuner:
    """
    Hyperparameter tuner for KAN-MAMMOTE Full encoder
    """
    
    def __init__(self, base_args, tuning_mode='comprehensive'):
        self.base_args = base_args
        self.tuning_mode = tuning_mode
        self.results = []
        
        # Define parameter search space
        if tuning_mode == 'quick':
            self.param_grid = {
                # Architecture parameters (using reasonable defaults)
                'expert_dim': [32, 64, 128, 256],
                'mamba_d_state': [64, 128, 256, 512],
                'mamba_expand': [2, 4, 8],
                'mamba_headdim': [16, 32, 64],
                'embedding_dim': [64, 128, 256],
                # Training hyperparameters
                'learning_rate': [1e-5,5e-5,1e-4, 5e-4,5e-3, 1e-3],
                'batch_size': [64, 128, 256,512],
                'weight_decay': [0.0]
            }
        elif tuning_mode == 'comprehensive':
            self.param_grid = {
                # Architecture parameters
                'expert_dim': [32, 64, 128, 256],
                'mamba_d_state': [64, 128, 256, 512],
                'mamba_expand': [2, 4, 8],
                'mamba_headdim': [16, 32, 64],
                'embedding_dim': [64, 128, 256],
                'encoder_dropout': [0.0, 0.1, 0.2],
                # Training hyperparameters
                'learning_rate': [5e-5, 1e-4, 2e-4, 5e-4, 1e-3],
                'batch_size': [32, 64, 128, 256, 512],
                #'weight_decay': [0.0, 1e-6, 1e-5, 1e-4, 1e-3]
            }
        elif tuning_mode == 'efficiency_focused':
            # Focus on smaller, more efficient configurations
            self.param_grid = {
                # Architecture parameters
                'expert_dim': [32, 64],
                'mamba_d_state': [64, 128, 256],
                'mamba_expand': [2, 4],
                'mamba_headdim': [16, 32],
                'embedding_dim': [64, 128],
                # Training hyperparameters
                'learning_rate': [1e-4, 5e-4],
                'batch_size': [128, 256],
                'weight_decay': [0.0, 1e-5]
            }
        elif tuning_mode == 'training_focused':
            # Focus primarily on training hyperparameters with fixed architecture
            self.param_grid = {
                # Fixed architecture (reasonable defaults)
                'expert_dim': [64, 128],
                'mamba_d_state': [128, 256],
                'mamba_expand': [2, 4],
                'mamba_headdim': [32],
                'embedding_dim': [128],
                # Extensive training hyperparameter search
                'learning_rate': [1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3],
                'batch_size': [16, 32, 64, 128, 256, 512],
                'weight_decay': [0.0, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
            }
        else:
            raise ValueError(f"Unknown tuning mode: {tuning_mode}")
        
        # Generate parameter combinations
        self.param_combinations = self._generate_param_combinations()
        
        print(f"🔧 KAN-MAMMOTE Full Hyperparameter Tuning")
        print(f"   Tuning mode: {tuning_mode}")
        print(f"   Parameter combinations: {len(self.param_combinations)}")
        print(f"   Architecture + Training hyperparameters included")
        print(f"   Estimated time per config: ~{base_args.epochs * 2} minutes")
        print(f"   Total estimated time: ~{len(self.param_combinations) * base_args.epochs * 2 / 60:.1f} hours")
    
    def _generate_param_combinations(self):
        """Generate all parameter combinations with validation"""
        combinations = []
        
        # Get all possible combinations
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        
        print(f"🔍 Debug: Parameter grid keys: {keys}")
        print(f"🔍 Debug: Parameter grid values: {[len(v) for v in values]}")
        
        total_combinations = 1
        for v in values:
            total_combinations *= len(v)
        print(f"🔍 Debug: Total possible combinations: {total_combinations}")
        
        valid_count = 0
        for combination in itertools.product(*values):
            params = dict(zip(keys, combination))
            
            # Validate parameter combination
            if self._is_valid_combination(params):
                combinations.append(params)
                valid_count += 1
            else:
                print(f"🔍 Debug: Invalid combination: {params}")
        
        print(f"🔍 Debug: Valid combinations: {valid_count}/{total_combinations}")
        return combinations
    
    def _is_valid_combination(self, params):
        """
        Validate parameter combinations to avoid invalid configurations
        """
        # Get required parameters with defaults for missing ones
        expert_dim = params.get('expert_dim', 64)
        mamba_headdim = params.get('mamba_headdim', 32)
        embedding_dim = params.get('embedding_dim', 128)
        mamba_d_state = params.get('mamba_d_state', 128)
        mamba_expand = params.get('mamba_expand', 2)
        learning_rate = params.get('learning_rate', 1e-4)
        batch_size = params.get('batch_size', 128)
        
        # For quick mode, be more permissive to ensure we have valid combinations
        if self.tuning_mode == 'quick':
            # Only apply essential rules for quick mode
            # Rule 1: mamba_headdim should divide expert_dim evenly
            if expert_dim % mamba_headdim != 0:
                return False
            
            # Rule 4: Ensure mamba_headdim is not larger than expert_dim
            if mamba_headdim > expert_dim:
                return False
                
            # Very basic memory sanity check
            param_product = expert_dim * mamba_d_state * mamba_expand * embedding_dim
            if param_product > 1_000_000_000:  # 1B parameter limit for quick mode
                return False
            
            return True
        
        # Full validation for other modes
        # Rule 1: mamba_headdim should divide expert_dim evenly
        if expert_dim % mamba_headdim != 0:
            return False
        
        # Rule 2: embedding_dim should be reasonable relative to expert_dim
        if embedding_dim < expert_dim // 4:
            return False
        
        # Rule 3: Avoid extremely large configurations (memory constraints)
        param_product = expert_dim * mamba_d_state * mamba_expand * embedding_dim
        if param_product > 500_000_000:  # More reasonable memory limit (500M vs 50M)
            return False
        
        # Rule 4: Ensure mamba_headdim is not larger than expert_dim
        if mamba_headdim > expert_dim:
            return False
        
        # Rule 5: Memory constraints based on batch size (relaxed significantly)
        # Larger batch sizes require smaller models
        memory_factor = batch_size * param_product
        if memory_factor > 10_000_000_000:  # 10B vs 100M - much more realistic for GPU memory
            return False
        
        # Rule 6: Very small learning rates with large batch sizes can be problematic (relaxed)
        if learning_rate < 5e-5 and batch_size > 512:  # More permissive
            return False
        
        # Rule 7: Very large learning rates with small batch sizes can be unstable (relaxed)
        if learning_rate > 2e-3 and batch_size < 32:  # More permissive
            return False
        
        return True
    
    def run_tuning(self, experiment_dir):
        """
        Run hyperparameter tuning experiment
        """
        print(f"\n🚀 Starting KAN-MAMMOTE Full hyperparameter tuning")
        print(f"   Total configurations to test: {len(self.param_combinations)}")
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tuning_dir = os.path.join(experiment_dir, f"kan_mammote_tuning_{timestamp}")
        os.makedirs(tuning_dir, exist_ok=True)
        
        models_dir = os.path.join(tuning_dir, "models")
        checkpoint_dir = os.path.join(tuning_dir, "checkpoints")
        
        print(f"📂 Tuning directory: {tuning_dir}")
        
        # Run experiments for each parameter combination
        for i, params in enumerate(self.param_combinations):
            print(f"\n{'='*80}")
            print(f"Configuration {i+1}/{len(self.param_combinations)}")
            print(f"Parameters: {params}")
            print(f"{'='*80}")
            
            # Update args with current parameters
            tuning_args = self._create_tuning_args(params)
            
            # Create a unique name for this configuration
            config_name = f"kan_mammote_full_{i+1:03d}"
            
            # Run experiment
            start_time = time.time()
            
            try:
                result = run_experiment(
                    encoder_name='kan_mammote_full',
                    args=tuning_args,
                    models_dir=models_dir,
                    checkpoint_dir=checkpoint_dir
                )
                
                if result:
                    # Add parameter configuration to result
                    result['parameters'] = params
                    result['config_id'] = i + 1
                    result['config_name'] = config_name
                    result['training_time_minutes'] = (time.time() - start_time) / 60
                    
                    self.results.append(result)
                    
                    print(f"✅ Configuration {i+1} completed successfully")
                    print(f"   Best Val MRR: {result['best_val_mrr']:.4f}")
                    print(f"   Best Val Acc: {result['best_val_acc']:.4f}")
                    print(f"   Training time: {result['training_time_minutes']:.1f} minutes")
                    
                else:
                    print(f"❌ Configuration {i+1} failed")
                
            except Exception as e:
                print(f"❌ Configuration {i+1} failed with error: {str(e)}")
                
                # Save failed configuration info
                failed_result = {
                    'config_id': i + 1,
                    'config_name': config_name,
                    'parameters': params,
                    'status': 'failed',
                    'error': str(e),
                    'training_time_minutes': (time.time() - start_time) / 60
                }
                self.results.append(failed_result)
            
            # Save intermediate results
            self._save_intermediate_results(tuning_dir)
        
        # Generate final analysis
        self._generate_tuning_analysis(tuning_dir)
        
        return self.results
    
    def _create_tuning_args(self, params):
        """Create args object with tuning parameters"""
        import copy
        tuning_args = copy.deepcopy(self.base_args)
        
        # Update with current parameter combination
        # Architecture parameters (use defaults if not in params)
        tuning_args.expert_dim = params.get('expert_dim', getattr(tuning_args, 'expert_dim', 64))
        tuning_args.mamba_d_state = params.get('mamba_d_state', getattr(tuning_args, 'mamba_d_state', 128))
        tuning_args.mamba_expand = params.get('mamba_expand', getattr(tuning_args, 'mamba_expand', 2))
        tuning_args.mamba_headdim = params.get('mamba_headdim', getattr(tuning_args, 'mamba_headdim', 32))
        tuning_args.embedding_dim = params.get('embedding_dim', getattr(tuning_args, 'embedding_dim', 128))
        
        # Training hyperparameters
        tuning_args.learning_rate = params.get('learning_rate', tuning_args.learning_rate)
        tuning_args.batch_size = params.get('batch_size', tuning_args.batch_size)
        tuning_args.weight_decay = params.get('weight_decay', tuning_args.weight_decay)
        
        return tuning_args
    
    def _save_intermediate_results(self, tuning_dir):
        """Save intermediate results to prevent data loss"""
        results_file = os.path.join(tuning_dir, "intermediate_results.json")
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
    
    def _generate_tuning_analysis(self, tuning_dir):
        """Generate comprehensive analysis of tuning results"""
        print(f"\n📊 Generating tuning analysis...")
        
        # Filter successful results
        successful_results = [r for r in self.results if 'best_val_mrr' in r]
        
        if not successful_results:
            print("❌ No successful configurations found!")
            return
        
        # Create analysis dataframe
        analysis_data = []
        for result in successful_results:
            params = result['parameters']
            
            analysis_data.append({
                'config_id': result['config_id'],
                'config_name': result['config_name'],
                # Architecture parameters (with defaults for missing ones)
                'expert_dim': params.get('expert_dim', 64),
                'mamba_d_state': params.get('mamba_d_state', 128),
                'mamba_expand': params.get('mamba_expand', 2),
                'mamba_headdim': params.get('mamba_headdim', 32),
                'embedding_dim': params.get('embedding_dim', 128),
                # Training hyperparameters
                'learning_rate': params.get('learning_rate', 1e-4),
                'batch_size': params.get('batch_size', 128),
                'weight_decay': params.get('weight_decay', 0.0),
                # Performance metrics
                'best_val_mrr': result['best_val_mrr'],
                'best_val_acc': result['best_val_acc'],
                'best_val_recall3': result['best_val_recall3'],
                'num_parameters': result['num_parameters'],
                'training_time_minutes': result['training_time_minutes'],
                'epochs_trained': len(result['history']['train_loss']),
                'encoder_dropout': params.get('encoder_dropout', 0.0)
            })
        
        df = pd.DataFrame(analysis_data)
        
        # Sort by best validation MRR
        df_sorted = df.sort_values('best_val_mrr', ascending=False).reset_index(drop=True)
        df_sorted['rank'] = df_sorted.index + 1
        
        # Save detailed results
        csv_file = os.path.join(tuning_dir, "tuning_results_detailed.csv")
        df_sorted.to_csv(csv_file, index=False)
        
        # Generate summary analysis
        self._generate_summary_analysis(df_sorted, tuning_dir)
        
        # Generate parameter impact analysis
        self._generate_parameter_impact_analysis(df_sorted, tuning_dir)
        
        # Save best configurations
        self._save_best_configurations(df_sorted, tuning_dir)
        
        print(f"📋 Tuning analysis saved to: {tuning_dir}")
    
    def _generate_summary_analysis(self, df, tuning_dir):
        """Generate summary analysis"""
        summary = []
        
        summary.append("🏆 KAN-MAMMOTE Full Hyperparameter Tuning Results")
        summary.append("=" * 60)
        summary.append("")
        
        # Best configuration
        best_config = df.iloc[0]
        summary.append(f"🥇 BEST CONFIGURATION:")
        summary.append(f"   Config ID: {best_config['config_id']}")
        summary.append(f"   Best Val MRR: {best_config['best_val_mrr']:.4f}")
        summary.append(f"   Best Val Acc: {best_config['best_val_acc']:.2f}%")
        summary.append(f"   Best Val Recall@3: {best_config['best_val_recall3']:.4f}")
        summary.append(f"   Parameters:")
        summary.append(f"     Architecture:")
        summary.append(f"       - expert_dim: {best_config['expert_dim']}")
        summary.append(f"       - mamba_d_state: {best_config['mamba_d_state']}")
        summary.append(f"       - mamba_expand: {best_config['mamba_expand']}")
        summary.append(f"       - mamba_headdim: {best_config['mamba_headdim']}")
        summary.append(f"       - embedding_dim: {best_config['embedding_dim']}")
        summary.append(f"     Training:")
        summary.append(f"       - learning_rate: {best_config['learning_rate']}")
        summary.append(f"       - batch_size: {best_config['batch_size']}")
        summary.append(f"       - weight_decay: {best_config['weight_decay']}")
        summary.append(f"   Model size: {best_config['num_parameters']:,} parameters")
        summary.append(f"   Training time: {best_config['training_time_minutes']:.1f} minutes")
        summary.append("")
        
        # Top 5 configurations
        summary.append(f"🏅 TOP 5 CONFIGURATIONS:")
        for i, (_, config) in enumerate(df.head(5).iterrows()):
            summary.append(f"   #{i+1}: MRR={config['best_val_mrr']:.4f}, "
                          f"expert_dim={config['expert_dim']}, "
                          f"lr={config['learning_rate']}, "
                          f"bs={config['batch_size']}, "
                          f"wd={config['weight_decay']}, "
                          f"params={config['num_parameters']:,}")
        summary.append("")
        
        # Performance statistics
        summary.append(f"📊 PERFORMANCE STATISTICS:")
        summary.append(f"   Best MRR: {df['best_val_mrr'].max():.4f}")
        summary.append(f"   Worst MRR: {df['best_val_mrr'].min():.4f}")
        summary.append(f"   Mean MRR: {df['best_val_mrr'].mean():.4f}")
        summary.append(f"   Std MRR: {df['best_val_mrr'].std():.4f}")
        summary.append(f"   MRR Improvement: {((df['best_val_mrr'].max() - df['best_val_mrr'].min()) / df['best_val_mrr'].min() * 100):.1f}%")
        summary.append("")
        
        # Parameter size statistics
        summary.append(f"📈 MODEL SIZE STATISTICS:")
        summary.append(f"   Smallest model: {df['num_parameters'].min():,} parameters")
        summary.append(f"   Largest model: {df['num_parameters'].max():,} parameters")
        summary.append(f"   Mean model size: {df['num_parameters'].mean():.0f} parameters")
        summary.append("")
        
        # Training time statistics
        summary.append(f"⏱️ TRAINING TIME STATISTICS:")
        summary.append(f"   Fastest training: {df['training_time_minutes'].min():.1f} minutes")
        summary.append(f"   Slowest training: {df['training_time_minutes'].max():.1f} minutes")
        summary.append(f"   Mean training time: {df['training_time_minutes'].mean():.1f} minutes")
        summary.append("")
        
        # Save summary
        summary_file = os.path.join(tuning_dir, "tuning_summary.txt")
        with open(summary_file, 'w') as f:
            f.write('\n'.join(summary))
        
        # Print to console
        print('\n'.join(summary))
    
    def _generate_parameter_impact_analysis(self, df, tuning_dir):
        """Analyze the impact of each parameter on performance"""
        impact_analysis = []
        
        impact_analysis.append("🔍 PARAMETER IMPACT ANALYSIS")
        impact_analysis.append("=" * 40)
        impact_analysis.append("")
        
        # Analyze each parameter (only those that vary in the experiment)
        all_architecture_params = ['expert_dim', 'mamba_d_state', 'mamba_expand', 'mamba_headdim', 'embedding_dim']
        all_training_params = ['learning_rate', 'batch_size', 'weight_decay']
        
        # Filter to only include parameters that actually vary
        architecture_params = [p for p in all_architecture_params if len(df[p].unique()) > 1]
        training_params = [p for p in all_training_params if len(df[p].unique()) > 1]
        
        if architecture_params:
            impact_analysis.append("🏗️ ARCHITECTURE PARAMETERS:")
            for param in architecture_params:
                param_stats = df.groupby(param)['best_val_mrr'].agg(['mean', 'std', 'min', 'max', 'count'])
                
                impact_analysis.append(f"📊 {param.upper()}:")
                for value in sorted(df[param].unique()):
                    stats = param_stats.loc[value]
                    impact_analysis.append(f"   {value}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                                         f"count={stats['count']}")
                
                # Find best value for this parameter
                best_value = param_stats['mean'].idxmax()
                best_mean = param_stats['mean'].max()
                impact_analysis.append(f"   🏆 Best {param}: {best_value} (mean MRR: {best_mean:.4f})")
                impact_analysis.append("")
        
        if training_params:
            impact_analysis.append("🎯 TRAINING HYPERPARAMETERS:")
            for param in training_params:
                param_stats = df.groupby(param)['best_val_mrr'].agg(['mean', 'std', 'min', 'max', 'count'])
                
                impact_analysis.append(f"📊 {param.upper()}:")
                for value in sorted(df[param].unique()):
                    stats = param_stats.loc[value]
                    if param == 'learning_rate':
                        value_str = f"{value:.0e}"  # Scientific notation for learning rate
                    elif param == 'weight_decay':
                        value_str = f"{value:.0e}" if value > 0 else "0.0"
                    else:
                        value_str = str(value)
                    impact_analysis.append(f"   {value_str}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                                         f"count={stats['count']}")
                
                # Find best value for this parameter
                best_value = param_stats['mean'].idxmax()
                best_mean = param_stats['mean'].max()
                if param == 'learning_rate' or param == 'weight_decay':
                    best_value_str = f"{best_value:.0e}" if best_value > 0 else "0.0"
                else:
                    best_value_str = str(best_value)
                impact_analysis.append(f"   🏆 Best {param}: {best_value_str} (mean MRR: {best_mean:.4f})")
                impact_analysis.append("")
        
        # Parameter correlations (only for parameters that vary)
        all_varying_params = architecture_params + training_params
        if all_varying_params:
            impact_analysis.append("📈 PARAMETER CORRELATIONS WITH MRR:")
            if architecture_params:
                impact_analysis.append("Architecture Parameters:")
                for param in architecture_params:
                    corr = df[param].corr(df['best_val_mrr'])
                    impact_analysis.append(f"   {param}: {corr:.3f}")
            if training_params:
                impact_analysis.append("Training Hyperparameters:")
                for param in training_params:
                    corr = df[param].corr(df['best_val_mrr'])
                    impact_analysis.append(f"   {param}: {corr:.3f}")
            impact_analysis.append("")
        
        # Save parameter impact analysis
        impact_file = os.path.join(tuning_dir, "parameter_impact_analysis.txt")
        with open(impact_file, 'w') as f:
            f.write('\n'.join(impact_analysis))
    
    def _save_best_configurations(self, df, tuning_dir):
        """Save the best configurations in easy-to-use format"""
        # Top 5 configurations
        top_5 = df.head(5)
        
        best_configs = {
            'best_configuration': {
                'architecture_parameters': {
                    'expert_dim': int(top_5.iloc[0]['expert_dim']),
                    'mamba_d_state': int(top_5.iloc[0]['mamba_d_state']),
                    'mamba_expand': int(top_5.iloc[0]['mamba_expand']),
                    'mamba_headdim': int(top_5.iloc[0]['mamba_headdim']),
                    'embedding_dim': int(top_5.iloc[0]['embedding_dim'])
                },
                'training_hyperparameters': {
                    'learning_rate': float(top_5.iloc[0]['learning_rate']),
                    'batch_size': int(top_5.iloc[0]['batch_size']),
                    'weight_decay': float(top_5.iloc[0]['weight_decay'])
                },
                'performance': {
                    'best_val_mrr': float(top_5.iloc[0]['best_val_mrr']),
                    'best_val_acc': float(top_5.iloc[0]['best_val_acc']),
                    'best_val_recall3': float(top_5.iloc[0]['best_val_recall3'])
                },
                'model_info': {
                    'num_parameters': int(top_5.iloc[0]['num_parameters']),
                    'training_time_minutes': float(top_5.iloc[0]['training_time_minutes'])
                }
            },
            'top_5_configurations': []
        }
        
        for i, (_, config) in enumerate(top_5.iterrows()):
            best_configs['top_5_configurations'].append({
                'rank': i + 1,
                'architecture_parameters': {
                    'expert_dim': int(config['expert_dim']),
                    'mamba_d_state': int(config['mamba_d_state']),
                    'mamba_expand': int(config['mamba_expand']),
                    'mamba_headdim': int(config['mamba_headdim']),
                    'embedding_dim': int(config['embedding_dim'])
                },
                'training_hyperparameters': {
                    'learning_rate': float(config['learning_rate']),
                    'batch_size': int(config['batch_size']),
                    'weight_decay': float(config['weight_decay'])
                },
                'performance': {
                    'best_val_mrr': float(config['best_val_mrr']),
                    'best_val_acc': float(config['best_val_acc']),
                    'best_val_recall3': float(config['best_val_recall3'])
                },
                'model_info': {
                    'num_parameters': int(config['num_parameters']),
                    'training_time_minutes': float(config['training_time_minutes'])
                }
            })
        
        # Save best configurations
        best_configs_file = os.path.join(tuning_dir, "best_configurations.json")
        with open(best_configs_file, 'w') as f:
            json.dump(best_configs, f, indent=2)
        
        print(f"🏆 Best configurations saved to: {best_configs_file}")


def main():
    parser = argparse.ArgumentParser(description='KAN-MAMMOTE Full Hyperparameter Tuning')
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, 
                        default='NeuralPointProcess-master/data/real/so',
                        help='Path to Stack Overflow data directory')
    parser.add_argument('--split', type=int, default=1, choices=[1,2,3,4,5],
                        help='Data split to use (1-5)')
    
    # Training parameters (base values, will be overridden during tuning)
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs per configuration (default: 50)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Base batch size (will be tuned if training_focused mode)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Base learning rate (will be tuned)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Base weight decay (will be tuned)')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='LSTM hidden dimension (default: 128)')
    
    # Tuning configuration
    parser.add_argument('--tuning_mode', type=str, default='comprehensive',
                        choices=['quick', 'comprehensive', 'efficiency_focused', 'training_focused'],
                        help='Tuning mode: quick (few configs), comprehensive (many configs), '
                             'efficiency_focused (small models), training_focused (training hyperparams)')
    parser.add_argument('--experiment_dir', type=str, default='kan_mammote_tuning',
                        help='Directory to save tuning results')
    
    # Optional training features
    parser.add_argument('--use_amp', action='store_true',
                        help='Enable CUDA Automatic Mixed Precision')
    parser.add_argument('--max_sequence_length', type=int, default=100,
                        help='Maximum sequence length (default: 100)')
    parser.add_argument('--min_sequence_length', type=int, default=3,
                        help='Minimum sequence length (default: 3)')
    parser.add_argument('--normalize_time', action='store_true', default=True,
                        help='Normalize timestamps')
    parser.add_argument('--use_proper_split', action='store_true',
                        help='Use proper 3-way split')
    parser.add_argument('--val_ratio', type=float, default=0.3,
                        help='Validation ratio for proper split')
    
    # Fixed parameters (not tuned)
    parser.add_argument('--mamba_d_conv', type=int, default=4,
                        help='Mamba convolution dimension (fixed)')
    parser.add_argument('--wavelet_type', type=str, default='shock',
                        help='Wavelet type (fixed)')
    parser.add_argument('--num_mixtures', type=int, default=16,
                        help='Number of mixtures (fixed)')
    parser.add_argument('--resume_training', action='store_false',
                        help='Disable resume training for clean tuning')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print(f"🔧 KAN-MAMMOTE Full Hyperparameter Tuning")
    print(f"📁 Data directory: {args.data_dir}")
    print(f"📊 Using split: {args.split}")
    print(f"🎯 Tuning mode: {args.tuning_mode}")
    print(f"⏱️ Epochs per configuration: {args.epochs}")
    
    # Verify kan_mammote_full is available
    available_encoders = get_available_encoders()
    if 'kan_mammote_full' not in available_encoders:
        print("❌ kan_mammote_full encoder not available!")
        print(f"Available encoders: {available_encoders}")
        return
    
    # Create tuner
    tuner = KANMAMMOTETuner(args, tuning_mode=args.tuning_mode)
    
    # Run tuning
    results = tuner.run_tuning(args.experiment_dir)
    
    print(f"\n✅ Hyperparameter tuning completed!")
    print(f"📊 Total configurations tested: {len(results)}")
    
    successful_results = [r for r in results if 'best_val_mrr' in r]
    if successful_results:
        best_result = max(successful_results, key=lambda x: x['best_val_mrr'])
        print(f"🏆 Best configuration achieved MRR: {best_result['best_val_mrr']:.4f}")
        print(f"   Best parameters: {best_result['parameters']}")
    else:
        print("❌ No successful configurations found!")


if __name__ == '__main__':
    main()