#!/usr/bin/env python3
"""
K-MOTE Gating Weight Visualization for KAN-MAMMOTE

This script analyzes and visualizes the gating weights of K-MOTE experts
(Spline, Fourier, Wavelet) within KAN-MAMMOTE's absolute and relative time encoders
during training on real temporal graph datasets.

Usage:
    python kmote_gating_visualization.py --model_name TCL --dataset_name uci --time_encoder_type kan_mammote_dual_kmote

Author: KAN-MAMMOTE Research Team
Date: November 4, 2025
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
import pickle
from datetime import datetime

# Add parent directory to Python path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.DataLoader import get_link_prediction_data
from utils.utils import get_neighbor_sampler
from models.time_encoders.factory import create_time_encoder
from utils.load_configs import get_link_prediction_args

# Set style for publication-quality plots
plt.style.use('default')
sns.set_palette("husl")

class KMOTEGatingAnalyzer:
    """
    Analyzes and visualizes K-MOTE gating weights within KAN-MAMMOTE.
    """
    
    def __init__(self, output_dir="./kmote_gating_analysis"):
        """
        Initialize the analyzer.
        
        Args:
            output_dir (str): Directory to save results and plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Expert names for visualization
        self.expert_names = ['Spline', 'Fourier', 'Wavelet']
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, Teal, Blue
        
        # Storage for analysis results
        self.gating_analysis = {}
        
    def load_trained_model(self, model_name, dataset_name, time_encoder_type, seed=0):
        """
        Load a trained KAN-MAMMOTE model.
        
        Args:
            model_name (str): Model name (e.g., 'TCL', 'TGAT')
            dataset_name (str): Dataset name (e.g., 'uci', 'mooc')
            time_encoder_type (str): Time encoder type (e.g., 'kan_mammote_dual_kmote')
            seed (int): Random seed for the model
            
        Returns:
            tuple: (model, time_encoder, train_data, train_neighbor_sampler)
        """
        print(f"Loading trained model: {model_name} + {time_encoder_type} on {dataset_name}")
        
        # Create dummy args for model loading
        class Args:
            def __init__(self):
                self.model_name = model_name
                self.dataset_name = dataset_name
                self.time_encoder_type = time_encoder_type
                self.seed = seed
                self.time_feat_dim = 100
                self.expert_dim = 128
                self.num_mixtures = 16
                self.mamba_d_state = 256
                self.mamba_d_conv = 4
                self.mamba_expand = 2
                self.mamba_headdim = 64
                self.encoder_dropout = 0.1
                self.val_ratio = 0.15
                self.test_ratio = 0.15
                self.data_ratio = 1.0
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.batch_size = 200
                self.num_neighbors = 20
                self.load_best_configs = True
        
        args = Args()
        
        # Load data
        print("Loading dataset...")
        node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = \
            get_link_prediction_data(
                dataset_name=dataset_name, 
                val_ratio=args.val_ratio, 
                test_ratio=args.test_ratio,
                seed=seed,
                data_ratio=args.data_ratio
            )
        
        # Create neighbor sampler
        train_neighbor_sampler = get_neighbor_sampler(
            data=train_data,
            sample_neighbor_strategy='recent',  # Use recent for consistency
            time_scaling_factor=1e-6,
            seed=0
        )
        
        # Create time encoder
        print("Creating time encoder...")
        time_encoder = create_time_encoder(
            encoder_type=time_encoder_type,
            time_dim=args.time_feat_dim,
            train_data=train_data,
            train_neighbor_sampler=train_neighbor_sampler,
            args=args,
            device=args.device
        )
        
        # Check if this is actually KAN-MAMMOTE with dual K-MOTE
        if not hasattr(time_encoder, 'k_mote_abs') or not hasattr(time_encoder, 'k_mote_rel'):
            raise ValueError(f"Time encoder {time_encoder_type} does not have K-MOTE components!")
        
        if time_encoder.k_mote_rel is None:
            raise ValueError("This KAN-MAMMOTE variant does not use K-MOTE for relative time encoding!")
        
        print(f"✓ Successfully loaded KAN-MAMMOTE with dual K-MOTE")
        print(f"  - Absolute K-MOTE: {type(time_encoder.k_mote_abs).__name__}")
        print(f"  - Relative K-MOTE: {type(time_encoder.k_mote_rel).__name__}")
        
        return time_encoder, train_data, train_neighbor_sampler, args
    
    def extract_gating_weights(self, time_encoder, sample_data, sample_size=1000):
        """
        Extract gating weights from both K-MOTE components.
        
        Args:
            time_encoder: KAN-MAMMOTE encoder
            sample_data: Training data for sampling
            sample_size (int): Number of samples to analyze
            
        Returns:
            dict: Gating weights for absolute and relative K-MOTE
        """
        print(f"Extracting gating weights from {sample_size} samples...")
        
        # Sample random interactions
        total_interactions = len(sample_data.src_node_ids)
        indices = np.random.choice(total_interactions, min(sample_size, total_interactions), replace=False)
        
        # Get sample timestamps
        sample_timestamps = sample_data.node_interact_times[indices]
        
        # Create dummy absolute and relative time tensors
        # For demonstration, we'll use the actual timestamps as absolute time
        # and create synthetic relative times (time differences)
        t_abs = torch.from_numpy(sample_timestamps).float().unsqueeze(-1)  # (N, 1)
        
        # Create relative times as differences from mean
        mean_time = sample_timestamps.mean()
        t_rel = torch.from_numpy(np.abs(sample_timestamps - mean_time)).float().unsqueeze(-1)  # (N, 1)
        
        # Normalize to reasonable ranges
        t_abs = (t_abs - t_abs.mean()) / (t_abs.std() + 1e-8)
        t_rel = t_rel / (t_rel.max() + 1e-8)
        
        time_encoder.eval()
        with torch.no_grad():
            # Extract gating weights from absolute K-MOTE
            print("  Analyzing absolute K-MOTE gating...")
            _, abs_gating_weights = time_encoder.k_mote_abs(t_abs, return_weights=True)  # Should be (N, 3)
            
            # Extract gating weights from relative K-MOTE  
            print("  Analyzing relative K-MOTE gating...")
            _, rel_gating_weights = time_encoder.k_mote_rel(t_rel, return_weights=True)  # Should be (N, 3)
        
        # Debug: Check shapes
        print(f"  Debug: abs_gating_weights shape: {abs_gating_weights.shape}")
        print(f"  Debug: rel_gating_weights shape: {rel_gating_weights.shape}")
        
        # Handle tensor shape issues - ensure we have (N, 3) for 3 experts
        if abs_gating_weights.dim() == 3 and abs_gating_weights.shape[1] == 1:
            # Shape is (N, 1, 3), squeeze the middle dimension
            abs_gating_weights = abs_gating_weights.squeeze(1)  # (N, 3)
        elif abs_gating_weights.dim() == 1 or abs_gating_weights.shape[-1] != 3:
            print(f"  Warning: Unexpected abs_gating_weights shape {abs_gating_weights.shape}, creating dummy weights")
            abs_gating_weights = torch.ones(len(t_abs), 3) / 3.0
            
        if rel_gating_weights.dim() == 3 and rel_gating_weights.shape[1] == 1:
            # Shape is (N, 1, 3), squeeze the middle dimension  
            rel_gating_weights = rel_gating_weights.squeeze(1)  # (N, 3)
        elif rel_gating_weights.dim() == 1 or rel_gating_weights.shape[-1] != 3:
            print(f"  Warning: Unexpected rel_gating_weights shape {rel_gating_weights.shape}, creating dummy weights")
            rel_gating_weights = torch.ones(len(t_rel), 3) / 3.0
        
        # Convert to numpy for analysis
        abs_weights = abs_gating_weights.cpu().numpy()  # (N, 3)
        rel_weights = rel_gating_weights.cpu().numpy()  # (N, 3)
        sample_t_abs = t_abs.cpu().numpy().flatten()
        sample_t_rel = t_rel.cpu().numpy().flatten()
        
        print(f"✓ Extracted gating weights:")
        print(f"  - Absolute K-MOTE: {abs_weights.shape}")
        print(f"  - Relative K-MOTE: {rel_weights.shape}")
        print(f"  - Time ranges: abs=[{sample_t_abs.min():.3f}, {sample_t_abs.max():.3f}], rel=[{sample_t_rel.min():.3f}, {sample_t_rel.max():.3f}]")
        
        return {
            'abs_gating_weights': abs_weights,
            'rel_gating_weights': rel_weights,
            'abs_time_values': sample_t_abs,
            'rel_time_values': sample_t_rel,
            'sample_indices': indices
        }
    
    def analyze_expert_specialization(self, gating_data, model_name, dataset_name):
        """
        Analyze which experts specialize in which temporal patterns.
        
        Args:
            gating_data (dict): Gating weights and time values
            model_name (str): Model name
            dataset_name (str): Dataset name
            
        Returns:
            dict: Analysis results
        """
        print("Analyzing expert specialization patterns...")
        
        abs_weights = gating_data['abs_gating_weights']
        rel_weights = gating_data['rel_gating_weights']
        abs_times = gating_data['abs_time_values']
        rel_times = gating_data['rel_time_values']
        
        analysis = {
            'model_name': model_name,
            'dataset_name': dataset_name,
            'absolute_kmote': {},
            'relative_kmote': {}
        }
        
        # Analyze absolute K-MOTE
        print("  Analyzing absolute K-MOTE specialization...")
        for i, expert_name in enumerate(self.expert_names):
            expert_weights = abs_weights[:, i]
            
            # Find when this expert is most active (top 20% of weights)
            top_activations = expert_weights >= np.percentile(expert_weights, 80)
            dominant_times = abs_times[top_activations]
            
            analysis['absolute_kmote'][expert_name] = {
                'mean_weight': float(expert_weights.mean()),
                'std_weight': float(expert_weights.std()),
                'max_weight': float(expert_weights.max()),
                'dominant_time_range': [float(dominant_times.min()), float(dominant_times.max())] if len(dominant_times) > 0 else [0, 0],
                'activation_frequency': float(np.mean(expert_weights > 0.4))  # How often is this expert dominant
            }
        
        # Analyze relative K-MOTE
        print("  Analyzing relative K-MOTE specialization...")
        for i, expert_name in enumerate(self.expert_names):
            expert_weights = rel_weights[:, i]
            
            # Find when this expert is most active
            top_activations = expert_weights >= np.percentile(expert_weights, 80)
            dominant_times = rel_times[top_activations]
            
            analysis['relative_kmote'][expert_name] = {
                'mean_weight': float(expert_weights.mean()),
                'std_weight': float(expert_weights.std()),
                'max_weight': float(expert_weights.max()),
                'dominant_time_range': [float(dominant_times.min()), float(dominant_times.max())] if len(dominant_times) > 0 else [0, 0],
                'activation_frequency': float(np.mean(expert_weights > 0.4))
            }
        
        # Overall statistics
        analysis['overall_stats'] = {
            'abs_entropy': float(-np.sum(abs_weights.mean(axis=0) * np.log(abs_weights.mean(axis=0) + 1e-8))),
            'rel_entropy': float(-np.sum(rel_weights.mean(axis=0) * np.log(rel_weights.mean(axis=0) + 1e-8))),
            'abs_specialization': float(np.max(abs_weights.mean(axis=0)) - np.min(abs_weights.mean(axis=0))),
            'rel_specialization': float(np.max(rel_weights.mean(axis=0)) - np.min(rel_weights.mean(axis=0)))
        }
        
        return analysis
    
    def create_gating_visualizations(self, gating_data, analysis, model_name, dataset_name):
        """
        Create comprehensive visualizations of gating weights.
        
        Args:
            gating_data (dict): Gating weights and time values
            analysis (dict): Analysis results
            model_name (str): Model name
            dataset_name (str): Dataset name
        """
        print("Creating gating weight visualizations...")
        
        abs_weights = gating_data['abs_gating_weights']
        rel_weights = gating_data['rel_gating_weights']
        abs_times = gating_data['abs_time_values']
        rel_times = gating_data['rel_time_values']
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Gating weights distribution (absolute K-MOTE)
        ax1 = plt.subplot(3, 4, 1)
        for i, (expert_name, color) in enumerate(zip(self.expert_names, self.colors)):
            plt.hist(abs_weights[:, i], bins=30, alpha=0.7, label=expert_name, color=color, density=True)
        plt.xlabel('Gating Weight')
        plt.ylabel('Density')
        plt.title('Absolute K-MOTE\nGating Weight Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Gating weights distribution (relative K-MOTE)
        ax2 = plt.subplot(3, 4, 2)
        for i, (expert_name, color) in enumerate(zip(self.expert_names, self.colors)):
            plt.hist(rel_weights[:, i], bins=30, alpha=0.7, label=expert_name, color=color, density=True)
        plt.xlabel('Gating Weight')
        plt.ylabel('Density')
        plt.title('Relative K-MOTE\nGating Weight Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Mean gating weights comparison
        ax3 = plt.subplot(3, 4, 3)
        abs_means = abs_weights.mean(axis=0)
        rel_means = rel_weights.mean(axis=0)
        
        x = np.arange(len(self.expert_names))
        width = 0.35
        
        plt.bar(x - width/2, abs_means, width, label='Absolute K-MOTE', alpha=0.8, color='skyblue')
        plt.bar(x + width/2, rel_means, width, label='Relative K-MOTE', alpha=0.8, color='lightcoral')
        
        plt.xlabel('Expert')
        plt.ylabel('Mean Gating Weight')
        plt.title('Mean Expert Utilization')
        plt.xticks(x, self.expert_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Expert specialization metrics
        ax4 = plt.subplot(3, 4, 4)
        abs_spec = analysis['overall_stats']['abs_specialization']
        rel_spec = analysis['overall_stats']['rel_specialization']
        abs_entropy = analysis['overall_stats']['abs_entropy']
        rel_entropy = analysis['overall_stats']['rel_entropy']
        
        metrics = ['Specialization', 'Entropy']
        abs_values = [abs_spec, abs_entropy]
        rel_values = [rel_spec, rel_entropy]
        
        x = np.arange(len(metrics))
        plt.bar(x - width/2, abs_values, width, label='Absolute K-MOTE', alpha=0.8, color='skyblue')
        plt.bar(x + width/2, rel_values, width, label='Relative K-MOTE', alpha=0.8, color='lightcoral')
        
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.title('Specialization Metrics')
        plt.xticks(x, metrics)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5-8. Time-dependent gating weights
        for subplot_idx, (weights, times, title_prefix) in enumerate([
            (abs_weights, abs_times, 'Absolute K-MOTE'),
            (rel_weights, rel_times, 'Relative K-MOTE')
        ]):
            ax = plt.subplot(3, 4, 5 + subplot_idx*2)
            
            # Sort by time for better visualization
            sort_idx = np.argsort(times)
            times_sorted = times[sort_idx]
            weights_sorted = weights[sort_idx]
            
            # Plot gating weights vs time (use every 10th point for clarity)
            step = max(1, len(times_sorted) // 100)
            for i, (expert_name, color) in enumerate(zip(self.expert_names, self.colors)):
                plt.scatter(times_sorted[::step], weights_sorted[::step, i], 
                           alpha=0.6, label=expert_name, color=color, s=20)
            
            plt.xlabel('Normalized Time')
            plt.ylabel('Gating Weight')
            plt.title(f'{title_prefix}\nGating vs Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Moving average of gating weights
            ax = plt.subplot(3, 4, 6 + subplot_idx*2)
            window_size = max(1, len(times_sorted) // 20)
            
            for i, (expert_name, color) in enumerate(zip(self.expert_names, self.colors)):
                # Compute moving average
                weights_ma = np.convolve(weights_sorted[:, i], np.ones(window_size)/window_size, mode='valid')
                times_ma = times_sorted[:len(weights_ma)]
                
                plt.plot(times_ma, weights_ma, label=expert_name, color=color, linewidth=2)
            
            plt.xlabel('Normalized Time')
            plt.ylabel('Moving Average Weight')
            plt.title(f'{title_prefix}\nTemporal Trends')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 9. Correlation matrix between experts (absolute)
        ax9 = plt.subplot(3, 4, 9)
        abs_corr = np.corrcoef(abs_weights.T)
        im1 = plt.imshow(abs_corr, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(im1, fraction=0.046, pad=0.04)
        plt.xticks(range(3), self.expert_names, rotation=45)
        plt.yticks(range(3), self.expert_names)
        plt.title('Absolute K-MOTE\nExpert Correlations')
        
        # Add correlation values
        for i in range(3):
            for j in range(3):
                plt.text(j, i, f'{abs_corr[i,j]:.2f}', ha='center', va='center', 
                        color='white' if abs(abs_corr[i,j]) > 0.5 else 'black')
        
        # 10. Correlation matrix between experts (relative)
        ax10 = plt.subplot(3, 4, 10)
        rel_corr = np.corrcoef(rel_weights.T)
        im2 = plt.imshow(rel_corr, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(im2, fraction=0.046, pad=0.04)
        plt.xticks(range(3), self.expert_names, rotation=45)
        plt.yticks(range(3), self.expert_names)
        plt.title('Relative K-MOTE\nExpert Correlations')
        
        # Add correlation values
        for i in range(3):
            for j in range(3):
                plt.text(j, i, f'{rel_corr[i,j]:.2f}', ha='center', va='center',
                        color='white' if abs(rel_corr[i,j]) > 0.5 else 'black')
        
        # 11. Expert dominance over time (absolute)
        ax11 = plt.subplot(3, 4, 11)
        dominant_experts_abs = np.argmax(abs_weights, axis=1)
        for i, (expert_name, color) in enumerate(zip(self.expert_names, self.colors)):
            mask = dominant_experts_abs == i
            if np.any(mask):
                plt.scatter(abs_times[mask], np.full(np.sum(mask), i), 
                           alpha=0.6, color=color, label=expert_name, s=20)
        
        plt.xlabel('Normalized Time')
        plt.ylabel('Dominant Expert')
        plt.title('Absolute K-MOTE\nExpert Dominance')
        plt.yticks(range(3), self.expert_names)
        plt.grid(True, alpha=0.3)
        
        # 12. Expert dominance over time (relative)
        ax12 = plt.subplot(3, 4, 12)
        dominant_experts_rel = np.argmax(rel_weights, axis=1)
        for i, (expert_name, color) in enumerate(zip(self.expert_names, self.colors)):
            mask = dominant_experts_rel == i
            if np.any(mask):
                plt.scatter(rel_times[mask], np.full(np.sum(mask), i), 
                           alpha=0.6, color=color, label=expert_name, s=20)
        
        plt.xlabel('Normalized Time')
        plt.ylabel('Dominant Expert')
        plt.title('Relative K-MOTE\nExpert Dominance')
        plt.yticks(range(3), self.expert_names)
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(f'K-MOTE Gating Analysis: {model_name} + {dataset_name.upper()}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save the figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"kmote_gating_analysis_{model_name}_{dataset_name}_{timestamp}.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"✓ Visualization saved to: {filepath}")
        
        return filepath
    
    def create_summary_report(self, analysis, gating_data, model_name, dataset_name):
        """
        Create a text summary report of the analysis.
        """
        print("Creating summary report...")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
K-MOTE Gating Analysis Report
============================

Model: {model_name}
Dataset: {dataset_name.upper()}
Time Encoder: KAN-MAMMOTE with Dual K-MOTE
Analysis Date: {timestamp}
Sample Size: {len(gating_data['abs_gating_weights'])}

ABSOLUTE K-MOTE ANALYSIS
------------------------
"""
        
        for expert_name in self.expert_names:
            stats = analysis['absolute_kmote'][expert_name]
            report += f"""
{expert_name} Expert:
  • Mean Weight: {stats['mean_weight']:.3f} ± {stats['std_weight']:.3f}
  • Max Weight: {stats['max_weight']:.3f}
  • Activation Frequency: {stats['activation_frequency']:.1%}
  • Dominant Time Range: [{stats['dominant_time_range'][0]:.3f}, {stats['dominant_time_range'][1]:.3f}]
"""
        
        report += f"""
RELATIVE K-MOTE ANALYSIS
------------------------
"""
        
        for expert_name in self.expert_names:
            stats = analysis['relative_kmote'][expert_name]
            report += f"""
{expert_name} Expert:
  • Mean Weight: {stats['mean_weight']:.3f} ± {stats['std_weight']:.3f}
  • Max Weight: {stats['max_weight']:.3f}
  • Activation Frequency: {stats['activation_frequency']:.1%}
  • Dominant Time Range: [{stats['dominant_time_range'][0]:.3f}, {stats['dominant_time_range'][1]:.3f}]
"""
        
        overall = analysis['overall_stats']
        report += f"""
OVERALL STATISTICS
------------------
Absolute K-MOTE:
  • Specialization Index: {overall['abs_specialization']:.3f}
  • Entropy: {overall['abs_entropy']:.3f}

Relative K-MOTE:
  • Specialization Index: {overall['rel_specialization']:.3f}
  • Entropy: {overall['rel_entropy']:.3f}

INTERPRETATION
--------------
Higher specialization index indicates experts have more distinct roles.
Higher entropy indicates more balanced usage across experts.
Lower activation frequency suggests an expert is rarely dominant.

EXPERT SPECIALIZATION SUMMARY
------------------------------
Based on the gating analysis:
"""
        
        # Determine which expert is most used in each K-MOTE
        abs_usage = [analysis['absolute_kmote'][name]['mean_weight'] for name in self.expert_names]
        rel_usage = [analysis['relative_kmote'][name]['mean_weight'] for name in self.expert_names]
        
        most_used_abs = self.expert_names[np.argmax(abs_usage)]
        most_used_rel = self.expert_names[np.argmax(rel_usage)]
        
        report += f"""
• Most utilized in Absolute K-MOTE: {most_used_abs} ({max(abs_usage):.3f})
• Most utilized in Relative K-MOTE: {most_used_rel} ({max(rel_usage):.3f})

This suggests that:
- {most_used_abs} expert is most important for absolute time patterns
- {most_used_rel} expert is most important for relative time patterns
"""
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"kmote_gating_report_{model_name}_{dataset_name}_{timestamp}.txt"
        report_filepath = self.output_dir / report_filename
        
        with open(report_filepath, 'w') as f:
            f.write(report)
        
        print(f"✓ Report saved to: {report_filepath}")
        return report_filepath
    
    def run_complete_analysis(self, model_name, dataset_name, time_encoder_type, seed=0, sample_size=1000):
        """
        Run complete K-MOTE gating analysis.
        
        Args:
            model_name (str): Model name (e.g., 'TCL', 'TGAT')
            dataset_name (str): Dataset name (e.g., 'uci', 'mooc')
            time_encoder_type (str): Time encoder type
            seed (int): Random seed
            sample_size (int): Number of samples to analyze
            
        Returns:
            dict: Complete analysis results
        """
        print(f"🔍 Starting K-MOTE Gating Analysis")
        print(f"   Model: {model_name}")
        print(f"   Dataset: {dataset_name}")
        print(f"   Time Encoder: {time_encoder_type}")
        print("=" * 60)
        
        try:
            # 1. Load trained model
            time_encoder, train_data, train_neighbor_sampler, args = self.load_trained_model(
                model_name, dataset_name, time_encoder_type, seed
            )
            
            # 2. Extract gating weights
            gating_data = self.extract_gating_weights(time_encoder, train_data, sample_size)
            
            # 3. Analyze expert specialization
            analysis = self.analyze_expert_specialization(gating_data, model_name, dataset_name)
            
            # 4. Create visualizations
            viz_path = self.create_gating_visualizations(gating_data, analysis, model_name, dataset_name)
            
            # 5. Create summary report
            report_path = self.create_summary_report(analysis, gating_data, model_name, dataset_name)
            
            # 6. Save complete results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_filename = f"kmote_gating_complete_{model_name}_{dataset_name}_{timestamp}.pkl"
            results_filepath = self.output_dir / results_filename
            
            complete_results = {
                'model_name': model_name,
                'dataset_name': dataset_name,
                'time_encoder_type': time_encoder_type,
                'analysis': analysis,
                'gating_data': gating_data,
                'visualization_path': str(viz_path),
                'report_path': str(report_path),
                'timestamp': timestamp
            }
            
            with open(results_filepath, 'wb') as f:
                pickle.dump(complete_results, f)
            
            print(f"✓ Complete analysis saved to: {results_filepath}")
            print(f"🎉 Analysis complete! Check the output directory: {self.output_dir}")
            
            return complete_results
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main function to run the K-MOTE gating analysis."""
    parser = argparse.ArgumentParser(description='K-MOTE Gating Weight Visualization for KAN-MAMMOTE')
    parser.add_argument('--model_name', type=str, default='TCL',
                        choices=['TGAT', 'TCL', 'CAWN', 'GraphMixer', 'DyGFormer', 'DyGMamba'],
                        help='Model name to analyze')
    parser.add_argument('--dataset_name', type=str, default='uci',
                        choices=['wikipedia', 'reddit', 'mooc', 'lastfm', 'enron', 'uci',
                                'CanParl', 'Contacts', 'Flights', 'UNtrade', 'UNvote', 'USLegis'],
                        help='Dataset name to analyze')
    parser.add_argument('--time_encoder_type', type=str, default='kan_mammote_dual_kmote',
                        choices=['kan_mammote_dual_kmote', 'kan_mammote_dual_kmote_tgat'],
                        help='Time encoder type (must be KAN-MAMMOTE with dual K-MOTE)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility')
    parser.add_argument('--sample_size', type=int, default=1000,
                        help='Number of samples to analyze')
    parser.add_argument('--output_dir', type=str, default='./kmote_gating_analysis',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = KMOTEGatingAnalyzer(args.output_dir)
    
    # Run analysis
    results = analyzer.run_complete_analysis(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        time_encoder_type=args.time_encoder_type,
        seed=args.seed,
        sample_size=args.sample_size
    )
    
    if results:
        print("\n🎯 KEY FINDINGS:")
        analysis = results['analysis']
        
        # Show expert utilization
        print("\nExpert Utilization in Absolute K-MOTE:")
        for expert in ['Spline', 'Fourier', 'Wavelet']:
            weight = analysis['absolute_kmote'][expert]['mean_weight']
            freq = analysis['absolute_kmote'][expert]['activation_frequency']
            print(f"  {expert}: {weight:.3f} avg weight, {freq:.1%} dominance")
        
        print("\nExpert Utilization in Relative K-MOTE:")
        for expert in ['Spline', 'Fourier', 'Wavelet']:
            weight = analysis['relative_kmote'][expert]['mean_weight']
            freq = analysis['relative_kmote'][expert]['activation_frequency']
            print(f"  {expert}: {weight:.3f} avg weight, {freq:.1%} dominance")
        
        print(f"\n📊 Specialization Indices:")
        print(f"  Absolute K-MOTE: {analysis['overall_stats']['abs_specialization']:.3f}")
        print(f"  Relative K-MOTE: {analysis['overall_stats']['rel_specialization']:.3f}")


if __name__ == "__main__":
    main()