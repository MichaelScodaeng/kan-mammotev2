#!/usr/bin/env python3
"""
Node-Level K-MOTE Gating Analysis

This script analyzes K-MOTE gating patterns for individual nodes in temporal graphs,
showing how different nodes utilize different experts (Spline, Fourier, Wavelet) 
for their temporal patterns.

Usage:
    python node_level_kmote_analysis.py --model_name TCL --dataset_name uci --node_id 42

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
sns.set_palette("Set2")

class NodeLevelKMOTEAnalyzer:
    """
    Analyzes K-MOTE gating patterns for individual nodes.
    """
    
    def __init__(self, output_dir="./node_kmote_analysis"):
        """
        Initialize the analyzer.
        
        Args:
            output_dir (str): Directory to save results and plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Expert names and colors
        self.expert_names = ['Spline', 'Fourier', 'Wavelet']
        self.colors = ['#E74C3C', '#3498DB', '#2ECC71']  # Red, Blue, Green
        
    def load_model_and_data(self, model_name, dataset_name, time_encoder_type, seed=0):
        """Load model and data for analysis."""
        print(f"Loading {model_name} + {time_encoder_type} on {dataset_name}...")
        
        # Create dummy args
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
            sample_neighbor_strategy='recent',
            time_scaling_factor=1e-6,
            seed=0
        )
        
        # Create time encoder
        time_encoder = create_time_encoder(
            encoder_type=time_encoder_type,
            time_dim=args.time_feat_dim,
            train_data=train_data,
            train_neighbor_sampler=train_neighbor_sampler,
            args=args,
            device=args.device
        )
        
        # Validate it's KAN-MAMMOTE with dual K-MOTE
        if not hasattr(time_encoder, 'k_mote_abs') or not hasattr(time_encoder, 'k_mote_rel'):
            raise ValueError(f"Time encoder {time_encoder_type} does not have K-MOTE components!")
        
        if time_encoder.k_mote_rel is None:
            raise ValueError("This KAN-MAMMOTE variant does not use K-MOTE for relative time encoding!")
        
        return time_encoder, train_data, full_data, args
    
    def get_node_interactions(self, data, node_id):
        """Get all interactions for a specific node."""
        # Find interactions where node_id is the source
        src_mask = data.src_node_ids == node_id
        src_interactions = {
            'timestamps': data.node_interact_times[src_mask],
            'partners': data.dst_node_ids[src_mask],
            'edge_ids': data.edge_ids[src_mask],
            'role': 'source'
        }
        
        # Find interactions where node_id is the destination
        dst_mask = data.dst_node_ids == node_id
        dst_interactions = {
            'timestamps': data.node_interact_times[dst_mask],
            'partners': data.src_node_ids[dst_mask], 
            'edge_ids': data.edge_ids[dst_mask],
            'role': 'destination'
        }
        
        # Combine and sort by timestamp
        all_timestamps = np.concatenate([src_interactions['timestamps'], dst_interactions['timestamps']])
        all_partners = np.concatenate([src_interactions['partners'], dst_interactions['partners']])
        all_edge_ids = np.concatenate([src_interactions['edge_ids'], dst_interactions['edge_ids']])
        all_roles = ['source'] * len(src_interactions['timestamps']) + ['destination'] * len(dst_interactions['timestamps'])
        
        # Sort by timestamp
        sort_idx = np.argsort(all_timestamps)
        
        return {
            'node_id': node_id,
            'timestamps': all_timestamps[sort_idx],
            'partners': all_partners[sort_idx],
            'edge_ids': all_edge_ids[sort_idx],
            'roles': np.array(all_roles)[sort_idx],
            'num_interactions': len(all_timestamps)
        }
    
    def analyze_node_gating_patterns(self, time_encoder, node_data):
        """Analyze K-MOTE gating patterns for a specific node."""
        print(f"Analyzing gating patterns for node {node_data['node_id']} ({node_data['num_interactions']} interactions)")
        
        timestamps = node_data['timestamps']
        
        if len(timestamps) < 2:
            print(f"Node {node_data['node_id']} has insufficient interactions (<2)")
            return None
        
        # Create absolute time features (normalized timestamps)
        t_abs_raw = (timestamps - timestamps.mean()) / (timestamps.std() + 1e-8)
        t_abs = torch.from_numpy(t_abs_raw).float().unsqueeze(-1)  # (N, 1)
        
        # Create relative time features (time differences)
        time_diffs = np.diff(timestamps)
        if len(time_diffs) == 0:
            return None
            
        # Pad to same length and normalize
        time_diffs_padded = np.concatenate([[time_diffs[0]], time_diffs])  # Repeat first diff
        t_rel_raw = time_diffs_padded / (np.max(time_diffs_padded) + 1e-8)
        t_rel = torch.from_numpy(t_rel_raw).float().unsqueeze(-1)  # (N, 1)
        
        time_encoder.eval()
        with torch.no_grad():
            # Get gating weights from both K-MOTE components
            _, abs_gating_weights = time_encoder.k_mote_abs(t_abs, return_weights=True)  # Should be (N, 3)
            _, rel_gating_weights = time_encoder.k_mote_rel(t_rel, return_weights=True)  # Should be (N, 3)
        
        # Debug: Check shapes
        print(f"Debug: abs_gating_weights shape: {abs_gating_weights.shape}")
        print(f"Debug: rel_gating_weights shape: {rel_gating_weights.shape}")
        
        # Handle tensor shape issues - ensure we have (N, 3) for 3 experts
        if abs_gating_weights.dim() == 3 and abs_gating_weights.shape[1] == 1:
            # Shape is (N, 1, 3), squeeze the middle dimension
            abs_gating_weights = abs_gating_weights.squeeze(1)  # (N, 3)
        elif abs_gating_weights.dim() == 1 or abs_gating_weights.shape[-1] != 3:
            print(f"Warning: Unexpected abs_gating_weights shape {abs_gating_weights.shape}, creating dummy weights")
            abs_gating_weights = torch.ones(len(t_abs), 3) / 3.0
            
        if rel_gating_weights.dim() == 3 and rel_gating_weights.shape[1] == 1:
            # Shape is (N, 1, 3), squeeze the middle dimension  
            rel_gating_weights = rel_gating_weights.squeeze(1)  # (N, 3)
        elif rel_gating_weights.dim() == 1 or rel_gating_weights.shape[-1] != 3:
            print(f"Warning: Unexpected rel_gating_weights shape {rel_gating_weights.shape}, creating dummy weights")
            rel_gating_weights = torch.ones(len(t_rel), 3) / 3.0
        
        # Convert to numpy
        abs_weights = abs_gating_weights.cpu().numpy()
        rel_weights = rel_gating_weights.cpu().numpy()
        
        return {
            'node_data': node_data,
            'abs_times': t_abs_raw,
            'rel_times': t_rel_raw,
            'abs_gating_weights': abs_weights,
            'rel_gating_weights': rel_weights,
            'raw_timestamps': timestamps,
            'time_differences': time_diffs_padded
        }
    
    def visualize_node_gating_timeline(self, analysis_results, model_name, dataset_name, time_range=None):
        """Create a detailed timeline visualization for a single node.
        
        Args:
            analysis_results: Analysis results from analyze_node_gating_patterns
            model_name: Model name
            dataset_name: Dataset name
            time_range: Tuple (start_time, end_time) to focus on specific time range, or None for full range
        """
        node_id = analysis_results['node_data']['node_id']
        timestamps = analysis_results['raw_timestamps']
        abs_weights = analysis_results['abs_gating_weights']
        rel_weights = analysis_results['rel_gating_weights']
        partners = analysis_results['node_data']['partners']
        roles = analysis_results['node_data']['roles']
        
        # Apply time range filter if specified
        if time_range is not None:
            start_time, end_time = time_range
            time_mask = (timestamps >= start_time) & (timestamps <= end_time)
            timestamps = timestamps[time_mask]
            abs_weights = abs_weights[time_mask]
            rel_weights = rel_weights[time_mask]
            partners = partners[time_mask]
            roles = roles[time_mask]
            
            if len(timestamps) == 0:
                print(f"❌ No interactions found in time range [{start_time}, {end_time}]")
                return None
                
            print(f"📊 Focusing on time range [{start_time}, {end_time}] with {len(timestamps)} interactions")
        
        # Create figure with subplots
        fig, axes = plt.subplots(4, 1, figsize=(16, 12))
        
        # 1. Timeline of interactions with partner diversity
        ax1 = axes[0]
        unique_partners = np.unique(partners)
        partner_colors = plt.cm.tab10(np.arange(len(unique_partners)) % 10)
        
        for i, partner in enumerate(unique_partners):
            mask = partners == partner
            partner_timestamps = timestamps[mask]
            partner_roles = roles[mask]
            
            # Different markers for source/destination
            for role, marker in [('source', 'o'), ('destination', '^')]:
                role_mask = partner_roles == role
                if np.any(role_mask):
                    ax1.scatter(partner_timestamps[role_mask], 
                              np.full(np.sum(role_mask), i),
                              c=[partner_colors[i]], marker=marker, s=50, alpha=0.7,
                              label=f'Partner {partner} ({role})' if i < 5 else "")
        
        ax1.set_xlabel('Timestamp')
        ax1.set_ylabel('Partner ID (Index)')
        ax1.set_title(f'Node {node_id} Interaction Timeline\n(○ = source, △ = destination)')
        if len(unique_partners) <= 5:
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Absolute K-MOTE gating weights over time
        ax2 = axes[1]
        for i, (expert_name, color) in enumerate(zip(self.expert_names, self.colors)):
            ax2.plot(timestamps, abs_weights[:, i], marker='o', color=color, 
                    label=expert_name, linewidth=2, markersize=4)
        
        ax2.set_xlabel('Timestamp')
        ax2.set_ylabel('Gating Weight')
        ax2.set_title('Absolute K-MOTE Gating Weights Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # 3. Relative K-MOTE gating weights over time
        ax3 = axes[2]
        for i, (expert_name, color) in enumerate(zip(self.expert_names, self.colors)):
            ax3.plot(timestamps, rel_weights[:, i], marker='s', color=color, 
                    label=expert_name, linewidth=2, markersize=4)
        
        ax3.set_xlabel('Timestamp')
        ax3.set_ylabel('Gating Weight')
        ax3.set_title('Relative K-MOTE Gating Weights Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # 4. Dominant expert over time
        ax4 = axes[3]
        dominant_abs = np.argmax(abs_weights, axis=1)
        dominant_rel = np.argmax(rel_weights, axis=1)
        
        # Create stacked bars showing dominant experts
        bar_width = (timestamps.max() - timestamps.min()) / len(timestamps) * 0.8
        
        for i, timestamp in enumerate(timestamps):
            # Absolute K-MOTE (bottom half)
            abs_expert = dominant_abs[i]
            ax4.bar(timestamp, -0.4, width=bar_width, bottom=-0.6, 
                   color=self.colors[abs_expert], alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Relative K-MOTE (top half)
            rel_expert = dominant_rel[i]
            ax4.bar(timestamp, 0.4, width=bar_width, bottom=0.6, 
                   color=self.colors[rel_expert], alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = []
        for i, (expert_name, color) in enumerate(zip(self.expert_names, self.colors)):
            legend_elements.append(Patch(facecolor=color, alpha=0.7, label=expert_name))
        ax4.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        ax4.set_xlabel('Timestamp')
        ax4.set_ylabel('K-MOTE Type')
        ax4.set_title('Dominant Experts Over Time')
        ax4.set_yticks([-1, 1])
        ax4.set_yticklabels(['Absolute\nK-MOTE', 'Relative\nK-MOTE'])
        ax4.grid(True, alpha=0.3, axis='x')
        ax4.axhline(y=0, color='black', linewidth=1)
        
        plt.suptitle(f'Node {node_id} K-MOTE Gating Analysis\n{model_name} + {dataset_name.upper()}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save the figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        time_suffix = f"_range_{start_time}_{end_time}" if time_range else "_full"
        filename = f"node_{node_id}_kmote_timeline_{model_name}_{dataset_name}{time_suffix}_{timestamp}.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        return filepath
    
    def create_node_summary(self, analysis_results, model_name, dataset_name):
        """Create a summary analysis for the node."""
        node_data = analysis_results['node_data']
        abs_weights = analysis_results['abs_gating_weights']
        rel_weights = analysis_results['rel_gating_weights']
        
        node_id = node_data['node_id']
        
        summary = f"""
Node-Level K-MOTE Analysis Summary
==================================

Node ID: {node_id}
Model: {model_name}
Dataset: {dataset_name.upper()}
Analysis Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Total Interactions: {node_data['num_interactions']}

INTERACTION PATTERN
-------------------
• Time Span: {node_data['timestamps'].min():.0f} to {node_data['timestamps'].max():.0f}
• Source Interactions: {np.sum(node_data['roles'] == 'source')}
• Destination Interactions: {np.sum(node_data['roles'] == 'destination')}
• Unique Partners: {len(np.unique(node_data['partners']))}

ABSOLUTE K-MOTE USAGE
---------------------
"""
        
        for i, expert_name in enumerate(self.expert_names):
            mean_weight = abs_weights[:, i].mean()
            max_weight = abs_weights[:, i].max()
            dominance_freq = np.mean(np.argmax(abs_weights, axis=1) == i)
            
            summary += f"""
{expert_name} Expert:
  • Mean Weight: {mean_weight:.3f}
  • Max Weight: {max_weight:.3f}
  • Dominance Frequency: {dominance_freq:.1%}
"""
        
        summary += f"""
RELATIVE K-MOTE USAGE
---------------------
"""
        
        for i, expert_name in enumerate(self.expert_names):
            mean_weight = rel_weights[:, i].mean()
            max_weight = rel_weights[:, i].max()
            dominance_freq = np.mean(np.argmax(rel_weights, axis=1) == i)
            
            summary += f"""
{expert_name} Expert:
  • Mean Weight: {mean_weight:.3f}
  • Max Weight: {max_weight:.3f}
  • Dominance Frequency: {dominance_freq:.1%}
"""
        
        # Expert specialization analysis
        abs_entropy = -np.sum(abs_weights.mean(axis=0) * np.log(abs_weights.mean(axis=0) + 1e-8))
        rel_entropy = -np.sum(rel_weights.mean(axis=0) * np.log(rel_weights.mean(axis=0) + 1e-8))
        
        most_used_abs = self.expert_names[np.argmax(abs_weights.mean(axis=0))]
        most_used_rel = self.expert_names[np.argmax(rel_weights.mean(axis=0))]
        
        summary += f"""
SPECIALIZATION ANALYSIS
-----------------------
• Absolute K-MOTE Entropy: {abs_entropy:.3f}
• Relative K-MOTE Entropy: {rel_entropy:.3f}
• Most Used Absolute Expert: {most_used_abs}
• Most Used Relative Expert: {most_used_rel}

INTERPRETATION
--------------
This node's temporal pattern shows:
"""
        
        if abs_entropy > 1.0:
            summary += "• Balanced usage of experts in absolute time encoding\n"
        else:
            summary += f"• Strong preference for {most_used_abs} expert in absolute time\n"
        
        if rel_entropy > 1.0:
            summary += "• Balanced usage of experts in relative time encoding\n"
        else:
            summary += f"• Strong preference for {most_used_rel} expert in relative time\n"
        
        # Save summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_filename = f"node_{node_id}_summary_{model_name}_{dataset_name}_{timestamp}.txt"
        summary_filepath = self.output_dir / summary_filename
        
        with open(summary_filepath, 'w') as f:
            f.write(summary)
        
        print(f"✓ Summary saved to: {summary_filepath}")
        return summary_filepath
    
    def analyze_node(self, model_name, dataset_name, time_encoder_type, node_id, seed=0, time_range=None):
        """Complete analysis for a single node.
        
        Args:
            model_name: Model name
            dataset_name: Dataset name  
            time_encoder_type: Time encoder type
            node_id: Node ID to analyze
            seed: Random seed
            time_range: Tuple (start_time, end_time) to focus on specific time range
        """
        print(f"🔍 Analyzing Node {node_id}")
        print(f"   Model: {model_name}")
        print(f"   Dataset: {dataset_name}")
        print(f"   Time Encoder: {time_encoder_type}")
        if time_range:
            print(f"   Time Range: [{time_range[0]}, {time_range[1]}]")
        print("=" * 50)
        
        try:
            # Load model and data
            time_encoder, train_data, full_data, args = self.load_model_and_data(
                model_name, dataset_name, time_encoder_type, seed
            )
            
            # Get node interactions
            node_data = self.get_node_interactions(full_data, node_id)
            
            if node_data['num_interactions'] < 2:
                print(f"❌ Node {node_id} has insufficient interactions ({node_data['num_interactions']})")
                return None
            
            print(f"✓ Found {node_data['num_interactions']} interactions for node {node_id}")
            
            # Analyze gating patterns
            analysis_results = self.analyze_node_gating_patterns(time_encoder, node_data)
            
            if analysis_results is None:
                print(f"❌ Failed to analyze node {node_id}")
                return None
            
            # Create visualizations
            viz_path = self.visualize_node_gating_timeline(analysis_results, model_name, dataset_name, time_range)
            
            # Create summary
            summary_path = self.create_node_summary(analysis_results, model_name, dataset_name)
            
            print(f"✅ Analysis complete for node {node_id}")
            print(f"   Visualization: {viz_path}")
            print(f"   Summary: {summary_path}")
            
            return {
                'node_id': node_id,
                'analysis_results': analysis_results,
                'visualization_path': str(viz_path),
                'summary_path': str(summary_path)
            }
            
        except Exception as e:
            print(f"❌ Analysis failed for node {node_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def find_interesting_nodes(self, data, min_interactions=10, max_nodes=5):
        """Find nodes with interesting interaction patterns for analysis."""
        print(f"Finding interesting nodes (min {min_interactions} interactions)...")
        
        # Count interactions per node
        node_counts = {}
        for node in np.unique(np.concatenate([data.src_node_ids, data.dst_node_ids])):
            src_count = np.sum(data.src_node_ids == node)
            dst_count = np.sum(data.dst_node_ids == node)
            node_counts[node] = src_count + dst_count
        
        # Filter nodes with sufficient interactions
        interesting_nodes = [node for node, count in node_counts.items() 
                           if count >= min_interactions]
        
        # Sort by interaction count (descending)
        interesting_nodes.sort(key=lambda x: node_counts[x], reverse=True)
        
        # Take top nodes
        selected_nodes = interesting_nodes[:max_nodes]
        
        print(f"Found {len(selected_nodes)} interesting nodes:")
        for node in selected_nodes:
            print(f"  Node {node}: {node_counts[node]} interactions")
        
        return selected_nodes


def main():
    """Main function for node-level K-MOTE analysis."""
    parser = argparse.ArgumentParser(description='Node-Level K-MOTE Gating Analysis')
    parser.add_argument('--model_name', type=str, default='TCL',
                        choices=['TGAT', 'TCL', 'CAWN', 'GraphMixer', 'DyGFormer', 'DyGMamba'],
                        help='Model name to analyze')
    parser.add_argument('--dataset_name', type=str, default='uci',
                        choices=['wikipedia', 'reddit', 'mooc', 'lastfm', 'enron', 'uci',
                                'CanParl', 'Contacts', 'Flights', 'UNtrade', 'UNvote', 'USLegis'],
                        help='Dataset name to analyze')
    parser.add_argument('--time_encoder_type', type=str, default='kan_mammote_dual_kmote',
                        choices=['kan_mammote_dual_kmote', 'kan_mammote_dual_kmote_tgat'],
                        help='Time encoder type')
    parser.add_argument('--node_id', type=int, default=None,
                        help='Specific node ID to analyze (if not provided, will find interesting nodes)')
    parser.add_argument('--auto_find_nodes', action='store_true', default=False,
                        help='Automatically find and analyze interesting nodes')
    parser.add_argument('--min_interactions', type=int, default=10,
                        help='Minimum interactions for a node to be considered interesting')
    parser.add_argument('--max_nodes', type=int, default=3,
                        help='Maximum number of nodes to analyze automatically')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./node_kmote_analysis',
                        help='Output directory')
    parser.add_argument('--time_range_start', type=float, default=None,
                        help='Start time for focusing on specific time range (optional)')
    parser.add_argument('--time_range_end', type=float, default=None,
                        help='End time for focusing on specific time range (optional)')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = NodeLevelKMOTEAnalyzer(args.output_dir)
    
    # Parse time range if provided
    time_range = None
    if args.time_range_start is not None and args.time_range_end is not None:
        time_range = (args.time_range_start, args.time_range_end)
        print(f"📊 Time range filter: [{time_range[0]}, {time_range[1]}]")
    elif args.time_range_start is not None or args.time_range_end is not None:
        print("❌ Both --time_range_start and --time_range_end must be provided together")
        return
    
    if args.node_id is not None:
        # Analyze specific node
        result = analyzer.analyze_node(
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            time_encoder_type=args.time_encoder_type,
            node_id=args.node_id,
            seed=args.seed,
            time_range=time_range
        )
        
    elif args.auto_find_nodes:
        # Find and analyze interesting nodes
        print("🔍 Finding interesting nodes for analysis...")
        
        # Load data to find interesting nodes
        from utils.DataLoader import get_link_prediction_data
        _, _, full_data, _, _, _, _, _ = get_link_prediction_data(
            dataset_name=args.dataset_name,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=args.seed,
            data_ratio=1.0
        )
        
        interesting_nodes = analyzer.find_interesting_nodes(
            full_data, args.min_interactions, args.max_nodes
        )
        
        results = []
        for node_id in interesting_nodes:
            print(f"\n🔍 Analyzing Node {node_id}...")
            result = analyzer.analyze_node(
                model_name=args.model_name,
                dataset_name=args.dataset_name,
                time_encoder_type=args.time_encoder_type,
                node_id=node_id,
                seed=args.seed,
                time_range=time_range
            )
            if result:
                results.append(result)
        
        print(f"\n🎉 Analyzed {len(results)} nodes successfully!")
        
    else:
        print("❌ Please specify either --node_id or --auto_find_nodes")
        return
    
    print(f"\n📁 Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()