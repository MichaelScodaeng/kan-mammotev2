#!/usr/bin/env python3
"""
Enhanced Node Temporal Pattern Analyzer
=======================================
Advanced tool for analyzing and visualizing temporal patterns in node interactions.
Allows custom node selection and detailed pattern analysis.

Usage:
    # Analyze specific nodes
    python enhanced_temporal_analyzer.py --nodes 69 1035 18 37
    
    # Find most interesting patterns automatically
    python enhanced_temporal_analyzer.py --auto_select 4
    
    # Analyze a specific node in detail
    python enhanced_temporal_analyzer.py --detailed_node 69
    
    # Export raw data for custom analysis
    python enhanced_temporal_analyzer.py --export_data --nodes 69 1035
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import entropy
import os
import sys
import argparse
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class TemporalPatternAnalyzer:
    """Advanced analyzer for temporal patterns in node interactions"""
    
    def __init__(self, dataset_path=None):
        if dataset_path is None:
            dataset_path = project_root / "processed_data" / "wikipedia" / "ml_wikipedia.csv"
        
        self.dataset_path = dataset_path
        self.df = None
        self.load_dataset()
    
    def load_dataset(self):
        """Load the Wikipedia temporal interaction dataset"""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
        
        print(f"📊 Loading dataset from {self.dataset_path}")
        self.df = pd.read_csv(self.dataset_path)
        
        print(f"   ├─ Total interactions: {len(self.df):,}")
        print(f"   ├─ Unique source nodes: {self.df['u'].nunique():,}")
        print(f"   ├─ Unique destination nodes: {self.df['i'].nunique():,}")
        print(f"   ├─ Time range: {self.df['ts'].min():.1f} - {self.df['ts'].max():.1f}")
        print(f"   └─ Dataset shape: {self.df.shape}")
    
    def get_node_interactions(self, node_id, as_source=True):
        """Extract interactions for a specific node"""
        if as_source:
            interactions = self.df[self.df['u'] == node_id].copy()
        else:
            interactions = self.df[self.df['i'] == node_id].copy()
        
        return interactions.sort_values('ts').reset_index(drop=True)
    
    def analyze_temporal_pattern(self, interactions, detailed=False):
        """Comprehensive temporal pattern analysis"""
        if len(interactions) < 2:
            return {'pattern_type': 'Insufficient data', 'num_interactions': len(interactions)}
        
        timestamps = interactions['ts'].values
        inter_arrivals = np.diff(timestamps)
        
        # Basic statistics
        mean_inter = np.mean(inter_arrivals)
        std_inter = np.std(inter_arrivals)
        cv_inter = std_inter / mean_inter if mean_inter > 0 else 0
        
        # Pattern classification
        if cv_inter < 0.5:
            pattern_type = "Regular"
        elif cv_inter > 2.0:
            pattern_type = "Highly Bursty"
        elif cv_inter > 1.0:
            pattern_type = "Bursty"
        else:
            pattern_type = "Mixed"
        
        analysis = {
            'pattern_type': pattern_type,
            'num_interactions': len(interactions),
            'cv_inter': cv_inter,
            'mean_inter': mean_inter,
            'std_inter': std_inter,
            'total_time_span': timestamps[-1] - timestamps[0],
            'rate_per_unit': len(interactions) / (timestamps[-1] - timestamps[0]) if timestamps[-1] > timestamps[0] else 0
        }
        
        if detailed:
            # Additional detailed analysis
            analysis.update({
                'median_inter': np.median(inter_arrivals),
                'min_inter': np.min(inter_arrivals),
                'max_inter': np.max(inter_arrivals),
                'skewness': self._skewness(inter_arrivals),
                'entropy': entropy(np.histogram(inter_arrivals, bins=20)[0] + 1),  # +1 to avoid log(0)
                'burstiness': self._burstiness_index(inter_arrivals)
            })
        
        return analysis
    
    def _skewness(self, data):
        """Calculate skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _burstiness_index(self, inter_arrivals):
        """Calculate burstiness index (Goh & Barabási, 2008)"""
        if len(inter_arrivals) < 2:
            return 0
        
        mean_inter = np.mean(inter_arrivals)
        std_inter = np.std(inter_arrivals)
        
        if mean_inter == 0:
            return 0
        
        cv = std_inter / mean_inter
        burstiness = (cv - 1) / (cv + 1)
        return burstiness
    
    def find_interesting_nodes(self, num_nodes=4, min_interactions=100, max_interactions=1000):
        """Find nodes with diverse temporal patterns"""
        print(f"🔍 Finding {num_nodes} nodes with diverse temporal patterns...")
        
        # Get candidate nodes
        node_counts = self.df['u'].value_counts()
        candidates = node_counts[
            (node_counts >= min_interactions) & 
            (node_counts <= max_interactions)
        ].index.tolist()
        
        print(f"   ├─ Analyzing {min(len(candidates), 100)} candidate nodes...")
        
        # Analyze patterns
        analyses = []
        for node_id in candidates[:100]:  # Limit for efficiency
            interactions = self.get_node_interactions(node_id)
            analysis = self.analyze_temporal_pattern(interactions)
            analysis['node_id'] = node_id
            analyses.append(analysis)
        
        # Select diverse nodes
        analyses.sort(key=lambda x: x['cv_inter'])
        
        selected = []
        
        # Select nodes with different pattern types
        pattern_types = ['Regular', 'Mixed', 'Bursty', 'Highly Bursty']
        
        for pattern_type in pattern_types:
            candidates_of_type = [a for a in analyses if a['pattern_type'] == pattern_type and a['node_id'] not in selected]
            if candidates_of_type and len(selected) < num_nodes:
                selected.append(candidates_of_type[0]['node_id'])
                print(f"   ├─ {pattern_type}: Node {candidates_of_type[0]['node_id']} (CV={candidates_of_type[0]['cv_inter']:.3f})")
        
        # Fill remaining slots
        while len(selected) < num_nodes and len(selected) < len(analyses):
            for analysis in analyses:
                if analysis['node_id'] not in selected:
                    selected.append(analysis['node_id'])
                    print(f"   ├─ Additional: Node {analysis['node_id']} (CV={analysis['cv_inter']:.3f})")
                    break
        
        print(f"   └─ Selected nodes: {selected}")
        return selected
    
    def create_publication_figure(self, node_ids, sigma=3, figsize=(16, 12)):
        """Create high-quality figure similar to academic publications"""
        print(f"📈 Creating publication-quality temporal pattern figure...")
        
        # Calculate grid dimensions
        n_nodes = len(node_ids)
        if n_nodes <= 4:
            rows, cols = 2, 2
        elif n_nodes <= 6:
            rows, cols = 2, 3
        elif n_nodes <= 9:
            rows, cols = 3, 3
        else:
            rows, cols = 4, 3
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_nodes == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Main title
        fig.suptitle('Temporal Interaction Patterns in Wikipedia Dataset\n' + 
                     'Comparison of Original and Smoothed Time Sequences (σ=3)',
                     fontsize=16, y=0.95)
        
        colors = {
            'original': '#FF6B35',    # Orange-red (similar to Figure 12)
            'smoothed': '#1E88E5',    # Blue
            'trend': '#4CAF50'        # Green for trend line
        }
        
        for idx, node_id in enumerate(node_ids):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # Get interaction data
            interactions = self.get_node_interactions(node_id)
            
            if len(interactions) < 10:
                ax.text(0.5, 0.5, f'Node {node_id}\\nInsufficient data\\n({len(interactions)} interactions)', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'Node {node_id}', fontsize=14)
                continue
            
            # Prepare data
            timestamps = interactions['ts'].values
            indices = np.arange(len(timestamps))
            
            # Apply smoothing
            if len(timestamps) >= 5:
                timestamps_smooth = gaussian_filter1d(timestamps, sigma=sigma)
            else:
                timestamps_smooth = timestamps
            
            # Plot original data (lightly)
            ax.plot(indices, timestamps, color='gray', alpha=0.3, linewidth=0.5, label='Original')
            
            # Plot smoothed data (main curve - this is your "orange line")
            ax.plot(indices, timestamps_smooth, 
                   color=colors['original'], linewidth=2.5, 
                   label='Smoothed (σ=3)', alpha=0.9)
            
            # Add trend line
            if len(indices) > 2:
                z = np.polyfit(indices, timestamps_smooth, 1)
                trend_line = np.poly1d(z)
                ax.plot(indices, trend_line(indices), 
                       color=colors['trend'], linewidth=1.5, 
                       linestyle='--', alpha=0.7, label='Trend')
            
            # Analyze pattern
            analysis = self.analyze_temporal_pattern(interactions, detailed=True)
            
            # Formatting
            ax.set_title(f'Node {node_id} - {analysis["pattern_type"]}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Interaction Index', fontsize=12)
            ax.set_ylabel('Time', fontsize=12)
            ax.grid(True, alpha=0.3, linewidth=0.5)
            
            # Statistics box
            stats_text = f"n = {analysis['num_interactions']}\n" + \
                        f"CV = {analysis['cv_inter']:.3f}\n" + \
                        f"Rate = {analysis['rate_per_unit']:.2e}"
            
            if 'burstiness' in analysis:
                stats_text += f"\nB = {analysis['burstiness']:.3f}"
            
            ax.text(0.02, 0.98, stats_text,
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                            edgecolor='gray', alpha=0.9),
                   fontsize=10, family='monospace')
            
            # Legend (only on first subplot)
            if idx == 0:
                ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        
        # Hide unused subplots
        for idx in range(len(node_ids), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.91)
        return fig
    
    def create_detailed_analysis_plot(self, node_id):
        """Create detailed analysis for a single node"""
        print(f"📊 Creating detailed analysis for Node {node_id}...")
        
        interactions = self.get_node_interactions(node_id)
        
        if len(interactions) < 10:
            print(f"   ❌ Insufficient data for Node {node_id} ({len(interactions)} interactions)")
            return None
        
        analysis = self.analyze_temporal_pattern(interactions, detailed=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        timestamps = interactions['ts'].values
        indices = np.arange(len(timestamps))
        inter_arrivals = np.diff(timestamps)
        
        # 1. Time series
        ax1 = axes[0, 0]
        ax1.plot(indices, timestamps, 'o-', color='#FF6B35', markersize=3, linewidth=1.5)
        ax1.set_title('Raw Time Series', fontweight='bold')
        ax1.set_xlabel('Interaction Index')
        ax1.set_ylabel('Time')
        ax1.grid(True, alpha=0.3)
        
        # 2. Smoothed time series
        ax2 = axes[0, 1]
        timestamps_smooth = gaussian_filter1d(timestamps, sigma=3)
        ax2.plot(indices, timestamps, color='gray', alpha=0.4, linewidth=1, label='Original')
        ax2.plot(indices, timestamps_smooth, color='#FF6B35', linewidth=2.5, label='Smoothed')
        ax2.set_title('Smoothed Time Series (σ=3)', fontweight='bold')
        ax2.set_xlabel('Interaction Index')
        ax2.set_ylabel('Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Inter-arrival times
        ax3 = axes[0, 2]
        ax3.plot(inter_arrivals, 'o-', color='#1E88E5', markersize=2, linewidth=1)
        ax3.set_title('Inter-arrival Times', fontweight='bold')
        ax3.set_xlabel('Event Index')
        ax3.set_ylabel('Inter-arrival Time')
        ax3.grid(True, alpha=0.3)
        
        # 4. Inter-arrival histogram
        ax4 = axes[1, 0]
        ax4.hist(inter_arrivals, bins=min(30, len(inter_arrivals)//2), 
                color='#4CAF50', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax4.set_title('Inter-arrival Distribution', fontweight='bold')
        ax4.set_xlabel('Inter-arrival Time')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
        
        # 5. Cumulative events
        ax5 = axes[1, 1]
        ax5.plot(timestamps, indices, 'o-', color='#9C27B0', markersize=2, linewidth=1.5)
        ax5.set_title('Cumulative Events Over Time', fontweight='bold')
        ax5.set_xlabel('Time')
        ax5.set_ylabel('Cumulative Event Count')
        ax5.grid(True, alpha=0.3)
        
        # 6. Statistics summary
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        stats_text = f"""NODE {node_id} ANALYSIS

Pattern Type: {analysis['pattern_type']}

Basic Statistics:
• Total Interactions: {analysis['num_interactions']:,}
• Time Span: {analysis['total_time_span']:.1f}
• Event Rate: {analysis['rate_per_unit']:.2e} events/time

Inter-arrival Statistics:
• Mean: {analysis['mean_inter']:.2f}
• Std Dev: {analysis['std_inter']:.2f}
• CV: {analysis['cv_inter']:.3f}
• Median: {analysis.get('median_inter', 0):.2f}
• Min: {analysis.get('min_inter', 0):.2f}
• Max: {analysis.get('max_inter', 0):.2f}

Advanced Metrics:
• Skewness: {analysis.get('skewness', 0):.3f}
• Entropy: {analysis.get('entropy', 0):.3f}
• Burstiness: {analysis.get('burstiness', 0):.3f}
"""
        
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, 
                verticalalignment='top', fontsize=11, family='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle(f'Detailed Temporal Analysis: Node {node_id}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        return fig, analysis
    
    def export_node_data(self, node_ids, output_dir):
        """Export raw interaction data for specified nodes"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        for node_id in node_ids:
            interactions = self.get_node_interactions(node_id)
            analysis = self.analyze_temporal_pattern(interactions, detailed=True)
            
            # Save raw data
            output_file = output_dir / f"node_{node_id}_interactions.csv"
            interactions.to_csv(output_file, index=False)
            
            # Save analysis
            analysis_file = output_dir / f"node_{node_id}_analysis.json"
            import json
            with open(analysis_file, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            
            print(f"   ✅ Exported Node {node_id}: {output_file}, {analysis_file}")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Enhanced Node Temporal Pattern Analyzer')
    parser.add_argument('--nodes', type=int, nargs='+', help='Specific node IDs to analyze')
    parser.add_argument('--auto_select', type=int, help='Automatically select N interesting nodes')
    parser.add_argument('--detailed_node', type=int, help='Create detailed analysis for specific node')
    parser.add_argument('--export_data', action='store_true', help='Export raw data')
    parser.add_argument('--output_dir', default='temporal_patterns', help='Output directory')
    parser.add_argument('--sigma', type=float, default=3.0, help='Gaussian filter sigma')
    
    args = parser.parse_args()
    
    print("🚀 Enhanced Node Temporal Pattern Analyzer")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = TemporalPatternAnalyzer()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Select nodes to analyze
    if args.nodes:
        selected_nodes = args.nodes
        print(f"📋 Analyzing specified nodes: {selected_nodes}")
    elif args.auto_select:
        selected_nodes = analyzer.find_interesting_nodes(args.auto_select)
    else:
        # Default: find 4 interesting nodes
        selected_nodes = analyzer.find_interesting_nodes(4)
    
    # Create main visualization
    if selected_nodes and not args.detailed_node:
        fig = analyzer.create_publication_figure(selected_nodes, sigma=args.sigma)
        output_file = output_dir / "wikipedia_temporal_patterns_enhanced.png"
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved main figure: {output_file}")
        plt.show()
    
    # Create detailed analysis
    if args.detailed_node:
        fig, analysis = analyzer.create_detailed_analysis_plot(args.detailed_node)
        if fig:
            output_file = output_dir / f"node_{args.detailed_node}_detailed_analysis.png"
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"   ✅ Saved detailed analysis: {output_file}")
            plt.show()
    
    # Export data
    if args.export_data:
        nodes_to_export = [args.detailed_node] if args.detailed_node else selected_nodes
        analyzer.export_node_data(nodes_to_export, output_dir / "exported_data")
        print(f"   ✅ Data exported to: {output_dir / 'exported_data'}")
    
    print(f"\n✨ Analysis complete! Output saved to: {output_dir}")

if __name__ == "__main__":
    main()