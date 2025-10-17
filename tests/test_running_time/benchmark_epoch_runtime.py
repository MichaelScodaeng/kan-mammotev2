#!/usr/bin/env python3
"""
Epoch Runtime Benchmarking Script
=================================

This script benchmarks the runtime per epoch for all combinations of:
- Models: TGAT, JODIE, TGN, DyGFormer, DyGMamba, TCL
- Datasets: wikipedia, reddit, mooc, lastfm, enron, SocialEvo, uci, CanParl, Contacts, Flights, UNtrade, UNvote, USLegis
- Time Encoders: mercer, lete, time2vec, kan_mammote_dual_kmote

Key Features:
1. Multi-level sampling for scalability (small → medium → large datasets)
2. Timeout protection to avoid hanging on very slow combinations
3. Memory monitoring to catch OOM issues early
4. Detailed CSV output with runtime estimates
5. Intelligent sampling strategies based on dataset size

Usage:
    python benchmark_epoch_runtime.py --quick_test
    python benchmark_epoch_runtime.py --full_benchmark
    python benchmark_epoch_runtime.py --timeout 300  # 5 minutes per combination
"""

import os
import sys
import time
import psutil
import signal
import argparse
import warnings
import pandas as pd
from datetime import datetime
from contextlib import contextmanager
import torch
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import required modules
from experiments.train_link_prediction import get_available_models, get_available_encoders
from utils.DataLoader import get_link_prediction_data
from utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils import get_neighbor_sampler, NegativeEdgeSampler
from utils.DataLoader import get_idx_data_loader
from models.time_encoders.factory import create_time_encoder
from utils.load_configs import get_link_prediction_args

# Import models
from models.gnn_backbones.TGAT import TGAT
from models.gnn_backbones.MemoryModel import MemoryModel
from models.gnn_backbones.CAWN import CAWN
from models.gnn_backbones.TCL import TCL
from models.gnn_backbones.GraphMixer import GraphMixer
from models.gnn_backbones.DyGFormer import DyGFormer

try:
    from models.gnn_backbones.DyGMamba import DyGMamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False

warnings.filterwarnings('ignore')

# Configuration
ALL_MODELS = ['TGAT', 'JODIE', 'TGN', 'DyGFormer', 'DyGMamba', 'TCL']
ALL_DATASETS = ['wikipedia', 'reddit', 'mooc', 'lastfm', 'enron', 'SocialEvo', 'uci', 
                'CanParl', 'Contacts', 'Flights', 'UNtrade', 'UNvote', 'USLegis']
ALL_ENCODERS = ['mercer', 'lete', 'time2vec', 'kan_mammote_dual_kmote']

# Quick test subsets for faster iteration
QUICK_MODELS = ['TGAT', 'TGN', 'DyGMamba']
QUICK_DATASETS = ['wikipedia', 'reddit', 'mooc']
QUICK_ENCODERS = ['mercer', 'kan_mammote_dual_kmote']

# Sampling strategies based on dataset size
SAMPLING_STRATEGIES = {
    'small': {'data_ratio': 1.0, 'max_batches': 5},      # Small datasets: full data, max 5 batches
    'medium': {'data_ratio': 0.3, 'max_batches': 10},    # Medium datasets: 30% data, max 10 batches  
    'large': {'data_ratio': 0.1, 'max_batches': 20},     # Large datasets: 10% data, max 20 batches
    'huge': {'data_ratio': 0.05, 'max_batches': 50}      # Huge datasets: 5% data, max 50 batches
}

# Dataset size estimates (number of edges)
DATASET_SIZES = {
    'CanParl': 'small',      # ~10K edges
    'Contacts': 'small',     # ~20K edges
    'Flights': 'medium',     # ~100K edges
    'UNtrade': 'small',      # ~50K edges  
    'UNvote': 'small',       # ~20K edges
    'USLegis': 'small',      # ~50K edges
    'enron': 'medium',       # ~200K edges
    'SocialEvo': 'medium',   # ~500K edges
    'uci': 'medium',         # ~200K edges
    'lastfm': 'large',       # ~1M edges
    'mooc': 'large',         # ~400K edges
    'reddit': 'huge',        # ~11M edges
    'wikipedia': 'huge'      # ~200M edges
}


class TimeoutException(Exception):
    """Custom exception for timeouts"""
    pass


@contextmanager
def timeout_context(seconds):
    """Context manager for timeout handling"""
    def timeout_handler(signum, frame):
        raise TimeoutException(f"Operation timed out after {seconds} seconds")
    
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Restore the old signal handler
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)


class MemoryMonitor:
    """Monitor memory usage during benchmarking"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = self.get_memory_mb()
        
    def get_memory_mb(self):
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
        
    def get_gpu_memory_mb(self):
        """Get GPU memory usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0
        
    def check_memory_limit(self, max_memory_mb=8000):
        """Check if memory usage exceeds limit"""
        current_memory = self.get_memory_mb()
        gpu_memory = self.get_gpu_memory_mb()
        
        if current_memory > max_memory_mb:
            raise MemoryError(f"CPU memory exceeded limit: {current_memory:.1f}MB > {max_memory_mb}MB")
        
        if gpu_memory > max_memory_mb:
            raise MemoryError(f"GPU memory exceeded limit: {gpu_memory:.1f}MB > {max_memory_mb}MB")
            
        return current_memory, gpu_memory


def get_sampling_strategy(dataset_name):
    """Get sampling strategy based on dataset size"""
    return SAMPLING_STRATEGIES.get(DATASET_SIZES.get(dataset_name, 'medium'), SAMPLING_STRATEGIES['medium'])


def create_model_instance(model_name, time_encoder, args):
    """Create model instance with proper configuration"""
    
    if model_name == 'TGAT':
        dynamic_backbone = TGAT(
            node_raw_features=args.node_raw_features,
            edge_raw_features=args.edge_raw_features,
            neighbor_sampler=args.neighbor_sampler,
            time_feat_dim=args.time_feat_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            device=args.device,
            time_encoder=time_encoder
        )
        
    elif model_name in ['JODIE', 'DyRep', 'TGN']:
        dynamic_backbone = MemoryModel(
            node_raw_features=args.node_raw_features,
            edge_raw_features=args.edge_raw_features,
            neighbor_sampler=args.neighbor_sampler,
            time_feat_dim=args.time_feat_dim,
            model_name=model_name,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            device=args.device,
            time_encoder=time_encoder
        )
        
    elif model_name == 'TCL':
        dynamic_backbone = TCL(
            node_raw_features=args.node_raw_features,
            edge_raw_features=args.edge_raw_features,
            neighbor_sampler=args.neighbor_sampler,
            time_feat_dim=args.time_feat_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            device=args.device,
            time_encoder=time_encoder
        )
        
    elif model_name == 'GraphMixer':
        dynamic_backbone = GraphMixer(
            node_raw_features=args.node_raw_features,
            edge_raw_features=args.edge_raw_features,
            neighbor_sampler=args.neighbor_sampler,
            time_feat_dim=args.time_feat_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            device=args.device,
            time_encoder=time_encoder
        )
        
    elif model_name == 'DyGMamba':
        if not MAMBA_AVAILABLE:
            raise ImportError("DyGMamba not available")
        dynamic_backbone = DyGMamba(
            node_raw_features=args.node_raw_features,
            edge_raw_features=args.edge_raw_features,
            neighbor_sampler=args.neighbor_sampler,
            time_feat_dim=args.time_feat_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            device=args.device,
            time_encoder=time_encoder
        )
        
    elif model_name == 'DyGFormer':
        dynamic_backbone = DyGFormer(
            node_raw_features=args.node_raw_features,
            edge_raw_features=args.edge_raw_features,
            neighbor_sampler=args.neighbor_sampler,
            time_feat_dim=args.time_feat_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            device=args.device,
            time_encoder=time_encoder
        )
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return dynamic_backbone


def benchmark_single_combination(model_name, dataset_name, encoder_name, args, timeout_seconds=300):
    """Benchmark a single model-dataset-encoder combination"""
    
    result = {
        'model': model_name,
        'dataset': dataset_name,
        'encoder': encoder_name,
        'status': 'failed',
        'epoch_time_seconds': None,
        'batches_processed': 0,
        'total_batches': 0,
        'memory_peak_mb': 0,
        'gpu_memory_peak_mb': 0,
        'parameters': 0,
        'data_edges': 0,
        'sampling_strategy': None,
        'error_message': None
    }
    
    memory_monitor = MemoryMonitor()
    
    try:
        print(f"\n🔧 Benchmarking: {model_name} + {dataset_name} + {encoder_name}")
        
        # Get sampling strategy
        strategy = get_sampling_strategy(dataset_name)
        result['sampling_strategy'] = DATASET_SIZES.get(dataset_name, 'medium')
        
        with timeout_context(timeout_seconds):
            
            # Load data with appropriate sampling
            print(f"   📊 Loading data with {strategy['data_ratio']*100:.1f}% sampling...")
            node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = \
                get_link_prediction_data(
                    dataset_name=dataset_name,
                    val_ratio=0.15,
                    test_ratio=0.15, 
                    seed=42,
                    data_ratio=strategy['data_ratio']
                )
            
            result['data_edges'] = len(train_data.src_node_ids)
            
            # Create neighbor sampler
            train_neighbor_sampler = get_neighbor_sampler(
                data=train_data,
                sample_neighbor_strategy='uniform',
                time_scaling_factor=0.0,
                seed=0
            )
            
            # Create time encoder
            print(f"   🕐 Creating time encoder: {encoder_name}")
            
            # Update args for this combination
            args.model_name = model_name
            args.dataset_name = dataset_name
            args.time_encoder_type = encoder_name
            args.node_raw_features = node_raw_features
            args.edge_raw_features = edge_raw_features
            args.neighbor_sampler = train_neighbor_sampler
            
            time_encoder = create_time_encoder(
                encoder_type=encoder_name,
                time_dim=args.time_feat_dim,
                train_data=train_data,
                train_neighbor_sampler=train_neighbor_sampler,
                args=args,
                device=args.device
            )
            
            # Create model
            print(f"   🤖 Creating model: {model_name}")
            dynamic_backbone = create_model_instance(model_name, time_encoder, args)
            
            # Add link predictor
            from models.gnn_backbones.modules import MergeLayer
            if model_name == 'DyGMamba':
                link_predictor = MergeLayer(input_dim1=dynamic_backbone.node_raw_features.shape[1], 
                                          input_dim2=dynamic_backbone.node_raw_features.shape[1], 
                                          hidden_dim=dynamic_backbone.node_raw_features.shape[1], 
                                          output_dim=1)
            else:
                link_predictor = MergeLayer(input_dim1=dynamic_backbone.node_raw_features.shape[1], 
                                          input_dim2=dynamic_backbone.node_raw_features.shape[1], 
                                          hidden_dim=dynamic_backbone.node_raw_features.shape[1], 
                                          output_dim=1)
            
            model = torch.nn.Sequential(dynamic_backbone, link_predictor)
            model = convert_to_gpu(model, device=args.device)
            
            result['parameters'] = get_parameter_sizes(model)
            
            # Create optimizer
            optimizer = create_optimizer(
                model=model,
                optimizer_name='Adam',
                learning_rate=0.0001,
                weight_decay=1e-5
            )
            
            # Create data loader with limited batches
            max_batches = min(strategy['max_batches'], (len(train_data.src_node_ids) + args.batch_size - 1) // args.batch_size)
            limited_indices = list(range(min(len(train_data.src_node_ids), max_batches * args.batch_size)))
            
            train_idx_data_loader = get_idx_data_loader(
                indices_list=limited_indices,
                batch_size=args.batch_size,
                shuffle=False
            )
            
            result['total_batches'] = len(train_idx_data_loader)
            
            # Create negative sampler
            train_neg_edge_sampler = NegativeEdgeSampler(
                src_node_ids=train_data.src_node_ids,
                dst_node_ids=train_data.dst_node_ids
            )
            
            print(f"   ⏱️  Starting epoch benchmark ({max_batches} batches)...")
            
            # Benchmark epoch time
            epoch_start_time = time.time()
            
            model.train()
            batches_processed = 0
            
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader):
                
                # Memory check every 5 batches
                if batch_idx % 5 == 0:
                    cpu_mem, gpu_mem = memory_monitor.check_memory_limit(max_memory_mb=16000)  # 16GB limit
                    result['memory_peak_mb'] = max(result['memory_peak_mb'], cpu_mem)
                    result['gpu_memory_peak_mb'] = max(result['gpu_memory_peak_mb'], gpu_mem)
                
                # Get batch data
                train_data_indices = train_data_indices.numpy()
                batch_src_node_ids = train_data.src_node_ids[train_data_indices]
                batch_dst_node_ids = train_data.dst_node_ids[train_data_indices]
                batch_node_interact_times = train_data.node_interact_times[train_data_indices]
                batch_edge_ids = train_data.edge_ids[train_data_indices]
                
                # Sample negative edges
                _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(size=len(train_data_indices))
                batch_neg_src_node_ids = batch_src_node_ids
                
                # Convert to tensors
                batch_src_node_ids = torch.from_numpy(batch_src_node_ids).long().to(args.device)
                batch_dst_node_ids = torch.from_numpy(batch_dst_node_ids).long().to(args.device)
                batch_neg_src_node_ids = torch.from_numpy(batch_neg_src_node_ids).long().to(args.device)
                batch_neg_dst_node_ids = torch.from_numpy(batch_neg_dst_node_ids).long().to(args.device)
                batch_node_interact_times = torch.from_numpy(batch_node_interact_times).float().to(args.device)
                batch_edge_ids = torch.from_numpy(batch_edge_ids).long().to(args.device)
                
                # Forward pass
                batch_src_node_embeddings, batch_dst_node_embeddings = model[0].compute_src_dst_node_temporal_embeddings(
                    src_node_ids=batch_src_node_ids,
                    dst_node_ids=batch_dst_node_ids,
                    node_interact_times=batch_node_interact_times,
                    edge_ids=batch_edge_ids
                )
                
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = model[0].compute_src_dst_node_temporal_embeddings(
                    src_node_ids=batch_neg_src_node_ids,
                    dst_node_ids=batch_neg_dst_node_ids,
                    node_interact_times=batch_node_interact_times,
                    edge_ids=batch_edge_ids
                )
                
                # Compute predictions
                pos_scores = model[1](batch_src_node_embeddings, batch_dst_node_embeddings).squeeze(dim=-1)
                neg_scores = model[1](batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings).squeeze(dim=-1)
                
                # Compute loss
                pos_labels = torch.ones_like(pos_scores)
                neg_labels = torch.zeros_like(neg_scores)
                
                labels = torch.cat([pos_labels, neg_labels], dim=0)
                predicts = torch.cat([pos_scores, neg_scores], dim=0)
                
                loss = torch.nn.functional.binary_cross_entropy_with_logits(input=predicts, target=labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                batches_processed += 1
                
                # Break early if we've processed enough batches
                if batches_processed >= max_batches:
                    break
            
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            
            result['status'] = 'success'
            result['epoch_time_seconds'] = epoch_time
            result['batches_processed'] = batches_processed
            
            print(f"   ✅ Success: {epoch_time:.2f}s for {batches_processed} batches")
            
    except TimeoutException:
        result['error_message'] = f'Timeout after {timeout_seconds}s'
        print(f"   ⏰ Timeout after {timeout_seconds}s")
        
    except MemoryError as e:
        result['error_message'] = f'Memory error: {str(e)}'
        print(f"   💾 Memory error: {str(e)}")
        
    except Exception as e:
        result['error_message'] = f'Error: {str(e)}'
        print(f"   ❌ Error: {str(e)}")
    
    finally:
        # Cleanup
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    return result


def run_benchmark(models, datasets, encoders, timeout_seconds=300, output_dir='benchmark_results'):
    """Run comprehensive benchmark"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results
    results = []
    total_combinations = len(models) * len(datasets) * len(encoders)
    current_combination = 0
    
    print(f"🚀 Starting runtime benchmark")
    print(f"   Models: {len(models)} ({models})")
    print(f"   Datasets: {len(datasets)} ({datasets})")
    print(f"   Encoders: {len(encoders)} ({encoders})")
    print(f"   Total combinations: {total_combinations}")
    print(f"   Timeout per combination: {timeout_seconds}s")
    print(f"   Output directory: {output_dir}")
    
    start_time = time.time()
    
    # Get default args
    args = get_link_prediction_args(is_evaluation=False)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for model_name in models:
        for dataset_name in datasets:
            for encoder_name in encoders:
                
                current_combination += 1
                print(f"\n{'='*60}")
                print(f"Progress: {current_combination}/{total_combinations}")
                
                # Skip unavailable combinations
                if model_name == 'DyGMamba' and not MAMBA_AVAILABLE:
                    print(f"⚠️  Skipping {model_name} (not available)")
                    continue
                
                # Set random seed for reproducibility
                set_random_seed(42)
                
                # Run benchmark
                result = benchmark_single_combination(
                    model_name, dataset_name, encoder_name, args, timeout_seconds
                )
                
                results.append(result)
                
                # Save intermediate results
                if len(results) % 10 == 0:  # Save every 10 results
                    save_results(results, output_dir)
    
    # Save final results
    save_results(results, output_dir)
    
    total_time = time.time() - start_time
    print(f"\n🎉 Benchmark completed in {total_time:.1f}s")
    print(f"📊 Results saved to {output_dir}/")
    
    return results


def save_results(results, output_dir):
    """Save results to CSV and generate summary"""
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    detailed_file = os.path.join(output_dir, f'epoch_runtime_detailed_{timestamp}.csv')
    df.to_csv(detailed_file, index=False)
    
    # Create summary by model
    model_summary = df.groupby('model').agg({
        'epoch_time_seconds': ['mean', 'std', 'min', 'max', 'count'],
        'parameters': 'mean',
        'memory_peak_mb': 'mean'
    }).round(2)
    
    model_summary_file = os.path.join(output_dir, f'epoch_runtime_by_model_{timestamp}.csv')
    model_summary.to_csv(model_summary_file)
    
    # Create summary by dataset
    dataset_summary = df.groupby('dataset').agg({
        'epoch_time_seconds': ['mean', 'std', 'min', 'max', 'count'],
        'data_edges': 'mean'
    }).round(2)
    
    dataset_summary_file = os.path.join(output_dir, f'epoch_runtime_by_dataset_{timestamp}.csv')
    dataset_summary.to_csv(dataset_summary_file)
    
    # Create summary by encoder
    encoder_summary = df.groupby('encoder').agg({
        'epoch_time_seconds': ['mean', 'std', 'min', 'max', 'count']
    }).round(2)
    
    encoder_summary_file = os.path.join(output_dir, f'epoch_runtime_by_encoder_{timestamp}.csv')
    encoder_summary.to_csv(encoder_summary_file)
    
    # Print summary
    print(f"\n📊 SUMMARY STATISTICS:")
    successful_results = df[df['status'] == 'success']
    
    if len(successful_results) > 0:
        print(f"   Successful combinations: {len(successful_results)}/{len(df)}")
        print(f"   Average epoch time: {successful_results['epoch_time_seconds'].mean():.2f}s")
        print(f"   Fastest combination: {successful_results['epoch_time_seconds'].min():.2f}s")
        print(f"   Slowest combination: {successful_results['epoch_time_seconds'].max():.2f}s")
        print(f"   Average parameters: {successful_results['parameters'].mean():.0f}")
        print(f"   Average memory: {successful_results['memory_peak_mb'].mean():.1f}MB")
    
    print(f"   Files saved:")
    print(f"     - {detailed_file}")
    print(f"     - {model_summary_file}")
    print(f"     - {dataset_summary_file}")
    print(f"     - {encoder_summary_file}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark epoch runtime for all model-dataset-encoder combinations')
    
    parser.add_argument('--quick_test', action='store_true',
                        help='Run quick test with subset of combinations')
    parser.add_argument('--full_benchmark', action='store_true',
                        help='Run full benchmark with all combinations')
    parser.add_argument('--timeout', type=int, default=300,
                        help='Timeout per combination in seconds (default: 300)')
    parser.add_argument('--models', nargs='+', choices=ALL_MODELS,
                        help='Specific models to benchmark')
    parser.add_argument('--datasets', nargs='+', choices=ALL_DATASETS,
                        help='Specific datasets to benchmark')
    parser.add_argument('--encoders', nargs='+', choices=ALL_ENCODERS,
                        help='Specific encoders to benchmark')
    parser.add_argument('--output_dir', default='benchmark_results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Determine combinations to test
    if args.quick_test:
        models = QUICK_MODELS
        datasets = QUICK_DATASETS
        encoders = QUICK_ENCODERS
        print("🔧 Running quick test with subset of combinations")
    elif args.full_benchmark:
        models = ALL_MODELS
        datasets = ALL_DATASETS
        encoders = ALL_ENCODERS
        print("🔧 Running full benchmark with all combinations")
    else:
        models = args.models or QUICK_MODELS
        datasets = args.datasets or QUICK_DATASETS
        encoders = args.encoders or QUICK_ENCODERS
        print("🔧 Running custom benchmark")
    
    # Run benchmark
    results = run_benchmark(
        models=models,
        datasets=datasets,
        encoders=encoders,
        timeout_seconds=args.timeout,
        output_dir=args.output_dir
    )
    
    print(f"\n✅ Benchmark completed! Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()