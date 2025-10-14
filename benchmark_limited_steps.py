#!/usr/bin/env python3
"""
Limited Steps Runtime Benchmark
===============================

This script runs all model-dataset-encoder combinations for just a few training steps
and extrapolates to estimate runtime per full epoch. Much faster than full benchmarking.

Usage:
    python benchmark_limited_steps.py [--steps N] [--timeout SECONDS]
    
    --steps N: Number of training steps to run (default: 3)
    --timeout SECONDS: Timeout per combination (default: 300)
"""

import os
import sys
import time
import torch
import psutil
import argparse
import pandas as pd
from datetime import datetime
import traceback
import gc

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from utils.load_configs import get_link_prediction_args
from utils import set_random_seed
from experiments.train_link_prediction import train_link_prediction

# All available combinations
ALL_MODELS = ['TGAT', 'JODIE', 'TGN', 'DyGFormer', 'DyGMamba', 'TCL']
ALL_DATASETS = ['wikipedia', 'reddit', 'mooc', 'lastfm', 'enron', 'SocialEvo', 
                'uci', 'CanParl', 'Contacts', 'Flights', 'UNtrade', 'UNvote', 'USLegis']
ALL_ENCODERS = ['mercer', 'lete', 'time2vec', 'kan_mammote_dual_kmote']

# Cache for real dataset edge counts (populated as needed)
DATASET_EDGE_CACHE = {}


def get_real_dataset_edges(dataset_name, args):
    """Get the actual number of edges from the real dataset"""
    
    # Check cache first
    cache_key = f"{dataset_name}_{getattr(args, 'data_ratio', 1.0)}"
    if cache_key in DATASET_EDGE_CACHE:
        return DATASET_EDGE_CACHE[cache_key]
    
    try:
        # Import here to avoid circular imports
        from train_link_prediction import get_data
        
        # Load the actual dataset
        node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data = get_data(
            dataset_name=dataset_name,
            use_validation=getattr(args, 'use_validation', True),
            different_new_nodes_between_val_and_test=getattr(args, 'different_new_nodes_between_val_and_test', False)
        )
        
        # Get actual edge counts
        total_edges = len(full_data.sources)
        train_edges = len(train_data.sources) 
        
        # Cache the result
        DATASET_EDGE_CACHE[cache_key] = {
            'total_edges': total_edges,
            'train_edges': train_edges
        }
        
        return DATASET_EDGE_CACHE[cache_key]
        
    except Exception as e:
        print(f"Warning: Could not load dataset {dataset_name}, using fallback estimate: {e}")
        # Fallback to reasonable estimates if data loading fails
        fallback_estimates = {
            'wikipedia': 157474, 'reddit': 672447, 'mooc': 411749, 'lastfm': 1293103,
            'enron': 125235, 'SocialEvo': 59835, 'uci': 20296, 'CanParl': 8821,
            'Contacts': 28244, 'Flights': 67536, 'UNtrade': 41317, 'UNvote': 13089,
            'USLegis': 15618
        }
        
        estimated_total = fallback_estimates.get(dataset_name, 50000)
        estimated_train = int(estimated_total * 0.7)  # Assume 70% train split
        
        return {
            'total_edges': estimated_total,
            'train_edges': estimated_train
        }


def preload_dataset_edge_counts(datasets, args):
    """Pre-load edge counts for all datasets to speed up benchmarking"""
    
    print("📊 Pre-loading real dataset edge counts...")
    
    for i, dataset in enumerate(datasets):
        print(f"  [{i+1}/{len(datasets)}] Loading {dataset}...", end=" ")
        try:
            dataset_info = get_real_dataset_edges(dataset, args)
            print(f"✅ {dataset_info['total_edges']:,} total edges, {dataset_info['train_edges']:,} train edges")
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    print("✅ Dataset edge counts pre-loaded\n")


class LimitedStepsBenchmark:
    """Benchmark runner that tests just a few training steps"""
    
    def __init__(self, num_steps=3, timeout_seconds=300):
        self.num_steps = num_steps
        self.timeout_seconds = timeout_seconds
        self.results = []
        
    def get_estimated_batches_per_epoch(self, dataset, args):
        """Estimate number of batches per epoch based on real dataset info and args"""
        
        # Get real dataset edge counts
        dataset_info = get_real_dataset_edges(dataset, args)
        
        # Use the actual training edges (already accounts for train/val/test split)
        train_edges = dataset_info['train_edges']
        total_edges = dataset_info['total_edges']
        
        # Apply data_ratio from args if it's less than 1.0
        data_ratio = getattr(args, 'data_ratio', 1.0)
        if data_ratio < 1.0:
            train_edges = int(train_edges * data_ratio)
        
        # Use actual batch_size from args
        batch_size = getattr(args, 'batch_size', 200)
        batches_per_epoch = max(1, train_edges // batch_size)
        
        return batches_per_epoch, train_edges, total_edges, batch_size
    
    def benchmark_single_combination(self, model, dataset, encoder, args):
        """Run limited steps for a single combination and extrapolate"""
        
        print(f"  🔧 Testing {model} + {dataset} + {encoder}")
        
        start_time = time.time()
        start_memory = psutil.virtual_memory().used / (1024**3)  # GB
        
        try:
            # Configure args
            args.model = model
            args.dataset = dataset
            args.time_encoder = encoder
            args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Set very low epochs and capture step timing
            original_epochs = getattr(args, 'max_epochs', 100)
            args.max_epochs = 1  # Just one epoch to measure steps
            
            # Create a custom trainer that stops after N steps
            step_times = []
            
            def limited_step_trainer():
                """Custom training function that stops after N steps"""
                nonlocal step_times
                
                # Import here to avoid issues
                try:
                    from train_link_prediction import get_neighbor_sampler, get_data, get_model_and_optimizer
                    from torch.utils.data import DataLoader
                    from utils.DataLoader import Data
                    from utils.loss import get_loss
                except ImportError as e:
                    raise ImportError(f"Failed to import required modules: {e}")
                
                set_random_seed(42)
                
                # Load data
                node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data = get_data(
                    dataset_name=args.dataset,
                    use_validation=args.use_validation,
                    different_new_nodes_between_val_and_test=args.different_new_nodes_between_val_and_test
                )
                
                # Get neighbor sampler
                train_neighbor_sampler = get_neighbor_sampler(data=train_data, sample_probability=args.sample_probability)
                
                # Get model
                dynamic_backbone, link_predictor, optimizer = get_model_and_optimizer(
                    args=args,
                    node_raw_features=node_raw_features,
                    edge_raw_features=edge_raw_features
                )
                
                # Create data loader
                train_data_loader = DataLoader(
                    Data(train_data),
                    batch_size=args.batch_size,
                    shuffle=True,
                    pin_memory=True
                )
                
                # Run limited steps
                dynamic_backbone.train()
                link_predictor.train()
                
                step_count = 0
                for batch_idx, input_data in enumerate(train_data_loader):
                    if step_count >= self.num_steps:
                        break
                    
                    step_start = time.time()
                    
                    # Move to device
                    input_data = input_data.to(args.device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    
                    # Get node embeddings
                    source_node_embedding, destination_node_embedding = dynamic_backbone.compute_temporal_embeddings(
                        source_nodes=input_data.sources,
                        destination_nodes=input_data.destinations,
                        destination_times=input_data.timestamps,
                        edge_idxs=input_data.edge_idxs,
                        n_neighbors=args.num_neighbors
                    )
                    
                    # Predict links
                    positive_probabilities = link_predictor(
                        source_node_embedding=source_node_embedding,
                        destination_node_embedding=destination_node_embedding
                    ).squeeze(dim=-1)
                    
                    # Negative sampling and prediction
                    negative_destinations = train_neighbor_sampler.get_historical_neighbors(
                        nodes=input_data.sources,
                        timestamps=input_data.timestamps,
                        n_neighbors=1
                    ).flatten()
                    
                    negative_destination_embedding = dynamic_backbone.compute_destination_embedding(
                        destination_nodes=negative_destinations,
                        destination_times=input_data.timestamps
                    )
                    
                    negative_probabilities = link_predictor(
                        source_node_embedding=source_node_embedding,
                        destination_node_embedding=negative_destination_embedding
                    ).squeeze(dim=-1)
                    
                    # Compute loss
                    loss = get_loss(
                        positive_probabilities=positive_probabilities,
                        negative_probabilities=negative_probabilities,
                        loss_func=args.loss_func
                    )
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    step_time = time.time() - step_start
                    step_times.append(step_time)
                    step_count += 1
                    
                    # Clear cache periodically
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                return step_times
            
            # Run with timeout
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Training timed out")
            
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout_seconds)
            
            try:
                step_times = limited_step_trainer()
                signal.alarm(0)  # Cancel timeout
            finally:
                signal.signal(signal.SIGALRM, old_handler)
            
            # Calculate metrics
            total_time = time.time() - start_time
            end_memory = psutil.virtual_memory().used / (1024**3)
            memory_used = end_memory - start_memory
            
            if step_times:
                avg_step_time = sum(step_times) / len(step_times)
                estimated_batches_per_epoch, train_edges, total_edges, batch_size = self.get_estimated_batches_per_epoch(dataset, args)
                estimated_epoch_time = avg_step_time * estimated_batches_per_epoch
                
                result = {
                    'model': model,
                    'dataset': dataset,
                    'encoder': encoder,
                    'status': 'success',
                    'steps_tested': len(step_times),
                    'avg_step_time_seconds': avg_step_time,
                    'estimated_batches_per_epoch': estimated_batches_per_epoch,
                    'estimated_epoch_time_seconds': estimated_epoch_time,
                    'total_test_time_seconds': total_time,
                    'memory_used_gb': memory_used,
                    'step_times': step_times,
                    'dataset_total_edges': total_edges,
                    'dataset_train_edges': train_edges,
                    'data_ratio': getattr(args, 'data_ratio', 1.0),
                    'batch_size': batch_size
                }
            else:
                result = {
                    'model': model,
                    'dataset': dataset,
                    'encoder': encoder,
                    'status': 'failed',
                    'error_message': 'No steps completed',
                    'total_test_time_seconds': total_time,
                    'memory_used_gb': memory_used
                }
            
        except Exception as e:
            total_time = time.time() - start_time
            end_memory = psutil.virtual_memory().used / (1024**3)
            memory_used = end_memory - start_memory
            
            error_msg = str(e)
            if "timeout" in error_msg.lower():
                error_msg = f"Timeout after {self.timeout_seconds}s"
            
            result = {
                'model': model,
                'dataset': dataset,
                'encoder': encoder,
                'status': 'failed',
                'error_message': error_msg,
                'total_test_time_seconds': total_time,
                'memory_used_gb': memory_used,
                'traceback': traceback.format_exc()
            }
        
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Report result
        if result['status'] == 'success':
            print(f"    ✅ {result['avg_step_time_seconds']:.3f}s/step → {result['estimated_epoch_time_seconds']:.1f}s/epoch")
        else:
            print(f"    ❌ Failed: {result['error_message']}")
        
        return result
    
    def run_all_combinations(self, max_combinations=None):
        """Run benchmark on all combinations"""
        
        print(f"🚀 Limited Steps Runtime Benchmark")
        print(f"   Steps per test: {self.num_steps}")
        print(f"   Timeout per test: {self.timeout_seconds}s")
        print("="*60)
        
        # Get base args
        args = get_link_prediction_args(is_evaluation=False)
        
        # Generate all combinations
        all_combinations = []
        for model in ALL_MODELS:
            for dataset in ALL_DATASETS:
                for encoder in ALL_ENCODERS:
                    all_combinations.append((model, dataset, encoder))
        
        if max_combinations:
            all_combinations = all_combinations[:max_combinations]
        
        # Pre-load dataset edge counts for accuracy
        unique_datasets = list(set([combo[1] for combo in all_combinations]))
        preload_dataset_edge_counts(unique_datasets, args)
        
        print(f"Testing {len(all_combinations)} combinations...")
        
        start_time = time.time()
        
        for i, (model, dataset, encoder) in enumerate(all_combinations):
            print(f"\n[{i+1}/{len(all_combinations)}] ", end="")
            
            result = self.benchmark_single_combination(model, dataset, encoder, args)
            self.results.append(result)
            
            # Progress update
            elapsed = time.time() - start_time
            if i > 0:
                avg_time_per_combo = elapsed / (i + 1)
                remaining_time = avg_time_per_combo * (len(all_combinations) - i - 1)
                print(f"    ⏱️  ETA: {remaining_time/60:.1f} minutes")
        
        print(f"\n✅ Completed {len(all_combinations)} combinations in {elapsed/60:.1f} minutes")
        
        return self.results
    
    def analyze_results(self):
        """Analyze and present results"""
        
        if not self.results:
            print("No results to analyze")
            return
        
        df = pd.DataFrame(self.results)
        successful_df = df[df['status'] == 'success'].copy()
        
        print(f"\n📊 ANALYSIS RESULTS")
        print("="*50)
        
        # Overall statistics
        total_combinations = len(df)
        successful_combinations = len(successful_df)
        success_rate = successful_combinations / total_combinations * 100
        
        print(f"Total combinations tested: {total_combinations}")
        print(f"Successful combinations: {successful_combinations} ({success_rate:.1f}%)")
        print(f"Failed combinations: {total_combinations - successful_combinations}")
        
        if successful_combinations == 0:
            print("❌ No successful combinations to analyze")
            return df
        
        # Runtime statistics
        print(f"\nEstimated epoch times:")
        print(f"  Average: {successful_df['estimated_epoch_time_seconds'].mean():.1f}s")
        print(f"  Median: {successful_df['estimated_epoch_time_seconds'].median():.1f}s")
        print(f"  Fastest: {successful_df['estimated_epoch_time_seconds'].min():.1f}s")
        print(f"  Slowest: {successful_df['estimated_epoch_time_seconds'].max():.1f}s")
        
        # Time for 100 epochs
        successful_df['estimated_100_epochs_hours'] = successful_df['estimated_epoch_time_seconds'] * 100 / 3600
        print(f"\nEstimated time for 100 epochs:")
        print(f"  Average: {successful_df['estimated_100_epochs_hours'].mean():.1f} hours")
        print(f"  Total for all successful: {successful_df['estimated_100_epochs_hours'].sum():.0f} hours")
        
        # By model
        if len(successful_df) > 0:
            print(f"\n📊 BY MODEL (avg estimated epoch time):")
            model_stats = successful_df.groupby('model')['estimated_epoch_time_seconds'].agg(['mean', 'count']).sort_values('mean')
            for model, (avg_time, count) in model_stats.iterrows():
                print(f"  {model:12s}: {avg_time:6.1f}s (n={count})")
        
        # By encoder
        if len(successful_df) > 0:
            print(f"\n📊 BY ENCODER (avg estimated epoch time):")
            encoder_stats = successful_df.groupby('encoder')['estimated_epoch_time_seconds'].agg(['mean', 'count']).sort_values('mean')
            for encoder, (avg_time, count) in encoder_stats.iterrows():
                print(f"  {encoder:25s}: {avg_time:6.1f}s (n={count})")
        
        # By dataset
        if len(successful_df) > 0:
            print(f"\n📊 BY DATASET (avg estimated epoch time):")
            dataset_stats = successful_df.groupby('dataset')['estimated_epoch_time_seconds'].agg(['mean', 'count']).sort_values('mean')
            for dataset, (avg_time, count) in dataset_stats.iterrows():
                print(f"  {dataset:12s}: {avg_time:6.1f}s (n={count})")
        
        # Top 10 fastest and slowest
        print(f"\n🚀 TOP 10 FASTEST COMBINATIONS:")
        fastest = successful_df.nsmallest(10, 'estimated_epoch_time_seconds')
        for _, row in fastest.iterrows():
            print(f"  {row['model']} + {row['dataset']} + {row['encoder']}: {row['estimated_epoch_time_seconds']:.1f}s")
        
        print(f"\n🐌 TOP 10 SLOWEST COMBINATIONS:")
        slowest = successful_df.nlargest(10, 'estimated_epoch_time_seconds')
        for _, row in slowest.iterrows():
            print(f"  {row['model']} + {row['dataset']} + {row['encoder']}: {row['estimated_epoch_time_seconds']:.1f}s")
        
        # Failed combinations
        failed_df = df[df['status'] == 'failed']
        if len(failed_df) > 0:
            print(f"\n❌ FAILED COMBINATIONS:")
            failure_reasons = failed_df.groupby('error_message').size().sort_values(ascending=False)
            for reason, count in failure_reasons.items():
                print(f"  {reason}: {count} combinations")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'limited_steps_benchmark_{timestamp}.csv'
        df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")
        
        return df


def main():
    """Main execution"""
    
    parser = argparse.ArgumentParser(description='Limited Steps Runtime Benchmark')
    parser.add_argument('--steps', type=int, default=3, help='Number of training steps to run (default: 3)')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout per combination in seconds (default: 300)')
    parser.add_argument('--max_combinations', type=int, help='Maximum number of combinations to test')
    
    args = parser.parse_args()
    
    # Create benchmark runner
    benchmark = LimitedStepsBenchmark(
        num_steps=args.steps,
        timeout_seconds=args.timeout
    )
    
    # Run benchmark
    results = benchmark.run_all_combinations(max_combinations=args.max_combinations)
    
    # Analyze results
    benchmark.analyze_results()


if __name__ == '__main__':
    main()