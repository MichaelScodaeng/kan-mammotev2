#!/usr/bin/env python3
"""
Analysis script for training time benchmark results
"""

import pandas as pd
import numpy as np
import sys
import os

def analyze_benchmark_results(csv_file):
    """Analyze benchmark results and create summary reports"""
    
    if not os.path.exists(csv_file):
        print(f"❌ Results file not found: {csv_file}")
        return
    
    print(f"📊 Analyzing benchmark results from: {csv_file}")
    
    # Load results
    df = pd.read_csv(csv_file)
    
    print(f"\n📈 OVERALL STATISTICS:")
    print(f"   Total combinations tested: {len(df)}")
    print(f"   Successful: {len(df[df['status'] == 'success'])} ({len(df[df['status'] == 'success'])/len(df)*100:.1f}%)")
    print(f"   Failed: {len(df[df['status'] == 'failed'])}")
    print(f"   Timeouts: {len(df[df['status'] == 'timeout'])}")
    print(f"   Exceptions: {len(df[df['status'] == 'exception'])}")
    
    # Focus on successful results
    success_df = df[df['status'] == 'success'].copy()
    
    if len(success_df) == 0:
        print("\n⚠️  No successful results to analyze!")
        return
    
    print(f"\n🎯 SUCCESS ANALYSIS ({len(success_df)} combinations):")
    
    # Overall statistics
    print(f"\n⏱️  TIMING STATISTICS:")
    print(f"   Average batch time: {success_df['avg_batch_time_seconds'].mean():.3f}s (std: {success_df['avg_batch_time_seconds'].std():.3f}s)")
    print(f"   Average epoch time: {success_df['estimated_epoch_time_minutes'].mean():.1f} minutes")
    print(f"   Average total training time: {success_df['estimated_total_time_hours'].mean():.1f} hours")
    print(f"   Average total training time: {success_df['estimated_total_time_days'].mean():.2f} days")
    
    # By model
    print(f"\n🤖 BY MODEL:")
    model_stats = success_df.groupby('model').agg({
        'estimated_total_time_hours': ['mean', 'std', 'min', 'max', 'count']
    }).round(2)
    model_stats.columns = ['Mean_Hours', 'Std_Hours', 'Min_Hours', 'Max_Hours', 'Count']
    model_stats = model_stats.sort_values('Mean_Hours')
    print(model_stats.to_string())
    
    # By encoder
    print(f"\n🔧 BY ENCODER:")
    encoder_stats = success_df.groupby('encoder').agg({
        'estimated_total_time_hours': ['mean', 'std', 'min', 'max', 'count']
    }).round(2)
    encoder_stats.columns = ['Mean_Hours', 'Std_Hours', 'Min_Hours', 'Max_Hours', 'Count']
    encoder_stats = encoder_stats.sort_values('Mean_Hours')
    print(encoder_stats.to_string())
    
    # By dataset
    print(f"\n📚 BY DATASET:")
    dataset_stats = success_df.groupby('dataset').agg({
        'estimated_total_time_hours': ['mean', 'std', 'min', 'max', 'count'],
        'training_data_size': 'first'
    }).round(2)
    dataset_stats.columns = ['Mean_Hours', 'Std_Hours', 'Min_Hours', 'Max_Hours', 'Count', 'Data_Size']
    dataset_stats = dataset_stats.sort_values('Mean_Hours')
    print(dataset_stats.to_string())
    
    # Top 10 fastest and slowest combinations
    if len(success_df) >= 10:
        print(f"\n🏃 TOP 10 FASTEST COMBINATIONS:")
        fastest = success_df.nsmallest(10, 'estimated_total_time_hours')[['dataset', 'model', 'encoder', 'estimated_total_time_hours']]
        fastest.columns = ['Dataset', 'Model', 'Encoder', 'Hours']
        print(fastest.to_string(index=False))
        
        print(f"\n🐌 TOP 10 SLOWEST COMBINATIONS:")
        slowest = success_df.nlargest(10, 'estimated_total_time_hours')[['dataset', 'model', 'encoder', 'estimated_total_time_hours']]
        slowest.columns = ['Dataset', 'Model', 'Encoder', 'Hours']
        print(slowest.to_string(index=False))
    
    # Model-Encoder combinations
    print(f"\n🔀 MODEL-ENCODER COMBINATIONS:")
    combo_stats = success_df.groupby(['model', 'encoder']).agg({
        'estimated_total_time_hours': ['mean', 'count']
    }).round(2)
    combo_stats.columns = ['Mean_Hours', 'Count']
    combo_stats = combo_stats.sort_values('Mean_Hours')
    print(combo_stats.to_string())
    
    # Failure analysis
    failed_df = df[df['status'] != 'success']
    if len(failed_df) > 0:
        print(f"\n❌ FAILURE ANALYSIS ({len(failed_df)} combinations):")
        
        # Failures by model
        model_failures = failed_df.groupby(['model', 'status']).size().unstack(fill_value=0)
        print(f"\nFailures by Model:")
        print(model_failures.to_string())
        
        # Failures by encoder
        encoder_failures = failed_df.groupby(['encoder', 'status']).size().unstack(fill_value=0)
        print(f"\nFailures by Encoder:")
        print(encoder_failures.to_string())
        
        # Common error patterns
        if 'error_message' in failed_df.columns:
            print(f"\nCommon Error Patterns:")
            error_counts = failed_df['error_message'].value_counts().head(5)
            for error, count in error_counts.items():
                print(f"   {count}x: {error[:100]}...")
    
    # Save summary to file
    summary_file = csv_file.replace('.csv', '_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Training Time Benchmark Analysis\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n")
        f.write(f"Source: {csv_file}\n\n")
        
        f.write(f"Overall Statistics:\n")
        f.write(f"  Total combinations: {len(df)}\n")
        f.write(f"  Successful: {len(success_df)} ({len(success_df)/len(df)*100:.1f}%)\n")
        f.write(f"  Failed: {len(df[df['status'] == 'failed'])}\n")
        f.write(f"  Timeouts: {len(df[df['status'] == 'timeout'])}\n\n")
        
        if len(success_df) > 0:
            f.write(f"Timing Summary (Successful Results):\n")
            f.write(f"  Average batch time: {success_df['avg_batch_time_seconds'].mean():.3f}s\n")
            f.write(f"  Average total time: {success_df['estimated_total_time_hours'].mean():.1f} hours\n")
            f.write(f"  Fastest combination: {success_df.loc[success_df['estimated_total_time_hours'].idxmin(), 'dataset']} + {success_df.loc[success_df['estimated_total_time_hours'].idxmin(), 'model']} + {success_df.loc[success_df['estimated_total_time_hours'].idxmin(), 'encoder']} ({success_df['estimated_total_time_hours'].min():.1f} hours)\n")
            f.write(f"  Slowest combination: {success_df.loc[success_df['estimated_total_time_hours'].idxmax(), 'dataset']} + {success_df.loc[success_df['estimated_total_time_hours'].idxmax(), 'model']} + {success_df.loc[success_df['estimated_total_time_hours'].idxmax(), 'encoder']} ({success_df['estimated_total_time_hours'].max():.1f} hours)\n")
    
    print(f"\n💾 Summary saved to: {summary_file}")

def main():
    """Main analysis function"""
    if len(sys.argv) != 2:
        print("Usage: python analyze_benchmark.py <csv_file>")
        print("Example: python analyze_benchmark.py full_training_time_benchmark.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    analyze_benchmark_results(csv_file)

if __name__ == "__main__":
    main()