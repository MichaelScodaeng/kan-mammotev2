import pandas as pd
import numpy as np
import os
from pathlib import Path

def check_null_values(data_dir='processed_data'):
    """
    Check for null values in all datasets
    
    Args:
        data_dir: Directory containing the processed datasets
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Data directory {data_dir} does not exist!")
        return
    
    # Get all dataset folders
    datasets = [d for d in data_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    print("=" * 80)
    print("CHECKING NULL VALUES IN DATASETS")
    print("=" * 80)
    
    for dataset_dir in sorted(datasets):
        dataset_name = dataset_dir.name
        print(f"\n📊 Dataset: {dataset_name}")
        print("-" * 50)
        
        # Check CSV files
        csv_files = list(dataset_dir.glob("*.csv"))
        for csv_file in csv_files:
            if csv_file.name.endswith('.csvZone.Identifier'):
                continue
                
            try:
                df = pd.read_csv(csv_file)
                null_count = df.isnull().sum().sum()
                print(f"  📄 {csv_file.name}: {df.shape[0]} rows, {df.shape[1]} cols")
                
                if null_count > 0:
                    print(f"      NULL VALUES FOUND: {null_count}")
                    # Show which columns have nulls
                    null_cols = df.isnull().sum()
                    for col, count in null_cols[null_cols > 0].items():
                        print(f"       - {col}: {count} nulls")
                else:
                    print(f"    SUCCESS: No null values")
                    
            except Exception as e:
                print(f"     Error reading {csv_file.name}: {e}")
        
        # Check NPY files
        npy_files = list(dataset_dir.glob("*.npy"))
        for npy_file in npy_files:
            if npy_file.name.endswith('.npyZone.Identifier'):
                continue
                
            try:
                data = np.load(npy_file)
                null_count = np.isnan(data).sum() if data.dtype.kind in ['f', 'c'] else 0
                print(f"  📊 {npy_file.name}: shape {data.shape}, dtype {data.dtype}")
                
                if null_count > 0:
                    print(f"      NaN VALUES FOUND: {null_count}")
                else:
                    print(f"    SUCCESS: No NaN values")
                    
            except Exception as e:
                print(f"     Error reading {npy_file.name}: {e}")

def check_specific_dataset(dataset_name, data_dir='processed_data'):
    """
    Check null values for a specific dataset
    
    Args:
        dataset_name: Name of the dataset to check
        data_dir: Directory containing the processed datasets
    """
    dataset_path = Path(data_dir) / dataset_name
    
    if not dataset_path.exists():
        print(f"Dataset {dataset_name} not found in {data_dir}")
        return
    
    print(f" Detailed analysis for dataset: {dataset_name}")
    print("=" * 60)
    
    # Check main CSV file
    main_csv = dataset_path / f"ml_{dataset_name}.csv"
    if main_csv.exists():
        df = pd.read_csv(main_csv)
        print(f"\n📄 Main CSV ({main_csv.name}):")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        # Check each column for nulls
        print("\n  Null analysis by column:")
        for col in df.columns:
            null_count = df[col].isnull().sum()
            total_count = len(df)
            percentage = (null_count / total_count) * 100
            
            if null_count > 0:
                print(f"      {col}: {null_count}/{total_count} ({percentage:.2f}%) nulls")
            else:
                print(f"    SUCCESS: {col}: No nulls")
        
        # Show sample of data with nulls if any
        if df.isnull().any().any():
            print("\n  Sample rows with null values:")
            null_rows = df[df.isnull().any(axis=1)]
            print(null_rows.head())
    
    # Check node features
    node_file = dataset_path / f"ml_{dataset_name}_node.npy"
    if node_file.exists():
        node_features = np.load(node_file)
        print(f"\n📊 Node features ({node_file.name}):")
        print(f"  Shape: {node_features.shape}")
        print(f"  Dtype: {node_features.dtype}")
        
        if node_features.dtype.kind in ['f', 'c']:
            nan_count = np.isnan(node_features).sum()
            if nan_count > 0:
                print(f"      NaN values: {nan_count}")
            else:
                print(f"     No NaN values")

def get_data_summary(data_dir='processed_data'):
    """
    Get a summary of all datasets
    """
    data_path = Path(data_dir)
    datasets = [d for d in data_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    print(" DATASET SUMMARY")
    print("=" * 60)
    
    for dataset_dir in sorted(datasets):
        dataset_name = dataset_dir.name
        main_csv = dataset_dir / f"ml_{dataset_name}.csv"
        
        if main_csv.exists():
            try:
                df = pd.read_csv(main_csv)
                has_nulls = df.isnull().any().any()
                status = "  HAS NULLS" if has_nulls else " CLEAN"
                print(f"{dataset_name:12} | {df.shape[0]:>8} rows | {df.shape[1]:>2} cols | {status}")
            except:
                print(f"{dataset_name:12} | ERROR READING")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Check for null values in datasets')
    parser.add_argument('--dataset', type=str, help='Check specific dataset')
    parser.add_argument('--summary', action='store_true', help='Show summary only')
    parser.add_argument('--data_dir', type=str, default='processed_data', help='Data directory')
    
    args = parser.parse_args()
    
    if args.summary:
        get_data_summary(args.data_dir)
    elif args.dataset:
        check_specific_dataset(args.dataset, args.data_dir)
    else:
        check_null_values(args.data_dir)