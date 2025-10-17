import pandas as pd
import numpy as np

def create_pivot_tables_by_encoder(csv_file_path):
    """
    Create pivot tables for each encoder showing:
    - Rows: datasets
    - Columns: models
    - Values: All Training Time
    """
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Clean column names (remove extra spaces)
    df.columns = df.columns.str.strip()
    
    # Get unique encoders
    encoders = df['encoder'].unique()
    
    print(f"Found {len(encoders)} unique encoders:")
    for i, encoder in enumerate(encoders, 1):
        print(f"{i}. {encoder}")
    
    print("\n" + "="*80 + "\n")
    
    # Create pivot table for each encoder
    for encoder in encoders:
        print(f"PIVOT TABLE FOR ENCODER: {encoder}")
        print("="*60)
        
        # Filter data for current encoder
        encoder_data = df[df['encoder'] == encoder].copy()
        
        # Create pivot table with models as rows and datasets as columns
        pivot = encoder_data.pivot_table(
            index='model', 
            columns='dataset', 
            values='All Training Time',
            aggfunc='first'  # Use first value if duplicates exist
        )
        
        # Round values to 2 decimal places for better readability
        pivot = pivot.round(2)
        
        # Display the pivot table
        print(pivot.to_string())
        
        # Show summary statistics
        print(f"\nSummary for {encoder}:")
        print(f"Number of models: {len(pivot.index)}")
        print(f"Number of datasets: {len(pivot.columns)}")
        print(f"Total experiments: {pivot.notna().sum().sum()}")
        
        # Show average training time by model
        print(f"\nAverage training time by model:")
        avg_by_model = pivot.mean(axis=1).round(2)
        for model, avg_time in avg_by_model.items():
            print(f"  {model}: {avg_time:.2f} hours")
        
        # Show average training time by dataset
        print(f"\nAverage training time by dataset:")
        avg_by_dataset = pivot.mean(axis=0).round(2)
        for dataset, avg_time in avg_by_dataset.items():
            print(f"  {dataset}: {avg_time:.2f} hours")
        
        print("\n" + "="*80 + "\n")
    
    return df, encoders

def save_pivot_tables_to_csv(csv_file_path, output_prefix="pivot_table_"):
    """
    Save each pivot table to a separate CSV file
    """
    df = pd.read_csv(csv_file_path)
    df.columns = df.columns.str.strip()
    
    encoders = df['encoder'].unique()
    
    for encoder in encoders:
        encoder_data = df[df['encoder'] == encoder].copy()
        
        pivot = encoder_data.pivot_table(
            index='model', 
            columns='dataset', 
            values='All Training Time',
            aggfunc='first'
        )
        
        pivot = pivot.round(2)
        
        # Clean encoder name for filename
        clean_encoder_name = encoder.replace('/', '_').replace(' ', '_')
        output_file = f"{output_prefix}{clean_encoder_name}.csv"
        
        pivot.to_csv(output_file)
        print(f"Saved pivot table for '{encoder}' to: {output_file}")

if __name__ == "__main__":
    # Path to your CSV file
    csv_file = "/home/s2516027/kan-mammotev2/training_time_analysis.csv"
    
    # Create and display pivot tables
    df, encoders = create_pivot_tables_by_encoder(csv_file)
    
    # Save pivot tables to separate CSV files
    print("Saving pivot tables to CSV files...")
    save_pivot_tables_to_csv(csv_file)
    
    print(f"\nOriginal data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")