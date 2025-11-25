"""
Metrics Logger Utility

This module provides utilities for logging and saving training/validation metrics
during model training. It saves epoch-wise metrics to CSV files for easy analysis
and visualization.
"""

import os
import csv
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime


class MetricsLogger:
    """
    A utility class for logging training and validation metrics per epoch.
    
    Features:
    - Saves metrics to CSV files
    - Supports multiple metrics (AP, ROC-AUC, loss, etc.)
    - Separate logging for train/val/test metrics
    - Easy to load for plotting and analysis
    """
    
    def __init__(self, save_dir: str, model_name: str, dataset_name: str, 
                 encoder_type: str, run_id: int = 0, save_model_name: str = None):
        """
        Initialize the metrics logger.
        
        Args:
            save_dir: Base directory to save metrics
            model_name: Name of the model (e.g., 'TGAT')
            dataset_name: Name of the dataset (e.g., 'wikipedia')
            encoder_type: Type of time encoder (e.g., 'KMM')
            run_id: Run/seed identifier for multiple runs
            save_model_name: Full save model name (overrides auto-generated name if provided)
                            This is used to support --save_model_name_suffix for isolated experiments
        """
        self.save_dir = save_dir
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.encoder_type = encoder_type
        self.run_id = run_id
        
        #  FIX: Use provided save_model_name if available (includes suffix for isolation)
        # Otherwise auto-generate for backward compatibility
        if save_model_name:
            dir_name = save_model_name  # Use full name with suffix (e.g., *_val_lastfm_jodie)
        else:
            dir_name = f"{model_name}_{encoder_type}_seed{run_id}"  # Legacy fallback
        
        # Create save directory
        self.metrics_dir = os.path.join(
            save_dir, model_name, dataset_name, dir_name
        )
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Initialize storage
        self.train_metrics = []
        self.val_metrics = []
        self.new_node_val_metrics = []
        self.test_metrics = []
        self.new_node_test_metrics = []
        self.test_periodic_metrics = []
        self.new_node_test_periodic_metrics = []
        
        # File paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.train_csv = os.path.join(self.metrics_dir, f"train_metrics_{timestamp}.csv")
        self.val_csv = os.path.join(self.metrics_dir, f"val_metrics_{timestamp}.csv")
        self.new_node_val_csv = os.path.join(self.metrics_dir, f"new_node_val_metrics_{timestamp}.csv")
        self.test_csv = os.path.join(self.metrics_dir, f"test_metrics_{timestamp}.csv")
        self.new_node_test_csv = os.path.join(self.metrics_dir, f"new_node_test_metrics_{timestamp}.csv")
        self.test_periodic_csv = os.path.join(self.metrics_dir, f"test_periodic_metrics_{timestamp}.csv")
        self.new_node_test_periodic_csv = os.path.join(self.metrics_dir, f"new_node_test_periodic_metrics_{timestamp}.csv")
        
        # Headers written flag
        self.train_header_written = False
        self.val_header_written = False
        self.new_node_val_header_written = False
        self.test_header_written = False
        self.new_node_test_header_written = False
        self.test_periodic_header_written = False
        self.new_node_test_periodic_header_written = False
        
        print(f"📊 MetricsLogger initialized:")
        print(f"   Save directory: {self.metrics_dir}")
        print(f"   Train metrics: {os.path.basename(self.train_csv)}")
        print(f"   Val metrics: {os.path.basename(self.val_csv)}")
        print(f"   New node val metrics: {os.path.basename(self.new_node_val_csv)}")
        print(f"   Test metrics: {os.path.basename(self.test_csv)}")
        print(f"   New node test metrics: {os.path.basename(self.new_node_test_csv)}")
        print(f"   Test periodic metrics: {os.path.basename(self.test_periodic_csv)}")
        print(f"   New node test periodic metrics: {os.path.basename(self.new_node_test_periodic_csv)}")
    
    def log_epoch_metrics(self, epoch: int, phase: str, metrics: Dict[str, float], 
                          loss: Optional[float] = None):
        """
        Log metrics for a single epoch.
        
        Args:
            epoch: Current epoch number
            phase: Training phase ('train', 'val', or 'test')
            metrics: Dictionary of metric names and values
            loss: Optional loss value
        """
        # Prepare metrics dict
        metrics_dict = {'epoch': epoch}
        if loss is not None:
            metrics_dict['loss'] = loss
        metrics_dict.update(metrics)
        
        # Add to appropriate storage
        if phase == 'train':
            self.train_metrics.append(metrics_dict)
            self._write_to_csv(metrics_dict, self.train_csv, self.train_header_written)
            self.train_header_written = True
        elif phase == 'val':
            self.val_metrics.append(metrics_dict)
            self._write_to_csv(metrics_dict, self.val_csv, self.val_header_written)
            self.val_header_written = True
        elif phase == 'new_node_val':
            self.new_node_val_metrics.append(metrics_dict)
            self._write_to_csv(metrics_dict, self.new_node_val_csv, self.new_node_val_header_written)
            self.new_node_val_header_written = True
        elif phase == 'test':
            self.test_metrics.append(metrics_dict)
            self._write_to_csv(metrics_dict, self.test_csv, self.test_header_written)
            self.test_header_written = True
        elif phase == 'new_node_test':
            self.new_node_test_metrics.append(metrics_dict)
            self._write_to_csv(metrics_dict, self.new_node_test_csv, self.new_node_test_header_written)
            self.new_node_test_header_written = True
        elif phase == 'test_periodic':
            self.test_periodic_metrics.append(metrics_dict)
            self._write_to_csv(metrics_dict, self.test_periodic_csv, self.test_periodic_header_written)
            self.test_periodic_header_written = True
        elif phase == 'new_node_test_periodic':
            self.new_node_test_periodic_metrics.append(metrics_dict)
            self._write_to_csv(metrics_dict, self.new_node_test_periodic_csv, self.new_node_test_periodic_header_written)
            self.new_node_test_periodic_header_written = True
        else:
            raise ValueError(f"Unknown phase: {phase}. Must be one of: 'train', 'val', 'new_node_val', 'test', 'new_node_test', 'test_periodic', 'new_node_test_periodic'")
    
    def _write_to_csv(self, metrics_dict: Dict, csv_path: str, header_written: bool):
        """Write a single row of metrics to CSV file."""
        try:
            with open(csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=metrics_dict.keys())
                if not header_written:
                    writer.writeheader()
                writer.writerow(metrics_dict)
        except IOError as e:
            print(f"Warning: Could not write to {csv_path}: {e}")
    
    def save_summary(self):
        """Save a summary of all metrics."""
        summary_path = os.path.join(self.metrics_dir, "metrics_summary.txt")
        
        try:
            with open(summary_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("TRAINING METRICS SUMMARY\n")
                f.write("=" * 80 + "\n")
                f.write(f"Model: {self.model_name}\n")
                f.write(f"Dataset: {self.dataset_name}\n")
                f.write(f"Encoder: {self.encoder_type}\n")
                f.write(f"Run ID: {self.run_id}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                if self.val_metrics:
                    f.write("VALIDATION METRICS (Best Epoch):\n")
                    f.write("-" * 80 + "\n")
                    
                    # Find best epoch based on average precision
                    best_epoch_idx = max(range(len(self.val_metrics)), 
                                       key=lambda i: self.val_metrics[i].get('average_precision', 0))
                    best_metrics = self.val_metrics[best_epoch_idx]
                    
                    for key, value in best_metrics.items():
                        f.write(f"  {key}: {value:.6f}\n")
                    f.write("\n")
                
                if self.new_node_val_metrics:
                    f.write("NEW NODE VALIDATION METRICS (Best Epoch):\n")
                    f.write("-" * 80 + "\n")
                    
                    # Find best epoch based on average precision
                    best_epoch_idx = max(range(len(self.new_node_val_metrics)), 
                                       key=lambda i: self.new_node_val_metrics[i].get('average_precision', 0))
                    best_metrics = self.new_node_val_metrics[best_epoch_idx]
                    
                    for key, value in best_metrics.items():
                        f.write(f"  {key}: {value:.6f}\n")
                    f.write("\n")
                
                if self.test_metrics:
                    f.write("TEST METRICS (Final):\n")
                    f.write("-" * 80 + "\n")
                    final_test = self.test_metrics[-1]
                    for key, value in final_test.items():
                        if key != 'epoch':
                            f.write(f"  {key}: {value:.6f}\n")
                    f.write("\n")
                
                if self.new_node_test_metrics:
                    f.write("NEW NODE TEST METRICS (Final):\n")
                    f.write("-" * 80 + "\n")
                    final_test = self.new_node_test_metrics[-1]
                    for key, value in final_test.items():
                        if key != 'epoch':
                            f.write(f"  {key}: {value:.6f}\n")
                    f.write("\n")
                
                if self.test_periodic_metrics:
                    f.write("TEST METRICS (Periodic During Training):\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Total periodic test evaluations: {len(self.test_periodic_metrics)}\n")
                    if self.test_periodic_metrics:
                        best_periodic_idx = max(range(len(self.test_periodic_metrics)), 
                                              key=lambda i: self.test_periodic_metrics[i].get('average_precision', 0))
                        best_periodic = self.test_periodic_metrics[best_periodic_idx]
                        f.write(f"Best periodic test performance:\n")
                        for key, value in best_periodic.items():
                            f.write(f"  {key}: {value:.6f}\n")
                    f.write("\n")
            
            print(f"📄 Metrics summary saved to: {summary_path}")
        except IOError as e:
            print(f"Warning: Could not save summary to {summary_path}: {e}")
    
    def load_metrics(self, phase: str = 'val') -> Optional[pd.DataFrame]:
        """
        Load metrics from CSV file as a pandas DataFrame.
        
        Args:
            phase: Which metrics to load ('train', 'val', 'new_node_val', 'test', 'new_node_test', 'test_periodic', 'new_node_test_periodic')
        
        Returns:
            DataFrame with metrics, or None if file doesn't exist
        """
        if phase == 'train':
            csv_path = self.train_csv
        elif phase == 'val':
            csv_path = self.val_csv
        elif phase == 'new_node_val':
            csv_path = self.new_node_val_csv
        elif phase == 'test':
            csv_path = self.test_csv
        elif phase == 'new_node_test':
            csv_path = self.new_node_test_csv
        elif phase == 'test_periodic':
            csv_path = self.test_periodic_csv
        elif phase == 'new_node_test_periodic':
            csv_path = self.new_node_test_periodic_csv
        else:
            raise ValueError(f"Unknown phase: {phase}")
        
        if os.path.exists(csv_path):
            try:
                return pd.read_csv(csv_path)
            except Exception as e:
                print(f"Warning: Could not load {csv_path}: {e}")
                return None
        else:
            print(f"Warning: File not found: {csv_path}")
            return None
    
    def get_best_epoch(self, metric_name: str = 'average_precision') -> Optional[int]:
        """
        Get the epoch number with the best validation metric.
        
        Args:
            metric_name: Name of the metric to optimize
        
        Returns:
            Best epoch number, or None if no validation metrics
        """
        if not self.val_metrics:
            return None
        
        best_idx = max(range(len(self.val_metrics)), 
                      key=lambda i: self.val_metrics[i].get(metric_name, 0))
        return self.val_metrics[best_idx]['epoch']


def create_metrics_logger(args, run_id: int = 0) -> MetricsLogger:
    """
    Factory function to create a MetricsLogger from training arguments.
    
    Args:
        args: Training arguments (argparse.Namespace)
        run_id: Run/seed identifier
    
    Returns:
        Configured MetricsLogger instance
    """
    # Use ablation_dir if provided, otherwise default to ./saved_metrics
    if hasattr(args, 'ablation_dir') and args.ablation_dir:
        save_dir = os.path.join(args.ablation_dir, "saved_metrics")
    else:
        save_dir = "./saved_metrics"
    
    #  FIX: Pass args.save_model_name if it exists (includes suffix for isolation)
    # This ensures validation experiments don't overwrite baseline metrics
    save_model_name = getattr(args, 'save_model_name', None)
    
    return MetricsLogger(
        save_dir=save_dir,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        encoder_type=args.time_encoder_type,
        run_id=run_id,
        save_model_name=save_model_name  #  NEW: Respects --save_model_name_suffix
    )
