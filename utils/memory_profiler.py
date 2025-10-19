"""
Memory profiling utilities for tracking GPU and CPU memory usage
Helps identify memory bottlenecks in neural network components
"""

import torch
import psutil
import os
from contextlib import contextmanager
from typing import Dict, Optional, List, Tuple
import numpy as np


class MemoryProfiler:
    """
    Comprehensive memory profiler for tracking GPU and CPU memory usage.
    
    Usage:
        profiler = MemoryProfiler(device='cuda')
        
        # Track a specific component
        with profiler.profile("k_mote_forward"):
            output = k_mote(input)
        
        # Get summary
        profiler.print_summary()
    """
    
    def __init__(self, device: str = 'cuda', enabled: bool = True):
        """
        Initialize memory profiler.
        
        Args:
            device: Device to profile ('cuda' or 'cpu')
            enabled: Whether profiling is enabled
        """
        self.device = device
        self.enabled = enabled
        self.profiles: Dict[str, List[Dict]] = {}
        self.current_context = None
        
        # Check if CUDA is available
        self.cuda_available = torch.cuda.is_available() and device == 'cuda'
        
        if self.cuda_available:
            self.device_id = torch.cuda.current_device()
            # Ensure CUDA is initialized
            torch.cuda.synchronize()
        
    def reset(self):
        """Clear all profiling data"""
        self.profiles = {}
        
    @contextmanager
    def profile(self, name: str, print_stats: bool = False):
        """
        Context manager for profiling a code block.
        
        Args:
            name: Name of the component being profiled
            print_stats: Whether to print stats after profiling
            
        Example:
            with profiler.profile("forward_pass"):
                output = model(input)
        """
        if not self.enabled:
            yield
            return
        
        # Initialize profile entry
        if name not in self.profiles:
            self.profiles[name] = []
        
        # Record memory before
        memory_before = self._get_memory_stats()
        
        # Set current context
        old_context = self.current_context
        self.current_context = name
        
        try:
            yield
        finally:
            # Restore old context
            self.current_context = old_context
            
            # Record memory after
            memory_after = self._get_memory_stats()
            
            # Calculate delta
            memory_delta = {
                'gpu_allocated_mb': memory_after['gpu_allocated_mb'] - memory_before['gpu_allocated_mb'],
                'gpu_reserved_mb': memory_after['gpu_reserved_mb'] - memory_before['gpu_reserved_mb'],
                'gpu_cached_mb': memory_after['gpu_cached_mb'] - memory_before['gpu_cached_mb'],
                'cpu_used_mb': memory_after['cpu_used_mb'] - memory_before['cpu_used_mb'],
                'cpu_percent': memory_after['cpu_percent'] - memory_before['cpu_percent'],
            }
            
            # Store profile
            profile_entry = {
                'name': name,
                'before': memory_before,
                'after': memory_after,
                'delta': memory_delta,
                'peak_gpu_mb': memory_after.get('gpu_peak_allocated_mb', 0),
            }
            
            self.profiles[name].append(profile_entry)
            
            if print_stats:
                self._print_profile_entry(profile_entry)
    
    def _get_memory_stats(self) -> Dict:
        """Get current memory statistics"""
        stats = {}
        
        # GPU memory (if available)
        if self.cuda_available:
            torch.cuda.synchronize()
            stats['gpu_allocated_mb'] = torch.cuda.memory_allocated(self.device_id) / 1024**2
            stats['gpu_reserved_mb'] = torch.cuda.memory_reserved(self.device_id) / 1024**2
            stats['gpu_cached_mb'] = torch.cuda.memory_cached(self.device_id) / 1024**2
            stats['gpu_peak_allocated_mb'] = torch.cuda.max_memory_allocated(self.device_id) / 1024**2
            stats['gpu_peak_reserved_mb'] = torch.cuda.max_memory_reserved(self.device_id) / 1024**2
        else:
            stats['gpu_allocated_mb'] = 0
            stats['gpu_reserved_mb'] = 0
            stats['gpu_cached_mb'] = 0
            stats['gpu_peak_allocated_mb'] = 0
            stats['gpu_peak_reserved_mb'] = 0
        
        # CPU memory
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        stats['cpu_used_mb'] = mem_info.rss / 1024**2
        stats['cpu_percent'] = process.memory_percent()
        
        return stats
    
    def _print_profile_entry(self, entry: Dict):
        """Print a single profile entry"""
        print(f"\n{'='*60}")
        print(f"📊 Memory Profile: {entry['name']}")
        print(f"{'='*60}")
        
        delta = entry['delta']
        
        print(f"GPU Memory Change:")
        print(f"  ├─ Allocated: {delta['gpu_allocated_mb']:+.2f} MB")
        print(f"  ├─ Reserved:  {delta['gpu_reserved_mb']:+.2f} MB")
        print(f"  └─ Cached:    {delta['gpu_cached_mb']:+.2f} MB")
        
        print(f"CPU Memory Change:")
        print(f"  ├─ Used:      {delta['cpu_used_mb']:+.2f} MB")
        print(f"  └─ Percent:   {delta['cpu_percent']:+.2f}%")
        
        print(f"Peak GPU Memory: {entry['peak_gpu_mb']:.2f} MB")
        print(f"{'='*60}\n")
    
    def print_summary(self, sort_by: str = 'gpu_allocated_mb'):
        """
        Print summary of all profiled components.
        
        Args:
            sort_by: Metric to sort by ('gpu_allocated_mb', 'gpu_reserved_mb', 'cpu_used_mb')
        """
        if not self.profiles:
            print("No profiling data available")
            return
        
        print(f"\n{'='*80}")
        print(f"📊 MEMORY PROFILING SUMMARY")
        print(f"{'='*80}\n")
        
        # Aggregate statistics per component
        aggregated = {}
        for name, entries in self.profiles.items():
            gpu_alloc = [e['delta']['gpu_allocated_mb'] for e in entries]
            gpu_reserved = [e['delta']['gpu_reserved_mb'] for e in entries]
            cpu_used = [e['delta']['cpu_used_mb'] for e in entries]
            peak_gpu = [e['peak_gpu_mb'] for e in entries]
            
            aggregated[name] = {
                'count': len(entries),
                'gpu_allocated_mb_avg': np.mean(gpu_alloc),
                'gpu_allocated_mb_max': np.max(gpu_alloc),
                'gpu_reserved_mb_avg': np.mean(gpu_reserved),
                'gpu_reserved_mb_max': np.max(gpu_reserved),
                'cpu_used_mb_avg': np.mean(cpu_used),
                'cpu_used_mb_max': np.max(cpu_used),
                'peak_gpu_mb_max': np.max(peak_gpu),
            }
        
        # Sort by specified metric
        sorted_components = sorted(
            aggregated.items(),
            key=lambda x: x[1][f'{sort_by}_max'],
            reverse=True
        )
        
        # Print table header
        print(f"{'Component':<30} {'Calls':<8} {'GPU Avg':<12} {'GPU Max':<12} {'Peak GPU':<12} {'CPU Avg':<12}")
        print(f"{'-'*30} {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
        
        # Print sorted results
        for name, stats in sorted_components:
            print(f"{name:<30} "
                  f"{stats['count']:<8} "
                  f"{stats['gpu_allocated_mb_avg']:>8.2f} MB  "
                  f"{stats['gpu_allocated_mb_max']:>8.2f} MB  "
                  f"{stats['peak_gpu_mb_max']:>8.2f} MB  "
                  f"{stats['cpu_used_mb_avg']:>8.2f} MB")
        
        print(f"\n{'-'*80}")
        
        # Print total memory usage
        total_gpu_avg = sum(s['gpu_allocated_mb_avg'] for s in aggregated.values())
        total_gpu_max = max(s['gpu_allocated_mb_max'] for s in aggregated.values())
        total_peak = max(s['peak_gpu_mb_max'] for s in aggregated.values())
        
        print(f"{'TOTAL':<30} {'':<8} {total_gpu_avg:>8.2f} MB  {total_gpu_max:>8.2f} MB  {total_peak:>8.2f} MB")
        print(f"{'='*80}\n")
        
        # Show top 5 memory hogs
        print(f"🔥 TOP 5 MEMORY CONSUMERS (by max GPU allocated):")
        for i, (name, stats) in enumerate(sorted_components[:5], 1):
            print(f"   {i}. {name}: {stats['gpu_allocated_mb_max']:.2f} MB (peak: {stats['peak_gpu_mb_max']:.2f} MB)")
        
        print(f"\n{'='*80}\n")
    
    def get_current_memory(self) -> Dict:
        """Get current memory usage snapshot"""
        return self._get_memory_stats()
    
    def reset_peak_memory(self):
        """Reset peak memory tracking"""
        if self.cuda_available:
            torch.cuda.reset_peak_memory_stats(self.device_id)
    
    def get_summary_dict(self) -> Dict:
        """
        Get profiling summary as a dictionary for JSON export.
        
        Returns:
            Dictionary with aggregated memory statistics per component
        """
        aggregated = {}
        
        for name, entries in self.profiles.items():
            if not entries:
                continue
            
            # Calculate aggregated statistics
            gpu_allocated_list = [e['delta']['gpu_allocated_mb'] for e in entries]
            gpu_reserved_list = [e['delta']['gpu_reserved_mb'] for e in entries]
            cpu_used_list = [e['delta']['cpu_used_mb'] for e in entries]
            peak_gpu_list = [e['peak_gpu_mb'] for e in entries]
            
            aggregated[name] = {
                'call_count': len(entries),
                'gpu_memory_mb': {
                    'allocated_avg': sum(gpu_allocated_list) / len(gpu_allocated_list),
                    'allocated_min': min(gpu_allocated_list),
                    'allocated_max': max(gpu_allocated_list),
                    'reserved_avg': sum(gpu_reserved_list) / len(gpu_reserved_list),
                    'peak_max': max(peak_gpu_list)
                },
                'cpu_memory_mb': {
                    'avg': sum(cpu_used_list) / len(cpu_used_list),
                    'min': min(cpu_used_list),
                    'max': max(cpu_used_list)
                }
            }
        
        # Add summary statistics
        if aggregated:
            summary = {
                'components': aggregated,
                'total_components': len(aggregated),
                'total_calls': sum(s['call_count'] for s in aggregated.values()),
                'max_gpu_memory_mb': max(s['gpu_memory_mb']['allocated_max'] for s in aggregated.values()) if aggregated else 0,
                'max_peak_gpu_mb': max(s['gpu_memory_mb']['peak_max'] for s in aggregated.values()) if aggregated else 0
            }
        else:
            summary = {
                'components': {},
                'total_components': 0,
                'total_calls': 0,
                'max_gpu_memory_mb': 0,
                'max_peak_gpu_mb': 0
            }
        
        return summary
    
    def export_to_csv(self, filename: str):
        """Export profiling data to CSV"""
        import csv
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Component', 'Call', 
                'GPU_Allocated_MB', 'GPU_Reserved_MB', 'GPU_Cached_MB',
                'CPU_Used_MB', 'CPU_Percent',
                'Peak_GPU_MB'
            ])
            
            for name, entries in self.profiles.items():
                for i, entry in enumerate(entries):
                    delta = entry['delta']
                    writer.writerow([
                        name, i,
                        delta['gpu_allocated_mb'],
                        delta['gpu_reserved_mb'],
                        delta['gpu_cached_mb'],
                        delta['cpu_used_mb'],
                        delta['cpu_percent'],
                        entry['peak_gpu_mb']
                    ])
        
        print(f"💾 Memory profile exported to: {filename}")


def print_memory_snapshot(device='cuda', label="Memory Snapshot"):
    """
    Quick utility to print current memory usage.
    
    Args:
        device: Device to check ('cuda' or 'cpu')
        label: Label for the snapshot
    """
    print(f"\n📸 {label}")
    print(f"{'-'*50}")
    
    if device == 'cuda' and torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        torch.cuda.synchronize()
        
        allocated = torch.cuda.memory_allocated(device_id) / 1024**2
        reserved = torch.cuda.memory_reserved(device_id) / 1024**2
        cached = torch.cuda.memory_cached(device_id) / 1024**2
        peak = torch.cuda.max_memory_allocated(device_id) / 1024**2
        
        print(f"GPU Memory:")
        print(f"  ├─ Allocated: {allocated:.2f} MB")
        print(f"  ├─ Reserved:  {reserved:.2f} MB")
        print(f"  ├─ Cached:    {cached:.2f} MB")
        print(f"  └─ Peak:      {peak:.2f} MB")
    
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    cpu_used = mem_info.rss / 1024**2
    cpu_percent = process.memory_percent()
    
    print(f"CPU Memory:")
    print(f"  ├─ Used:      {cpu_used:.2f} MB")
    print(f"  └─ Percent:   {cpu_percent:.2f}%")
    
    print(f"{'-'*50}\n")


# Convenience function for quick profiling
@contextmanager
def profile_memory(name: str, device='cuda', print_stats=True):
    """
    Standalone context manager for quick memory profiling.
    
    Example:
        with profile_memory("my_operation"):
            result = expensive_operation()
    """
    profiler = MemoryProfiler(device=device)
    with profiler.profile(name, print_stats=print_stats):
        yield
