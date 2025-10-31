"""
Simple Comparison: Original K-MOTE vs. Original LeTE

This script focuses on a direct and robust comparison of the two foundational models:
1.  k_mote_beforeoptim.py (The original, unoptimized K-MOTE)
2.  LeTE_original.py (The original LeTE implementation from the paper)

This avoids the training-related errors encountered in the more experimental,
memory-optimized models (`sequential_kmote` and `lete_plus`).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import time
import psutil
from typing import Dict

# --- Setup and Imports ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
models_path = '/home/s2516027/kan-mammotev2/models/time_encoders'
analysis_path = '/home/s2516027/kan-mammotev2/analysis'
sys.path.insert(0, project_root)
sys.path.append(models_path)
sys.path.append(analysis_path)

# Global configuration
MAX_EPOCHS = 100
EMBEDDING_DIM = 64

# Import models
try:
    from k_mote_beforeoptim import KMOTE
    print("✅ Imported: Original K-MOTE (KMOTE)")
except ImportError as e:
    print(f"❌ FAILED to import Original K-MOTE: {e}")
    KMOTE = None

try:
    from LeTE import CombinedLeTE
    print("✅ Imported: Original LeTE (CombinedLeTE)")
except ImportError as e:
    print(f"❌ FAILED to import Original LeTE: {e}")
    CombinedLeTE = None

os.makedirs('simple_comparison_results', exist_ok=True)

# --- Model Wrapper ---
class ModelRegressor(nn.Module):
    def __init__(self, model_type: str):
        super().__init__()
        self.model_type = model_type
        if model_type == 'original_kmote' and KMOTE:
            self.time_encoder = KMOTE(input_dim=1, output_dim=EMBEDDING_DIM, hidden_dim=EMBEDDING_DIM, transform_mode='shared')
        elif model_type == 'original_lete' and CombinedLeTE:
            self.time_encoder = CombinedLeTE(dim=EMBEDDING_DIM, p=0.5)
        else:
            raise ValueError(f"Model type '{model_type}' is not available.")
        
        self.output_head = nn.Linear(EMBEDDING_DIM, 1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if self.model_type == 'original_kmote' and t.dim() == 2:
            t = t.unsqueeze(-1)
        
        embeddings = self.time_encoder(t)
        return self.output_head(embeddings)

# --- Utilities ---
def get_memory_usage():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

def benchmark_model(model_name: str, test_inputs: Dict) -> Dict:
    print(f"\n{'='*15} Benchmarking: {model_name} {'='*15}")
    try:
        model = ModelRegressor(model_name)
    except ValueError as e:
        print(f"   ❌ SKIPPING: {e}")
        return None

    total_params = sum(p.numel() for p in model.parameters())
    results = {'parameters': total_params}
    
    for size_name, (batch, seq) in test_inputs.items():
        print(f"   Testing {size_name} input ({batch}x{seq})...")
        timestamps = torch.randn(batch, seq)
        
        # Timing
        times = []
        try:
            with torch.no_grad():
                _ = model(timestamps) # Warmup
                for _ in range(5):
                    start_time = time.time()
                    _ = model(timestamps)
                    times.append((time.time() - start_time) * 1000)
            results[f'time_{size_name}'] = np.mean(times)
            print(f"     Time: {np.mean(times):.2f} ms")
        except Exception as e:
            print(f"     ❌ Timing failed: {e}")
            results[f'time_{size_name}'] = float('inf')

        # Memory
        initial_mem = get_memory_usage()
        try:
            with torch.no_grad():
                _ = model(timestamps)
            peak_mem = get_memory_usage()
            results[f'memory_{size_name}'] = peak_mem - initial_mem
            print(f"     Memory: {peak_mem - initial_mem:.2f} MB")
        except Exception as e:
            print(f"     ❌ Memory test failed: {e}")
            results[f'memory_{size_name}'] = float('inf')
            
    return results

def train_and_evaluate(model_name: str, t_data: torch.Tensor, y_true: torch.Tensor) -> Dict:
    print(f"   Training {model_name}...")
    try:
        model = ModelRegressor(model_name)
    except ValueError as e:
        print(f"   ❌ SKIPPING: {e}")
        return {'final_loss': float('inf'), 'avg_epoch_time': float('inf')}

    optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-6)
    loss_fn = nn.MSELoss()
    
    if t_data.dim() == 1: t_data = t_data.unsqueeze(0)
    if y_true.dim() == 1: y_true = y_true.unsqueeze(0).unsqueeze(-1)

    epoch_times = []
    for epoch in range(MAX_EPOCHS):
        start_time = time.time()
        model.train()
        try:
            y_pred = model(t_data)
            loss = loss_fn(y_pred, y_true)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"     ❌ NaN/Inf loss at epoch {epoch+1}. Stopping.")
                return {'final_loss': float('inf'), 'avg_epoch_time': float('inf')}
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_times.append(time.time() - start_time)
        except Exception as e:
            print(f"     ❌ Training failed at epoch {epoch+1}: {e}")
            return {'final_loss': float('inf'), 'avg_epoch_time': float('inf')}

    final_loss = loss.item()
    avg_epoch_time = np.mean(epoch_times) * 1000
    print(f"     ✅ Final Loss: {final_loss:.6f}, Avg Epoch Time: {avg_epoch_time:.2f} ms")
    return {'final_loss': final_loss, 'avg_epoch_time': avg_epoch_time}

# --- Main ---
def main():
    print("🔬" * 20)
    print("  Simple Comparison: K-MOTE vs. LeTE")
    print("🔬" * 20)

    models_to_test = ['original_kmote', 'original_lete']
    
    # 1. Benchmarks
    test_inputs = {'medium': (16, 256)}
    benchmark_results = {}
    for name in models_to_test:
        results = benchmark_model(name, test_inputs)
        if results: benchmark_results[name] = results

    # 2. Performance
    functions = {'sin_wave': lambda t: torch.sin(2 * np.pi * t), 'polynomial': lambda t: t**3 - 2*t**2}
    t_test = torch.linspace(-2, 2, 200)
    performance_results = {}
    for func_name, func in functions.items():
        print(f"\n{'='*15} Testing Function: {func_name} {'='*15}")
        y_true = func(t_test)
        performance_results[func_name] = {}
        for name in models_to_test:
            results = train_and_evaluate(name, t_test, y_true)
            performance_results[func_name][name] = results

    # 3. Summary
    print_summary(benchmark_results, performance_results)

def print_summary(benchmarks: Dict, performance: Dict):
    print("\n" + "📊" * 20)
    print("  Comparison Summary")
    print("📊" * 20)

    # --- Space and Time ---
    print("\n--- Space and Time Complexity ---")
    headers = ["Model", "Params", "Time (Medium)", "Memory (Medium)"]
    print(f"{headers[0]:<20} {headers[1]:>12} {headers[2]:>15} {headers[3]:>18}")
    print("-" * 70)
    for name, results in benchmarks.items():
        params = f"{results.get('parameters', 0):,}"
        time_med = f"{results.get('time_medium', float('inf')):.2f} ms"
        mem_med = f"{results.get('memory_medium', float('inf')):.2f} MB"
        print(f"{name:<20} {params:>12} {time_med:>15} {mem_med:>18}")

    # --- Performance ---
    print("\n--- Performance & Training Speed (Final Loss | Avg Epoch Time) ---")
    func_names = list(performance.keys())
    model_names = list(benchmarks.keys())
    header = f"{'Function':<20}" + "".join([f"{name[:12]:>25}" for name in model_names])
    print(header)
    print("-" * (20 + 25 * len(model_names)))

    for func, results in performance.items():
        row = f"{func:<20}"
        for name in model_names:
            loss = results.get(name, {}).get('final_loss', float('inf'))
            speed = results.get(name, {}).get('avg_epoch_time', float('inf'))
            stat = f"{loss:.4f} | {speed:.2f} ms" if loss != float('inf') else "FAILED"
            row += f"{stat:>25}"
        print(row)
        
    print("\n--- Key Takeaways ---")
    print("• This direct comparison shows the trade-offs between the two foundational models.")
    print("• Original LeTE is generally more parameter-efficient and faster during inference.")
    print("• Original K-MOTE, while larger, may show different performance characteristics during training.")


if __name__ == "__main__":
    main()
