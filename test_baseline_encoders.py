#!/usr/bin/env python3
"""
Test script for baseline time encoders
Tests all baseline encoders and compares with KAN-MAMMOTE
"""

import torch
import torch.nn as nn
import traceback
import time
from models.time_encoders.baseline_encoders import (
    BochnerTimeEncoder, MercerTimeEncoder, RandomTimeEncoder,
    PositionalTimeEncoder, LearnableTimeEncoder, OriginalTimeEncoder,
    create_baseline_encoder
)

def test_baseline_encoders():
    """Test all baseline time encoders"""
    print("=== Testing Baseline Time Encoders ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test configuration
    batch_size = 8
    seq_len = 128
    embedding_dim = 256
    
    # Create test data
    t_abs = torch.randn(batch_size, seq_len, device=device)
    
    encoders = {
        'Bochner': BochnerTimeEncoder(embedding_dim),
        'Mercer': MercerTimeEncoder(embedding_dim, expand_dim=8),
        'Random': RandomTimeEncoder(embedding_dim),
        'Positional': PositionalTimeEncoder(embedding_dim),
        'Learnable': LearnableTimeEncoder(embedding_dim),
        'Original': OriginalTimeEncoder(embedding_dim)
    }
    
    print(f"Input shape: {t_abs.shape}")
    print(f"Expected output shape: ({batch_size}, {seq_len}, {embedding_dim})")
    print()
    
    results = {}
    
    for name, encoder in encoders.items():
        try:
            encoder = encoder.to(device)
            
            # Forward pass
            with torch.no_grad():
                start_time = time.time()
                output = encoder(t_abs)
                end_time = time.time()
            
            # Check shape
            expected_shape = (batch_size, seq_len, embedding_dim)
            assert output.shape == expected_shape, f"Shape mismatch for {name}!"
            
            # Check for NaN/Inf
            assert torch.isfinite(output).all(), f"Non-finite values in {name}!"
            
            # Parameter count
            param_count = sum(p.numel() for p in encoder.parameters())
            forward_time = (end_time - start_time) * 1000  # ms
            
            results[name] = {
                'success': True,
                'params': param_count,
                'time': forward_time,
                'output_shape': output.shape
            }
            
            print(f"✓ {name:12} - Shape: {output.shape}, Params: {param_count:,}, Time: {forward_time:.2f}ms")
            
        except Exception as e:
            results[name] = {'success': False, 'error': str(e)}
            print(f"✗ {name:12} - Failed: {e}")
    
    return results

def test_factory_function():
    """Test the factory function for creating encoders"""
    print("\n=== Testing Factory Function ===")
    
    encoder_types = ['bochner', 'mercer', 'random', 'positional', 'learnable', 'original']
    num_units = 64
    
    for encoder_type in encoder_types:
        try:
            encoder = create_baseline_encoder(encoder_type, num_units)
            param_count = sum(p.numel() for p in encoder.parameters())
            print(f"✓ {encoder_type:12} - Created successfully, Params: {param_count:,}")
        except Exception as e:
            print(f"✗ {encoder_type:12} - Failed: {e}")

def test_gradient_flow():
    """Test gradient flow through encoders"""
    print("\n=== Testing Gradient Flow ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 4
    seq_len = 32
    embedding_dim = 64
    
    # Create test data
    t_abs = torch.randn(batch_size, seq_len, device=device, requires_grad=True)
    target = torch.randn(batch_size, seq_len, embedding_dim, device=device)
    
    encoders = {
        'Bochner': BochnerTimeEncoder(embedding_dim),
        'Mercer': MercerTimeEncoder(embedding_dim),
        'Learnable': LearnableTimeEncoder(embedding_dim),
        'Original': OriginalTimeEncoder(embedding_dim)
    }
    
    for name, encoder in encoders.items():
        try:
            encoder = encoder.to(device)
            encoder.train()
            
            # Forward pass
            output = encoder(t_abs)
            
            # Compute loss
            loss = nn.MSELoss()(output, target)
            
            # Backward pass
            loss.backward(retain_graph=True)
            
            # Check gradients
            has_grad = any(param.grad is not None for param in encoder.parameters() if param.requires_grad)
            
            if has_grad:
                print(f"✓ {name:12} - Gradient flow successful, Loss: {loss.item():.6f}")
            else:
                print(f"⚠ {name:12} - No gradients found")
                
        except Exception as e:
            print(f"✗ {name:12} - Gradient test failed: {e}")

def compare_with_kan_mammote():
    """Compare baseline encoders with KAN-MAMMOTE"""
    print("\n=== Comparing with KAN-MAMMOTE ===")
    
    try:
        from models.time_encoders.kan_mammote import KAN_MAMMOTE
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Configuration
        batch_size = 8
        seq_len = 128
        embedding_dim = 256
        
        # Create models
        models = {
            'KAN-MAMMOTE': KAN_MAMMOTE(embedding_dim=embedding_dim, expert_dim=128, num_mixtures=32),
            'Bochner': BochnerTimeEncoder(embedding_dim),
            'Mercer': MercerTimeEncoder(embedding_dim),
            'Original': OriginalTimeEncoder(embedding_dim),
        }
        
        # Test data
        t_abs = torch.randn(batch_size, seq_len, 1, device=device)
        t_rel = torch.rand(batch_size, seq_len, 1, device=device) + 0.1
        
        print(f"{'Model':<15} {'Parameters':<12} {'Output Shape':<20} {'Time (ms)':<12} {'Status'}")
        print("-" * 75)
        
        for name, model in models.items():
            try:
                model = model.to(device)
                
                with torch.no_grad():
                    start_time = time.time()
                    
                    if name == 'KAN-MAMMOTE':
                        output = model(t_abs, t_rel)
                    else:
                        output = model(t_abs.squeeze(-1))
                    
                    end_time = time.time()
                    forward_time = (end_time - start_time) * 1000
                
                param_count = sum(p.numel() for p in model.parameters())
                
                print(f"{name:<15} {param_count:<12,} {str(output.shape):<20} {forward_time:<12.2f} ✓")
                
            except Exception as e:
                print(f"{name:<15} {'N/A':<12} {'N/A':<20} {'N/A':<12} ✗ ({e})")
        
        print("\n✅ Comparison completed!")
        
    except ImportError:
        print("KAN-MAMMOTE not available for comparison")

def test_different_input_shapes():
    """Test encoders with different input shapes"""
    print("\n=== Testing Different Input Shapes ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding_dim = 64
    
    # Test different input shapes
    test_cases = [
        ("2D Input", torch.randn(4, 32, device=device)),
        ("3D Input (dim=1)", torch.randn(4, 32, 1, device=device)),
        ("Small batch", torch.randn(2, 16, device=device)),
        ("Large sequence", torch.randn(4, 256, device=device)),
    ]
    
    encoder = BochnerTimeEncoder(embedding_dim).to(device)
    
    for case_name, test_input in test_cases:
        try:
            with torch.no_grad():
                output = encoder(test_input)
            
            expected_seq_len = test_input.shape[1]
            expected_shape = (test_input.shape[0], expected_seq_len, embedding_dim)
            
            assert output.shape == expected_shape, f"Shape mismatch for {case_name}!"
            print(f"✓ {case_name:15} - {test_input.shape} → {output.shape}")
            
        except Exception as e:
            print(f"✗ {case_name:15} - Failed: {e}")

def performance_benchmark():
    """Benchmark performance of different encoders"""
    print("\n=== Performance Benchmark ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test configuration
    batch_size = 32
    seq_len = 256
    embedding_dim = 256
    num_runs = 100
    
    encoders = {
        'Bochner': BochnerTimeEncoder(embedding_dim),
        'Mercer': MercerTimeEncoder(embedding_dim),
        'Learnable': LearnableTimeEncoder(embedding_dim),
        'Original': OriginalTimeEncoder(embedding_dim)
    }
    
    test_input = torch.randn(batch_size, seq_len, device=device)
    
    print(f"Benchmarking with batch_size={batch_size}, seq_len={seq_len}, num_runs={num_runs}")
    print(f"{'Encoder':<12} {'Avg Time (ms)':<15} {'Throughput (tokens/s)':<20} {'Memory (MB)':<12}")
    print("-" * 70)
    
    for name, encoder in encoders.items():
        try:
            encoder = encoder.to(device)
            
            # Warmup
            for _ in range(5):
                with torch.no_grad():
                    _ = encoder(test_input)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            
            # Benchmark
            start_time = time.time()
            for _ in range(num_runs):
                with torch.no_grad():
                    _ = encoder(test_input)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            avg_time = (end_time - start_time) / num_runs * 1000  # ms
            throughput = batch_size * seq_len / (avg_time / 1000)  # tokens/sec
            
            # Memory usage
            if device.type == 'cuda':
                memory_used = torch.cuda.max_memory_allocated() / 1024**2  # MB
            else:
                memory_used = 0
            
            print(f"{name:<12} {avg_time:<15.2f} {throughput:<20.0f} {memory_used:<12.1f}")
            
        except Exception as e:
            print(f"{name:<12} Failed: {e}")

if __name__ == "__main__":
    print("🔧 Baseline Time Encoders Test Suite")
    print("=" * 50)
    
    # Run all tests
    results = test_baseline_encoders()
    
    if all(r.get('success', False) for r in results.values()):
        print("\n✅ All baseline encoders working correctly!")
        
        # Additional tests
        test_factory_function()
        test_gradient_flow()
        test_different_input_shapes()
        compare_with_kan_mammote()
        
        # Performance benchmark
        if torch.cuda.is_available():
            performance_benchmark()
        
        print("\n" + "=" * 50)
        print("🎯 Next Steps:")
        print("1. Integrate baselines into your experiment pipeline")
        print("2. Update factory.py to include baseline encoders") 
        print("3. Run comparative experiments:")
        print("   python experiments/train_link_prediction.py --time_encoder_type bochner")
        print("   python experiments/train_link_prediction.py --time_encoder_type kan_mammote")
        print("\n📝 Usage example:")
        print("from models.time_encoders.baseline_encoders import create_baseline_encoder")
        print("encoder = create_baseline_encoder('bochner', embedding_dim=256)")
        print("output = encoder(time_data)")
        
    else:
        print("\n❌ Some baseline encoders failed. Check error messages above.")
    
    print("\n🎉 Baseline encoders are ready for experiments!")
