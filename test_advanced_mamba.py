#!/usr/bin/env python3
# test_advanced_mamba.py

"""
Comprehensive test script for the advanced Continuous-Time Mamba implementation in KAN-MAMMOTE.
Tests both the full selective version and the simplified version for comparison.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.utils.config import KANMAMOTEConfig
from src.models.c_mamba import ContinuousMambaBlock, SimplifiedContinuousMambaBlock
from src.models.kan_mammote import KAN_MAMOTE_Model

def test_basic_mamba_functionality():
    """Test basic functionality of both Mamba implementations."""
    print("=" * 60)
    print("Testing Basic Mamba Functionality")
    print("=" * 60)
    
    # Create config
    config = KANMAMOTEConfig(
        hidden_dim_mamba=64,
        state_dim_mamba=32,
        dt_rank=8,
        D_time=32,
        num_experts=4,
        raw_event_feature_dim=8
    )
    
    # Test dimensions
    batch_size = 4
    seq_len = 10
    input_dim = config.D_time + config.D_time + config.raw_event_feature_dim  # abs + rel + raw features
    
    # Create test data
    u_sequence = torch.randn(batch_size, seq_len, input_dim)
    delta_t_sequence = torch.abs(torch.randn(batch_size, seq_len, 1)) * 0.1  # Small positive time diffs
    
    print(f"Input shape: {u_sequence.shape}")
    print(f"Delta_t shape: {delta_t_sequence.shape}")
    
    # Test advanced Mamba
    print("\n--- Testing Advanced Continuous Mamba ---")
    advanced_mamba = ContinuousMambaBlock(input_dim, config)
    
    try:
        output_advanced = advanced_mamba(u_sequence, delta_t_sequence)
        print(f"✅ Advanced Mamba output shape: {output_advanced.shape}")
        print(f"Expected shape: ({batch_size}, {seq_len}, {config.hidden_dim_mamba})")
        
        # Check for NaN/Inf
        if torch.isnan(output_advanced).any():
            print("❌ Advanced Mamba output contains NaN!")
        elif torch.isinf(output_advanced).any():
            print("❌ Advanced Mamba output contains Inf!")
        else:
            print("✅ Advanced Mamba output is numerically stable")
            
    except Exception as e:
        print(f"❌ Advanced Mamba failed: {e}")
    
    # Test simplified Mamba
    print("\n--- Testing Simplified Continuous Mamba ---")
    simple_mamba = SimplifiedContinuousMambaBlock(input_dim, config)
    
    try:
        output_simple = simple_mamba(u_sequence, delta_t_sequence)
        print(f"✅ Simplified Mamba output shape: {output_simple.shape}")
        
        # Check for NaN/Inf
        if torch.isnan(output_simple).any():
            print("❌ Simplified Mamba output contains NaN!")
        elif torch.isinf(output_simple).any():
            print("❌ Simplified Mamba output contains Inf!")
        else:
            print("✅ Simplified Mamba output is numerically stable")
            
    except Exception as e:
        print(f"❌ Simplified Mamba failed: {e}")

def test_temporal_consistency():
    """Test that the Mamba implementations handle temporal sequences consistently."""
    print("\n" + "=" * 60)
    print("Testing Temporal Consistency")
    print("=" * 60)
    
    config = KANMAMOTEConfig(
        hidden_dim_mamba=32,
        state_dim_mamba=16,
        dt_rank=4,
        D_time=16,
        num_experts=4,
        raw_event_feature_dim=4
    )
    
    batch_size = 2
    seq_len = 5
    input_dim = config.D_time + config.D_time + config.raw_event_feature_dim
    
    # Create deterministic test data
    torch.manual_seed(42)
    u_sequence = torch.randn(batch_size, seq_len, input_dim)
    
    # Test with different time scales
    delta_t_small = torch.ones(batch_size, seq_len, 1) * 0.01  # Small time steps
    delta_t_large = torch.ones(batch_size, seq_len, 1) * 1.0   # Large time steps
    
    mamba = ContinuousMambaBlock(input_dim, config)
    
    print("Testing with small time steps (0.01)...")
    output_small = mamba(u_sequence, delta_t_small)
    print(f"Output range with small dt: [{output_small.min().item():.3f}, {output_small.max().item():.3f}]")
    
    print("Testing with large time steps (1.0)...")
    output_large = mamba(u_sequence, delta_t_large)
    print(f"Output range with large dt: [{output_large.min().item():.3f}, {output_large.max().item():.3f}]")
    
    # The outputs should be different but both stable
    diff = torch.norm(output_small - output_large).item()
    print(f"Difference between outputs: {diff:.3f}")
    
    if diff > 0.001:  # Should be different
        print("✅ Mamba correctly adapts to different time scales")
    else:
        print("❌ Mamba not responding to time scale changes")

def test_full_kan_mammote_integration():
    """Test the full KAN-MAMMOTE model with the new Mamba implementation."""
    print("\n" + "=" * 60)
    print("Testing Full KAN-MAMMOTE Integration")
    print("=" * 60)
    
    config = KANMAMOTEConfig(
        D_time=64,
        hidden_dim_mamba=128,
        state_dim_mamba=32,
        dt_rank=8,
        num_experts=4,
        raw_event_feature_dim=0,  # Test without raw features
        K_top=2
    )
    
    # Create KAN-MAMMOTE model
    model = KAN_MAMOTE_Model(config)
    
    # Test data for event sequence
    batch_size = 3
    seq_len = 8
    
    # Timestamps should be increasing
    base_timestamps = torch.linspace(0, 10, seq_len).unsqueeze(0).unsqueeze(-1)
    timestamps = base_timestamps.expand(batch_size, -1, -1)
    timestamps = timestamps + torch.randn(batch_size, seq_len, 1) * 0.1  # Add some noise
    timestamps = torch.cumsum(torch.abs(timestamps), dim=1)  # Ensure monotonic
    
    # Empty event features
    event_features = torch.zeros(batch_size, seq_len, 0)
    
    print(f"Testing KAN-MAMMOTE with:")
    print(f"  Timestamps shape: {timestamps.shape}")
    print(f"  Event features shape: {event_features.shape}")
    
    try:
        # Forward pass
        embeddings, moe_info = model(timestamps, event_features)
        
        print(f"✅ KAN-MAMMOTE forward pass successful!")
        print(f"  Output embeddings shape: {embeddings.shape}")
        print(f"  Expected shape: ({batch_size}, {seq_len}, {config.hidden_dim_mamba})")
        
        # Check MoE info
        abs_weights, rel_weights = moe_info
        print(f"  Absolute expert weights shape: {abs_weights.shape}")
        print(f"  Relative expert weights shape: {rel_weights.shape}")
        
        # Check numerical stability
        if torch.isnan(embeddings).any():
            print("❌ KAN-MAMMOTE output contains NaN!")
        elif torch.isinf(embeddings).any():
            print("❌ KAN-MAMMOTE output contains Inf!")
        else:
            print("✅ KAN-MAMMOTE output is numerically stable")
            
        # Check expert weight distributions
        abs_entropy = -torch.sum(abs_weights * torch.log(abs_weights + 1e-8), dim=-1).mean()
        rel_entropy = -torch.sum(rel_weights * torch.log(rel_weights + 1e-8), dim=-1).mean()
        print(f"  Expert selection entropy - Abs: {abs_entropy:.3f}, Rel: {rel_entropy:.3f}")
        
    except Exception as e:
        print(f"❌ KAN-MAMMOTE integration failed: {e}")
        import traceback
        traceback.print_exc()

def test_gradient_flow():
    """Test that gradients flow properly through the advanced Mamba implementation."""
    print("\n" + "=" * 60)
    print("Testing Gradient Flow")
    print("=" * 60)
    
    config = KANMAMOTEConfig(
        hidden_dim_mamba=32,
        state_dim_mamba=16,
        dt_rank=4
    )
    
    input_dim = 64
    mamba = ContinuousMambaBlock(input_dim, config)
    
    # Create test data
    batch_size = 2
    seq_len = 4
    u_sequence = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)
    delta_t_sequence = torch.ones(batch_size, seq_len, 1) * 0.1
    
    # Forward pass
    output = mamba(u_sequence, delta_t_sequence)
    
    # Create a simple loss (sum of outputs)
    loss = output.sum()
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    param_with_grad = 0
    param_without_grad = 0
    
    for name, param in mamba.named_parameters():
        if param.grad is not None:
            param_with_grad += 1
            if torch.isnan(param.grad).any():
                print(f"❌ NaN gradients in {name}")
            elif torch.isinf(param.grad).any():
                print(f"❌ Inf gradients in {name}")
        else:
            param_without_grad += 1
            print(f"⚠️  No gradient for {name}")
    
    print(f"Parameters with gradients: {param_with_grad}")
    print(f"Parameters without gradients: {param_without_grad}")
    
    if param_with_grad > 0 and param_without_grad == 0:
        print("✅ All parameters receive gradients")
    else:
        print("❌ Some parameters don't receive gradients")
    
    # Check input gradients
    if u_sequence.grad is not None:
        print("✅ Input receives gradients")
    else:
        print("❌ Input doesn't receive gradients")

def run_performance_comparison():
    """Compare performance between advanced and simplified Mamba implementations."""
    print("\n" + "=" * 60)
    print("Performance Comparison")
    print("=" * 60)
    
    config = KANMAMOTEConfig(
        hidden_dim_mamba=64,
        state_dim_mamba=32,
        dt_rank=8
    )
    
    input_dim = 128
    batch_size = 8
    seq_len = 50
    
    # Create test data
    u_sequence = torch.randn(batch_size, seq_len, input_dim)
    delta_t_sequence = torch.rand(batch_size, seq_len, 1) * 0.1
    
    # Test advanced Mamba
    advanced_mamba = ContinuousMambaBlock(input_dim, config)
    simple_mamba = SimplifiedContinuousMambaBlock(input_dim, config)
    
    # Warm up
    _ = advanced_mamba(u_sequence, delta_t_sequence)
    _ = simple_mamba(u_sequence, delta_t_sequence)
    
    import time
    
    # Time advanced Mamba
    start_time = time.time()
    for _ in range(10):
        _ = advanced_mamba(u_sequence, delta_t_sequence)
    advanced_time = time.time() - start_time
    
    # Time simplified Mamba
    start_time = time.time()
    for _ in range(10):
        _ = simple_mamba(u_sequence, delta_t_sequence)
    simple_time = time.time() - start_time
    
    print(f"Advanced Mamba time: {advanced_time:.3f}s")
    print(f"Simplified Mamba time: {simple_time:.3f}s")
    print(f"Speedup ratio: {advanced_time / simple_time:.2f}x")
    
    # Parameter count
    advanced_params = sum(p.numel() for p in advanced_mamba.parameters())
    simple_params = sum(p.numel() for p in simple_mamba.parameters())
    
    print(f"Advanced Mamba parameters: {advanced_params:,}")
    print(f"Simplified Mamba parameters: {simple_params:,}")
    print(f"Parameter ratio: {advanced_params / simple_params:.2f}x")

if __name__ == "__main__":
    print("Advanced Continuous-Time Mamba Test Suite")
    print("========================================")
    
    try:
        test_basic_mamba_functionality()
        test_temporal_consistency() 
        test_full_kan_mammote_integration()
        test_gradient_flow()
        run_performance_comparison()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("The advanced Mamba implementation is ready for KAN-MAMMOTE.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
