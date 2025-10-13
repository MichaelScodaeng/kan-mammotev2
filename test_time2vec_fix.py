#!/usr/bin/env python3
"""
Test script to verify Time2Vec encoder fix for BCELoss compatibility.
This script tests whether the Time2Vec encoder outputs are bounded properly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from models.time_encoders.time2vec_encoder import Time2VecEncoder

def test_time2vec_bounds():
    """Test that Time2Vec encoder outputs are bounded and don't cause BCELoss issues."""
    print("Testing Time2Vec encoder bounds...")
    
    # Create encoder
    time_dim = 64
    encoder = Time2VecEncoder(time_dim=time_dim, activation='sin')
    
    # Test with various input ranges
    test_cases = [
        ("Small values", torch.tensor([0.1, 0.5, 1.0])),
        ("Medium values", torch.tensor([10.0, 50.0, 100.0])),
        ("Large values", torch.tensor([1000.0, 5000.0, 10000.0])),
        ("Very large values", torch.tensor([100000.0, 500000.0, 1000000.0])),
        ("Negative values", torch.tensor([-100.0, -50.0, -10.0])),
        ("Mixed values", torch.tensor([-1000.0, 0.0, 1000.0, 10000.0])),
    ]
    
    all_passed = True
    
    for test_name, timestamps in test_cases:
        print(f"\n  Testing {test_name}: {timestamps.tolist()}")
        
        # Test encoding
        encoded = encoder(timestamps=timestamps)
        
        # Check output bounds
        min_val = encoded.min().item()
        max_val = encoded.max().item()
        mean_val = encoded.mean().item()
        std_val = encoded.std().item()
        
        print(f"    Output shape: {encoded.shape}")
        print(f"    Output range: [{min_val:.4f}, {max_val:.4f}]")
        print(f"    Mean: {mean_val:.4f}, Std: {std_val:.4f}")
        
        # Check if values are reasonable (not extreme)
        if abs(min_val) > 100 or abs(max_val) > 100:
            print(f"    ❌ FAIL: Values too extreme!")
            all_passed = False
        else:
            print(f"    ✅ PASS: Values within reasonable bounds")
        
        # Test that values don't contain NaN or Inf
        if torch.isnan(encoded).any() or torch.isinf(encoded).any():
            print(f"    ❌ FAIL: Contains NaN or Inf!")
            all_passed = False
        else:
            print(f"    ✅ PASS: No NaN or Inf values")
    
    return all_passed

def test_bceLoss_compatibility():
    """Test that Time2Vec encoder can be used in a simple model with BCELoss."""
    print("\nTesting BCELoss compatibility...")
    
    # Create a simple model that uses Time2Vec encoder
    time_dim = 32
    encoder = Time2VecEncoder(time_dim=time_dim, activation='sin')
    classifier = torch.nn.Sequential(
        torch.nn.Linear(time_dim, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 1),
        torch.nn.Sigmoid()  # This is crucial for BCELoss
    )
    
    # Create some test data
    batch_size = 10
    timestamps = torch.randn(batch_size) * 1000  # Random timestamps
    targets = torch.randint(0, 2, (batch_size, 1)).float()
    
    print(f"  Input timestamps range: [{timestamps.min():.2f}, {timestamps.max():.2f}]")
    
    # Forward pass
    try:
        time_features = encoder(timestamps=timestamps)
        predictions = classifier(time_features)
        
        print(f"  Time features shape: {time_features.shape}")
        print(f"  Time features range: [{time_features.min():.4f}, {time_features.max():.4f}]")
        print(f"  Predictions shape: {predictions.shape}")
        print(f"  Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
        
        # Test BCELoss
        loss_fn = torch.nn.BCELoss()
        loss = loss_fn(predictions, targets)
        
        print(f"  BCELoss: {loss.item():.4f}")
        print(f"  ✅ PASS: BCELoss computed successfully!")
        
        # Test backward pass
        loss.backward()
        print(f"  ✅ PASS: Backward pass successful!")
        
        return True
        
    except Exception as e:
        print(f"  ❌ FAIL: Error during forward/backward pass: {e}")
        return False

def test_dual_stream_interface():
    """Test the dual-stream interface (t_abs, t_rel) used by DyGMamba."""
    print("\nTesting dual-stream interface...")
    
    time_dim = 32
    encoder = Time2VecEncoder(time_dim=time_dim, activation='sin')
    
    # Test data similar to what DyGMamba would pass
    batch_size = 5
    seq_len = 10
    
    # Simulate absolute and relative times
    t_abs = torch.rand(batch_size, seq_len, 1) * 10000  # Absolute timestamps
    t_rel = torch.rand(batch_size, seq_len, 1) * 100    # Relative time differences
    
    try:
        # Test dual-stream interface
        encoded_dual = encoder(t_abs=t_abs, t_rel=t_rel)
        print(f"  Dual-stream output shape: {encoded_dual.shape}")
        print(f"  Dual-stream output range: [{encoded_dual.min():.4f}, {encoded_dual.max():.4f}]")
        
        # Test single timestamp interface
        timestamps = t_rel.squeeze(-1)
        encoded_single = encoder(timestamps=timestamps)
        print(f"  Single-stream output shape: {encoded_single.shape}")
        print(f"  Single-stream output range: [{encoded_single.min():.4f}, {encoded_single.max():.4f}]")
        
        print(f"  ✅ PASS: Both interfaces work correctly!")
        return True
        
    except Exception as e:
        print(f"  ❌ FAIL: Error in dual-stream interface: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Testing Time2Vec encoder fixes for DyGMamba + BCELoss compatibility\n")
    
    # Run all tests
    test1_passed = test_time2vec_bounds()
    test2_passed = test_bceLoss_compatibility()
    test3_passed = test_dual_stream_interface()
    
    print(f"\n{'='*60}")
    print(f"TEST RESULTS:")
    print(f"  Bounds test: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"  BCELoss test: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    print(f"  Dual-stream test: {'✅ PASS' if test3_passed else '❌ FAIL'}")
    
    if all([test1_passed, test2_passed, test3_passed]):
        print(f"\n🎉 ALL TESTS PASSED! Time2Vec encoder should now work with DyGMamba + BCELoss")
    else:
        print(f"\n❌ SOME TESTS FAILED! Further fixes may be needed.")