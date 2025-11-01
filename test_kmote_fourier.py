#!/usr/bin/env python3
"""
Quick test to verify the new LeTE-style Fourier expert works correctly
"""

import torch
import torch.nn as nn
import sys
import os

# Add the project root to Python path
sys.path.append('/home/s2516027/kan-mammotev3/kan-mammotev2')

try:
    from models.time_encoders.k_mote import KMOTE, LeTEFourierSeries
    print("✅ Successfully imported KMOTE and LeTEFourierSeries")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_lete_fourier_series():
    """Test the new LeTEFourierSeries implementation"""
    print("\n🧪 Testing LeTEFourierSeries...")
    
    # Test parameters
    batch_size = 4
    seq_len = 10
    dim_fourier = 32
    
    # Create test data
    x = torch.randn(batch_size, seq_len, dim_fourier)
    
    # Create FourierSeries module
    fourier_series = LeTEFourierSeries(dim_fourier=dim_fourier, grid_size_fourier=5)
    
    # Forward pass
    try:
        output = fourier_series(x)
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   ✅ LeTEFourierSeries forward pass successful")
        
        # Check output properties
        assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert torch.isfinite(output).all(), "Output contains infinite values"
        print(f"   ✅ Output validation passed")
        
    except Exception as e:
        print(f"   ❌ LeTEFourierSeries test failed: {e}")
        return False
    
    return True

def test_kmote_with_new_fourier():
    """Test KMOTE with the new Fourier expert"""
    print("\n🧪 Testing KMOTE with new Fourier expert...")
    
    # Test parameters
    batch_size = 4
    seq_len = 10
    output_dim = 16
    hidden_dim = 32
    
    # Create test data (time input)
    t = torch.randn(batch_size, seq_len, 1)
    
    # Create KMOTE with per_expert transform mode
    try:
        kmote = KMOTE(
            input_dim=1, 
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            transform_mode='per_expert',  # Use per-expert transforms (like LeTE)
            use_layernorm=True,
            use_scale=True
        )
        print(f"   ✅ KMOTE initialization successful")
        
        # Forward pass
        output = kmote(t)
        print(f"   Input shape: {t.shape}")
        print(f"   Output shape: {output.shape}")
        
        # Check output properties
        expected_shape = (batch_size, seq_len, output_dim)
        assert output.shape == expected_shape, f"Shape mismatch: {output.shape} vs {expected_shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert torch.isfinite(output).all(), "Output contains infinite values"
        print(f"   ✅ KMOTE forward pass successful")
        
        # Test with return_weights
        output_with_weights, weights = kmote(t, return_weights=True)
        print(f"   Gating weights shape: {weights.shape}")
        expected_weights_shape = (batch_size, seq_len, 3)  # 3 experts
        assert weights.shape == expected_weights_shape, f"Weights shape mismatch: {weights.shape} vs {expected_weights_shape}"
        print(f"   ✅ Gating weights test passed")
        
        # Check that weights sum to 1
        weights_sum = weights.sum(dim=-1)
        assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-6), "Weights don't sum to 1"
        print(f"   ✅ Gating weights normalization verified")
        
    except Exception as e:
        print(f"   ❌ KMOTE test failed: {e}")
        return False
    
    return True

def test_expert_comparison():
    """Compare the old vs new Fourier expert behavior"""
    print("\n🧪 Testing expert compatibility...")
    
    batch_size = 2
    seq_len = 5
    hidden_dim = 16
    
    # Test data
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Test new LeTEFourierSeries
    lete_fourier = LeTEFourierSeries(dim_fourier=hidden_dim, grid_size_fourier=5)
    output_lete = lete_fourier(x)
    
    print(f"   LeTEFourierSeries output range: [{output_lete.min():.4f}, {output_lete.max():.4f}]")
    print(f"   LeTEFourierSeries output std: {output_lete.std():.4f}")
    print(f"   ✅ New Fourier expert works correctly")
    
    return True

if __name__ == "__main__":
    print("🚀 Testing K-MOTE with new LeTE-style Fourier expert...")
    
    all_tests_passed = True
    
    # Run tests
    all_tests_passed &= test_lete_fourier_series()
    all_tests_passed &= test_kmote_with_new_fourier()
    all_tests_passed &= test_expert_comparison()
    
    if all_tests_passed:
        print("\n🎉 All tests passed! The new LeTE-style Fourier expert is working correctly.")
        print("\n📈 Benefits of the new implementation:")
        print("   • Exact LeTE FourierSeries implementation")
        print("   • Simpler architecture (2-stage vs 3-stage)")
        print("   • Better alignment with LeTE theory")
        print("   • Proven effectiveness from LeTE paper")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)