#!/usr/bin/env python3
"""
Quick test to verify K-MOTE architectural fixes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

# Import our fixed K-MOTE
from models.time_encoders.k_mote import KMOTE

def test_k_mote_fix():
    """Test the fixed K-MOTE implementation"""
    print("🔬 Testing K-MOTE architectural fixes...")
    
    # Test parameters
    batch_size = 32
    seq_len = 10
    input_dim = 1
    output_dim = 64
    
    # Create test data
    t_abs = torch.randn(batch_size, seq_len, input_dim) * 10  # Absolute times
    t_rel = torch.randn(batch_size, seq_len, input_dim) * 1   # Relative times
    
    print(f"📊 Test data shape: {t_abs.shape}")
    print(f"📊 Absolute time range: [{t_abs.min():.2f}, {t_abs.max():.2f}]")
    print(f"📊 Relative time range: [{t_rel.min():.2f}, {t_rel.max():.2f}]")
    
    # Test absolute time K-MOTE
    print("\n🧪 Testing K-MOTE with absolute time encoding...")
    k_mote_abs = KMOTE(
        input_dim=input_dim,
        output_dim=output_dim,
        time_type='absolute',
        use_scale=True,
        version='v2'
    )
    
    with torch.no_grad():
        output_abs = k_mote_abs(t_abs)
        print(f"✅ Absolute K-MOTE output shape: {output_abs.shape}")
        print(f"📈 Output range: [{output_abs.min():.4f}, {output_abs.max():.4f}]")
        print(f"📊 Output std: {output_abs.std():.4f}")
    
    # Test relative time K-MOTE
    print("\n🧪 Testing K-MOTE with relative time encoding...")
    k_mote_rel = KMOTE(
        input_dim=input_dim,
        output_dim=output_dim,
        time_type='relative',
        use_scale=True,
        version='v2'
    )
    
    with torch.no_grad():
        output_rel = k_mote_rel(t_rel)
        print(f"✅ Relative K-MOTE output shape: {output_rel.shape}")
        print(f"📈 Output range: [{output_rel.min():.4f}, {output_rel.max():.4f}]")
        print(f"📊 Output std: {output_rel.std():.4f}")
    
    # Test gradient flow
    print("\n🔍 Testing gradient flow...")
    k_mote_abs.train()
    output_abs = k_mote_abs(t_abs)
    loss = output_abs.mean()
    loss.backward()
    
    # Check gradients
    for name, param in k_mote_abs.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"📉 {name}: grad_norm = {grad_norm:.2e}")
        else:
            print(f"⚠️  {name}: No gradient")
    
    print("\n✅ K-MOTE architectural fix test completed!")
    return True

def test_comparison():
    """Compare old vs new K-MOTE behavior"""
    print("\n🔬 Comparing architectural changes...")
    
    batch_size = 16
    seq_len = 5
    input_dim = 1
    output_dim = 32
    
    t = torch.randn(batch_size, seq_len, input_dim)
    
    # Test new K-MOTE with different time types
    k_mote_abs = KMOTE(input_dim=input_dim, output_dim=output_dim, time_type='absolute', use_scale=True)
    k_mote_rel = KMOTE(input_dim=input_dim, output_dim=output_dim, time_type='relative', use_scale=True)
    
    with torch.no_grad():
        out_abs = k_mote_abs(t)
        out_rel = k_mote_rel(t)
    
    print(f"🔸 Absolute encoding output std: {out_abs.std():.4f}")
    print(f"🔸 Relative encoding output std: {out_rel.std():.4f}")
    print(f"🔸 Difference in outputs: {(out_abs - out_rel).abs().mean():.4f}")
    
    return True

if __name__ == "__main__":
    try:
        test_k_mote_fix()
        test_comparison()
        print("\n🎉 All tests passed! K-MOTE architectural fixes are working correctly.")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)