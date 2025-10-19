#!/usr/bin/env python3
"""
Quick test to verify K-MOTE registration fix
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.time_encoders.factory import get_available_encoders, create_time_encoder
import torch

print("🧪 Testing K-MOTE Registration Fix\n")

# Test 1: Check if 'k_mote' is in available encoders
print("Test 1: Check available encoders")
available = get_available_encoders()
if 'k_mote' in available:
    print("✅ PASS: 'k_mote' found in available encoders")
else:
    print("❌ FAIL: 'k_mote' NOT in available encoders")
    print(f"Available: {available}")
    sys.exit(1)

# Test 2: Try to create 'k_mote' encoder
print("\nTest 2: Create 'k_mote' encoder")
try:
    encoder = create_time_encoder(
        encoder_type='k_mote',
        time_dim=32,
        transform_mode='adapter',
        adapter_type='affine'
    )
    print("✅ PASS: K-MOTE encoder created successfully")
    print(f"   Encoder type: {type(encoder).__name__}")
except Exception as e:
    print(f"❌ FAIL: Could not create K-MOTE encoder: {e}")
    sys.exit(1)

# Test 3: Test forward pass with single input
print("\nTest 3: Test forward pass")
try:
    x = torch.randn(2, 10, 1)  # (batch=2, seq_len=10, input_dim=1)
    output = encoder(x)
    print(f"✅ PASS: Forward pass successful")
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {output.shape}")
    
    expected_shape = (2, 10, 32)  # (batch, seq_len, time_dim)
    if output.shape == expected_shape:
        print(f"✅ PASS: Output shape matches expected {expected_shape}")
    else:
        print(f"❌ FAIL: Output shape {output.shape} != expected {expected_shape}")
        sys.exit(1)
except Exception as e:
    print(f"❌ FAIL: Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test different transform modes
print("\nTest 4: Test different transform modes")
for mode in ['adapter', 'shared', 'per_expert']:
    try:
        enc = create_time_encoder(
            encoder_type='k_mote',
            time_dim=16,
            transform_mode=mode
        )
        x_test = torch.randn(1, 5, 1)
        out_test = enc(x_test)
        print(f"✅ PASS: transform_mode='{mode}' works (output shape: {out_test.shape})")
    except Exception as e:
        print(f"❌ FAIL: transform_mode='{mode}' failed: {e}")
        sys.exit(1)

print("\n🎉 All tests passed! K-MOTE registration fix is working correctly.")
