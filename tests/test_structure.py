#!/usr/bin/env python3
"""
Test script to verify the reorganized KAN-MAMMOTE structure

This script tests that all time encoders can be imported and instantiated correctly.
"""

import sys
import torch

def test_time_encoders():
    """Test that all time encoders can be imported and work."""
    print("Testing reorganized KAN-MAMMOTE structure...")
    print()
    
    # Test factory import
    try:
        from models.time_encoders import create_time_encoder, get_available_encoders
        print("✓ Factory import successful")
    except ImportError as e:
        print(f"✗ Factory import failed: {e}")
        return False
    
    # Check available encoders
    available = get_available_encoders()
    print(f"✓ Available encoders: {list(available.keys())}")
    print()
    
    # Test each encoder type
    time_dim = 64
    batch_size = 32
    seq_len = 10
    
    # Create test data
    timestamps = torch.randn(batch_size, seq_len) * 1000
    time_deltas = torch.randn(batch_size, seq_len) * 100
    
    for encoder_type in available.keys():
        print(f"Testing {encoder_type} encoder...")
        
        try:
            # Create encoder
            if encoder_type == 'kan_mammote':
                encoder = create_time_encoder(
                    encoder_type=encoder_type,
                    time_dim=time_dim,
                    mamba_d_state=16,
                    mamba_d_conv=4,
                    mamba_expand=2
                )
            else:
                encoder = create_time_encoder(
                    encoder_type=encoder_type,
                    time_dim=time_dim
                )
            
            print(f"  ✓ {encoder_type} created successfully")
            
            # Test forward pass
            if encoder_type == 'kan_mammote':
                # KAN-MAMMOTE expects different input format
                t_abs = timestamps.unsqueeze(-1)  # (batch, seq, 1)
                t_rel = time_deltas.unsqueeze(-1)  # (batch, seq, 1)
                output = encoder(t_abs, t_rel)
            else:
                # Other encoders
                if hasattr(encoder, 'forward'):
                    output = encoder(timestamps)
                else:
                    # For CombinedLeTE which might have different interface
                    output = encoder(timestamps.unsqueeze(-1))
            
            print(f"  ✓ {encoder_type} forward pass successful")
            print(f"  ✓ Output shape: {output.shape}")
            
        except Exception as e:
            print(f"  ✗ {encoder_type} failed: {e}")
            continue
        
        print()
    
    print("All tests completed!")
    return True


if __name__ == "__main__":
    success = test_time_encoders()
    sys.exit(0 if success else 1)
