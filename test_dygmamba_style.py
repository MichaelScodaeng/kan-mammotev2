#!/usr/bin/env python3
"""
Test script for the DyGMamba-style Continuous Mamba implementation in KAN-MAMMOTE.
"""

import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.append('.')

from src.utils.config import KANMAMOTEConfig
from src.models.c_mamba import ContinuousMambaBlock, SimplifiedContinuousMambaBlock, MAMBA_AVAILABLE

def test_continuous_mamba():
    """Test the continuous Mamba implementation."""
    print("Testing DyGMamba-style Continuous Mamba implementation...")
    print(f"Mamba SSM available: {MAMBA_AVAILABLE}")
    
    # Create test configuration
    config = KANMAMOTEConfig(
        D_time=32,
        hidden_dim_mamba=64,
        state_dim_mamba=16,
        num_mamba_layers=2,
        gamma=0.5,
        num_experts=4,
        K_top=2,
        use_aux_features_router=False,
        raw_event_feature_dim=0
    )
    
    # Test parameters
    batch_size = 4
    seq_len = 10
    input_dim = 32  # Should match K-MOTE output dimension
    
    # Create test data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Random input sequence (simulating K-MOTE output)
    u_sequence = torch.randn(batch_size, seq_len, input_dim, device=device)
    
    # Timestamps (sorted for realistic time series)
    timestamps = torch.sort(torch.rand(batch_size, seq_len, 1, device=device), dim=1)[0]
    
    print(f"Input sequence shape: {u_sequence.shape}")
    print(f"Timestamps shape: {timestamps.shape}")
    
    # Test 1: Full ContinuousMambaBlock
    print("\n=== Testing ContinuousMambaBlock ===")
    try:
        mamba_block = ContinuousMambaBlock(input_dim=input_dim, config=config).to(device)
        
        # Forward pass
        with torch.no_grad():
            output = mamba_block(u_sequence, timestamps)
        
        print(f"✓ ContinuousMambaBlock output shape: {output.shape}")
        print(f"✓ Expected shape: ({batch_size}, {seq_len}, {config.hidden_dim_mamba})")
        
        # Check if output shape is correct
        expected_shape = (batch_size, seq_len, config.hidden_dim_mamba)
        assert output.shape == expected_shape, f"Shape mismatch: {output.shape} vs {expected_shape}"
        
        print("✓ ContinuousMambaBlock test passed!")
        
    except Exception as e:
        print(f"✗ ContinuousMambaBlock test failed: {e}")
    
    # Test 2: Simplified version
    print("\n=== Testing SimplifiedContinuousMambaBlock ===")
    try:
        simplified_block = SimplifiedContinuousMambaBlock(input_dim=input_dim, config=config).to(device)
        
        # Forward pass
        with torch.no_grad():
            output_simplified = simplified_block(u_sequence, timestamps)
        
        print(f"✓ SimplifiedContinuousMambaBlock output shape: {output_simplified.shape}")
        
        # Check if output shape is correct
        expected_shape = (batch_size, seq_len, config.hidden_dim_mamba)
        assert output_simplified.shape == expected_shape, f"Shape mismatch: {output_simplified.shape} vs {expected_shape}"
        
        print("✓ SimplifiedContinuousMambaBlock test passed!")
        
    except Exception as e:
        print(f"✗ SimplifiedContinuousMambaBlock test failed: {e}")
    
    # Test 3: Gradient flow
    print("\n=== Testing Gradient Flow ===")
    try:
        model = SimplifiedContinuousMambaBlock(input_dim=input_dim, config=config).to(device)
        model.train()
        
        # Create target for loss computation
        target = torch.randn(batch_size, seq_len, config.hidden_dim_mamba, device=device)
        
        # Forward pass
        output = model(u_sequence, timestamps)
        
        # Compute loss and backward pass
        loss = nn.MSELoss()(output, target)
        loss.backward()
        
        # Check if gradients exist
        has_gradients = any(p.grad is not None for p in model.parameters())
        assert has_gradients, "No gradients found!"
        
        print(f"✓ Loss value: {loss.item():.4f}")
        print("✓ Gradient flow test passed!")
        
    except Exception as e:
        print(f"✗ Gradient flow test failed: {e}")

def test_time_encoder():
    """Test the TimeEncoder component."""
    print("\n=== Testing TimeEncoder ===")
    try:
        from src.models.c_mamba import TimeEncoder
        
        time_dim = 64
        encoder = TimeEncoder(time_dim=time_dim)
        
        batch_size = 4
        seq_len = 10
        
        # Test with 2D timestamps
        timestamps_2d = torch.randn(batch_size, seq_len)
        output_2d = encoder(timestamps_2d)
        
        print(f"✓ TimeEncoder 2D input shape: {timestamps_2d.shape}")
        print(f"✓ TimeEncoder 2D output shape: {output_2d.shape}")
        
        # Test with 3D timestamps
        timestamps_3d = torch.randn(batch_size, seq_len, 1)
        output_3d = encoder(timestamps_3d)
        
        print(f"✓ TimeEncoder 3D input shape: {timestamps_3d.shape}")
        print(f"✓ TimeEncoder 3D output shape: {output_3d.shape}")
        
        # Check shapes
        expected_shape = (batch_size, seq_len, time_dim)
        assert output_2d.shape == expected_shape
        assert output_3d.shape == expected_shape
        
        print("✓ TimeEncoder test passed!")
        
    except Exception as e:
        print(f"✗ TimeEncoder test failed: {e}")

def test_integration_with_kan_mammote():
    """Test integration with the full KAN-MAMMOTE model."""
    print("\n=== Testing KAN-MAMMOTE Integration ===")
    try:
        from src.models.kan_mammote import KAN_MAMOTE_Model
        
        # Create configuration
        config = KANMAMOTEConfig(
            D_time=32,
            hidden_dim_mamba=64,
            state_dim_mamba=16,
            num_mamba_layers=2,
            gamma=0.5,
            num_experts=4,
            K_top=2,
            use_aux_features_router=False,
            raw_event_feature_dim=8  # Some event features
        )
        
        # Create model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = KAN_MAMOTE_Model(config).to(device)
        
        # Test data
        batch_size = 4
        seq_len = 10
        
        timestamps = torch.sort(torch.rand(batch_size, seq_len, 1, device=device), dim=1)[0]
        event_features = torch.randn(batch_size, seq_len, config.raw_event_feature_dim, device=device)
        
        print(f"Input timestamps shape: {timestamps.shape}")
        print(f"Input event features shape: {event_features.shape}")
        
        # Forward pass
        with torch.no_grad():
            embeddings, moe_info = model(timestamps, event_features)
        
        print(f"✓ KAN-MAMMOTE embeddings shape: {embeddings.shape}")
        print(f"✓ MoE info type: {type(moe_info)}")
        
        expected_shape = (batch_size, seq_len, config.hidden_dim_mamba)
        assert embeddings.shape == expected_shape, f"Shape mismatch: {embeddings.shape} vs {expected_shape}"
        
        print("✓ KAN-MAMMOTE integration test passed!")
        
    except Exception as e:
        print(f"✗ KAN-MAMMOTE integration test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=" * 50)
    print("DyGMamba-style Continuous Mamba Test Suite")
    print("=" * 50)
    
    # Run all tests
    test_time_encoder()
    test_continuous_mamba()
    test_integration_with_kan_mammote()
    
    print("\n" + "=" * 50)
    print("Test suite completed!")
    print("=" * 50)
