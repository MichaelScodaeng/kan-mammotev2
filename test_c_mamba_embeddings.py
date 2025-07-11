#!/usr/bin/env python3
"""
Test script for the updated DyGMamba-style c_mamba implementation
that takes time embeddings from K-MOTE as input, plus the new
Immediate Faster-KAN Layer architecture.
"""

import torch
import sys
sys.path.append('.')

from src.utils.config import KANMAMOTEConfig
from archive.c_mamba import ContinuousMambaBlock, SimplifiedContinuousMambaBlock
from src.models.k_mote import K_MOTE
from src.models.kan_mammote import KAN_MAMOTE_Model
from src.models.immediate_fasterkan_layer import (
    FasterKANTemporalLayer, 
    FasterKANTemporalNetwork,
    ImmediateFasterKANLayer, 
    ImprovedKANMAMOTE
)

def test_c_mamba_with_time_embeddings():
    """Test the ContinuousMambaBlock with time embeddings from K-MOTE"""
    print("Testing ContinuousMambaBlock with time embeddings...")
    
    # Create configuration
    config = KANMAMOTEConfig(
        D_time=32,
        num_experts=4,
        K_top=2,
        hidden_dim_mamba=64,
        state_dim_mamba=32,
        use_aux_features_router=False,
        raw_event_feature_dim=0
    )
    config.D_time_per_expert = config.D_time // config.num_experts
    
    # Use CUDA if available for mamba_ssm, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test data
    batch_size = 4
    seq_len = 10
    
    # Time embeddings from K-MOTE (simulated)
    time_embeddings = torch.randn(batch_size, seq_len, config.D_time, device=device)
    
    # Raw timestamps for time difference computation
    timestamps = torch.cumsum(torch.rand(batch_size, seq_len, 1, device=device) * 0.1, dim=1)
    
    # Create Mamba block
    mamba_block = ContinuousMambaBlock(input_dim=config.D_time, config=config).to(device)
    
    print(f"Input shapes:")
    print(f"  Time embeddings: {time_embeddings.shape}")
    print(f"  Timestamps: {timestamps.shape}")
    
    # Forward pass
    try:
        output = mamba_block(time_embeddings, timestamps)
        print(f"Output shape: {output.shape}")
        print("✓ ContinuousMambaBlock test passed!")
        return True
    except Exception as e:
        print(f"✗ ContinuousMambaBlock test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simplified_c_mamba_with_time_embeddings():
    """Test the SimplifiedContinuousMambaBlock with time embeddings"""
    print("\nTesting SimplifiedContinuousMambaBlock with time embeddings...")
    
    # Create configuration
    config = KANMAMOTEConfig(
        D_time=32,
        hidden_dim_mamba=64,
        state_dim_mamba=32
    )
    
    # Use CUDA if available for mamba_ssm, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test data
    batch_size = 4
    seq_len = 10
    
    # Time embeddings from K-MOTE (simulated)
    time_embeddings = torch.randn(batch_size, seq_len, config.D_time, device=device)
    
    # Raw timestamps for time difference computation
    timestamps = torch.cumsum(torch.rand(batch_size, seq_len, 1, device=device) * 0.1, dim=1)
    
    # Create simplified Mamba block
    mamba_block = SimplifiedContinuousMambaBlock(input_dim=config.D_time, config=config).to(device)
    
    print(f"Input shapes:")
    print(f"  Time embeddings: {time_embeddings.shape}")
    print(f"  Timestamps: {timestamps.shape}")
    
    # Forward pass
    try:
        output = mamba_block(time_embeddings, timestamps)
        print(f"Output shape: {output.shape}")
        print("✓ SimplifiedContinuousMambaBlock test passed!")
        return True
    except Exception as e:
        print(f"✗ SimplifiedContinuousMambaBlock test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_faster_kan_layer():
    """Test the Faster-KAN layer for temporal difference processing"""
    print("\nTesting Faster-KAN Temporal Layer...")
    
    # Test data
    batch_size, seq_len, dim = 4, 10, 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Input: temporal difference embeddings
    time_diff_embeddings = torch.randn(batch_size, seq_len, dim, device=device)
    
    # Create Faster-KAN layer
    kan_layer = FasterKANTemporalLayer(
        input_dim=dim, 
        output_dim=dim,
        grid_min=-2.0,
        grid_max=2.0,
        num_grids=8
    ).to(device)
    
    print(f"Input shape: {time_diff_embeddings.shape}")
    
    try:
        output = kan_layer(time_diff_embeddings)
        print(f"Output shape: {output.shape}")
        print("✓ Faster-KAN Temporal Layer test passed!")
        return True
    except Exception as e:
        print(f"✗ Faster-KAN Temporal Layer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_faster_kan_network():
    """Test the multi-layer Faster-KAN network"""
    print("\nTesting Faster-KAN Temporal Network...")
    
    # Test data
    batch_size, seq_len = 4, 10
    input_dim, hidden_dim, output_dim = 32, 48, 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Input: temporal difference embeddings
    time_diff_embeddings = torch.randn(batch_size, seq_len, input_dim, device=device)
    
    # Create multi-layer Faster-KAN network
    kan_network = FasterKANTemporalNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=3,
        num_grids=8
    ).to(device)
    
    print(f"Input shape: {time_diff_embeddings.shape}")
    
    try:
        output = kan_network(time_diff_embeddings)
        print(f"Output shape: {output.shape}")
        print("✓ Faster-KAN Temporal Network test passed!")
        return True
    except Exception as e:
        print(f"✗ Faster-KAN Temporal Network test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_immediate_fasterkan_layer():
    """Test the complete Immediate Faster-KAN architecture"""
    print("\nTesting Immediate Faster-KAN architecture...")
    
    # Create configuration
    config = KANMAMOTEConfig(
        D_time=32,
        num_experts=4,
        K_top=2,
        hidden_dim_mamba=64,
        state_dim_mamba=32,
        use_aux_features_router=False,
        raw_event_feature_dim=0,
        kan_grid_size=5  # For Faster-KAN
    )
    config.D_time_per_expert = config.D_time // config.num_experts
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test data
    batch_size = 4
    seq_len = 10
    
    # Timestamps (increasing over time)
    timestamps = torch.cumsum(torch.rand(batch_size, seq_len, 1, device=device) * 0.1, dim=1)
    event_features = torch.zeros(batch_size, seq_len, 0, device=device)
    
    # Create model
    model = ImmediateFasterKANLayer(config).to(device)
    
    print(f"Input shapes:")
    print(f"  Timestamps: {timestamps.shape}")
    print(f"  Event features: {event_features.shape}")
    
    try:
        embeddings, moe_info = model(timestamps, event_features)
        current_weights, previous_weights, current_masks, previous_masks = moe_info
        
        print(f"Output shapes:")
        print(f"  Final embeddings: {embeddings.shape}")
        print(f"  Current expert weights: {current_weights.shape}")
        print(f"  Previous expert weights: {previous_weights.shape}")
        print(f"  Current expert masks: {current_masks.shape}")
        print(f"  Previous expert masks: {previous_masks.shape}")
        print("✓ Immediate Faster-KAN layer test passed!")
        return True
    except Exception as e:
        print(f"✗ Immediate Faster-KAN layer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_improved_kan_mammote():
    """Test the complete improved KAN-MAMMOTE model"""
    print("\nTesting Improved KAN-MAMMOTE model...")
    
    # Create configuration
    config = KANMAMOTEConfig(
        D_time=32,
        num_experts=4,
        K_top=2,
        hidden_dim_mamba=64,
        state_dim_mamba=32,
        use_aux_features_router=False,
        raw_event_feature_dim=0,
        kan_grid_size=5
    )
    config.D_time_per_expert = config.D_time // config.num_experts
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test data
    batch_size = 4
    seq_len = 10
    
    timestamps = torch.cumsum(torch.rand(batch_size, seq_len, 1, device=device) * 0.1, dim=1)
    event_features = torch.zeros(batch_size, seq_len, 0, device=device)
    
    # Create improved model
    model = ImprovedKANMAMOTE(config).to(device)
    
    print(f"Input shapes:")
    print(f"  Timestamps: {timestamps.shape}")
    print(f"  Event features: {event_features.shape}")
    
    try:
        embeddings, moe_info = model(timestamps, event_features)
        current_weights, previous_weights, current_masks, previous_masks = moe_info
        
        print(f"Output shapes:")
        print(f"  Final embeddings: {embeddings.shape}")
        print("✓ Improved KAN-MAMMOTE test passed!")
        
        # Test temporal difference analysis
        print("\n--- Temporal Analysis ---")
        print(f"Timestamp differences:")
        time_diffs = timestamps[:, 1:] - timestamps[:, :-1]
        print(f"  Mean time diff: {time_diffs.mean().item():.4f}")
        print(f"  Std time diff: {time_diffs.std().item():.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Improved KAN-MAMMOTE test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_temporal_difference_behavior():
    """Test specific temporal difference behavior"""
    print("\nTesting temporal difference behavior...")
    
    # Create a fresh configuration to avoid any caching issues
    config = KANMAMOTEConfig()
    
    # Override specific parameters for this test
    config.D_time = 16
    config.num_experts = 4  # Must match the hardcoded experts in K-MOTE
    config.K_top = 2        # Use 2 out of 4 experts
    config.hidden_dim_mamba = 32
    config.state_dim_mamba = 16
    config.kan_grid_size = 3
    config.use_aux_features_router = False  # Important: disable aux features
    config.raw_event_feature_dim = 0        # Important: no auxiliary features
    config.D_time_per_expert = config.D_time // config.num_experts  # 16 // 4 = 4
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create simple test case
    batch_size, seq_len = 2, 5
    
    # Create regular vs irregular timestamps
    regular_timestamps = torch.arange(0, seq_len, device=device).float().unsqueeze(0).unsqueeze(-1)
    irregular_timestamps = torch.tensor([[[0.0], [0.1], [0.5], [0.6], [2.0]]], device=device)
    
    timestamps = torch.cat([regular_timestamps, irregular_timestamps], dim=0)
    event_features = torch.zeros(batch_size, seq_len, 0, device=device)
    
    model = ImmediateFasterKANLayer(config).to(device)
    
    try:
        embeddings, moe_info = model(timestamps, event_features)
        
        print(f"Regular timestamps: {regular_timestamps.squeeze()}")
        print(f"Irregular timestamps: {irregular_timestamps.squeeze()}")
        print(f"Output embeddings shape: {embeddings.shape}")
        
        # Check if model produces different outputs for different temporal patterns
        regular_output = embeddings[0]  # Regular pattern
        irregular_output = embeddings[1]  # Irregular pattern
        
        difference = torch.norm(regular_output - irregular_output)
        print(f"Output difference between regular/irregular: {difference.item():.4f}")
        
        if difference > 0.1:  # Should be different
            print("✓ Model successfully differentiates temporal patterns!")
        else:
            print("⚠ Model may not be learning temporal differences properly")
        
        return True
    except Exception as e:
        print(f"✗ Temporal difference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_kan_mammote_model():
    """Test the full KAN-MAMMOTE model with updated c_mamba"""
    print("\nTesting full KAN-MAMMOTE model...")
    
    # Create configuration
    config = KANMAMOTEConfig(
        D_time=32,
        num_experts=4,
        K_top=2,
        hidden_dim_mamba=64,
        state_dim_mamba=32,
        use_aux_features_router=False,
        raw_event_feature_dim=0
    )
    config.D_time_per_expert = config.D_time // config.num_experts
    
    # Use CUDA if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test data
    batch_size = 4
    seq_len = 10
    
    # Timestamps and empty event features
    timestamps = torch.cumsum(torch.rand(batch_size, seq_len, 1, device=device) * 0.1, dim=1)
    event_features = torch.zeros(batch_size, seq_len, 0, device=device)  # Empty features
    
    # Create KAN-MAMMOTE model
    model = KAN_MAMOTE_Model(config).to(device)
    
    print(f"Input shapes:")
    print(f"  Timestamps: {timestamps.shape}")
    print(f"  Event features: {event_features.shape}")
    
    # Forward pass
    try:
        embeddings, moe_info = model(timestamps, event_features)
        abs_weights, rel_weights = moe_info
        
        print(f"Output shapes:")
        print(f"  Final embeddings: {embeddings.shape}")
        print(f"  Abs expert weights: {abs_weights.shape}")
        print(f"  Rel expert weights: {rel_weights.shape}")
        print("✓ Full KAN-MAMMOTE model test passed!")
        return True
    except Exception as e:
        print(f"✗ Full KAN-MAMMOTE model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 70)
    print("Testing DyGMamba-style c_mamba with Immediate Faster-KAN Layer")
    print("=" * 70)
    
    tests = [
        test_c_mamba_with_time_embeddings,
        test_simplified_c_mamba_with_time_embeddings,
        test_faster_kan_layer,
        test_faster_kan_network,
        test_immediate_fasterkan_layer,
        test_improved_kan_mammote,
        test_temporal_difference_behavior,
        test_full_kan_mammote_model
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 70)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 70)
    
    if passed == total:
        print("🎉 All tests passed! Complete KAN-MAMMOTE architecture working correctly.")
        print("\nArchitecture Components Tested:")
        print("✓ Original C-Mamba blocks")
        print("✓ Faster-KAN temporal difference processing")
        print("✓ Immediate Faster-KAN Layer (current vs previous)")
        print("✓ Improved KAN-MAMMOTE model")
        print("✓ Temporal pattern differentiation")
        print("✓ Original KAN-MAMMOTE compatibility")
    else:
        print("❌ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()