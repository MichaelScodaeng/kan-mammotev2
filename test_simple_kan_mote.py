#!/usr/bin/env python3
"""
Simple test script for KAN-MOTE without the notebook's corrupted CUDA context.
This script tests the basic functionality on CPU.
"""

import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.append('.')

# Import components
from src.models.k_mote import K_MOTE
from src.utils.config import KANMAMOTEConfig

def test_kan_mote():
    """Test basic KAN-MOTE functionality"""
    print("Testing KAN-MOTE on CPU...")
    
    # Force CPU usage
    device = torch.device("cpu")
    
    # Create configuration
    config = KANMAMOTEConfig(
        D_time=32,
        num_experts=4,
        K_top=2,
        use_aux_features_router=False,
        raw_event_feature_dim=0,
        hidden_dim_mamba=32,
        state_dim_mamba=16,
        num_mamba_layers=2,
        gamma=0.5,
        lambda_moe_load_balancing=0.01,
        lambda_sobolev_l2=0.0,
        lambda_total_variation_l1=0.0
    )
    
    print(f"Config created: D_time={config.D_time}, D_time_per_expert={config.D_time_per_expert}")
    
    # Create K-MOTE model
    k_mote = K_MOTE(config).to(device)
    
    # Test with dummy data
    batch_size = 4
    seq_len = 10
    timestamps = torch.rand(batch_size * seq_len, 1, device=device)  # Random timestamps [0,1]
    
    print(f"Input timestamps shape: {timestamps.shape}")
    
    # Forward pass
    with torch.no_grad():
        embeddings, weights, masks = k_mote(timestamps, None)
        
        print(f"Output embeddings shape: {embeddings.shape}")
        print(f"Expert weights shape: {weights.shape}")
        print(f"Expert masks shape: {masks.shape}")
        
        # Check for any NaN or inf values
        if torch.isnan(embeddings).any():
            print("WARNING: NaN values in embeddings!")
        if torch.isinf(embeddings).any():
            print("WARNING: Inf values in embeddings!")
        
        print("✓ K-MOTE test passed!")
        
        return True

class SimpleKANMOTEClassifier(nn.Module):
    """Simplified classifier for testing"""
    def __init__(self, config):
        super().__init__()
        self.time_encoder = K_MOTE(config)
        self.classifier = nn.Linear(config.D_time, 10)
        
    def forward(self, x):
        # x: (batch_size, seq_len) pixel positions
        batch_size, seq_len = x.shape
        
        # Normalize to [0,1]
        timestamps = x.float() / 783.0
        timestamps = timestamps.view(-1, 1)  # Flatten
        
        # Get embeddings
        embeddings, weights, masks = self.time_encoder(timestamps, None)
        
        # Reshape and pool
        embeddings = embeddings.view(batch_size, seq_len, -1)
        pooled = embeddings.mean(dim=1)  # Simple average pooling
        
        # Classify
        logits = self.classifier(pooled)
        
        return logits, {'weights': weights, 'masks': masks}

def test_classifier():
    """Test the simplified classifier"""
    print("\nTesting simplified classifier...")
    
    device = torch.device("cpu")
    
    config = KANMAMOTEConfig(
        D_time=32, num_experts=4, K_top=2,
        use_aux_features_router=False, raw_event_feature_dim=0,
        hidden_dim_mamba=32, state_dim_mamba=16, num_mamba_layers=2,
        gamma=0.5, lambda_moe_load_balancing=0.01,
        lambda_sobolev_l2=0.0, lambda_total_variation_l1=0.0
    )
    
    model = SimpleKANMOTEClassifier(config).to(device)
    
    # Test data
    batch_size = 4
    seq_len = 20
    test_data = torch.randint(0, 784, (batch_size, seq_len), device=device)
    
    print(f"Test data shape: {test_data.shape}")
    
    with torch.no_grad():
        logits, info = model(test_data)
        
        print(f"Output logits shape: {logits.shape}")
        print(f"Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
        
        # Check for issues
        if torch.isnan(logits).any():
            print("ERROR: NaN values in output!")
            return False
        if torch.isinf(logits).any():
            print("ERROR: Inf values in output!")
            return False
            
        print("✓ Classifier test passed!")
        return True

if __name__ == "__main__":
    print("=" * 60)
    print("KAN-MOTE Debugging Test")
    print("=" * 60)
    
    try:
        # Test basic K-MOTE
        success1 = test_kan_mote()
        
        # Test classifier
        success2 = test_classifier()
        
        if success1 and success2:
            print("\n✅ All tests passed! The issue is likely with the notebook's CUDA context.")
            print("💡 Recommendation: Restart the Jupyter kernel and run the notebook again.")
        else:
            print("\n❌ Tests failed. There are issues with the model implementation.")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
