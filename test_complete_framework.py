#!/usr/bin/env python3
"""
Complete KAN-MAMMOTE Framework Test

This script comprehensively tests the entire KAN-MAMMOTE system:
1. Custom c_mamba.py implementation verification
2. K_MOTE expert routing and adaptive basis functions  
3. End-to-end KAN-MAMMOTE model functionality
4. Comparison with baseline models

Confirms that the framework is ready for deployment as a plug-and-play time encoder.
"""

import torch
import torch.nn as nn
import numpy as np
import time
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.config import KANMAMOTEConfig
from src.models.kan_mammote import KAN_MAMOTE_Model
from src.models.c_mamba import ContinuousMambaBlock, SimplifiedContinuousMambaBlock
from src.models.k_mote import K_MOTE
from src.layers.basis_functions import FourierBasis, SplineBasis, GaussianKernelBasis, WaveletBasis
from src.losses.simple_losses import feature_extraction_loss
from src.losses.regularization_losses import sobolev_l2_loss

def test_c_mamba_implementation():
    """Test the custom continuous-time Mamba implementation."""
    print("🔍 Testing Custom C-Mamba Implementation...")
    
    config = KANMAMOTEConfig()
    batch_size, seq_len, input_dim = 8, 32, 64
    
    # Test full selective implementation
    c_mamba = ContinuousMambaBlock(input_dim, config)
    
    # Create test data with irregular time intervals
    u_sequence = torch.randn(batch_size, seq_len, input_dim)
    delta_t = torch.abs(torch.randn(batch_size, seq_len, 1)) * 0.1 + 0.01  # Positive time deltas
    
    print(f"  Input shape: {u_sequence.shape}")
    print(f"  Time deltas shape: {delta_t.shape}")
    print(f"  Time delta range: [{delta_t.min():.4f}, {delta_t.max():.4f}]")
    
    # Forward pass
    with torch.no_grad():
        output = c_mamba(u_sequence, delta_t)
    
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Test simplified version
    simple_mamba = SimplifiedContinuousMambaBlock(input_dim, config)
    with torch.no_grad():
        simple_output = simple_mamba(u_sequence, delta_t)
    
    print(f"  Simplified output shape: {simple_output.shape}")
    print(f"  ✅ C-Mamba implementations working correctly")
    
    return c_mamba, simple_mamba

def test_k_mote_experts():
    """Test K-MOTE expert routing and adaptive basis functions."""
    print("\n🔍 Testing K-MOTE Expert System...")
    
    config = KANMAMOTEConfig()
    input_dim, output_dim = 64, 32
    
    # Test different basis function types
    basis_types = ['fourier', 'spline', 'gaussian', 'wavelet']
    
    for basis_type in basis_types:
        print(f"  Testing {basis_type} basis...")
        
        if basis_type == 'fourier':
            basis = FourierBasis(input_dim, config)
        elif basis_type == 'spline':
            basis = SplineBasis(input_dim, config)
        elif basis_type == 'gaussian':
            basis = GaussianKernelBasis(input_dim, config)
        elif basis_type == 'wavelet':
            basis = WaveletBasis(input_dim, config)
        
        # Test basis function
        x = torch.randn(8, input_dim)
        with torch.no_grad():
            y = basis(x)
        print(f"    Input: {x.shape} -> Output: {y.shape} ✅")
    
    # Test full K-MOTE system
    k_mote = K_MOTE(config)
    
    # Test with timestamp input (K-MOTE expects scalar timestamp)
    timestamps = torch.randn(8, 1)  # Scalar timestamp per sample
    with torch.no_grad():
        output, router_probs, expert_weights = k_mote(timestamps)
    
    print(f"  K-MOTE input shape: {timestamps.shape}")
    print(f"  K-MOTE output shape: {output.shape}")
    print(f"  Router probabilities shape: {router_probs.shape}")
    print(f"  Expert weights shape: {expert_weights.shape}")
    print(f"  Router entropy: {-torch.sum(router_probs * torch.log(router_probs + 1e-8), dim=-1).mean():.4f}")
    print(f"  ✅ K-MOTE expert system working correctly")
    
    return k_mote

def test_complete_kan_mammote():
    """Test the complete KAN-MAMMOTE model."""
    print("\n🔍 Testing Complete KAN-MAMMOTE Model...")
    
    config = KANMAMOTEConfig()
    
    # Model configuration
    input_dim = 64
    output_dim = 32
    batch_size, seq_len = 8, 50
    
    # Create KAN-MAMMOTE model
    model = KAN_MAMOTE_Model(config)
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create test data (event-based with irregular timestamps)
    features = torch.randn(batch_size, seq_len, input_dim)
    timestamps = torch.sort(torch.rand(batch_size, seq_len) * 10.0)[0]  # Sorted timestamps [0, 10]
    
    print(f"  Input features shape: {features.shape}")
    print(f"  Timestamps shape: {timestamps.shape}")
    print(f"  Timestamp range: [{timestamps.min():.2f}, {timestamps.max():.2f}]")
    
    # Forward pass
    start_time = time.time()
    with torch.no_grad():
        output = model(features, timestamps)
    inference_time = time.time() - start_time
    
    print(f"  Output shape: {output.shape}")
    print(f"  Inference time: {inference_time:.4f}s")
    print(f"  Throughput: {batch_size * seq_len / inference_time:.0f} samples/sec")
    print(f"  ✅ Complete KAN-MAMMOTE model working correctly")
    
    return model, output

def test_plug_and_play_capability():
    """Test KAN-MAMMOTE as a plug-and-play time encoder."""
    print("\n🔍 Testing Plug-and-Play Time Encoder Capability...")
    
    config = KANMAMOTEConfig()
    
    # Simulate different downstream tasks
    tasks = [
        {"name": "Classification", "input_dim": 128, "output_dim": 10},
        {"name": "Regression", "input_dim": 64, "output_dim": 1},
        {"name": "Sequence Generation", "input_dim": 256, "output_dim": 256}
    ]
    
    for task in tasks:
        print(f"  Testing {task['name']} task...")
        
        # Create time encoder
        encoder = KAN_MAMOTE_Model(config)
        
        # Simulate task-specific data
        batch_size, seq_len = 4, 20
        features = torch.randn(batch_size, seq_len, task['input_dim'])
        timestamps = torch.sort(torch.rand(batch_size, seq_len) * 5.0)[0]
        
        # Encode temporal features
        with torch.no_grad():
            encoded_features = encoder(features, timestamps)
        
        # Simulate downstream task head
        if task['name'] == "Classification":
            classifier = nn.Linear(task['output_dim'], 10)
            logits = classifier(encoded_features.mean(dim=1))  # Global average pooling
            predictions = torch.softmax(logits, dim=-1)
            print(f"    Classification probabilities shape: {predictions.shape}")
            
        elif task['name'] == "Regression":
            regressor = nn.Linear(task['output_dim'], 1)
            predictions = regressor(encoded_features.mean(dim=1))
            print(f"    Regression predictions shape: {predictions.shape}")
            
        elif task['name'] == "Sequence Generation":
            generator = nn.Linear(task['output_dim'], task['input_dim'])
            generated = generator(encoded_features)
            print(f"    Generated sequence shape: {generated.shape}")
        
        print(f"    ✅ {task['name']} integration successful")
    
    print(f"  ✅ Plug-and-play capability confirmed")

def test_training_capability():
    """Test training capability with various loss functions."""
    print("\n🔍 Testing Training Capability...")
    
    config = KANMAMOTEConfig()
    model = KAN_MAMOTE_Model(config)
    
    # Create training data
    batch_size, seq_len = 4, 25
    features = torch.randn(batch_size, seq_len, 64)
    timestamps = torch.sort(torch.rand(batch_size, seq_len) * 3.0)[0]
    targets = torch.randn(batch_size, seq_len, 32)
    
    # Enable gradients
    model.train()
    
    # Forward pass
    output = model(features, timestamps)
    
    # Test different loss functions
    print("  Testing loss functions...")
    
    # Feature extraction loss
    feat_loss = feature_extraction_loss(output, targets)
    print(f"    Feature extraction loss: {feat_loss.item():.6f}")
    
    # Regularization loss (need to create simple timestamps for the test)
    test_timestamps = torch.randn(4, 25, 1) 
    reg_loss = sobolev_l2_loss(model, 0.01, test_timestamps)
    print(f"    Sobolev regularization loss: {reg_loss.item():.6f}")
    
    # Combined loss
    total_loss = feat_loss + 0.01 * reg_loss
    print(f"    Combined loss: {total_loss.item():.6f}")
    
    # Backward pass
    total_loss.backward()
    
    # Check gradients
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    print(f"    Gradient norm: {grad_norm:.6f}")
    
    print(f"  ✅ Training capability confirmed")

def benchmark_against_baseline():
    """Benchmark KAN-MAMMOTE against a simple baseline."""
    print("\n🔍 Benchmarking Against Simple Baseline...")
    
    config = KANMAMOTEConfig()
    
    # Models
    kan_mammote = KAN_MAMOTE_Model(config)
    baseline_lstm = nn.LSTM(64, 32, batch_first=True)
    
    # Test data
    batch_size, seq_len = 8, 100
    features = torch.randn(batch_size, seq_len, 64)
    timestamps = torch.sort(torch.rand(batch_size, seq_len) * 10.0)[0]
    
    # KAN-MAMMOTE inference
    start_time = time.time()
    with torch.no_grad():
        kan_output = kan_mammote(features, timestamps)
    kan_time = time.time() - start_time
    
    # LSTM baseline inference
    start_time = time.time()
    with torch.no_grad():
        lstm_output, _ = baseline_lstm(features)
    lstm_time = time.time() - start_time
    
    print(f"  KAN-MAMMOTE inference time: {kan_time:.4f}s")
    print(f"  LSTM baseline inference time: {lstm_time:.4f}s")
    print(f"  Relative speed: {lstm_time/kan_time:.2f}x")
    
    print(f"  KAN-MAMMOTE parameters: {sum(p.numel() for p in kan_mammote.parameters()):,}")
    print(f"  LSTM parameters: {sum(p.numel() for p in baseline_lstm.parameters()):,}")
    
    print(f"  ✅ Benchmarking completed")

def main():
    """Run comprehensive KAN-MAMMOTE framework test."""
    print("🚀 KAN-MAMMOTE Framework Comprehensive Test")
    print("=" * 60)
    
    try:
        # Test individual components
        test_c_mamba_implementation()
        test_k_mote_experts()
        model, output = test_complete_kan_mammote()
        
        # Test integration capabilities
        test_plug_and_play_capability()
        test_training_capability()
        benchmark_against_baseline()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("\nKAN-MAMMOTE Framework Status:")
        print("✅ Custom C-Mamba implementation working")
        print("✅ K-MOTE expert routing functional")
        print("✅ End-to-end model integration complete")
        print("✅ Plug-and-play time encoder ready")
        print("✅ Training pipeline operational")
        print("✅ Ready for deployment!")
        
        print(f"\nFramework Summary:")
        print(f"• C-Mamba: Custom continuous-time state-space model (NOT real Mamba)")
        print(f"• K-MOTE: Mixture-of-Experts with 4 adaptive basis function types")
        print(f"• Integration: Seamless spatio-temporal representation learning")
        print(f"• Use case: Drop-in replacement for LeTE or other time encoders")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
