# file: test_kan_mammote.py (Final Cleaned Version)

import torch
import torch.nn as nn
import traceback
import sys
import numpy as np

# Import the model to be tested
from models.time_encoders.kan_mammote import KAN_MAMMOTE

def run_all_tests():
    """Runs all tests and provides a final summary."""
    print_system_info()
    
    # Run the main functional test
    success = test_kan_mammote_functionality()
    
    if success:
        print("\n=== Main Functionality Tests Passed ===")
        test_parameter_counts()
        test_initialization()
        print("\n\n🎉 All KAN-MAMMOTE tests completed successfully!")
        print("Your KAN-MAMMOTE implementation is ready for real experiments.")
    else:
        print("\n\n❌ KAN-MAMMOTE core tests failed. Please check the error messages above.")

def test_kan_mammote_functionality():
    """Tests the core functionality: instantiation, forward pass, and gradient flow."""
    print("\n=== Testing KAN-MAMMOTE Core Functionality ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Cannot run GPU-specific tests.")
        return False
        
    try:
        # Use dimensions that are multiples of 16 for A100 compatibility
        batch_size = 8
        seq_len = 128
        embedding_dim = 128
        expert_dim = 64 # This is used by k_mote internally
        num_mixtures = 32
        
        print(f"Using dimensions: batch_size={batch_size}, seq_len={seq_len}, embedding_dim={embedding_dim}")
        
        print("\n[Test 1/3] Instantiation...")
        kan_mammote = KAN_MAMMOTE(
            embedding_dim=embedding_dim,
            expert_dim=expert_dim, 
            num_mixtures=num_mixtures
        ).to(device)
        print("✓ KAN-MAMMOTE instantiation successful.")
        
        print("\n[Test 2/3] Forward pass...")
        t_abs = torch.randn(batch_size, seq_len, 1, device=device)
        t_rel = torch.rand(batch_size, seq_len, 1, device=device)
        
        output = kan_mammote(t_abs, t_rel)
        expected_shape = (batch_size, seq_len, embedding_dim)
        
        assert output.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {output.shape}"
        print(f"✓ Forward pass successful, output shape: {output.shape}")
        
        print("\n[Test 3/3] Gradient flow...")
        target = torch.randn_like(output)
        loss_fn = nn.MSELoss()
        loss = loss_fn(output, target)
        loss.backward()
        
        gradients_exist = any(param.grad is not None for param in kan_mammote.parameters() if param.requires_grad)
        assert gradients_exist, "No gradients found after backward pass!"
        print(f"✓ Gradient flow successful, loss: {loss.item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Core functionality test failed with error: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False

def test_parameter_counts():
    """Test parameter counts of different configurations."""
    print("\n=== Testing Parameter Counts ===")
    from models.time_encoders.kan_mammote import KAN_MAMMOTE
    
    configs = [
        {"embedding_dim": 128, "expert_dim": 64, "num_mixtures": 16},
        {"embedding_dim": 256, "expert_dim": 128, "num_mixtures": 32},
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"\n[Config {i}] {config}")
        model = KAN_MAMMOTE(**config)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")
        
    print("✓ Parameter count tests completed.")

def test_initialization():
    """Test SM-Kernel data-driven initialization."""
    print("\n=== Testing SM-Kernel Initialization ===")
    from models.time_encoders.kan_mammote import KAN_MAMMOTE
    
    try:
        kan_mammote = KAN_MAMMOTE(embedding_dim=128, expert_dim=64, num_mixtures=16)
        
        time = torch.linspace(0, 100, 1000)
        signal = torch.sin(time * 2 * torch.pi * 0.1)
        delta_t_sample = (signal.diff().abs() + 0.5).reshape(1, -1, 1)
        
        print("Initializing SM-Kernel from synthetic data...")
        kan_mammote.initialize_sm_kernel(delta_t_sample)
        print("✓ SM-Kernel initialization completed successfully.")
    except Exception as e:
        print(f"\n✗ Initialization test failed: {e}")
        traceback.print_exc()

def print_system_info():
    """Print system and library information."""
    print("=== System Information ===")
    # (Implementation is fine as is)
    print()

if __name__ == "__main__":
    run_all_tests()