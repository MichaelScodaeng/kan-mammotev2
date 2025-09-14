#!/usr/bin/env python3
# file: test_kan_mammote.py

import torch
import torch.nn as nn
import traceback

def test_kan_mammote():
    """Test the KAN-MAMMOTE framework comprehensively"""
    print("=== Testing KAN-MAMMOTE Framework ===\n")
    
    try:
        # Test 1: Import test
        print("[Test 1/5] Testing imports...")
        from models.time_encoders.kan_mammote import KAN_MAMMOTE
        print("✓ KAN-MAMMOTE import successful")
        
        # Test 2: Basic instantiation
        print("\n[Test 2/5] Testing instantiation...")
        embedding_dim = 64
        expert_dim = 32
        num_mixtures = 16
        
        kan_mammote = KAN_MAMMOTE(
            embedding_dim=embedding_dim,
            expert_dim=expert_dim,
            num_mixtures=num_mixtures
        )
        print("✓ KAN-MAMMOTE instantiation successful")
        
        # Test 3: Forward pass with dummy data
        print("\n[Test 3/5] Testing forward pass...")
        batch_size = 4
        seq_len = 10
        
        # Create dummy input tensors
        t_abs = torch.randn(batch_size, seq_len, 1)  # Absolute timestamps
        t_rel = torch.rand(batch_size, seq_len, 1) + 0.1  # Relative time differences (positive)
        
        output = kan_mammote(t_abs, t_rel)
        expected_shape = (batch_size, seq_len, embedding_dim)
        
        assert output.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {output.shape}"
        print(f"✓ Forward pass successful, output shape: {output.shape}")
        
        # Test 4: Component accessibility
        print("\n[Test 4/5] Testing component accessibility...")
        
        # Test K-MOTE component
        k_mote_output = kan_mammote.k_mote(t_abs)
        expected_k_mote_shape = (batch_size, seq_len, expert_dim)
        assert k_mote_output.shape == expected_k_mote_shape, f"K-MOTE shape mismatch! Expected {expected_k_mote_shape}, got {k_mote_output.shape}"
        print(f"✓ K-MOTE component working, output shape: {k_mote_output.shape}")
        
        # Test SM-Kernel component
        sm_kernel_output = kan_mammote.sm_kernel(t_rel)
        expected_sm_kernel_shape = (batch_size, seq_len, num_mixtures)
        assert sm_kernel_output.shape == expected_sm_kernel_shape, f"SM-Kernel shape mismatch! Expected {expected_sm_kernel_shape}, got {sm_kernel_output.shape}"
        print(f"✓ SM-Kernel component working, output shape: {sm_kernel_output.shape}")
        
        # Test 5: Gradient flow
        print("\n[Test 5/5] Testing gradient flow...")
        
        # Create a simple loss and check if gradients flow through
        target = torch.randn_like(output)
        loss_fn = nn.MSELoss()
        loss = loss_fn(output, target)
        
        # Check gradients before backward
        initial_k_mote_grad = None
        for param in kan_mammote.k_mote.parameters():
            if param.requires_grad:
                initial_k_mote_grad = param.grad
                break
        
        loss.backward()
        
        # Check if gradients exist after backward
        gradients_exist = False
        for param in kan_mammote.parameters():
            if param.grad is not None:
                gradients_exist = True
                break
        
        assert gradients_exist, "No gradients found after backward pass!"
        print(f"✓ Gradient flow successful, loss: {loss.item():.6f}")
        
        print("\n=== All KAN-MAMMOTE tests passed! ===")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False

def test_parameter_counts():
    """Test parameter counts of different configurations"""
    print("\n=== Testing Parameter Counts ===")
    
    try:
        configs = [
            {"embedding_dim": 32, "expert_dim": 16, "num_mixtures": 8},
            {"embedding_dim": 64, "expert_dim": 32, "num_mixtures": 16},
            {"embedding_dim": 128, "expert_dim": 64, "num_mixtures": 32},
        ]
        
        for i, config in enumerate(configs, 1):
            print(f"\n[Config {i}] {config}")
            from models.time_encoders.kan_mammote import KAN_MAMMOTE
            
            model = KAN_MAMMOTE(**config)
            
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            
        print("\n✓ Parameter count tests completed")
        
    except Exception as e:
        print(f"\n✗ Parameter count test failed: {e}")
        traceback.print_exc()

def test_initialization():
    """Test SM-Kernel data-driven initialization"""
    print("\n=== Testing SM-Kernel Initialization ===")
    
    try:
        from models.time_encoders.kan_mammote import KAN_MAMMOTE
        
        kan_mammote = KAN_MAMMOTE(embedding_dim=64, expert_dim=32, num_mixtures=16)
        
        # Create synthetic data for initialization
        batch_size = 2
        seq_len = 100
        
        # Create delta_t with some periodic pattern
        time = torch.linspace(0, 10, seq_len)
        periodic_signal = torch.sin(2 * torch.pi * 0.1 * time)  # 0.1 Hz
        delta_t_sample = (periodic_signal.abs() + 0.5).repeat(batch_size, 1).unsqueeze(-1)
        
        print("Initializing SM-Kernel from synthetic data...")
        kan_mammote.initialize_sm_kernel(delta_t_sample)
        print("✓ SM-Kernel initialization completed")
        
    except Exception as e:
        print(f"\n✗ Initialization test failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    import torch
    if torch._C._GLIBCXX_USE_CXX11_ABI:
        print("Install the cxx11abi TRUE variant")
    else:
        print("Install the cxx11abi FALSE variant")
    import sys, torch
    print("Python:", sys.version.split()[0])        # e.g., 3.11.x  → cp311
    print("Torch:", torch.__version__)              # e.g., 2.5.1   → torch2.5
    print("Torch CUDA:", torch.version.cuda)        # e.g., 12.1    → cu12 / cu121
    print("ABI:", "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE")
    success = test_kan_mammote()
    
    if success:
        test_parameter_counts()
        test_initialization()
        print("\n🎉 All KAN-MAMMOTE tests completed successfully!")
    else:
        print("\n❌ KAN-MAMMOTE tests failed. Please check the error messages above.")
