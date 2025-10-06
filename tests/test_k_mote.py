# file: test_k_mote.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import sys
import traceback

# Import the modules to be tested
# This assumes the script is run from the project's root directory
from models.time_encoders.k_mote import SplineKANLayer, FourierKANLayer, WaveletKANLayer, KMOTE

def run_all_tests():
    """Runs all individual tests and provides a final summary."""
    print("--- Running K-MOTE Component Tests ---")
    
    # Run tests for each expert component first
    expert1_ok = test_expert(SplineKANLayer, "Spline KAN (B-Spline)")
    expert2_ok = test_expert(SplineKANLayer, "Spline KAN (RBF)", basis_function='rbf')
    expert3_ok = test_expert(FourierKANLayer, "Fourier KAN")
    expert4_ok = test_expert(WaveletKANLayer, "Wavelet KAN")
    
    all_experts_ok = all([expert1_ok, expert2_ok, expert3_ok, expert4_ok])
    
    if not all_experts_ok:
        print("\n❌ One or more expert tests failed. Aborting full module test.")
        return

    print("\n--- All Expert Component Tests Passed ---")
    
    # If experts are okay, test the full KMOTE module
    kmote_ok = test_kmote_module()
    
    if kmote_ok:
        print("\n\n🎉 All K-MOTE tests completed successfully!")
    else:
        print("\n\n❌ K-MOTE module tests failed. Please check the error messages above.")

def test_expert(expert_class, expert_name, **kwargs):
    """A generic test function for any expert layer."""
    print(f"\n--- Testing Expert: {expert_name} ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Test 1: Instantiation
        print(f"[1/3] Instantiating {expert_name}...")
        input_dim, output_dim = 1, 32
        expert = expert_class(input_dim=input_dim, output_dim=output_dim, **kwargs).to(device)
        print("✓ Instantiation successful.")

        # Test 2: Forward Pass
        print(f"[2/3] Testing forward pass...")
        batch_size, seq_len = 8, 128
        dummy_input = torch.randn(batch_size, seq_len, input_dim, device=device)
        output = expert(dummy_input)
        
        expected_shape = (batch_size, seq_len, output_dim)
        assert output.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {output.shape}"
        print(f"✓ Forward pass successful. Output shape: {output.shape}")

        # Test 3: Gradient Flow
        print(f"[3/3] Testing gradient flow...")
        target = torch.randn_like(output)
        loss = F.mse_loss(output, target)
        loss.backward()
        
        grad_found = any(p.grad is not None for p in expert.parameters() if p.requires_grad)
        assert grad_found, "No gradients were computed!"
        print("✓ Gradient flow successful.")
        
        return True
        
    except Exception as e:
        print(f"✗ Test FAILED for {expert_name} with error: {e}")
        traceback.print_exc()
        return False

def test_kmote_module():
    """Tests the full KMOTE module."""
    print(f"\n--- Testing Full KMOTE Module ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Test 1: Instantiation
        print(f"[1/3] Instantiating KMOTE...")
        input_dim, output_dim = 1, 64
        kmote = KMOTE(input_dim=input_dim, output_dim=output_dim).to(device)
        print("✓ Instantiation successful.")

        # Test 2: Forward Pass
        print(f"[2/3] Testing forward pass...")
        batch_size, seq_len = 8, 128
        dummy_input = torch.randn(batch_size, seq_len, input_dim, device=device)
        output = kmote(dummy_input)
        
        expected_shape = (batch_size, seq_len, output_dim)
        assert output.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {output.shape}"
        print(f"✓ Forward pass successful. Output shape: {output.shape}")

        # Test 3: Gradient Flow (End-to-End)
        print(f"[3/3] Testing gradient flow...")
        target = torch.randn_like(output)
        loss = F.mse_loss(output, target)
        loss.backward()
        
        # Check for gradients in both the gating network and the experts
        assert kmote.gating_network[0].weight.grad is not None, "No gradients in gating network."
        assert kmote.experts[-1].parameters().__next__().grad is not None, "No gradients in the last expert."
        print("✓ End-to-end gradient flow successful.")
        
        return True
        
    except Exception as e:
        print(f"✗ Test FAILED for KMOTE module with error: {e}")
        traceback.print_exc()
        return False

if __name__ == '__main__':
    run_all_tests()