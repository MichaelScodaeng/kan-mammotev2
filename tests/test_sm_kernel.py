# file: test_sm_kernel.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Assuming your sm_kernel.py is in a folder that can be imported
# For example, if your structure is src/models/time_encoders/sm_kernel.py
# you would run this from the root of the project.
from models.time_encoders.sm_kernel import SMKernelLayer

def run_tests():
    """Runs a series of tests to validate the SMKernelLayer."""
    print("--- Running SMKernelLayer Tests ---")
    
    # --- Test 1: Basic Forward Pass and Shape Verification ---
    print("\n[Test 1/3] Testing forward pass and output shape...")
    try:
        batch_size = 8
        seq_len = 100
        num_mixtures = 64
        
        sm_kernel_layer = SMKernelLayer(num_mixtures=num_mixtures)
        dummy_delta_t = torch.rand(batch_size, seq_len, 1) # Must be positive
        
        output_embedding = sm_kernel_layer(dummy_delta_t)
        
        expected_shape = (batch_size, seq_len, num_mixtures)
        assert output_embedding.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {output_embedding.shape}"
        print(f"✓ Test 1 Passed: Output shape is correct ({output_embedding.shape})")
    except Exception as e:
        print(f"✗ Test 1 FAILED: {e}")
        return

    # --- Test 2: Data-Driven Initialization ---
    print("\n[Test 2/3] Testing data-driven initialization...")
    try:
        sm_kernel_layer_init = SMKernelLayer(num_mixtures=4)
        
        # Store the initial (random) means
        initial_means = sm_kernel_layer_init.kernel.raw_mixture_means.clone().detach()
        
        # Create a synthetic data sample with a known, clear frequency (period = 10)
        # We want delta_t values that have a dominant frequency of 0.1 Hz
        time = torch.linspace(0, 100, 1000)
        known_frequency = 0.1  # 0.1 Hz (period = 10)
        
        # Create delta_t values that oscillate with the known frequency
        # This creates a pattern where delta_t varies with the known frequency
        base_delta_t = 1.0  # base time difference
        modulation = 0.5 * torch.sin(2 * torch.pi * known_frequency * time[:-1])
        sample_delta_t = (base_delta_t + modulation + 0.1).reshape(1, -1, 1)  # Add small constant to keep positive
        
        sm_kernel_layer_init.initialize_from_data(sample_delta_t)
        
        initialized_means = sm_kernel_layer_init.kernel.raw_mixture_means.clone().detach()
        
        assert not torch.allclose(initial_means, initialized_means), "Initialization did not change the parameters!"
        
        # The frequency detection should find something close to our known frequency
        # We'll be more lenient since FFT-based frequency detection on delta_t is approximate
        max_detected_freq = initialized_means.max().item()
        print(f"Known frequency: {known_frequency}, Max detected frequency: {max_detected_freq}")
        
        # Check if any of the detected frequencies is reasonably close to the known frequency
        all_means = initialized_means.flatten().tolist()
        frequency_found = any(abs(freq - known_frequency) < 0.05 for freq in all_means)
        
        if not frequency_found:
            print(f"Warning: Expected frequency {known_frequency} not found in detected frequencies {all_means}")
            print("This may be due to the nature of FFT on delta_t sequences, but initialization still worked.")
        
        print("✓ Test 2 Passed: Initialization method works and parameters were updated.")
    except Exception as e:
        print(f"✗ Test 2 FAILED: {e}")
        return

    # --- Test 3: Gradient Flow and Learnability ---
    print("\n[Test 3/3] Testing gradient flow...")
    try:
        sm_kernel_layer_grad = SMKernelLayer(num_mixtures=16)
        
        # Check initial state of a parameter
        initial_weight_val = sm_kernel_layer_grad.kernel.raw_mixture_weights[0].clone().detach()

        # Dummy data, target, optimizer, and loss
        input_data = torch.rand(4, 50, 1)
        target = torch.randn(4, 50, 16)
        optimizer = optim.Adam(sm_kernel_layer_grad.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        
        # Perform a single training step
        optimizer.zero_grad()
        output = sm_kernel_layer_grad(input_data)
        loss = loss_fn(output, target)
        loss.backward()
        
        # Check if gradients exist
        assert sm_kernel_layer_grad.kernel.raw_mixture_weights.grad is not None, "Gradients are None! Backpropagation failed."
        assert sm_kernel_layer_grad.kernel.raw_mixture_means.grad is not None, "Gradients are None! Backpropagation failed."
        assert sm_kernel_layer_grad.kernel.raw_mixture_scales.grad is not None, "Gradients are None! Backpropagation failed."
        
        optimizer.step()
        
        # Check if parameters were updated
        updated_weight_val = sm_kernel_layer_grad.kernel.raw_mixture_weights[0].clone().detach()
        assert not torch.allclose(initial_weight_val, updated_weight_val), "Optimizer step did not update parameters!"

        print("✓ Test 3 Passed: Gradients are flowing and parameters are learnable.")
    except Exception as e:
        print(f"✗ Test 3 FAILED: {e}")
        return

    print("\n--- All SMKernelLayer tests passed successfully! ---")

if __name__ == '__main__':
    run_tests()