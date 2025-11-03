#!/usr/bin/env python3

import torch
import torch.nn as nn

def test_repeat_gradient_flow():
    """Test if repeat() operations break gradient flow"""
    print("="*60)
    print("Testing repeat() gradient flow behavior")
    print("="*60)
    
    # Test 1: Direct repeat() gradient flow
    x = torch.randn(2, 3, 4, requires_grad=True)
    print(f"1. Original tensor requires_grad: {x.requires_grad}")
    
    # Using repeat() - creates new tensor
    y = x.repeat(1, 1, 2)
    print(f"2. After repeat() requires_grad: {y.requires_grad}")
    print(f"3. Gradient function: {y.grad_fn}")
    print(f"4. Are they the same tensor? {x is y}")
    
    # Test gradient flow
    loss = y.sum()
    loss.backward()
    print(f"5. Original tensor got gradients: {x.grad is not None}")
    print(f"6. Original gradient shape: {x.grad.shape if x.grad is not None else None}")
    
    # Test 2: Memory efficiency
    print("\n" + "="*60)
    print("Testing memory efficiency")
    print("="*60)
    
    # Create a larger tensor
    large_x = torch.randn(100, 50, 64, requires_grad=True)
    print(f"Original tensor memory: {large_x.element_size() * large_x.numel() / 1024 / 1024:.2f} MB")
    
    # Repeat creates new memory
    repeated = large_x.repeat(1, 1, 4)  # 4x expansion
    print(f"Repeated tensor memory: {repeated.element_size() * repeated.numel() / 1024 / 1024:.2f} MB")
    print(f"Memory multiplication factor: {repeated.numel() / large_x.numel()}")
    
    # Test 3: Alternative approaches
    print("\n" + "="*60)
    print("Testing alternatives to repeat()")
    print("="*60)
    
    # Alternative 1: expand() - doesn't create new memory (only works with singleton dims)
    x_test = torch.randn(2, 3, 1, requires_grad=True)  # Note: last dim must be 1 for expansion
    expanded = x_test.expand(2, 3, 8)  # Now this works
    print(f"expand() creates new tensor: {x_test is expanded}")
    print(f"expand() gradient function: {expanded.grad_fn}")
    print(f"expand() shares memory: {expanded.data_ptr() == x_test.data_ptr()}")  # Check memory sharing
    
    # Alternative 2: Using broadcasting
    x_broadcast = torch.randn(2, 3, 1, requires_grad=True)
    target_shape = torch.randn(2, 3, 8)
    broadcasted = x_broadcast + torch.zeros_like(target_shape)
    print(f"Broadcasting gradient function: {broadcasted.grad_fn}")
    
def test_controllable_mamba_issue():
    """Test the specific issue in ControllableMamba2"""
    print("\n" + "="*60)
    print("Testing ControllableMamba2 specific issue")
    print("="*60)
    
    # Simulate the problematic scenario
    batch_size, seq_len = 4, 32
    nheads_modulator = 8
    nheads_mamba = 16  # Different from modulator
    
    # Simulate gamma/beta from modulator
    gamma = torch.randn(batch_size, seq_len, nheads_modulator, requires_grad=True)
    beta = torch.randn(batch_size, seq_len, nheads_modulator, requires_grad=True)
    
    # Simulate dt_content from Mamba2
    dt_content = torch.randn(batch_size, seq_len, nheads_mamba, requires_grad=True)
    
    print(f"gamma shape: {gamma.shape}")
    print(f"dt_content shape: {dt_content.shape}")
    print(f"Dimension mismatch: {gamma.shape[-1] != dt_content.shape[-1]}")
    
    # Simulate the problematic repeat() fix
    if gamma.shape[-1] != dt_content.shape[-1]:
        target_dim = dt_content.shape[-1]
        if gamma.shape[-1] < target_dim:
            print(f"Applying repeat() fix: {gamma.shape[-1]} -> {target_dim}")
            gamma_repeated = gamma.repeat(1, 1, target_dim // gamma.shape[-1])
            beta_repeated = beta.repeat(1, 1, target_dim // beta.shape[-1])
            
            print(f"gamma_repeated shape: {gamma_repeated.shape}")
            print(f"Memory increase factor: {gamma_repeated.numel() / gamma.numel()}")
            
            # Test if gradients flow back
            dt_fused = gamma_repeated * dt_content + beta_repeated
            loss = dt_fused.sum()
            loss.backward()
            
            print(f"Original gamma got gradients: {gamma.grad is not None}")
            print(f"Original beta got gradients: {beta.grad is not None}")
            print(f"dt_content got gradients: {dt_content.grad is not None}")

if __name__ == "__main__":
    test_repeat_gradient_flow()
    test_controllable_mamba_issue()