#!/usr/bin/env python3
"""
Test script to demonstrate the Time2Vec numerical instability issue.
This script shows how the linear component v2 can produce unbounded values.
"""

import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.time_encoders.periodic_activations import SineActivation, CosineActivation, t2v

def test_original_time2vec():
    """Test the original periodic_activations.py implementation"""
    print("🧪 Testing Original Time2Vec Implementation")
    print("=" * 50)
    
    # Create a SineActivation layer
    time_dim = 64
    sine_layer = SineActivation(1, time_dim)
    
    # Test with various input values
    test_inputs = [
        torch.tensor([[0.1]]),      # Small value
        torch.tensor([[1.0]]),      # Normal value  
        torch.tensor([[10.0]]),     # Large value
        torch.tensor([[100.0]]),    # Very large value
        torch.tensor([[1000.0]]),   # Extremely large value
    ]
    
    print("Input Value | Min Output | Max Output | Range")
    print("-" * 45)
    
    for i, input_val in enumerate(test_inputs):
        with torch.no_grad():
            output = sine_layer(input_val)
            min_val = output.min().item()
            max_val = output.max().item()
            range_val = max_val - min_val
            
            print(f"{input_val.item():10.1f} | {min_val:10.3f} | {max_val:10.3f} | {range_val:10.3f}")
            
            # Check for extreme values that could cause BCELoss issues
            if abs(min_val) > 10 or abs(max_val) > 10:
                print(f"  ⚠️  WARNING: Extreme values detected! This could cause BCELoss issues.")
            
            if range_val > 50:
                print(f"  ⚠️  WARNING: Very large range detected!")

def test_individual_components():
    """Test the individual components v1 and v2 separately"""
    print("\n🔍 Analyzing Individual Components")
    print("=" * 50)
    
    # Create parameters similar to SineActivation
    w0 = torch.randn(1, 1)  # For linear component
    b0 = torch.randn(1)     # For linear component  
    w = torch.randn(1, 63)  # For periodic component
    b = torch.randn(63)     # For periodic component
    
    print(f"w0 = {w0.item():.3f}, b0 = {b0.item():.3f}")
    
    test_inputs = [1.0, 10.0, 100.0, 1000.0]
    
    print("\nInput | v1 (sin) range | v2 (linear) value")
    print("-" * 40)
    
    for input_val in test_inputs:
        tau = torch.tensor([[input_val]])
        
        # Periodic component (bounded to [-1, 1])
        v1 = torch.sin(torch.matmul(tau, w) + b)
        v1_min, v1_max = v1.min().item(), v1.max().item()
        
        # Linear component (unbounded!)
        v2 = torch.matmul(tau, w0) + b0
        v2_val = v2.item()
        
        print(f"{input_val:5.0f} | [{v1_min:6.3f}, {v1_max:6.3f}] | {v2_val:12.3f}")
        
        if abs(v2_val) > 10:
            print(f"      ⚠️  Linear component v2 is getting large!")

if __name__ == "__main__":
    test_original_time2vec()
    test_individual_components()
    
    print("\n📊 CONCLUSION:")
    print("The original Time2Vec implementation CAN produce unbounded values")
    print("due to the linear component v2 = tau * w0 + b0.")
    print("When tau (timestamp) becomes large, v2 becomes arbitrarily large,")
    print("which can cause numerical instability in downstream losses like BCELoss.")