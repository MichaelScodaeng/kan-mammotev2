#!/usr/bin/env python3
"""
Test different solutions for Time2Vec numerical stability issues.
"""

import torch
import torch.nn as nn
import numpy as np

def t2v_original(tau, f, out_features, w, b, w0, b0, arg=None):
    """Original Time2Vec implementation"""
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], -1)

def t2v_layernorm(tau, f, out_features, w, b, w0, b0, arg=None):
    """Time2Vec with layer normalization applied to the entire output"""
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    output = torch.cat([v1, v2], -1)
    # Apply layer normalization to entire output
    return torch.nn.functional.layer_norm(output, output.shape[-1:])

class SineActivationOriginal(nn.Module):
    """Original sine activation with random initialization"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.out_features = out_features
        self.w0 = nn.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.w = nn.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.Parameter(torch.randn(out_features-1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v_original(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)

class SineActivationSmallInit(nn.Module):
    """Sine activation with small initialization for linear component"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.out_features = out_features
        
        # Small initialization for linear component
        self.w0 = nn.Parameter(torch.randn(in_features, 1) * 0.01)  # Much smaller
        self.b0 = nn.Parameter(torch.randn(1) * 0.01)  # Much smaller
        
        # Normal initialization for periodic components
        self.w = nn.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.Parameter(torch.randn(out_features-1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v_original(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)

class SineActivationLayerNorm(nn.Module):
    """Sine activation with layer normalization"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.out_features = out_features
        self.w0 = nn.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.w = nn.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.Parameter(torch.randn(out_features-1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v_layernorm(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)

class SineActivationControlledInit(nn.Module):
    """Sine activation with controlled initialization like original TimeEncoder"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.out_features = out_features
        
        # Controlled initialization similar to original TimeEncoder
        # Use small frequencies for both periodic and linear components
        freq_scale = 1 / (10 ** np.linspace(0, 3, out_features))
        
        # Initialize weights with controlled frequencies
        self.w0 = nn.Parameter(torch.tensor(freq_scale[:1]).float().unsqueeze(0))  # (1, 1)
        self.b0 = nn.Parameter(torch.zeros(1))  # Start with zero bias
        
        self.w = nn.Parameter(torch.tensor(freq_scale[1:]).float().unsqueeze(0))  # (1, out_features-1)
        self.b = nn.Parameter(torch.zeros(out_features-1))  # Start with zero bias
        
        self.f = torch.sin

    def forward(self, tau):
        return t2v_original(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)

def test_implementation(impl_class, impl_name, test_values):
    """Test a specific implementation"""
    print(f"\n🧪 Testing {impl_name}")
    print("=" * 60)
    
    model = impl_class(1, 64)
    
    print("Input Value | Min Output | Max Output | Range     | Status")
    print("-" * 60)
    
    for val in test_values:
        input_tensor = torch.tensor([[val]], dtype=torch.float32)
        with torch.no_grad():
            output = model(input_tensor)
            min_val = output.min().item()
            max_val = output.max().item()
            range_val = max_val - min_val
            
            # Check for potential issues
            status = "✅ OK"
            if abs(min_val) > 10 or abs(max_val) > 10:
                status = "⚠️  LARGE"
            elif range_val > 10:
                status = "⚠️  WIDE"
            
            print(f"{val:>10} | {min_val:>9.3f} | {max_val:>9.3f} | {range_val:>8.3f} | {status}")
    
    # Test with very large timestamp to see behavior
    large_input = torch.tensor([[10000.0]], dtype=torch.float32)
    with torch.no_grad():
        large_output = model(large_input)
        print(f"\n📏 Extreme test (input=10000): range = {large_output.max().item() - large_output.min().item():.3f}")

def main():
    print("🔬 Testing Different Time2Vec Solutions")
    print("=" * 60)
    
    test_values = [0.1, 1.0, 10.0, 100.0, 1000.0]
    
    # Test all implementations
    test_implementation(SineActivationOriginal, "Original Random Init", test_values)
    test_implementation(SineActivationSmallInit, "Small Initialization", test_values)
    test_implementation(SineActivationLayerNorm, "Layer Normalization", test_values)
    test_implementation(SineActivationControlledInit, "Controlled Initialization", test_values)
    
    print(f"\n📊 CONCLUSIONS:")
    print("- Original: Can produce unbounded values")
    print("- Small Init: Reduces magnitude but doesn't solve fundamental issue")
    print("- Layer Norm: Ensures bounded output regardless of input")
    print("- Controlled Init: Uses frequency-based initialization like TimeEncoder")

if __name__ == "__main__":
    main()