#!/usr/bin/env python3
"""
Manual parameter count analysis based on architecture definitions
"""

def linear_params(in_features, out_features, bias=True):
    """Calculate parameters for a Linear layer"""
    params = in_features * out_features
    if bias:
        params += out_features
    return params

def conv1d_params(in_channels, out_channels, kernel_size, groups=1, bias=True):
    """Calculate parameters for a Conv1d layer"""
    params = (in_channels // groups) * out_channels * kernel_size
    if bias:
        params += out_channels
    return params

def layernorm_params(features):
    """Calculate parameters for LayerNorm"""
    return 2 * features  # weight + bias

def analyze_kan_mammote():
    """Analyze KAN-MAMMOTE parameter count based on your log"""
    print("=== KAN-MAMMOTE DUAL PARAMETER ANALYSIS ===")
    print("(Based on actual architecture from your training log)")
    print()
    
    # From your log: time_base_transform
    time_transform = linear_params(1, 64, bias=True)
    print(f"time_base_transform: {time_transform:,} params")
    
    # Expert scales and shifts (3 experts, 64 dims each)
    expert_scales_shifts = 64 * 3 * 2  # scales + shifts
    print(f"expert_scales + expert_shifts (3 experts): {expert_scales_shifts:,} params")
    
    # Expert networks (from your log)
    # Expert 0: LeTESpline + Linear(64→128)
    expert_0 = 0 + linear_params(64, 128)  # LeTESpline has no learnable params
    
    # Expert 1: EfficientFourierKAN + Linear(64→128) 
    fourier_kan = linear_params(64, 64) + linear_params(64, 64)  # input + output projection
    expert_1 = fourier_kan + linear_params(64, 128)
    
    # Expert 2: EnhancedWaveletKAN + Linear(64→128)
    wavelet_kan = linear_params(320, 64)  # From your log: 320→64
    expert_2 = wavelet_kan + linear_params(64, 128)
    
    experts_total = expert_0 + expert_1 + expert_2
    print(f"Expert networks:")
    print(f"  Expert 0 (LeTESpline): {expert_0:,} params")
    print(f"  Expert 1 (FourierKAN): {expert_1:,} params") 
    print(f"  Expert 2 (WaveletKAN): {expert_2:,} params")
    print(f"  Total experts: {experts_total:,} params")
    
    # Gating network: Linear(64→64) + GELU + Linear(64→3)
    gating = linear_params(64, 64) + linear_params(64, 3)
    print(f"Gating network: {gating:,} params")
    
    # LayerNorm
    layer_norm = layernorm_params(128)
    print(f"LayerNorm: {layer_norm:,} params")
    
    # Single KMOTE total
    single_kmote = time_transform + expert_scales_shifts + experts_total + gating + layer_norm
    print(f"\nSingle KMOTE total: {single_kmote:,} params")
    
    # Mamba2 components (from your log)
    mamba_in_proj = linear_params(128, 1288, bias=False)
    mamba_conv1d = conv1d_params(768, 768, 4, groups=768)  # groups=768 means depthwise
    mamba_out_proj = linear_params(512, 128, bias=False)
    mamba_total = mamba_in_proj + mamba_conv1d + mamba_out_proj
    
    print(f"\nMamba2 components:")
    print(f"  in_proj: {mamba_in_proj:,} params")
    print(f"  conv1d: {mamba_conv1d:,} params") 
    print(f"  out_proj: {mamba_out_proj:,} params")
    print(f"  Total Mamba2: {mamba_total:,} params")
    
    # Modulator head: Linear(128→64) + GELU + Dropout + Linear(64→16)
    modulator = linear_params(128, 64) + linear_params(64, 16)
    print(f"Modulator head: {modulator:,} params")
    
    # Output projection: Linear(128→100) + LayerNorm(100)
    output_proj = linear_params(128, 100) + layernorm_params(100)
    print(f"Output projection: {output_proj:,} params")
    
    # Single KAN-MAMMOTE instance
    single_kan_mammote = single_kmote * 2 + mamba_total + modulator + output_proj  # 2 KMOTEs (abs+rel)
    print(f"\nSingle KAN-MAMMOTE: {single_kan_mammote:,} params")
    
    # You have 2 KAN-MAMMOTE instances in your model
    total_kan_mammote = single_kan_mammote * 2
    print(f"Total KAN-MAMMOTE (2 instances): {total_kan_mammote:,} params")
    
    # Convert to MB
    size_mb = total_kan_mammote * 4 / (1024 * 1024)
    print(f"Total KAN-MAMMOTE size: {size_mb:.2f} MB")
    
    return total_kan_mammote, size_mb

def compare_with_simple_encoders():
    """Compare with simpler time encoders"""
    print("\n=== COMPARISON WITH SIMPLE ENCODERS ===")
    
    # Original TGN time encoder (sinusoidal + learned)
    original = 100  # Just time_feat_dim parameters
    print(f"Original (sinusoidal): {original:,} params ({original*4/1024/1024:.4f} MB)")
    
    # Time2Vec: Linear(1→dim) + cos/sin
    time2vec = linear_params(1, 100)
    print(f"Time2Vec: {time2vec:,} params ({time2vec*4/1024/1024:.4f} MB)")
    
    # LETE: Learnable time embedding  
    lete = 50000  # Typical learnable embedding size
    print(f"LETE (estimated): {lete:,} params ({lete*4/1024/1024:.4f} MB)")
    
    # Your KAN-MAMMOTE
    kan_params, kan_size = analyze_kan_mammote()
    
    print(f"\n=== RELATIVE SIZES ===")
    print(f"KAN-MAMMOTE vs Original: {kan_params/original:.0f}x larger")
    print(f"KAN-MAMMOTE vs Time2Vec: {kan_params/time2vec:.0f}x larger") 
    print(f"KAN-MAMMOTE vs LETE: {kan_params/lete:.1f}x larger")

if __name__ == "__main__":
    compare_with_simple_encoders()