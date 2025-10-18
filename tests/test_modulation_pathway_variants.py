#!/usr/bin/env python3
"""
Test script to verify both modulation pathway variants work correctly
"""

import torch
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.time_encoders.kan_mammote import KAN_MAMMOTE

def test_modulation_pathway_variants():
    """Test both separate and combined modulation pathways"""
    
    print("="*80)
    print("Testing Modulation Pathway Variants")
    print("="*80)
    
    # Test parameters
    batch_size = 2
    seq_len = 10
    embedding_dim = 64
    expert_dim = 64
    
    # Create dummy inputs
    t_abs = torch.randn(batch_size, seq_len, 1)
    t_rel = torch.randn(batch_size, seq_len, 1).abs()
    
    print(f"\n📊 Test Configuration:")
    print(f"   Batch size: {batch_size}")
    print(f"   Sequence length: {seq_len}")
    print(f"   Embedding dim: {embedding_dim}")
    print(f"   Expert dim: {expert_dim}")
    
    # ===== VARIANT 1: Separate Modulation Pathway (DEFAULT) =====
    print("\n" + "="*80)
    print("🔬 VARIANT 1: Separate Modulation Pathway (DEFAULT)")
    print("   Content pathway: u_k (pure absolute)")
    print("   Modulation pathway: relative time → FiLM gates")
    print("="*80)
    
    model_separate = KAN_MAMMOTE(
        embedding_dim=embedding_dim,
        expert_dim=expert_dim,
        fusion_strategy='mamba',
        use_controllable_mamba=True,
        separate_modulation_pathway=True,  # DEFAULT
        wavelet_type='shock'
    )
    
    # Forward pass
    output_separate = model_separate(t_abs, t_rel, debug=False)
    
    print(f"\n✅ Variant 1 Output:")
    print(f"   Shape: {output_separate.shape}")
    print(f"   Range: [{output_separate.min().item():.3f}, {output_separate.max().item():.3f}]")
    print(f"   Mean: {output_separate.mean().item():.3f}")
    print(f"   Std: {output_separate.std().item():.3f}")
    
    # ===== VARIANT 2: Combined Pathway (LEGACY) =====
    print("\n" + "="*80)
    print("🔬 VARIANT 2: Combined Pathway (LEGACY)")
    print("   Content pathway: u_k + fusion_features (absolute + relative)")
    print("   Modulation pathway: relative time → FiLM gates")
    print("="*80)
    
    model_combined = KAN_MAMMOTE(
        embedding_dim=embedding_dim,
        expert_dim=expert_dim,
        fusion_strategy='mamba',
        use_controllable_mamba=True,
        separate_modulation_pathway=False,  # LEGACY
        wavelet_type='shock'
    )
    
    # Forward pass
    output_combined = model_combined(t_abs, t_rel, debug=False)
    
    print(f"\n✅ Variant 2 Output:")
    print(f"   Shape: {output_combined.shape}")
    print(f"   Range: [{output_combined.min().item():.3f}, {output_combined.max().item():.3f}]")
    print(f"   Mean: {output_combined.mean().item():.3f}")
    print(f"   Std: {output_combined.std().item():.3f}")
    
    # ===== COMPARISON =====
    print("\n" + "="*80)
    print("📊 COMPARISON")
    print("="*80)
    
    # Check shapes match
    assert output_separate.shape == output_combined.shape, "Output shapes must match!"
    print(f"✅ Output shapes match: {output_separate.shape}")
    
    # Check outputs are different (they should be due to different architectures)
    difference = (output_separate - output_combined).abs().mean().item()
    print(f"\n📏 Mean absolute difference: {difference:.6f}")
    
    if difference > 1e-6:
        print(f"✅ Outputs are different (as expected for different architectures)")
    else:
        print(f"⚠️  Outputs are very similar (unexpected!)")
    
    # ===== VANILLA MAMBA2 (should use combined pathway) =====
    print("\n" + "="*80)
    print("🔬 VANILLA MAMBA2 (should always use combined pathway)")
    print("="*80)
    
    model_vanilla = KAN_MAMMOTE(
        embedding_dim=embedding_dim,
        expert_dim=expert_dim,
        fusion_strategy='mamba',
        use_controllable_mamba=False,  # Vanilla Mamba2
        separate_modulation_pathway=True,  # This should be ignored for vanilla
        wavelet_type='shock'
    )
    
    # Forward pass
    output_vanilla = model_vanilla(t_abs, t_rel, debug=False)
    
    print(f"\n✅ Vanilla Mamba2 Output:")
    print(f"   Shape: {output_vanilla.shape}")
    print(f"   Note: Vanilla always uses combined pathway (u_k + fusion_features)")
    
    print("\n" + "="*80)
    print("✅ All tests passed!")
    print("="*80)

if __name__ == "__main__":
    test_modulation_pathway_variants()
