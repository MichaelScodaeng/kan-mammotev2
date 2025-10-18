#!/usr/bin/env python3
"""
Quick test script to verify all KAN-MAMMOTE variants work correctly
"""

import torch
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.time_encoders.kan_mammote import KAN_MAMMOTE

def test_variant(name, **kwargs):
    """Test a single KAN-MAMMOTE variant"""
    print(f"\n{'='*80}")
    print(f"Testing: {name}")
    print(f"{'='*80}")
    
    try:
        # Create model
        model = KAN_MAMMOTE(
            embedding_dim=128,
            expert_dim=64,
            **kwargs
        )
        
        # Create dummy input
        batch_size = 2
        seq_len = 10
        t_abs = torch.randn(batch_size, seq_len, 1)
        t_rel = torch.randn(batch_size, seq_len, 1).abs()
        
        # Forward pass
        output = model(t_abs, t_rel)
        
        # Check output shape
        expected_shape = (batch_size, seq_len, 128)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        # Check output is not NaN
        assert not torch.isnan(output).any(), "Output contains NaN"
        
        print(f"✅ PASS: Output shape {output.shape}")
        print(f"   Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Test all KAN-MAMMOTE variants"""
    
    print("🧪 Testing KAN-MAMMOTE Variants")
    print("=" * 80)
    
    results = {}
    
    # Test all fusion strategies with K-MOTE (default)
    variants = [
        ("Default (Mamba + ControllableMamba2 + K-MOTE)", {}),
        ("Mamba + Vanilla Mamba2 + K-MOTE", {"use_controllable_mamba": False}),
        ("Concat Fusion + K-MOTE", {"fusion_strategy": "concat"}),
        ("Weighted Fusion + K-MOTE", {"fusion_strategy": "weighted"}),
        ("Attention Fusion + K-MOTE", {"fusion_strategy": "attention"}),
        ("Mamba + ControllableMamba2 + SM-Kernel (legacy)", {"use_kmote_for_relative": False, "num_mixtures": 64}),
    ]
    
    for name, kwargs in variants:
        results[name] = test_variant(name, **kwargs)
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 SUMMARY")
    print(f"{'='*80}")
    
    passed = sum(results.values())
    total = len(results)
    
    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\n{'='*80}")
    print(f"Results: {passed}/{total} tests passed")
    print(f"{'='*80}")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
