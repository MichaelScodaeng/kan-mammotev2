#!/usr/bin/env python3
"""
Real parameter count comparison for different time encoders
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from models.time_encoders.factory import create_time_encoder

def count_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_size_mb(param_count):
    """Convert parameter count to MB (assuming float32)"""
    return param_count * 4 / (1024 * 1024)

def test_time_encoder(encoder_type, time_dim=100):
    """Test a specific time encoder and return parameter count"""
    try:
        # Create encoder
        encoder = create_time_encoder(
            time_encoder_type=encoder_type,
            time_feat_dim=time_dim,
            # KAN-MAMMOTE specific params
            expert_dim=128,
            num_mixtures=16,
            mamba_d_state=128,
            mamba_d_conv=4,
            mamba_expand=4,
            mamba_headdim=64
        )
        
        param_count = count_parameters(encoder)
        size_mb = get_size_mb(param_count)
        
        # Test with dummy input
        batch_size = 10
        seq_len = 32
        dummy_times = torch.randn(batch_size, seq_len, 1)
        
        with torch.no_grad():
            output = encoder(dummy_times)
            output_shape = output.shape
        
        return {
            'params': param_count,
            'size_mb': size_mb,
            'output_shape': output_shape,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'params': 0,
            'size_mb': 0,
            'output_shape': None,
            'success': False,
            'error': str(e)
        }

def main():
    print("=== REAL TIME ENCODER PARAMETER COMPARISON ===")
    print()
    
    # List of encoders to test
    encoders_to_test = [
        'original',
        'time2vec', 
        'lete',
        'mercer',
        'bochner',
        'kan_mammote_dual_kmote',
        'kan_mammote'
    ]
    
    results = {}
    
    print(f"{'Encoder':<25} | {'Parameters':<12} | {'Size (MB)':<10} | {'Status'}")
    print("-" * 70)
    
    for encoder_type in encoders_to_test:
        result = test_time_encoder(encoder_type)
        results[encoder_type] = result
        
        if result['success']:
            params_str = f"{result['params']:,}"
            size_str = f"{result['size_mb']:.3f}"
            status = "✅ OK"
        else:
            params_str = "N/A"
            size_str = "N/A" 
            status = f"❌ {result['error'][:20]}..."
        
        print(f"{encoder_type:<25} | {params_str:<12} | {size_str:<10} | {status}")
    
    print()
    print("=== ANALYSIS ===")
    
    # Find successful results
    successful = {k: v for k, v in results.items() if v['success']}
    
    if 'kan_mammote_dual_kmote' in successful:
        kan_mammote = successful['kan_mammote_dual_kmote']
        print(f"KAN-MAMMOTE Dual: {kan_mammote['params']:,} parameters ({kan_mammote['size_mb']:.2f} MB)")
        
        # Compare with others
        for name, result in successful.items():
            if name != 'kan_mammote_dual_kmote':
                ratio = kan_mammote['params'] / result['params'] if result['params'] > 0 else float('inf')
                print(f"  vs {name}: {ratio:.1f}x larger")
    
    # Show sizes in context
    print()
    print("=== SIZE CONTEXT ===")
    print("Modern ML Model References:")
    print("• BERT-base: ~110M params (~440 MB)")
    print("• ResNet-50: ~25M params (~100 MB)")  
    print("• MobileNet: ~4M params (~16 MB)")
    print("• Your full TGN model: ~6M params (~24 MB)")
    
    if 'kan_mammote_dual_kmote' in successful:
        kan_size = successful['kan_mammote_dual_kmote']['size_mb']
        print(f"• Your KAN-MAMMOTE: ~{kan_size:.1f} MB ({kan_size/440*100:.1f}% of BERT-base)")

if __name__ == "__main__":
    main()