#!/usr/bin/env python3
"""
Test script for Bochner and Mercer time encoders
"""

import torch
import sys
import os

# Add the project root to path
sys.path.append('/home/s2516027/kan-mammotev2')

def test_encoders():
    """Test Bochner and Mercer encoders"""
    print("Testing Bochner and Mercer Time Encoders")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test parameters
    batch_size = 8
    seq_len = 32
    time_dim = 256
    
    # Create test data
    timestamps = torch.randn(batch_size, seq_len, device=device)
    
    # Test Bochner encoder
    print("\n1. Testing Bochner Encoder:")
    try:
        from models.time_encoders.bochner_encoder import BochnerTimeEncoder
        
        bochner = BochnerTimeEncoder(time_dim=time_dim, device=device).to(device)
        
        with torch.no_grad():
            output = bochner(timestamps)
        
        assert output.shape == (batch_size, seq_len, time_dim), f"Shape mismatch: {output.shape}"
        assert torch.isfinite(output).all(), "Non-finite values detected"
        
        param_count = sum(p.numel() for p in bochner.parameters())
        print(f"   ✓ Shape: {output.shape}")
        print(f"   ✓ Parameters: {param_count:,}")
        print(f"   ✓ Output range: [{output.min():.3f}, {output.max():.3f}]")
        
    except Exception as e:
        print(f"   ✗ Bochner failed: {e}")
        return False
    
    # Test Mercer encoder
    print("\n2. Testing Mercer Encoder:")
    try:
        from models.time_encoders.mercer_encoder import MercerTimeEncoder
        
        mercer = MercerTimeEncoder(time_dim=time_dim, device=device).to(device)
        
        with torch.no_grad():
            output = mercer(timestamps)
        
        assert output.shape == (batch_size, seq_len, time_dim), f"Shape mismatch: {output.shape}"
        assert torch.isfinite(output).all(), "Non-finite values detected"
        
        param_count = sum(p.numel() for p in mercer.parameters())
        print(f"   ✓ Shape: {output.shape}")
        print(f"   ✓ Parameters: {param_count:,}")
        print(f"   ✓ Output range: [{output.min():.3f}, {output.max():.3f}]")
        
    except Exception as e:
        print(f"   ✗ Mercer failed: {e}")
        return False
    
    # Test factory function
    print("\n3. Testing Factory Function:")
    try:
        from models.time_encoders.factory import create_time_encoder, get_available_encoders
        
        available = get_available_encoders()
        print(f"   Available encoders: {list(available.keys())}")
        
        # Test creating each encoder type
        for encoder_type in ['bochner', 'mercer']:
            encoder = create_time_encoder(encoder_type, time_dim=64, device=device)
            test_input = torch.randn(4, 16, device=device)
            
            with torch.no_grad():
                output = encoder(test_input)
            
            print(f"   ✓ {encoder_type}: {output.shape}")
        
    except Exception as e:
        print(f"   ✗ Factory test failed: {e}")
        return False
    
    return True

def test_gradient_flow():
    """Test gradient flow through encoders"""
    print("\n4. Testing Gradient Flow:")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        from models.time_encoders.bochner_encoder import BochnerTimeEncoder
        from models.time_encoders.mercer_encoder import MercerTimeEncoder
        import torch.nn as nn
        
        # Test data
        timestamps = torch.randn(4, 16, device=device, requires_grad=True)
        target = torch.randn(4, 16, 64, device=device)
        
        encoders = {
            'Bochner': BochnerTimeEncoder(64, device=device),
            'Mercer': MercerTimeEncoder(64, device=device)
        }
        
        for name, encoder in encoders.items():
            encoder = encoder.to(device)
            encoder.train()
            
            # Forward pass
            output = encoder(timestamps)
            loss = nn.MSELoss()(output, target)
            
            # Backward pass
            loss.backward(retain_graph=True)
            
            # Check gradients
            has_grad = any(p.grad is not None for p in encoder.parameters() if p.requires_grad)
            
            if has_grad:
                print(f"   ✓ {name}: Gradients computed, Loss: {loss.item():.6f}")
            else:
                print(f"   ⚠ {name}: No gradients found")
        
    except Exception as e:
        print(f"   ✗ Gradient test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🔧 Testing Baseline Time Encoders")
    
    success = test_encoders()
    if success:
        success = test_gradient_flow()
    
    if success:
        print("\n" + "=" * 50)
        print("✅ All tests passed!")
        print("\n📝 Usage examples:")
        print("from models.time_encoders.factory import create_time_encoder")
        print("encoder = create_time_encoder('bochner', time_dim=256)")
        print("output = encoder(timestamps)")
        print("\n🎯 Ready for comparative experiments!")
    else:
        print("\n❌ Some tests failed. Check error messages above.")
