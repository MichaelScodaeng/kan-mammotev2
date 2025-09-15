#!/usr/bin/env python3

"""
Test script to verify the encoder fixes work properly.
This tests the LeTE encoder fix for the dimension broadcasting issue.
"""

import torch
import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_lete_encoder():
    """Test LeTE encoder with various input shapes."""
    print("Testing LeTE encoder...")
    
    try:
        from models.time_encoders.lete_encoder import LeTE
        
        # Create encoder
        encoder = LeTE(time_dim=100, device='cpu')
        print("✓ LeTE encoder created successfully")
        
        # Test different input shapes
        test_cases = [
            torch.randn(32, 1),      # (batch_size, 1)
            torch.randn(32, 20, 1),  # (batch_size, seq_len, 1)
            torch.randn(32),         # (batch_size,)
            torch.randn(32, 64),     # (batch_size, features) - this was causing the error
        ]
        
        for i, test_input in enumerate(test_cases):
            print(f"Testing input shape {i+1}: {test_input.shape}")
            try:
                output = encoder(timestamps=test_input)
                print(f"  ✓ Output shape: {output.shape}")
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                return False
                
        print("✓ All LeTE tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ LeTE encoder test failed: {e}")
        return False

def test_factory_wrapper():
    """Test the factory wrapper with different encoders."""
    print("Testing factory wrapper...")
    
    try:
        from models.time_encoders.factory import create_time_encoder
        
        # Test available encoders
        test_encoders = ['original', 'lete']
        
        for encoder_type in test_encoders:
            print(f"Testing {encoder_type} encoder...")
            try:
                encoder = create_time_encoder(encoder_type, time_dim=100, device='cpu')
                print(f"  ✓ {encoder_type} encoder created successfully")
                
                # Test with sample input that was causing issues
                test_input = torch.randn(32, 64)  # This shape was problematic
                print(f"  Testing input shape: {test_input.shape}")
                
                # Test both interfaces
                output1 = encoder(timestamps=test_input)
                print(f"    ✓ Timestamps interface output: {output1.shape}")
                
                output2 = encoder(t_abs=test_input)
                print(f"    ✓ t_abs interface output: {output2.shape}")
                
            except Exception as e:
                print(f"  ✗ {encoder_type} failed: {e}")
                return False
        
        print("✓ All factory wrapper tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Factory wrapper test failed: {e}")
        return False

def test_dual_stream_interface():
    """Test the dual-stream interface that TGAT uses."""
    print("Testing dual-stream interface...")
    
    try:
        from models.time_encoders.factory import create_time_encoder
        
        # Create encoder
        encoder = create_time_encoder('lete', time_dim=100, device='cpu')
        
        # Test the problematic case from TGAT
        batch_size = 32
        num_neighbors = 20
        
        # Simulate the TGAT scenario
        neighbor_t_abs = torch.randn(batch_size, num_neighbors)  # This was (32, 20)
        neighbor_t_rel = torch.randn(batch_size, num_neighbors)  # This was (32, 20)
        
        print(f"Testing TGAT scenario:")
        print(f"  neighbor_t_abs shape: {neighbor_t_abs.shape}")
        print(f"  neighbor_t_rel shape: {neighbor_t_rel.shape}")
        
        # This was the failing call
        output = encoder(t_abs=neighbor_t_abs, t_rel=neighbor_t_rel)
        print(f"  ✓ Output shape: {output.shape}")
        
        # Verify output is proper for TGAT
        assert output.dim() == 2, f"Expected 2D output, got {output.dim()}D"
        assert output.shape[0] == batch_size, f"Expected batch size {batch_size}, got {output.shape[0]}"
        
        print("✓ Dual-stream interface test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Dual-stream interface test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing KAN-MAMMOTE Encoder Fixes")
    print("=" * 60)
    
    all_passed = True
    
    # Test individual components
    all_passed &= test_lete_encoder()
    print()
    
    all_passed &= test_factory_wrapper()
    print()
    
    all_passed &= test_dual_stream_interface()
    print()
    
    if all_passed:
        print("🎉 All tests passed! The encoder fixes are working correctly.")
        print("You can now run the training script with different encoders:")
        print("  python experiments/train_link_prediction.py --time_encoder_type lete")
        print("  python experiments/train_link_prediction.py --time_encoder_type kan_mammote")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
