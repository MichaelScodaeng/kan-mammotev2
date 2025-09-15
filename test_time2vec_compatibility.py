#!/usr/bin/env python3
"""
Test Time2Vec compatibility with factory and base encoder system.
"""

import torch
import sys
import traceback

def test_time2vec_compatibility():
    """Test Time2Vec compatibility with factory and base encoder."""
    print("=== Testing Time2Vec Compatibility ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    time_dim = 128
    
    try:
        # Test 1: Direct instantiation
        print("\n[Test 1] Direct instantiation...")
        from models.time_encoders.time2vec_encoder import Time2VecEncoder
        encoder = Time2VecEncoder(time_dim=time_dim, device=device, activation='sin')
        print(f"✓ Direct instantiation successful")
        
        # Test 2: Factory creation
        print("\n[Test 2] Factory creation...")
        from models.time_encoders.factory import create_time_encoder
        encoder_factory = create_time_encoder('time2vec', time_dim=time_dim, device=device, activation='cos')
        print(f"✓ Factory creation successful")
        
        # Test 3: Forward pass with different input shapes
        print("\n[Test 3] Forward pass tests...")
        
        # Shape 1: (batch_size,)
        timestamps_1d = torch.randn(8, device=device)
        output_1d = encoder(timestamps_1d)
        print(f"✓ 1D input {timestamps_1d.shape} → {output_1d.shape}")
        assert output_1d.shape == (8, time_dim), f"Expected shape (8, {time_dim}), got {output_1d.shape}"
        
        # Shape 2: (batch_size, seq_len)
        timestamps_2d = torch.randn(8, 64, device=device)
        output_2d = encoder(timestamps_2d)
        print(f"✓ 2D input {timestamps_2d.shape} → {output_2d.shape}")
        assert output_2d.shape == (8, 64, time_dim), f"Expected shape (8, 64, {time_dim}), got {output_2d.shape}"
        
        # Shape 3: (batch_size, seq_len, 1)
        timestamps_3d = torch.randn(8, 64, 1, device=device)
        output_3d = encoder(timestamps_3d)
        print(f"✓ 3D input {timestamps_3d.shape} → {output_3d.shape}")
        assert output_3d.shape == (8, 64, time_dim), f"Expected shape (8, 64, {time_dim}), got {output_3d.shape}"
        
        # Test 4: Interface compliance
        print("\n[Test 4] Interface compliance...")
        config = encoder.get_config()
        print(f"✓ Config method works: {config}")
        assert 'type' in config and 'time_dim' in config, "Config missing required fields"
        
        # Test 5: Device management
        print("\n[Test 5] Device management...")
        if torch.cuda.is_available():
            encoder.to_device('cuda')
            print(f"✓ Device transfer successful")
        
        # Test 6: Gradient flow
        print("\n[Test 6] Gradient flow test...")
        encoder.train()
        test_input = torch.randn(4, 32, device=device, requires_grad=True)
        output = encoder(test_input)
        loss = output.sum()
        loss.backward()
        print(f"✓ Gradient flow successful, Loss: {loss.item():.6f}")
        
        # Test 7: Both activation types
        print("\n[Test 7] Testing both activation types...")
        sin_encoder = create_time_encoder('time2vec', time_dim=64, activation='sin')
        cos_encoder = create_time_encoder('time2vec', time_dim=64, activation='cos')
        
        test_data = torch.randn(4, 16)
        sin_output = sin_encoder(test_data)
        cos_output = cos_encoder(test_data)
        
        print(f"✓ Sin activation: {test_data.shape} → {sin_output.shape}")
        print(f"✓ Cos activation: {test_data.shape} → {cos_output.shape}")
        
        # Test 8: Factory list
        print("\n[Test 8] Factory integration...")
        from models.time_encoders.factory import get_available_encoders
        available = get_available_encoders()
        assert 'time2vec' in available, "time2vec not in available encoders"
        print(f"✓ time2vec in available encoders: {available['time2vec']}")
        
        print("\n✅ All Time2Vec compatibility tests passed!")
        
        # Summary
        print("\n" + "="*50)
        print("📊 Test Summary:")
        print(f"  - Direct instantiation: ✓")
        print(f"  - Factory creation: ✓")
        print(f"  - Multiple input shapes: ✓")
        print(f"  - Interface compliance: ✓")
        print(f"  - Device management: ✓")
        print(f"  - Gradient flow: ✓")
        print(f"  - Activation types: ✓")
        print(f"  - Factory integration: ✓")
        
        print("\n🎯 Usage Examples:")
        print("from models.time_encoders.factory import create_time_encoder")
        print("encoder = create_time_encoder('time2vec', time_dim=256, activation='sin')")
        print("timestamps = torch.randn(32, 128, 1)")
        print("encoded = encoder(timestamps)  # Shape: (32, 128, 256)")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Time2Vec compatibility test failed: {e}")
        traceback.print_exc()
        return False

def test_performance():
    """Quick performance test."""
    print("\n=== Performance Test ===")
    
    try:
        import time
        from models.time_encoders.factory import create_time_encoder
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        encoder = create_time_encoder('time2vec', time_dim=256, device=device)
        
        # Large batch test
        batch_size = 32
        seq_len = 256
        test_data = torch.randn(batch_size, seq_len, device=device)
        
        # Warmup
        for _ in range(5):
            _ = encoder(test_data)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        num_runs = 100
        
        for _ in range(num_runs):
            with torch.no_grad():
                _ = encoder(test_data)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_runs * 1000  # ms
        throughput = batch_size * seq_len / (avg_time / 1000)  # tokens/sec
        
        print(f"✓ Performance test completed:")
        print(f"  - Input shape: {test_data.shape}")
        print(f"  - Average time: {avg_time:.2f} ms")
        print(f"  - Throughput: {throughput:.0f} tokens/sec")
        
    except Exception as e:
        print(f"✗ Performance test failed: {e}")

if __name__ == "__main__":
    print("🔧 Time2Vec Encoder Compatibility Test Suite")
    print("=" * 50)
    
    success = test_time2vec_compatibility()
    
    if success:
        test_performance()
        print("\n🎉 Time2Vec encoder is now fully compatible with your framework!")
    else:
        print("\n❌ Time2Vec compatibility test failed. Check errors above.")
        sys.exit(1)
