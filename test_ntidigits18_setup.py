#!/usr/bin/env python3
"""
Quick test script for NTIDIGITS18 experiment setup
This script tests if all dependencies are available and the dataset can be loaded
"""

import sys
import os
import torch

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_imports():
    """Test if all required imports are available"""
    print("🧪 Testing imports...")
    
    try:
        import tonic
        from tonic.datasets import NTIDIGITS18
        print("✅ Tonic library available")
    except ImportError as e:
        print(f"❌ Tonic library error: {e}")
        return False
    
    try:
        from models.time_encoders.kan_mammote import KAN_MAMMOTE
        from models.time_encoders.ablation_encoders import KMOTEAbsOnly
        print("✅ KAN-MAMMOTE encoders available")
    except ImportError as e:
        print(f"❌ KAN-MAMMOTE import error: {e}")
        return False
    
    try:
        from models.time_encoders.lete_encoder import LeTE
        print("✅ LeTE encoder available")
    except ImportError:
        print("⚠️ LeTE encoder not available (optional)")
    
    return True

def test_dataset_loading():
    """Test basic dataset loading without full processing"""
    print("\n📊 Testing dataset loading...")
    
    try:
        # Test creating dataset object (should work even on CPU)
        dataset = NTIDIGITS18(
            save_to='./data',
            train=True,
            download=False  # Don't download in test
        )
        print(f"✅ Dataset object created: {len(dataset)} samples")
        
        # Try to get one sample
        if len(dataset) > 0:
            events, label = dataset[0]
            print(f"✅ Sample 0: events shape = {events.shape if hasattr(events, 'shape') else 'N/A'}, label = {label}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading error: {e}")
        return False

def test_model_creation():
    """Test if model can be created on CPU"""
    print("\n🔧 Testing model creation...")
    
    try:
        from experiments.ntidigits18_experiment import DigitClassifier
        
        # Test creating model on CPU
        model = DigitClassifier(
            encoder_type='lstm_only',
            embedding_dim=32,
            hidden_dim=64,
            num_classes=11,
            num_channels=64
        )
        
        print(f"✅ Model created: {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test forward pass with dummy data
        batch_size, seq_len = 2, 10
        dummy_times = torch.randn(batch_size, seq_len)
        dummy_channels = torch.randint(0, 64, (batch_size, seq_len))
        dummy_lengths = torch.tensor([seq_len, seq_len])
        
        with torch.no_grad():
            outputs = model(dummy_times, dummy_channels, dummy_lengths)
            print(f"✅ Forward pass successful: output shape = {outputs.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 NTIDIGITS18 Experiment Setup Test")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    success = True
    
    # Test imports
    if not test_imports():
        success = False
    
    # Test dataset loading
    if not test_dataset_loading():
        success = False
    
    # Test model creation
    if not test_model_creation():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed! Ready for GPU training.")
        print("\nTo run the full experiment when you have GPU access:")
        print("python experiments/ntidigits18_experiment.py --epochs 50 --batch_size 128")
        print("python experiments/ntidigits18_experiment.py --encoders lstm_only k_mote_abs kan_mammote_full")
    else:
        print("❌ Some tests failed. Please fix the issues before running on GPU.")

if __name__ == '__main__':
    main()