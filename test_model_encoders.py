#!/usr/bin/env python3
"""
Test script to verify that all GNN backbone models can accept and use different time encoders.
"""

import sys
import os
import torch
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.time_encoders.factory import create_time_encoder, get_available_encoders
from models.gnn_backbones.TGAT import TGAT
from models.gnn_backbones.CAWN import CAWN
from models.gnn_backbones.TCL import TCL
from models.gnn_backbones.GraphMixer import GraphMixer
from models.gnn_backbones.DyGFormer import DyGFormer
from models.gnn_backbones.DyGMamba import DyGMamba
from utils.utils import NeighborSampler

def test_model_with_encoders():
    """Test that all models can accept different time encoders."""
    
    print("🧪 Testing GNN Backbone Models with Different Time Encoders")
    print("=" * 70)
    
    # Create dummy data
    num_nodes = 100
    num_edges = 200
    node_feat_dim = 172
    edge_feat_dim = 172
    time_feat_dim = 100
    
    node_raw_features = np.random.randn(num_nodes + 1, node_feat_dim).astype(np.float32)
    edge_raw_features = np.random.randn(num_edges + 1, edge_feat_dim).astype(np.float32)
    
    # Create dummy neighbor sampler (basic implementation)
    class DummyNeighborSampler:
        def __init__(self):
            pass
    
    neighbor_sampler = DummyNeighborSampler()
    
    # Test encoders that should work
    test_encoders = ['original', 'kan_mammote', 'kan_mammote_lite']
    
    # Get available encoders
    available_encoders = get_available_encoders()
    print(f"Available encoders: {available_encoders}")
    
    # Models to test
    models_to_test = {
        'TGAT': {
            'class': TGAT,
            'params': {
                'node_raw_features': node_raw_features,
                'edge_raw_features': edge_raw_features,
                'neighbor_sampler': neighbor_sampler,
                'time_feat_dim': time_feat_dim,
                'num_layers': 2,
                'num_heads': 2,
                'dropout': 0.1,
                'device': 'cpu'
            }
        },
        'CAWN': {
            'class': CAWN,
            'params': {
                'node_raw_features': node_raw_features,
                'edge_raw_features': edge_raw_features,
                'neighbor_sampler': neighbor_sampler,
                'time_feat_dim': time_feat_dim,
                'position_feat_dim': 172,
                'walk_length': 2,
                'num_walk_heads': 8,
                'dropout': 0.1,
                'device': 'cpu'
            }
        },
        'TCL': {
            'class': TCL,
            'params': {
                'node_raw_features': node_raw_features,
                'edge_raw_features': edge_raw_features,
                'neighbor_sampler': neighbor_sampler,
                'time_feat_dim': time_feat_dim,
                'num_layers': 2,
                'num_heads': 2,
                'num_depths': 20,
                'dropout': 0.1,
                'device': 'cpu'
            }
        },
        'GraphMixer': {
            'class': GraphMixer,
            'params': {
                'node_raw_features': node_raw_features,
                'edge_raw_features': edge_raw_features,
                'neighbor_sampler': neighbor_sampler,
                'time_feat_dim': time_feat_dim,
                'num_tokens': 20,
                'num_layers': 2,
                'dropout': 0.1,
                'device': 'cpu'
            }
        },
        'DyGFormer': {
            'class': DyGFormer,
            'params': {
                'node_raw_features': node_raw_features,
                'edge_raw_features': edge_raw_features,
                'neighbor_sampler': neighbor_sampler,
                'time_feat_dim': time_feat_dim,
                'channel_embedding_dim': 50,
                'patch_size': 1,
                'num_layers': 2,
                'num_heads': 2,
                'dropout': 0.1,
                'max_input_sequence_length': 512,
                'device': 'cpu'
            }
        },
        'DyGMamba': {
            'class': DyGMamba,
            'params': {
                'node_raw_features': node_raw_features,
                'edge_raw_features': edge_raw_features,
                'neighbor_sampler': neighbor_sampler,
                'time_feat_dim': time_feat_dim,
                'channel_embedding_dim': 50,
                'patch_size': 1,
                'num_layers': 2,
                'num_heads': 2,
                'dropout': 0.1,
                'gamma': 0.5,
                'max_input_sequence_length': 512,
                'max_interaction_times': 10,
                'device': 'cpu'
            }
        }
    }
    
    # Test each model with each encoder
    results = {}
    
    for encoder_name in test_encoders:
        if encoder_name not in available_encoders:
            print(f"⚠️ Encoder '{encoder_name}' not available, skipping...")
            continue
            
        print(f"\n🔧 Testing with {encoder_name} encoder:")
        print("-" * 50)
        
        # Create the encoder
        try:
            time_encoder = create_time_encoder(
                encoder_type=encoder_name,
                time_dim=time_feat_dim,
                device='cpu',
                expert_dim=64,  # For KAN-MAMMOTE
                num_mixtures=10  # For KAN-MAMMOTE
            )
            print(f"✅ Successfully created {encoder_name} encoder")
        except Exception as e:
            print(f"❌ Failed to create {encoder_name} encoder: {e}")
            continue
        
        results[encoder_name] = {}
        
        # Test each model
        for model_name, model_config in models_to_test.items():
            try:
                print(f"  Testing {model_name}...", end=" ")
                
                # Add time_encoder to params
                params = model_config['params'].copy()
                params['time_encoder'] = time_encoder
                
                # Create model
                model = model_config['class'](**params)
                
                print("✅ Success")
                results[encoder_name][model_name] = "✅ Success"
                
            except Exception as e:
                print(f"❌ Failed: {e}")
                results[encoder_name][model_name] = f"❌ Failed: {e}"
    
    # Print summary
    print(f"\n📊 SUMMARY")
    print("=" * 70)
    
    for encoder_name, model_results in results.items():
        print(f"\n{encoder_name.upper()} ENCODER:")
        for model_name, result in model_results.items():
            print(f"  {model_name}: {result}")
    
    # Count successes
    total_tests = len(results) * len(models_to_test)
    successful_tests = sum(1 for encoder_results in results.values() 
                          for result in encoder_results.values() 
                          if result.startswith("✅"))
    
    print(f"\n🎯 OVERALL RESULT: {successful_tests}/{total_tests} tests passed")
    
    if successful_tests == total_tests:
        print("🎉 All tests passed! All models can use different time encoders.")
        return True
    else:
        print("⚠️ Some tests failed. Check the results above.")
        return False

if __name__ == "__main__":
    success = test_model_with_encoders()
    sys.exit(0 if success else 1)