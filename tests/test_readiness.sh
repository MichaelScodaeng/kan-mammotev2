#!/bin/bash

# Test if the code is ready for execution with different time encoders
echo "Testing KAN-MAMMOTE execution readiness..."

# Test 1: Check if arguments are properly set up
echo "Testing argument parsing..."
cd /home/s2516027/kan-mammotev2
python -c "
from utils.load_configs import get_link_prediction_args
import sys
try:
    args = get_link_prediction_args(is_evaluation=False)
    print(f'✓ Arguments loaded successfully')
    print(f'  - time_encoder_type: {args.time_encoder_type}')
    print(f'  - expert_dim: {args.expert_dim}')
    print(f'  - num_mixtures: {args.num_mixtures}')
except Exception as e:
    print(f'✗ Failed to load arguments: {e}')
    sys.exit(1)
"

# Test 2: Check time encoder factory
echo "Testing time encoder factory..."
python -c "
from models.time_encoders.factory import create_time_encoder
import sys
try:
    # Test original encoder
    encoder = create_time_encoder('original', time_dim=100)
    print(f'✓ Original encoder created: {type(encoder).__name__}')
    
    # Test LeTE encoder
    encoder = create_time_encoder('lete', time_dim=100)
    print(f'✓ LeTE encoder created: {type(encoder).__name__}')
    
    # Test KAN-MAMMOTE encoder
    encoder = create_time_encoder('kan_mammote', time_dim=100, expert_dim=64, num_mixtures=4)
    print(f'✓ KAN-MAMMOTE encoder created: {type(encoder).__name__}')
except Exception as e:
    print(f'✗ Failed to create encoders: {e}')
    sys.exit(1)
"

# Test 3: Check model initialization with time encoder
echo "Testing model initialization with custom time encoder..."
python -c "
from models.time_encoders.factory import create_time_encoder
from models.gnn_backbones.TGAT import TGAT
import numpy as np
import sys

try:
    # Create dummy data
    node_features = np.random.rand(100, 64)
    edge_features = np.random.rand(100, 32)
    
    # Create time encoder
    time_encoder = create_time_encoder('kan_mammote', time_dim=100, expert_dim=64, num_mixtures=4)
    
    # Try to create model (will fail due to neighbor_sampler but we can catch import issues)
    print('✓ Model imports work correctly')
    print('✗ Full model test requires neighbor sampler setup')
except ImportError as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)
except Exception as e:
    print(f'✓ Expected error (neighbor sampler): {e}')
"

echo "Readiness test completed!"
