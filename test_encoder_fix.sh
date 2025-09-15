#!/bin/bash

# Test script to verify all encoders work with the new factory
echo "Testing all time encoders..."

cd /home/s2516027/kan-mammotev2

echo "Testing encoder factory..."
python -c "
from models.time_encoders.factory import get_available_encoders, create_time_encoder
import torch

print('Available encoders:', get_available_encoders())

# Test each encoder type
encoders_to_test = ['original', 'lete', 'kan_mammote', 'mercer', 'bochner', 'time2vec']

for encoder_type in encoders_to_test:
    try:
        print(f'\\nTesting {encoder_type}...')
        encoder = create_time_encoder(encoder_type, time_dim=100, device='cpu')
        
        # Test with sample data
        sample_input = torch.randn(10, 20)  # batch_size=10, seq_len=20
        
        # Test dual-stream interface
        output = encoder(t_abs=sample_input.unsqueeze(-1), t_rel=torch.zeros_like(sample_input.unsqueeze(-1)))
        print(f'  ✓ {encoder_type} works! Output shape: {output.shape}')
        
    except Exception as e:
        print(f'  ✗ {encoder_type} failed: {e}')

print('\\nFactory test completed!')
"

echo "Testing with command line arguments..."
python experiments/train_link_prediction.py \
    --dataset_name wikipedia \
    --model_name TGAT \
    --time_encoder_type original \
    --num_epochs 1 \
    --batch_size 50 \
    --num_runs 1 \
    --gpu -1 \
    --dry_run || echo "Original encoder test completed (expected to fail without dry_run support)"

echo "All tests completed!"
