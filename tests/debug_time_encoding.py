#!/usr/bin/env python3
"""
Debug script to identify the learning issue in Event-Based MNIST experiment
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from models.time_encoders.ablation_encoders import SMKernelOnly
from event_based_mnist_experiment import EventBasedMNIST, collate_fn, TimeEncoderClassifier
from torch.utils.data import DataLoader

def debug_time_encoding():
    """Debug the time encoding process"""
    print("🔍 Debugging Time Encoding Issues")
    print("=" * 50)
    
    # Create a small sample dataset
    print("\n1. Creating sample dataset...")
    dataset = EventBasedMNIST(root='./data', train=True, threshold=0.9, max_events=10, download=True)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    # Get one batch
    sequences, labels, lengths = next(iter(loader))
    print(f"Batch shape: {sequences.shape}")
    print(f"Labels: {labels}")
    print(f"Lengths: {lengths}")
    print(f"Sample sequence: {sequences[0]}")
    
    # Debug time generation (Option 1)
    print("\n2. Debugging time generation...")
    batch_size, seq_len = sequences.shape
    
    # Generate absolute and relative time as in the model
    t_abs = sequences.unsqueeze(-1).float()  # pixel positions
    t_rel = torch.zeros_like(t_abs)
    t_rel[:, 1:, 0] = sequences[:, 1:] - sequences[:, :-1]  # differences
    t_rel[:, 0, 0] = 0  # first position
    
    print(f"t_abs shape: {t_abs.shape}")
    print(f"t_rel shape: {t_rel.shape}")
    print(f"Sample t_abs[0]: {t_abs[0].squeeze()}")
    print(f"Sample t_rel[0]: {t_rel[0].squeeze()}")
    
    # Check if relative time values are reasonable
    print(f"t_rel min: {t_rel.min()}, max: {t_rel.max()}, mean: {t_rel.mean()}")
    print(f"t_rel non-zero ratio: {(t_rel != 0).float().mean()}")
    
    # Debug SM-Kernel encoder
    print("\n3. Debugging SM-Kernel encoder...")
    sm_encoder = SMKernelOnly(embedding_dim=32, num_mixtures=12)
    
    # Initialize the SM-Kernel (this might be missing!)
    print("Initializing SM-Kernel from data...")
    sm_encoder.initialize_sm_kernel(t_rel)
    
    # Forward pass
    with torch.no_grad():
        sm_output = sm_encoder(t_abs, t_rel)
        print(f"SM-Kernel output shape: {sm_output.shape}")
        print(f"SM-Kernel output range: [{sm_output.min():.4f}, {sm_output.max():.4f}]")
        print(f"SM-Kernel output mean: {sm_output.mean():.4f}, std: {sm_output.std():.4f}")
        
        # Check if output is all zeros or constants
        if torch.allclose(sm_output, torch.zeros_like(sm_output), atol=1e-6):
            print("❌ SM-Kernel output is all zeros!")
        elif torch.allclose(sm_output, sm_output[0:1], atol=1e-6):
            print("❌ SM-Kernel output is constant across batch!")
        else:
            print("✅ SM-Kernel output varies across batch")

def debug_model_learning():
    """Debug if the model can learn on a simple task"""
    print("\n4. Debugging model learning capability...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create a very simple model
    model = TimeEncoderClassifier(
        encoder_type='sm_kernel_only',
        embedding_dim=32,
        hidden_dim=64,
        num_classes=10
    ).to(device)
    
    # Create some dummy data
    batch_size = 8
    seq_len = 10
    sequences = torch.randint(0, 784, (batch_size, seq_len))
    labels = torch.randint(0, 10, (batch_size,))
    lengths = torch.full((batch_size,), seq_len)
    
    sequences, labels = sequences.to(device), labels.to(device)
    
    # Check if model can overfit on this small batch
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("Testing if model can overfit on small batch...")
    initial_loss = None
    
    for epoch in range(20):
        optimizer.zero_grad()
        outputs = model(sequences, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if epoch == 0:
            initial_loss = loss.item()
        
        if epoch % 5 == 0:
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == labels).float().mean()
            print(f"Epoch {epoch}: Loss {loss.item():.4f}, Acc {accuracy:.4f}")
    
    final_loss = loss.item()
    print(f"Loss change: {initial_loss:.4f} -> {final_loss:.4f}")
    
    if final_loss < initial_loss * 0.5:
        print("✅ Model can learn (loss decreased significantly)")
    else:
        print("❌ Model cannot learn (loss barely changed)")
        
    # Check gradients
    total_grad_norm = 0
    param_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm
            param_count += 1
            if grad_norm < 1e-6:
                print(f"⚠️  Very small gradient for {name}: {grad_norm}")
    
    avg_grad_norm = total_grad_norm / param_count if param_count > 0 else 0
    print(f"Average gradient norm: {avg_grad_norm}")

def debug_data_distribution():
    """Debug the data distribution"""
    print("\n5. Debugging data distribution...")
    
    dataset = EventBasedMNIST(root='./data', train=True, threshold=0.9, max_events=50)
    
    # Check label distribution
    labels = [dataset[i][1] for i in range(min(1000, len(dataset)))]
    unique, counts = np.unique(labels, return_counts=True)
    print("Label distribution in first 1000 samples:")
    for label, count in zip(unique, counts):
        print(f"  Class {label}: {count} samples ({count/len(labels)*100:.1f}%)")
    
    # Check sequence length distribution
    lengths = [len(dataset[i][0]) for i in range(min(1000, len(dataset)))]
    print(f"Sequence length stats: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")

if __name__ == '__main__':
    debug_time_encoding()
    debug_model_learning()
    debug_data_distribution()