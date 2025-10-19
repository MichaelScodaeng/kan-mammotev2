#!/usr/bin/env python3


# Global training configuration
MAX_EPOCHS = 2

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import argparse
from datetime import datetime
import json
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
import pandas as pd
import csv
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import time encoders
from models.time_encoders.factory import create_time_encoder
from models.time_encoders.ablation_encoders import (
    SMKernelOnly, KMOTEAbsOnly, KMOTERelOnly, DualStreamBaseline
)
from models.time_encoders.kan_mammote_lite import KAN_MAMMOTE_Lite
from models.time_encoders.kan_mammote import KAN_MAMMOTE

# Import optional encoders
try:
    from models.time_encoders.lete_encoder import LeTE
    LETE_AVAILABLE = True
except ImportError:
    LETE_AVAILABLE = False

# Create LeTE Relative Time variant
class LeTERelativeTime(nn.Module):
    """LeTE variant that uses relative time differences instead of absolute positions
    
    NOTE: This wrapper converts absolute pixel positions to relative differences.
    All positions become relative (including first position = 0), and values are
    shifted to positive range [0, ~800] for compatibility with LeTE's learned frequencies.
    """
    def __init__(self, time_dim):
        super().__init__()
        if not LETE_AVAILABLE:
            raise ImportError("LeTE not available")
        self.lete = LeTE(time_dim=time_dim)
        
    def forward(self, x):
        """
        Convert absolute positions to relative differences before LeTE encoding
        Args:
            x: (batch, seq_len) - absolute pixel positions [0-783]
        Returns:
            embeddings: (batch, seq_len, time_dim)
        """
        # ===== FIX: Make all positions consistently relative =====
        batch_size, seq_len = x.shape
        rel_times = torch.zeros_like(x, dtype=torch.float32)
        
        # All positions become relative differences (consistent signal)
        rel_times[:, 0] = 0.0  # First event has no predecessor
        if seq_len > 1:
            rel_times[:, 1:] = x[:, 1:].float() - x[:, :-1].float()
        
        # Shift to positive range (LeTE uses learned frequencies that expect positive values)
        # Max difference is 783, so range is [-783, 783]
        #rel_times = rel_times + 400.0  # Shift to approximately [0, 800] range
        # ===== END FIX =====
        
        return self.lete(rel_times)

# Create Time2Vec Relative Time variant
class Time2VecRelativeTime(nn.Module):
    """Time2Vec variant that uses relative time differences instead of absolute positions"""
    def __init__(self, time_dim):
        super().__init__()
        if not TIME2VEC_AVAILABLE:
            raise ImportError("Time2Vec not available")
        self.time2vec = Time2VecEncoder(time_dim=time_dim)
        
    def forward(self, x):
        """
        Convert absolute positions to relative differences before Time2Vec encoding
        Args:
            x: (batch, seq_len) - absolute pixel positions [0-783]
        Returns:
            embeddings: (batch, seq_len, time_dim)
        """
        # ===== FIX: Make all positions consistently relative =====
        batch_size, seq_len = x.shape
        rel_times = torch.zeros_like(x, dtype=torch.float32)
        
        # All positions become relative differences (consistent signal)
        rel_times[:, 0] = 0.0  # First event has no predecessor
        if seq_len > 1:
            rel_times[:, 1:] = x[:, 1:].float() - x[:, :-1].float()
        
        # Shift to positive range (Time2Vec uses periodic functions that work better with positive values)
        #rel_times = rel_times + 400.0  # Shift to approximately [0, 800] range
        # ===== END FIX =====
        
        return self.time2vec(rel_times)

# Create Mercer Relative Time variant
class MercerRelativeTime(nn.Module):
    """Mercer variant that uses relative time differences instead of absolute positions"""
    def __init__(self, time_dim):
        super().__init__()
        if not MERCER_AVAILABLE:
            raise ImportError("Mercer not available")
        self.mercer = MercerTimeEncoder(time_dim=time_dim)
        
    def forward(self, x):
        """
        Convert absolute positions to relative differences before Mercer encoding
        Args:
            x: (batch, seq_len) - absolute pixel positions [0-783]
        Returns:
            embeddings: (batch, seq_len, time_dim)
        """
        # ===== FIX: Make all positions consistently relative =====
        batch_size, seq_len = x.shape
        rel_times = torch.zeros_like(x, dtype=torch.float32)
        
        # All positions become relative differences (consistent signal)
        rel_times[:, 0] = 0.0  # First event has no predecessor
        if seq_len > 1:
            rel_times[:, 1:] = x[:, 1:].float() - x[:, :-1].float()
        
        # Shift to positive range (Mercer kernel expects positive values)
        #rel_times = rel_times + 400.0  # Shift to approximately [0, 800] range
        # ===== END FIX =====
        
        return self.mercer(rel_times)

# Create K-MOTE Absolute Time variants (using absolute pixel positions only)
class KMOTEAbsoluteTime(nn.Module):
    """K-MOTE variant that uses absolute positions (default adapter mode)"""
    def __init__(self, time_dim, **kwargs):
        super().__init__()
        self.kmote = create_time_encoder('k_mote', time_dim=time_dim, 
                                        transform_mode='adapter', adapter_type='affine', **kwargs)
        
    def forward(self, x):
        """x: (batch, seq_len) - absolute pixel positions [0-783]"""
        return self.kmote(x.float())

class KMOTESharedAbsoluteTime(nn.Module):
    """K-MOTE (shared transform) variant that uses absolute positions"""
    def __init__(self, time_dim, **kwargs):
        super().__init__()
        self.kmote = create_time_encoder('k_mote', time_dim=time_dim, 
                                        transform_mode='shared', **kwargs)
        
    def forward(self, x):
        """x: (batch, seq_len) - absolute pixel positions [0-783]"""
        return self.kmote(x.float())

class KMOTEPerExpertAbsoluteTime(nn.Module):
    """K-MOTE (per-expert transform) variant that uses absolute positions"""
    def __init__(self, time_dim, **kwargs):
        super().__init__()
        self.kmote = create_time_encoder('k_mote', time_dim=time_dim, 
                                        transform_mode='per_expert', **kwargs)
        
    def forward(self, x):
        """x: (batch, seq_len) - absolute pixel positions [0-783]"""
        return self.kmote(x.float())

class KMOTEAdapterAffineAbsoluteTime(nn.Module):
    """K-MOTE (adapter affine) variant that uses absolute positions"""
    def __init__(self, time_dim, **kwargs):
        super().__init__()
        self.kmote = create_time_encoder('k_mote', time_dim=time_dim, 
                                        transform_mode='adapter', adapter_type='affine', **kwargs)
        
    def forward(self, x):
        """x: (batch, seq_len) - absolute pixel positions [0-783]"""
        return self.kmote(x.float())

class KMOTEAdapterLinearAbsoluteTime(nn.Module):
    """K-MOTE (adapter linear) variant that uses absolute positions"""
    def __init__(self, time_dim, **kwargs):
        super().__init__()
        self.kmote = create_time_encoder('k_mote', time_dim=time_dim, 
                                        transform_mode='adapter', adapter_type='linear', **kwargs)
        
    def forward(self, x):
        """x: (batch, seq_len) - absolute pixel positions [0-783]"""
        return self.kmote(x.float())

# Create K-MOTE Relative Time variants (using time differences)
class KMOTERelativeTime(nn.Module):
    """K-MOTE variant that uses relative time differences (default adapter mode)"""
    def __init__(self, time_dim, **kwargs):
        super().__init__()
        self.kmote = create_time_encoder('k_mote', time_dim=time_dim, 
                                        transform_mode='adapter', adapter_type='affine', **kwargs)
        
    def forward(self, x):
        """Convert absolute positions to relative differences before K-MOTE encoding"""
        batch_size, seq_len = x.shape
        rel_times = torch.zeros_like(x, dtype=torch.float32)
        rel_times[:, 0] = 0.0
        if seq_len > 1:
            rel_times[:, 1:] = x[:, 1:].float() - x[:, :-1].float()
        return self.kmote(rel_times)

class KMOTESharedRelativeTime(nn.Module):
    """K-MOTE (shared transform) variant that uses relative time differences"""
    def __init__(self, time_dim, **kwargs):
        super().__init__()
        self.kmote = create_time_encoder('k_mote', time_dim=time_dim, 
                                        transform_mode='shared', **kwargs)
        
    def forward(self, x):
        """Convert absolute positions to relative differences before K-MOTE encoding"""
        batch_size, seq_len = x.shape
        rel_times = torch.zeros_like(x, dtype=torch.float32)
        rel_times[:, 0] = 0.0
        if seq_len > 1:
            rel_times[:, 1:] = x[:, 1:].float() - x[:, :-1].float()
        return self.kmote(rel_times)

class KMOTEPerExpertRelativeTime(nn.Module):
    """K-MOTE (per-expert transform) variant that uses relative time differences"""
    def __init__(self, time_dim, **kwargs):
        super().__init__()
        self.kmote = create_time_encoder('k_mote', time_dim=time_dim, 
                                        transform_mode='per_expert', **kwargs)
        
    def forward(self, x):
        """Convert absolute positions to relative differences before K-MOTE encoding"""
        batch_size, seq_len = x.shape
        rel_times = torch.zeros_like(x, dtype=torch.float32)
        rel_times[:, 0] = 0.0
        if seq_len > 1:
            rel_times[:, 1:] = x[:, 1:].float() - x[:, :-1].float()
        return self.kmote(rel_times)

class KMOTEAdapterAffineRelativeTime(nn.Module):
    """K-MOTE (adapter affine) variant that uses relative time differences"""
    def __init__(self, time_dim, **kwargs):
        super().__init__()
        self.kmote = create_time_encoder('k_mote', time_dim=time_dim, 
                                        transform_mode='adapter', adapter_type='affine', **kwargs)
        
    def forward(self, x):
        """Convert absolute positions to relative differences before K-MOTE encoding"""
        batch_size, seq_len = x.shape
        rel_times = torch.zeros_like(x, dtype=torch.float32)
        rel_times[:, 0] = 0.0
        if seq_len > 1:
            rel_times[:, 1:] = x[:, 1:].float() - x[:, :-1].float()
        return self.kmote(rel_times)

class KMOTEAdapterLinearRelativeTime(nn.Module):
    """K-MOTE (adapter linear) variant that uses relative time differences"""
    def __init__(self, time_dim, **kwargs):
        super().__init__()
        self.kmote = create_time_encoder('k_mote', time_dim=time_dim, 
                                        transform_mode='adapter', adapter_type='linear', **kwargs)
        
    def forward(self, x):
        """Convert absolute positions to relative differences before K-MOTE encoding"""
        batch_size, seq_len = x.shape
        rel_times = torch.zeros_like(x, dtype=torch.float32)
        rel_times[:, 0] = 0.0
        if seq_len > 1:
            rel_times[:, 1:] = x[:, 1:].float() - x[:, :-1].float()
        return self.kmote(rel_times)

try:
    from models.time_encoders.mercer_encoder import MercerTimeEncoder
    MERCER_AVAILABLE = True
except ImportError:
    MERCER_AVAILABLE = False

try:
    from models.time_encoders.bochner_encoder import BochnerTimeEncoder
    BOCHNER_AVAILABLE = True
except ImportError:
    BOCHNER_AVAILABLE = False

try:
    from models.time_encoders.time2vec_encoder import Time2VecEncoder
    TIME2VEC_AVAILABLE = True
except ImportError:
    TIME2VEC_AVAILABLE = False


class EventBasedMNIST(Dataset):
    """
    Convert MNIST images to event sequences based on pixel brightness threshold.
    Each event is a pixel position that exceeds the threshold.
    """
    def __init__(self, root, train=True, threshold=0.9, max_events=None, transform=None, download=True, normalize_positions=True):
        super(EventBasedMNIST, self).__init__()
        
        # Load MNIST dataset
        
        self.threshold = threshold
        self.max_events = max_events  # None = use all events (matching paper)
        self.normalize_positions = normalize_positions  # Normalize pixel positions to [0,1]
        self.transform = transform
        # Convert images to event sequences
        self.event_data = []
        self.labels = []
        self.data = datasets.MNIST(root=root, train=train, transform=transform, download=download)
        print(f"Converting MNIST to event sequences (threshold={threshold}, max_events={max_events})...")
        for img, label in self.data:
            img_flat = img.view(-1)  # (784,)
            events = torch.nonzero(img_flat > self.threshold).squeeze()
            events = torch.sort(events).values
            self.event_data.append(events)
            self.labels.append(label)
    
    def __len__(self):
        return len(self.event_data)
    
    def __getitem__(self, idx):
        return self.event_data[idx], self.labels[idx]

def custom_collate_fn(batch):
    events_list = []
    labels_list = []
    lengths = []
    for events, label in batch:
        events_list.append(events)
        labels_list.append(label)
        lengths.append(events.shape[0])
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)
    padded_events = pad_sequence(events_list, batch_first=True, padding_value=0)  # (batch, max_len)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return padded_events, lengths, labels_tensor


"""
def collate_fn(batch):
    # Custom collate function to handle variable sequence lengths (matching LeTE implementation)
    sequences, labels = zip(*batch)
    
    # Get sequence lengths
    lengths = [len(seq) for seq in sequences]
    
    # ✅ Use pad_sequence with padding_value=0 (matching LeTE exactly)
    
    padded_sequences = torch_pad_sequence(sequences, batch_first=True, padding_value=0)
    
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    
    return padded_sequences, labels_tensor, lengths_tensor
"""

class PlainLSTMEncoder(nn.Module):
    """
    Plain embedding for LSTM baseline (no time encoding).
    Treats pixel positions as categorical indices to be embedded.
    This matches the "LSTM" baseline in the paper (without specialized time encoding).
    """
    def __init__(self, embedding_dim, max_position=784):
        super().__init__()
        self.embedding_dim = embedding_dim
        # Embedding table: pixel position (0-783) -> embedding vector
        # padding_idx=-1 for padded positions
        self.embedding = nn.Embedding(max_position, embedding_dim, padding_idx=0)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len) - pixel positions [0-783] or -1 for padding
        Returns:
            embeddings: (batch, seq_len, embedding_dim)
        """
        # Convert to long for embedding lookup
        x_long = x.long()
        # Replace -1 padding with 0 temporarily (padding_idx handles it)
        x_long = torch.where(x_long == -1, torch.zeros_like(x_long), x_long)
        return self.embedding(x_long)


class TimeEncoderClassifier(nn.Module):
    """
    LSTM classifier with different time encoders for event sequences
    """
    def __init__(self, encoder_type='lete', embedding_dim=32, hidden_dim=128, num_classes=10, **encoder_kwargs):
        super().__init__()
        self.encoder_type = encoder_type
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Create time encoder
        self.time_encoder = self._create_time_encoder(encoder_type, embedding_dim, **encoder_kwargs)
        
        # LSTM + classifier
        self.lstm = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_dim, 
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def _create_time_encoder(self, encoder_type, embedding_dim, **kwargs):
        """Create the appropriate time encoder"""
        if encoder_type == 'lstm_only':
            return PlainLSTMEncoder(embedding_dim=embedding_dim)
        elif encoder_type == 'lete':
            if not LETE_AVAILABLE:
                raise ImportError("LeTE not available")
            return LeTE(time_dim=embedding_dim)
        elif encoder_type in ['lete_rel', 'lete_relative']:
            return LeTERelativeTime(time_dim=embedding_dim)
        elif encoder_type == 'time2vec':
            if not TIME2VEC_AVAILABLE:
                raise ImportError("Time2Vec not available")
            return Time2VecEncoder(time_dim=embedding_dim)
        elif encoder_type in ['time2vec_rel', 'time2vec_relative']:
            return Time2VecRelativeTime(time_dim=embedding_dim)
        elif encoder_type == 'mercer':
            if not MERCER_AVAILABLE:
                raise ImportError("Mercer not available")
            return MercerTimeEncoder(time_dim=embedding_dim)
        elif encoder_type in ['mercer_rel', 'mercer_relative']:
            return MercerRelativeTime(time_dim=embedding_dim)
        elif encoder_type == 'bochner':
            if not BOCHNER_AVAILABLE:
                raise ImportError("Bochner not available")
            return BochnerTimeEncoder(time_dim=embedding_dim)
        # K-MOTE variants with different transform modes
        elif encoder_type == 'k_mote':
            # Default: adapter mode with affine adapters
            return create_time_encoder('k_mote', time_dim=embedding_dim, 
                                      transform_mode='adapter', adapter_type='affine', **kwargs)
        elif encoder_type == 'k_mote_shared':
            # Shared transform (MoE approach)
            return create_time_encoder('k_mote', time_dim=embedding_dim, 
                                      transform_mode='shared', **kwargs)
        elif encoder_type == 'k_mote_per_expert':
            # Per-expert transforms (LeTE-style)
            return create_time_encoder('k_mote', time_dim=embedding_dim, 
                                      transform_mode='per_expert', **kwargs)
        elif encoder_type == 'k_mote_adapter_affine':
            # Adapter mode with affine adapters (same as default)
            return create_time_encoder('k_mote', time_dim=embedding_dim, 
                                      transform_mode='adapter', adapter_type='affine', **kwargs)
        elif encoder_type == 'k_mote_adapter_linear':
            # Adapter mode with linear adapters
            return create_time_encoder('k_mote', time_dim=embedding_dim, 
                                      transform_mode='adapter', adapter_type='linear', **kwargs)
        
        # K-MOTE Absolute Time variants (uses absolute pixel positions)
        elif encoder_type in ['k_mote_abs', 'k_mote_absolute']:
            return KMOTEAbsoluteTime(time_dim=embedding_dim, **kwargs)
        elif encoder_type in ['k_mote_shared_abs', 'k_mote_shared_absolute']:
            return KMOTESharedAbsoluteTime(time_dim=embedding_dim, **kwargs)
        elif encoder_type in ['k_mote_per_expert_abs', 'k_mote_per_expert_absolute']:
            return KMOTEPerExpertAbsoluteTime(time_dim=embedding_dim, **kwargs)
        elif encoder_type in ['k_mote_adapter_affine_abs', 'k_mote_adapter_affine_absolute']:
            return KMOTEAdapterAffineAbsoluteTime(time_dim=embedding_dim, **kwargs)
        elif encoder_type in ['k_mote_adapter_linear_abs', 'k_mote_adapter_linear_absolute']:
            return KMOTEAdapterLinearAbsoluteTime(time_dim=embedding_dim, **kwargs)
        
        # K-MOTE Relative Time variants (uses time differences)
        elif encoder_type in ['k_mote_rel', 'k_mote_relative']:
            return KMOTERelativeTime(time_dim=embedding_dim, **kwargs)
        elif encoder_type in ['k_mote_shared_rel', 'k_mote_shared_relative']:
            return KMOTESharedRelativeTime(time_dim=embedding_dim, **kwargs)
        elif encoder_type in ['k_mote_per_expert_rel', 'k_mote_per_expert_relative']:
            return KMOTEPerExpertRelativeTime(time_dim=embedding_dim, **kwargs)
        elif encoder_type in ['k_mote_adapter_affine_rel', 'k_mote_adapter_affine_relative']:
            return KMOTEAdapterAffineRelativeTime(time_dim=embedding_dim, **kwargs)
        elif encoder_type in ['k_mote_adapter_linear_rel', 'k_mote_adapter_linear_relative']:
            return KMOTEAdapterLinearRelativeTime(time_dim=embedding_dim, **kwargs)
        
        # Ablation encoders
        elif encoder_type == 'sm_kernel_only':
            return SMKernelOnly(embedding_dim=embedding_dim, **kwargs)
        elif encoder_type == 'kmote_abs_only':
            return KMOTEAbsOnly(embedding_dim=embedding_dim, **kwargs)
        elif encoder_type == 'kmote_rel_only':
            return KMOTERelOnly(embedding_dim=embedding_dim, **kwargs)
        elif encoder_type == 'dual_stream_baseline':
            return DualStreamBaseline(embedding_dim=embedding_dim, **kwargs)
        # Full encoders
        elif encoder_type == 'kan_mammote_lite':
            return KAN_MAMMOTE_Lite(embedding_dim=embedding_dim, **kwargs)
        elif encoder_type in ['kan_mammote_lite_concat', 'kan_mammote_lite_weighted', 
                              'kan_mammote_lite_attention', 'kan_mammote_dual_kmote']:
            return KAN_MAMMOTE_Lite(embedding_dim=embedding_dim, fusion_strategy=encoder_type.replace('kan_mammote_lite_', ''), **kwargs)
        elif encoder_type == 'kan_mammote_full':
            # Default: K-MOTE for relative, controllable Mamba2, mamba fusion
            return KAN_MAMMOTE(embedding_dim=embedding_dim,expert_dim = 64, mamba_headdim=16)
        # KAN-MAMMOTE variants with different fusion strategies
        elif encoder_type == 'kan_mammote_concat':
            return KAN_MAMMOTE(embedding_dim=embedding_dim, fusion_strategy='concat', **kwargs)
        elif encoder_type == 'kan_mammote_weighted':
            return KAN_MAMMOTE(embedding_dim=embedding_dim, fusion_strategy='weighted', **kwargs)
        elif encoder_type == 'kan_mammote_attention':
            return KAN_MAMMOTE(embedding_dim=embedding_dim, fusion_strategy='attention', **kwargs)
        # KAN-MAMMOTE with vanilla Mamba2 (no FiLM modulation)
        elif encoder_type == 'kan_mammote_vanilla_mamba':
            return KAN_MAMMOTE(embedding_dim=embedding_dim, use_controllable_mamba=False, **kwargs)
        # KAN-MAMMOTE with SM-kernel (legacy, for ablation)
        elif encoder_type == 'kan_mammote_sm_kernel':
            return KAN_MAMMOTE(embedding_dim=embedding_dim, use_kmote_for_relative=False, **kwargs)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    def _needs_both_times(self, encoder_type):
        """Check if encoder needs both absolute and relative time"""
        dual_time_encoders = [
            'sm_kernel_only', 'kmote_abs_only', 'kmote_rel_only',
            'dual_stream_baseline', 'kan_mammote_lite', 'kan_mammote_full',
            'kan_mammote_lite_concat', 'kan_mammote_lite_weighted',
            'kan_mammote_lite_attention', 'kan_mammote_dual_kmote',
            'kan_mammote_concat', 'kan_mammote_weighted', 'kan_mammote_attention',
            'kan_mammote_vanilla_mamba', 'kan_mammote_sm_kernel'
        ]
        return encoder_type in dual_time_encoders
    
    def forward(self, x, lengths):
        """
        Forward pass
        
        Args:
            x: (batch, seq_len) - pixel positions [0-783]
            lengths: (batch,) - actual sequence lengths
            
        Returns:
            logits: (batch, num_classes)
        """
        batch_size, seq_len = x.shape
        
        # ===== CRITICAL FIX: Convert to float BEFORE encoder =====
        # Original LeTE code does: padded_events.float().to(device)
        # This ensures the encoder receives float tensors, not long tensors
        x_float = x.float()
        # ===== END CRITICAL FIX =====
        
        # Check encoder type and process accordingly
        if self.encoder_type == 'lstm_only':
            # LSTM baseline uses categorical embedding (expects long tensor)
            embedded = self.time_encoder(x)  # Keep as long for embedding lookup
            
            if not hasattr(self, '_debug_printed'):
                print(f"🔍 DEBUG - LSTM-Only Baseline (no time encoding):")
                print(f"  Input range: [{x.min():.1f}, {x.max():.1f}]")
                print(f"  Embedding output shape: {embedded.shape}")
                print(f"  Using categorical embedding table (784 positions)")
                self._debug_printed = True
        
        elif self._needs_both_times(self.encoder_type):
            # Dual-time encoders need both absolute and relative
            t_abs = x_float.unsqueeze(-1)  # (batch, seq_len, 1)
            
            # Generate relative time (differences between consecutive events)
            t_rel = torch.zeros_like(t_abs)
            t_rel[:, 1:, 0] = x_float[:, 1:] - x_float[:, :-1]
            t_rel[:, 0, 0] = 0  # First position has no predecessor
            
            # Move to device
            t_abs = t_abs.to(next(self.parameters()).device)
            t_rel = t_rel.to(next(self.parameters()).device)
            
            # Debug on first batch only
            if not hasattr(self, '_debug_printed'):
                print(f"🔍 DEBUG - Input Statistics (RAW - matching paper):")
                print(f"  t_abs range: [{t_abs.min():.1f}, {t_abs.max():.1f}], mean: {t_abs.mean():.1f}")
                print(f"  t_rel range: [{t_rel.min():.1f}, {t_rel.max():.1f}], mean: {t_rel.mean():.1f}")
                self._debug_printed = True
            
            # Forward with RAW values (no normalization)
            embedded = self.time_encoder(t_abs, t_rel)
            
        else:
            # Single input encoders (LeTE, Mercer, Time2Vec, etc.)
            # ===== CRITICAL: NO DEBUG PRINT HERE =====
            # Original LeTE code does NOT print the tensor
            # ===== END CRITICAL =====
            
            if not hasattr(self, '_debug_printed'):
                print(f"🔍 DEBUG - Input Statistics (RAW for {self.encoder_type}):")
                print(f"  x_float range: [{x_float.min():.1f}, {x_float.max():.1f}], mean: {x_float.mean():.1f}")
                self._debug_printed = True
            
            # ===== CRITICAL FIX: Pass float tensor directly =====
            # Original LeTE: embedded = self.time_encoder(x) where x is already float
            embedded = self.time_encoder(x_float)
            # ===== END CRITICAL FIX =====
        
        # ===== CRITICAL FIX: Pack sequences EXACTLY like LeTE =====
        # Original LeTE uses: pack_padded_sequence(..., enforce_sorted=False)
        # This is ESSENTIAL for handling variable-length sequences correctly
        packed = pack_padded_sequence(
            embedded, 
            lengths.cpu(),  # Move to CPU for pack_padded_sequence
            batch_first=True, 
            enforce_sorted=False  # CRITICAL: allows unsorted lengths
        )
        # ===== END CRITICAL FIX =====
        
        # LSTM forward
        _, (h_n, c_n) = self.lstm(packed)
        
        # ===== CRITICAL FIX: Use last hidden state correctly =====
        # Original LeTE: h_n = h_n[-1]  (takes last layer's hidden state)
        h_n = h_n[-1]  # (batch, hidden_dim)
        # ===== END CRITICAL FIX =====
        
        # Classifier
        logits = self.fc(h_n)  # (batch, num_classes)
        
        return logits


def train_model(model, train_loader, val_loader, num_epochs, device, encoder_name, models_dir='.', checkpoint_dir=None, resume_from_checkpoint=False):
    """
    Train the model with proper evaluation
    """
    # ===== CRITICAL FIX: Use Adam with lr=1e-3 (matching LeTE exactly) =====
    # Original LeTE: optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(f"🔧 Using Adam optimizer: lr=0.001 (matching LeTE)")
    # ===== END CRITICAL FIX =====
    
    criterion = nn.CrossEntropyLoss()
    
    # Initialize tracking
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    best_val_acc = 0.0
    start_epoch = 0
    
    # Resume from checkpoint if requested
    if resume_from_checkpoint and checkpoint_dir:
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir, encoder_name)
        if latest_checkpoint:
            checkpoint_info = load_checkpoint(latest_checkpoint, model, optimizer)
            start_epoch = checkpoint_info['epoch'] + 1
            best_val_acc = checkpoint_info['best_val_acc']
            history = checkpoint_info['history']
            print(f"✅ Resumed from epoch {start_epoch} with best val acc: {best_val_acc:.2f}%")
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        # ===== Training Phase =====
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        # ===== CRITICAL FIX: Use tqdm EXACTLY like LeTE =====
        # Original LeTE wraps the dataloader with tqdm for progress bars
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        # ===== END CRITICAL FIX =====
        
        for batch_idx, (padded_events, lengths, labels) in enumerate(train_pbar):
            # ===== CRITICAL FIX: Move to device BEFORE forward pass =====
            # Original LeTE: padded_events.float().to(device)
            padded_events = padded_events.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            # ===== END CRITICAL FIX =====
            
            # Debug first batch only
            if batch_idx == 0 and epoch == start_epoch:
                print(f"🔍 DEBUG - Input Statistics (RAW - matching paper):")
                print(f"  t_abs range: [{padded_events.min():.1f}, {padded_events.max():.1f}], mean: {padded_events.float().mean():.1f}")
                # For relative time calculation
                t_rel_debug = torch.zeros_like(padded_events.float())
                t_rel_debug[:, 1:] = padded_events[:, 1:].float() - padded_events[:, :-1].float()
                print(f"  t_rel range: [{t_rel_debug.min():.1f}, {t_rel_debug.max():.1f}], mean: {t_rel_debug.mean():.1f}")
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(padded_events, lengths)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # ===== ADD: Gradient clipping for stability =====
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # ===== END ADD =====
            
            # ===== OPTIONAL: Gradient health monitoring (first batch only) =====
            if batch_idx == 0 and epoch == start_epoch:
                total_params = 0
                small_grads = 0
                zero_grads = 0
                grad_norms = []
                
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        total_params += 1
                        grad_norm = param.grad.norm().item()
                        grad_norms.append(grad_norm)
                        
                        if grad_norm < 1e-8:
                            zero_grads += 1
                        elif grad_norm < 1e-6:
                            small_grads += 1
                            print(f"⚠️  Very small gradient for {name}: {grad_norm:.2e}")
                
                if grad_norms:
                    avg_grad_norm = sum(grad_norms) / len(grad_norms)
                    print(f"🔍 DEBUG - Average gradient norm: {avg_grad_norm:.6f}")
                    print(f"📊 Gradient health: {small_grads} small + {zero_grads} zero / {total_params} total ({100*(small_grads+zero_grads)/total_params:.1f}% unhealthy)")
            # ===== END OPTIONAL =====
            
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
            # Update progress bar
            current_acc = 100.0 * total_correct / total_samples
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
        
        # Calculate epoch metrics
        train_loss = total_loss / total_samples
        train_acc = 100.0 * total_correct / total_samples
        
        # ===== Validation Phase =====
        val_loss, val_acc = evaluate_model(model, val_loader, device, criterion)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(models_dir, f'best_model_{encoder_name}.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"🔥 New best validation accuracy: {val_acc:.2f}%")
            print(f"💾 Model saved to: {best_model_path}")
        
        # Save checkpoint
        if checkpoint_dir:
            checkpoint_path = save_checkpoint(
                model, optimizer, epoch, history, best_val_acc, encoder_name, checkpoint_dir
            )
            print(f"💾 Best checkpoint saved to: {checkpoint_path}")
    
    return history, best_val_acc


def evaluate_model(model, data_loader, device, criterion):
    """
    Evaluate model on a dataset (matches LeTE's evaluate function exactly)
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # ===== CRITICAL FIX: Use tqdm for validation too =====
    val_pbar = tqdm(data_loader, desc=f'Epoch {model.training}/{model.training} [Val]')
    # Get current epoch from somewhere, or just use a generic description
    val_pbar.set_description('[Val]')
    # ===== END CRITICAL FIX =====
    
    with torch.no_grad():
        for padded_events, lengths, labels in val_pbar:
            # Move to device
            padded_events = padded_events.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(padded_events, lengths)
            loss = criterion(outputs, labels)
            
            # Track metrics
            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
            # Update progress bar
            current_acc = 100.0 * total_correct / total_samples
            val_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
    
    avg_loss = total_loss / total_samples
    acc = 100.0 * total_correct / total_samples
    
    return avg_loss, acc

def resume_specific_encoder(experiment_dir, encoder_name, additional_epochs=50):
    """
    Resume training for a specific encoder from its latest checkpoint
    
    Args:
        experiment_dir: Path to experiment directory containing checkpoints
        encoder_name: Name of the encoder to resume
        additional_epochs: Number of additional epochs to train
    """
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir, encoder_name)
    
    if not latest_checkpoint:
        print(f"❌ No checkpoint found for {encoder_name} in {checkpoint_dir}")
        return None
    
    print(f"🔄 Resuming {encoder_name} from {latest_checkpoint}")
    
    # Load checkpoint to get training configuration
    checkpoint = torch.load(latest_checkpoint, map_location='cpu')
    completed_epochs = checkpoint['epoch'] + 1
    
    print(f"📊 Checkpoint info:")
    print(f"  - Completed epochs: {completed_epochs}")
    print(f"  - Best validation accuracy: {checkpoint['best_val_acc']:.2f}%")
    print(f"  - Will train for {additional_epochs} more epochs")
    
    return {
        'checkpoint_path': latest_checkpoint,
        'completed_epochs': completed_epochs,
        'best_val_acc': checkpoint['best_val_acc'],
        'history': checkpoint['history']
    }


def get_available_encoders():
    """Get list of available encoders based on imports"""
    # Always available encoders (no external dependency)
    encoders = [
        #'lstm_only',  # Baseline
        '''
        # K-MOTE variants with different transform modes (default: uses absolute positions)
        'k_mote',  # Default: adapter mode with affine adapters
        'k_mote_shared',  # Shared transform (MoE approach)
        'k_mote_per_expert',  # Per-expert transforms (LeTE-style)
        'k_mote_adapter_affine',  # Adapter mode with affine (same as default)
        'k_mote_adapter_linear',  # Adapter mode with linear adapters
        '''
        # K-MOTE Absolute Time variants (explicitly using absolute pixel positions)
        #'k_mote_abs',  # Default adapter with absolute time
        'k_mote_shared_abs',  # Shared transform with absolute time
        'k_mote_per_expert_abs',  # Per-expert with absolute time
        'k_mote_adapter_affine_abs',  # Affine adapter with absolute time
        'k_mote_adapter_linear_abs',  # Linear adapter with absolute time
        
        # K-MOTE Relative Time variants (using time differences)
        #'k_mote_rel',  # Default adapter with relative time
        'k_mote_shared_rel',  # Shared transform with relative time
        'k_mote_per_expert_rel',  # Per-expert with relative time
        'k_mote_adapter_affine_rel',  # Affine adapter with relative time
        'k_mote_adapter_linear_rel',  # Linear adapter with relative time
        
        # Ablation study encoders
        #'kmote_abs_only', 'kmote_rel_only', 
        #'sm_kernel_only',
        #'dual_stream_baseline',
        # KAN-MAMMOTE Lite variants (without Mamba)
        #'kan_mammote_lite', 'kan_mammote_lite_concat', 'kan_mammote_lite_weighted', 
        #'kan_mammote_lite_attention', 'kan_mammote_dual_kmote',
        # KAN-MAMMOTE Full variants (with different fusion strategies)
        #'kan_mammote_full',  # Default: K-MOTE relative + ControllableMamba2 + mamba fusion
        #'kan_mammote_concat',  # K-MOTE relative + concat fusion
        #'kan_mammote_weighted',  # K-MOTE relative + weighted fusion
        #'kan_mammote_attention',  # K-MOTE relative + attention fusion
        #'kan_mammote_vanilla_mamba',  # K-MOTE relative + vanilla Mamba2 + mamba fusion
        #'kan_mammote_sm_kernel',  # SM-kernel (legacy) + ControllableMamba2 + mamba fusion
    ]

    
    # Optional encoders (require imports)
    if LETE_AVAILABLE:
        encoders.extend(['lete', 'lete_relative'])
    '''
    if MERCER_AVAILABLE:
        encoders.extend(['mercer', 'mercer_relative'])
    if TIME2VEC_AVAILABLE:
        encoders.extend(['time2vec', 'time2vec_relative'])
    '''
    """
    if BOCHNER_AVAILABLE:
        encoders.append('bochner')
    """
    
    return encoders


def save_checkpoint(model, optimizer, epoch, history, best_val_acc, encoder_name, checkpoint_dir):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'best_val_acc': best_val_acc,
        'encoder_name': encoder_name
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{encoder_name}_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Also save as latest checkpoint
    latest_path = os.path.join(checkpoint_dir, f'checkpoint_{encoder_name}_latest.pth')
    torch.save(checkpoint, latest_path)
    
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load training checkpoint"""
    print(f"📂 Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'epoch': checkpoint['epoch'],
        'history': checkpoint['history'],
        'best_val_acc': checkpoint['best_val_acc'],
        'encoder_name': checkpoint['encoder_name']
    }


def find_latest_checkpoint(checkpoint_dir, encoder_name):
    """Find the latest checkpoint for a given encoder"""
    latest_path = os.path.join(checkpoint_dir, f'checkpoint_{encoder_name}_latest.pth')
    
    if os.path.exists(latest_path):
        return latest_path
    
    # Look for numbered checkpoints
    checkpoint_files = []
    if os.path.exists(checkpoint_dir):
        for file in os.listdir(checkpoint_dir):
            if file.startswith(f'checkpoint_{encoder_name}_epoch_') and file.endswith('.pth'):
                epoch_num = int(file.split('_epoch_')[1].split('.pth')[0])
                checkpoint_files.append((epoch_num, os.path.join(checkpoint_dir, file)))
    
    if checkpoint_files:
        # Return the checkpoint with the highest epoch number
        checkpoint_files.sort(key=lambda x: x[0], reverse=True)
        return checkpoint_files[0][1]
    
    return None




def run_experiment(encoder_name, args, models_dir='.', checkpoint_dir=None):
    """Run experiment for a specific encoder"""
    print(f"\n{'='*60}")
    print(f"Running experiment: {encoder_name}")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    transform = transforms.ToTensor()
    # Create datasets with input normalization (fixes gradient scaling issues)
    train_dataset = EventBasedMNIST(
        root='./data', 
        train=True, 
        threshold=args.threshold,
        max_events=args.max_events,
        transform=transform,
        download=True,
        normalize_positions=False  # ✅ Enable position normalization
    )
    
    val_dataset = EventBasedMNIST(
        root='./data', 
        train=False, 
        threshold=args.threshold,
        max_events=args.max_events,
        transform=transform,
        download=True,
        normalize_positions=False  # ✅ Enable position normalization
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=custom_collate_fn,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=custom_collate_fn,
        num_workers=2
    )
    
    # Create model
    try:
        model = TimeEncoderClassifier(
            encoder_type=encoder_name,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_classes=10,
            expert_dim=args.expert_dim,
            mamba_d_state=args.mamba_d_state,
            mamba_d_conv=args.mamba_d_conv,
            mamba_expand=args.mamba_expand,
            wavelet_type=args.wavelet_type
        ).to(device)
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train model
        history, best_val_acc = train_model(
            model, train_loader, val_loader, 
            args.epochs, device, encoder_name, models_dir=models_dir, 
            checkpoint_dir=checkpoint_dir, resume_from_checkpoint=args.resume_training
        )
        
        # ✅ Return best_val_acc (not final_val_acc) for fair comparison
        return {
            'encoder': encoder_name,
            'best_val_acc': best_val_acc,  # ← This is what we compare!
            'final_train_acc': history['train_acc'][-1],
            'final_val_acc': history['val_acc'][-1],
            'history': history,
            'success': True
        }
        
    except Exception as e:
        print(f"❌ Error with {encoder_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'encoder': encoder_name,
            'error': str(e),
            'success': False
        }


def test_simple_classification(encoder_type='sm_kernel_only', epochs=10):
    """Test if the model can learn a simple pattern (for debugging)"""
    print(f"\n🧪 Testing simple pattern learning with {encoder_type}...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create simple synthetic data
    # Pattern: sequences starting with low numbers = class 0, high numbers = class 1
    sequences = []
    labels = []
    
    for i in range(1000):
        if i < 500:
            # Class 0: sequences starting with low pixel positions (0-100)
            seq = torch.randint(0, 100, (10,))
            label = 0
        else:
            # Class 1: sequences starting with high pixel positions (600-784)
            seq = torch.randint(600, 784, (10,))
            label = 1
        
        sequences.append(seq)
        labels.append(label)
    
    # Test if model can learn this simple pattern
    model = TimeEncoderClassifier(
        encoder_type=encoder_type,
        embedding_dim=32,
        hidden_dim=64,
        num_classes=2
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print("Testing pattern: low pixel positions (0-100) = class 0, high (600-784) = class 1")
    
    # Train for a few epochs
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for i in range(0, len(sequences), 32):
            batch_seq = torch.stack(sequences[i:i+32])
            batch_labels = torch.tensor(labels[i:i+32])
            batch_lengths = torch.tensor([10] * len(batch_labels))
            
            batch_seq, batch_labels = batch_seq.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_seq, batch_lengths)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
        
        if epoch % 5 == 0:
            acc = 100. * correct / total
            print(f"Epoch {epoch}: Loss {total_loss:.4f}, Acc {acc:.2f}%")
    
    final_acc = 100. * correct / total
    if final_acc > 80:
        print("✅ Model can learn simple patterns! Fixes are working.")
        return True
    else:
        print("❌ Model still cannot learn simple patterns")
        return False


def plot_training_curves(results, save_path='mnist_training_curves.png'):
    """Plot training curves like Figure 3 in the paper"""
    
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("⚠️  No successful results to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Colors for different encoders
    colors = plt.cm.tab10(np.linspace(0, 1, len(successful_results)))
    
    for idx, result in enumerate(successful_results):
        encoder_name = result['encoder']
        history = result['history']
        
        # Plot validation accuracy (left) - convert to 0-1 range for paper style
        epochs = range(1, len(history['val_acc']) + 1)
        val_acc_normalized = [acc / 100.0 for acc in history['val_acc']]
        ax1.plot(epochs, val_acc_normalized, 
                label=f"LSTM+{encoder_name}" if encoder_name != 'lstm_only' else 'LSTM', 
                color=colors[idx], linewidth=2)
        
        # Plot validation loss (right)
        ax2.plot(epochs, history['val_loss'], 
                label=f"LSTM+{encoder_name}" if encoder_name != 'lstm_only' else 'LSTM',
                color=colors[idx], linewidth=2)
    
    # Format left plot (accuracy)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('(a) Testing Accuracy', fontsize=14)
    ax1.legend(fontsize=10, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.5, 1.0])
    
    # Format right plot (loss)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('(b) Testing Loss', fontsize=14)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Training curves saved to: {save_path}")
    plt.close()


def save_results_to_csv(results, save_path='mnist_results.csv'):
    """Save results to CSV format for easy analysis"""
    
    csv_data = []
    
    for result in results:
        if result['success']:
            row = {
                'encoder': result['encoder'],
                'best_val_acc': result['best_val_acc'],
                'final_train_acc': result['final_train_acc'],
                'final_val_acc': result['final_val_acc'],
                'num_epochs': len(result['history']['train_acc']),
                'status': 'SUCCESS'
            }
        else:
            row = {
                'encoder': result['encoder'],
                'best_val_acc': 'N/A',
                'final_train_acc': 'N/A',
                'final_val_acc': 'N/A',
                'num_epochs': 0,
                'status': f"FAILED: {result.get('error', 'Unknown error')}"
            }
        csv_data.append(row)
    
    # Write to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(save_path, index=False)
    print(f"📄 Results saved to CSV: {save_path}")
    
    return df


def save_epoch_history_to_csv(results, save_dir='mnist_epoch_history'):
    """Save epoch-by-epoch training history for each encoder"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    for result in results:
        if not result['success']:
            continue
        
        encoder_name = result['encoder']
        history = result['history']
        
        # Create CSV with epoch history
        csv_path = os.path.join(save_dir, f'{encoder_name}_history.csv')
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
            
            for epoch in range(len(history['train_loss'])):
                writer.writerow([
                    epoch + 1,
                    history['train_loss'][epoch],
                    history['train_acc'][epoch],
                    history['val_loss'][epoch],
                    history['val_acc'][epoch]
                ])
        
        print(f"  📄 Saved {encoder_name} epoch history to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Event-Based MNIST Time Encoder Comparison')
    
    # Dataset parameters
    parser.add_argument('--threshold', type=float, default=0.9,
                        help='Brightness threshold for event generation (default: 0.9, matching paper)')
    parser.add_argument('--max_events', type=int, default=None,
                        help='Maximum events per sequence (default: None = use all, matching paper)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=MAX_EPOCHS,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size (default: 512)')
    parser.add_argument('--embedding_dim', type=int, default=32,
                        help='Time embedding dimension (default: 32)')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='LSTM hidden dimension (default: 128)')
    
    parser.add_argument('--expert_dim', type=int, default=64,
                        help='Expert dimension for K-MOTE (default: 64)')
    parser.add_argument('--mamba_d_state', type=int, default=16,
                        help='Mamba state dimension (default: 16)')
    parser.add_argument('--mamba_d_conv', type=int, default=4,
                        help='Mamba convolution dimension (default: 4)')
    parser.add_argument('--mamba_expand', type=int, default=2,
                        help='Mamba expansion factor (default: 2)')
    parser.add_argument('--wavelet_type', type=str, default='shock',
                        help='Wavelet type for K-MOTE (default: shock)')
    
    # Experiment control
    parser.add_argument('--encoders', nargs='+',
                        help='Specific encoders to test (default: all available)')
    parser.add_argument('--save_results', type=str, default='mnist_time_encoder_results.json',
                        help='File to save results (default: mnist_time_encoder_results.json)')
    parser.add_argument('--experiment_dir', type=str, default='mnist_experiments',
                        help='Directory to save all experiment outputs (default: mnist_experiments)')
    
    # Checkpoint system
    parser.add_argument('--resume_training', action='store_true',
                        help='Resume training from latest checkpoint if available')
    parser.add_argument('--resume_experiment', type=str,
                        help='Resume training from specific experiment directory (e.g., mnist_experiments/run_20251017_143022)')
    parser.add_argument('--resume_encoder', type=str,
                        help='Resume training for specific encoder only (use with --resume_experiment)')
    parser.add_argument('--additional_epochs', type=int, default=MAX_EPOCHS,
                        help='Additional epochs to train when resuming (default: 50)')
    parser.add_argument('--checkpoint_every', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--no_checkpoints', action='store_true',
                        help='Disable checkpoint saving (not recommended for long training)')
    
    args = parser.parse_args()
    
    # Handle resume experiment mode
    if args.resume_experiment:
        if not os.path.exists(args.resume_experiment):
            print(f"❌ Experiment directory not found: {args.resume_experiment}")
            return
        
        if args.resume_encoder:
            # Resume specific encoder
            print(f"🔄 Resuming encoder '{args.resume_encoder}' from experiment: {args.resume_experiment}")
            checkpoint_info = resume_specific_encoder(args.resume_experiment, args.resume_encoder, args.additional_epochs)
            if checkpoint_info:
                print("✅ Use the following command to continue training:")
                print(f"python {sys.argv[0]} --resume_training --experiment_dir {args.resume_experiment} --encoders {args.resume_encoder} --epochs {checkpoint_info['completed_epochs'] + args.additional_epochs}")
            return
        else:
            # List available checkpoints
            checkpoint_dir = os.path.join(args.resume_experiment, "checkpoints")
            if os.path.exists(checkpoint_dir):
                print(f"📁 Available checkpoints in {args.resume_experiment}:")
                checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth') and 'latest' in f]
                encoders_with_checkpoints = []
                for file in checkpoint_files:
                    encoder_name = file.replace('checkpoint_', '').replace('_latest.pth', '')
                    encoders_with_checkpoints.append(encoder_name)
                    checkpoint_path = os.path.join(checkpoint_dir, file)
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    print(f"  - {encoder_name}: Epoch {checkpoint['epoch']}, Best Val Acc: {checkpoint['best_val_acc']:.2f}%")
                
                print(f"\n💡 To resume a specific encoder, use:")
                print(f"python {sys.argv[0]} --resume_experiment {args.resume_experiment} --resume_encoder <encoder_name>")
                print(f"Available encoders: {', '.join(encoders_with_checkpoints)}")
            else:
                print(f"❌ No checkpoints directory found in {args.resume_experiment}")
            return
    
    # Get available encoders
    available_encoders = get_available_encoders()
    print(f"Available encoders: {available_encoders}")
    
    # Filter encoders if specified
    if args.encoders:
        encoders_to_test = [enc for enc in args.encoders if enc in available_encoders]
        if not encoders_to_test:
            print("❌ No valid encoders specified!")
            return
    else:
        encoders_to_test = available_encoders
    '''
    # ===== TEST SIMPLE LEARNING FIRST (using first encoder in the list) =====
    print("🔬 Running preliminary learning test...")
    test_encoder = encoders_to_test[0]
    if not test_simple_classification(encoder_type=test_encoder, epochs=args.epochs):
        print(f"❌ {test_encoder} cannot learn simple patterns - check implementation!")
        print("⚠️  Proceeding with full experiments anyway (issue might be test-specific)...")
    else:
        print(f"✅ {test_encoder} can learn - proceeding with full experiments...")
    # ===== END PRELIMINARY TEST =====
    
    print(f"\n🧪 Event-Based MNIST Time Encoder Comparison")
    print(f"=" * 60)
    print(f"Threshold: {args.threshold}")
    print(f"Max Events: {args.max_events}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Embedding Dim: {args.embedding_dim}")
    print(f"Hidden Dim: {args.hidden_dim}")
    print(f"Testing {len(encoders_to_test)} encoders: {encoders_to_test}")
    '''
    # Run experiments
    results = []
    start_time = datetime.now()
    timestamp = "no_timestamp"
    
    # Create experiment directory structure
    experiment_folder = os.path.join(args.experiment_dir, f"run_{timestamp}")
    models_dir = os.path.join(experiment_folder, "models")
    history_dir = os.path.join(experiment_folder, "epoch_history")
    checkpoint_dir = os.path.join(experiment_folder, "checkpoints") if not args.no_checkpoints else None
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"📁 Experiment folder: {experiment_folder}")
    if checkpoint_dir:
        print(f"💾 Checkpoints will be saved to: {checkpoint_dir}")
        if args.resume_training:
            print("🔄 Resume training mode enabled")
    else:
        print("⚠️  Checkpoint saving disabled")
    
    print(f"\n🧪 Event-Based MNIST Time Encoder Comparison")
    print(f"=" * 60)
    print(f"Timestamp: {timestamp}")
    print(f"Threshold: {args.threshold}")
    print(f"Max Events: {args.max_events}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Embedding Dim: {args.embedding_dim}")
    print(f"Hidden Dim: {args.hidden_dim}")
    print(f"Testing {len(encoders_to_test)} encoders: {encoders_to_test}")
    
    for i, encoder_name in enumerate(encoders_to_test, 1):
        print(f"\n[{i}/{len(encoders_to_test)}] Testing {encoder_name}...")
        result = run_experiment(encoder_name, args, models_dir=models_dir, checkpoint_dir=checkpoint_dir)
        results.append(result)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Generate file paths within experiment folder
    base_name = 'mnist_time_encoder_results'
    json_path = os.path.join(experiment_folder, f"{base_name}.json")
    csv_path = os.path.join(experiment_folder, f"{base_name}.csv")
    plot_path = os.path.join(experiment_folder, f"{base_name}_curves.png")
    
    # Save results to JSON
    experiment_info = {
        'timestamp': timestamp,
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration_seconds': duration.total_seconds(),
        'parameters': vars(args),
        'results': results
    }
    
    with open(json_path, 'w') as f:
        json.dump(experiment_info, f, indent=2)
    print(f"\n💾 Results saved to JSON: {json_path}")
    
    # ✅ NEW: Save results to CSV
    print(f"\n📊 Saving results to CSV...")
    save_results_to_csv(results, csv_path)
    
    # ✅ NEW: Save epoch history for each encoder
    print(f"\n📁 Saving epoch histories...")
    save_epoch_history_to_csv(results, history_dir)
    
    # ✅ NEW: Generate plots
    print(f"\n📈 Generating training curves...")
    plot_training_curves(results, plot_path)
    
    # Print summary
    print(f"\n{'='*80}")
    print("EXPERIMENT RESULTS SUMMARY (Comparing Best Val Acc)")
    print(f"{'='*80}")
    print(f"{'Encoder':<25} {'Status':<10} {'Best Val Acc':<15}")
    print(f"{'-'*80}")
    
    successful_results = [r for r in results if r['success']]
    
    for result in results:
        if result['success']:
            status = "✅ SUCCESS"
            best_acc = f"{result['best_val_acc']:.2f}%"
        else:
            status = "❌ FAILED"
            best_acc = "N/A"
        
        print(f"{result['encoder']:<25} {status:<10} {best_acc:<15}")
    
    print(f"{'-'*80}")
    print(f"Total: {len(results)} experiments, {len(successful_results)} successful")
    print(f"Duration: {duration}")
    print(f"\n📁 Output Files:")
    print(f"  JSON:     {json_path}")
    print(f"  CSV:      {csv_path}")
    print(f"  Plot:     {plot_path}")
    print(f"  History:  {history_dir}/")
    
    if successful_results:
        # Find best encoder (by best val acc)
        best_result = max(successful_results, key=lambda x: x['best_val_acc'])
        print(f"\n🏆 Best Encoder: {best_result['encoder']}")
        print(f"   Best Val Accuracy: {best_result['best_val_acc']:.2f}%")
    
    print(f"{'='*80}")


if __name__ == '__main__':
    main()