#!/usr/bin/env python3
"""
Stack Overflow Badge Prediction Experiment
==========================================

Adapts the event-based MNIST experiment for Stack Overflow badge sequences.
Task: Next badge prediction from temporal badge sequences.

This experiment tests different time encoders on real temporal data from Stack Overflow,
where users earn badges over time. The goal is to predict the next badge a user will earn
given their historical badge sequence and timestamps.

Key differences from MNIST:
- Real temporal sequences instead of synthetic pixel events
- 22 badge types instead of 784 pixel positions  
- Variable sequence lengths
- Actual timestamps with meaningful temporal patterns
- Next badge prediction (22-class classification) instead of digit classification

Usage:
    python experiments/stackoverflow_badge_prediction.py [options]
    
Example:
    python experiments/stackoverflow_badge_prediction.py --epochs 200 --batch_size 256 --split 1
"""

# Global training configuration
MAX_EPOCHS = 400
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-2
EARLY_STOPPING_PATIENCE = 20
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
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
from bitsandbytes.optim import AdamW8bit

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import global debug control
from debug_config import should_debug_model  # 🔍 Global debug control

# Import time encoders
from models.time_encoders.factory import create_time_encoder
from models.time_encoders.ablation_encoders import (
    SMKernelOnly, KMOTEAbsOnly, KMOTERelOnly, DualStreamBaseline
)
from models.time_encoders.kan_mammote_lite import KAN_MAMMOTE_Lite
from models.time_encoders.kan_mammote import KAN_MAMMOTE

# Import FTE (Fourier Time Encoder) from GNN modules
from models.gnn_backbones.modules import TimeEncoder as FTE

# Import optional encoders
try:
    from models.time_encoders.lete_encoder import LeTE
    LETE_AVAILABLE = True
except ImportError:
    LETE_AVAILABLE = False

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


class StackOverflowDataset(Dataset):
    """
    Stack Overflow badge sequences for next badge prediction.
    
    Each sequence represents badges earned by a user over time.
    Task: Given a sequence of badges and timestamps, predict the next badge.
    """
    def __init__(self, data_dir, split_id=1, train=True, max_sequence_length=None, 
                 min_sequence_length=3, normalize_time=True):
        """
        Args:
            data_dir: Path to SO data directory
            split_id: Which data split to use (1-5)
            train: True for training data, False for test data
            max_sequence_length: Maximum sequence length (None = no limit)
            min_sequence_length: Minimum sequence length for training (default: 3)
            normalize_time: Whether to normalize timestamps
        """
        super(StackOverflowDataset, self).__init__()
        
        self.max_sequence_length = max_sequence_length
        self.min_sequence_length = min_sequence_length
        self.normalize_time = normalize_time
        
        # Load data files
        split_suffix = f"-{split_id}"
        mode = "train" if train else "test"
        
        event_file = os.path.join(data_dir, f"event{split_suffix}-{mode}.txt")
        time_file = os.path.join(data_dir, f"time{split_suffix}-{mode}.txt")
        
        if not os.path.exists(event_file) or not os.path.exists(time_file):
            raise FileNotFoundError(f"Data files not found: {event_file}, {time_file}")
        
        # Read sequences
        self.sequences = []
        self.timestamps = []
        
        with open(event_file, 'r') as f:
            event_lines = f.readlines()
        
        with open(time_file, 'r') as f:
            time_lines = f.readlines()
        
        print(f"Loading {'training' if train else 'test'} data from split {split_id}...")
        
        # Track badge statistics
        all_badges = set()
        
        for event_line, time_line in zip(event_lines, time_lines):
            # Parse badge sequence
            badges = [int(x) for x in event_line.strip().split()]
            times = [float(x) for x in time_line.strip().split()]
            
            if len(badges) != len(times):
                continue  # Skip malformed sequences
            
            if len(badges) < self.min_sequence_length:
                continue  # Skip sequences that are too short
            
            # Track all badge IDs
            all_badges.update(badges)
            
            # Truncate if needed
            if self.max_sequence_length and len(badges) > self.max_sequence_length:
                badges = badges[:self.max_sequence_length]
                times = times[:self.max_sequence_length]
            
            self.sequences.append(badges)
            self.timestamps.append(times)
        
        # Convert to tensors and normalize times if requested
        self.sequences = [torch.tensor(seq, dtype=torch.long) for seq in self.sequences]
        self.timestamps = [torch.tensor(times, dtype=torch.float) for times in self.timestamps]
        
        if self.normalize_time:
            self._normalize_timestamps()
        
        print(f"Loaded {len(self.sequences)} sequences")
        if len(self.sequences) > 0:
            seq_lengths = [len(seq) for seq in self.sequences]
            print(f"Sequence length stats: min={min(seq_lengths)}, max={max(seq_lengths)}, "
                  f"mean={np.mean(seq_lengths):.1f}")
            
            # Get badge statistics
            badge_counts = {}
            for seq in self.sequences:
                for badge in seq:
                    badge_counts[badge.item()] = badge_counts.get(badge.item(), 0) + 1
            
            unique_badges = sorted(badge_counts.keys())
            print(f"Found {len(unique_badges)} unique badges: {unique_badges}")
            print(f"Badge ID range: {min(unique_badges)} to {max(unique_badges)}")
            
            # Store dataset info for model creation
            self.num_unique_badges = len(unique_badges)
            self.max_badge_id = max(unique_badges)
            self.min_badge_id = min(unique_badges)
            
            # Create badge ID to classification index mapping
            self.badge_id_to_class = {badge_id: idx for idx, badge_id in enumerate(unique_badges)}
            self.class_to_badge_id = {idx: badge_id for badge_id, idx in self.badge_id_to_class.items()}
    
    def _normalize_timestamps(self):
        """Normalize timestamps to [0, 1] range within each sequence"""
        for i, times in enumerate(self.timestamps):
            if len(times) > 1:
                t_min, t_max = times.min(), times.max()
                if t_max > t_min:
                    self.timestamps[i] = (times - t_min) / (t_max - t_min)
                else:
                    self.timestamps[i] = torch.zeros_like(times)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Returns a training example for next badge prediction.
        
        Returns:
            input_badges: Badge sequence (excluding last badge) [seq_len-1]
            input_times: Timestamp sequence (excluding last timestamp) [seq_len-1] 
            target_badge_class: Next badge class index to predict (last badge converted to class) [1]
        """
        badges = self.sequences[idx]
        times = self.timestamps[idx]
        
        # For next badge prediction, use all but last badge as input
        # and last badge as target
        if len(badges) == 1:
            # Handle edge case of single badge
            input_badges = badges
            input_times = times
            target_badge = badges[-1]  # Predict the same badge
        else:
            input_badges = badges[:-1]
            input_times = times[:-1]
            target_badge = badges[-1]
        
        # Convert target badge ID to classification index
        target_class = self.badge_id_to_class[target_badge.item()]
        
        return input_badges, input_times, torch.tensor(target_class, dtype=torch.long)


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable sequence lengths
    
    IMPORTANT: This function pads sequences to the max length in each batch.
    This is why you see many trailing zeros in the debug output - they're padding values,
    not actual data. The 'lengths' tensor tracks the real sequence lengths.
    """
    input_badges_list = []
    input_times_list = []
    target_classes_list = []
    lengths = []
    
    for input_badges, input_times, target_class in batch:
        input_badges_list.append(input_badges)
        input_times_list.append(input_times)
        target_classes_list.append(target_class)
        lengths.append(len(input_badges))
    
    # Pad sequences to max length in batch with zeros
    # This creates trailing zeros for shorter sequences!
    padded_badges = pad_sequence(input_badges_list, batch_first=True, padding_value=0)
    padded_times = pad_sequence(input_times_list, batch_first=True, padding_value=0.0)
    
    target_classes = torch.stack(target_classes_list)
    lengths = torch.tensor(lengths, dtype=torch.long)
    
    return padded_badges, padded_times, lengths, target_classes


def create_train_val_split(dataset, val_ratio=0.3, random_seed=42):
    """
    Create train/validation split from a dataset
    
    Args:
        dataset: StackOverflowDataset instance
        val_ratio: Ratio of data to use for validation (default: 0.3)
        random_seed: Random seed for reproducible splits
        
    Returns:
        train_indices, val_indices: Lists of indices for train and validation sets
    """
    torch.manual_seed(random_seed)
    
    total_size = len(dataset)
    indices = torch.randperm(total_size).tolist()
    
    val_size = int(val_ratio * total_size)
    train_size = total_size - val_size
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    print(f"📊 Created train/val split: {train_size} train, {val_size} val")
    
    return train_indices, val_indices


class PlainLSTMEncoder(nn.Module):
    """
    Plain embedding for LSTM baseline (no time encoding).
    Treats badge IDs as categorical indices to be embedded.
    """
    def __init__(self, embedding_dim, max_badge_id=22):  # Handle the actual max badge ID
        super(PlainLSTMEncoder, self).__init__()
        # Add 1 for padding idx 0, so vocab size = max_badge_id + 1
        self.embedding = nn.Embedding(max_badge_id + 1, embedding_dim, padding_idx=0)
        
    def forward(self, x):
        """
        Args:
            x: Badge sequence [batch, seq_len]
        Returns:
            Embedded badges [batch, seq_len, embedding_dim]
        """
        return self.embedding(x)


# Create LeTE (LeTE) Absolute Time variant
class LeTEAbsoluteTime(nn.Module):
    """LeTE variant that uses absolute timestamps (direct LeTE encoding)"""
    def __init__(self, time_dim, **kwargs):
        super().__init__()
        if not LETE_AVAILABLE:
            raise ImportError("LeTE not available")
        self.lete = LeTE(time_dim=time_dim)
        
    def forward(self, x):
        """x: (batch, seq_len, 1) - absolute timestamps"""
        # Remove the last dimension if present, LeTE expects (batch, seq_len)
        if x.dim() == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        return self.lete(x.float())

# Create LeTE (LeTE) Relative Time variant  
class LeTERelativeTime(nn.Module):
    """LeTE variant that uses relative time differences (GNN pattern)"""
    def __init__(self, time_dim, **kwargs):
        super().__init__()
        if not LETE_AVAILABLE:
            raise ImportError("LeTE not available")
        self.lete = LeTE(time_dim=time_dim)
        
    def forward(self, t_abs, t_rel):
        """
        Args:
            t_abs: (batch, seq_len, 1) - absolute timestamps
            t_rel: (batch, seq_len, 1) - relative timestamps (GNN pattern)
        Returns:
            LeTE encoding of relative timestamps
        """
        # Use relative time (GNN pattern: latest_time - current_time)
        if t_rel.dim() == 3 and t_rel.shape[-1] == 1:
            t_rel = t_rel.squeeze(-1)
        return self.lete(t_rel.float())

# Create Mercer Absolute Time variant
class MercerAbsoluteTime(nn.Module):
    """Mercer variant that uses absolute timestamps (direct Mercer encoding)"""
    def __init__(self, time_dim, **kwargs):
        super().__init__()
        if not MERCER_AVAILABLE:
            raise ImportError("Mercer encoder not available")
        self.mercer = MercerTimeEncoder(time_dim=time_dim)
        
    def forward(self, x):
        """x: (batch, seq_len, 1) - absolute timestamps"""
        # Remove the last dimension if present, Mercer expects (batch, seq_len)
        if x.dim() == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        return self.mercer(x.float())

# Create Mercer Relative Time variant  
class MercerRelativeTime(nn.Module):
    """Mercer variant that uses relative time differences (GNN pattern)"""
    def __init__(self, time_dim, **kwargs):
        super().__init__()
        if not MERCER_AVAILABLE:
            raise ImportError("Mercer encoder not available")
        self.mercer = MercerTimeEncoder(time_dim=time_dim)
        
    def forward(self, t_abs, t_rel):
        """
        Args:
            t_abs: (batch, seq_len, 1) - absolute timestamps
            t_rel: (batch, seq_len, 1) - relative timestamps (GNN pattern)
        Returns:
            Mercer encoding of relative timestamps
        """
        # Use relative time (GNN pattern: latest_time - current_time)
        if t_rel.dim() == 3 and t_rel.shape[-1] == 1:
            t_rel = t_rel.squeeze(-1)
        return self.mercer(t_rel.float())

# Create FTE (Fourier Time Encoder) Absolute Time variant
class FTEAbsoluteTime(nn.Module):
    """FTE variant that uses absolute timestamps (Fourier Time Encoder from GNN modules)"""
    def __init__(self, time_dim, **kwargs):
        super().__init__()
        self.fte = FTE(time_dim=time_dim, parameter_requires_grad=True)
        
    def forward(self, x):
        """x: (batch, seq_len, 1) - absolute timestamps"""
        # Remove the last dimension if present, FTE expects (batch, seq_len)
        if x.dim() == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        return self.fte(x.float())

# Create FTE (Fourier Time Encoder) Relative Time variant  
class FTERelativeTime(nn.Module):
    """FTE variant that uses relative time differences (Fourier Time Encoder from GNN modules)"""
    def __init__(self, time_dim, **kwargs):
        super().__init__()
        self.fte = FTE(time_dim=time_dim, parameter_requires_grad=True)
        
    def forward(self, t_abs, t_rel):
        """
        Args:
            t_abs: (batch, seq_len, 1) - absolute timestamps
            t_rel: (batch, seq_len, 1) - relative timestamps (GNN pattern)
        Returns:
            FTE encoding of relative timestamps
        """
        # Use relative time (GNN pattern: latest_time - current_time)
        if t_rel.dim() == 3 and t_rel.shape[-1] == 1:
            t_rel = t_rel.squeeze(-1)
        return self.fte(t_rel.float())


class BadgePredictionModel(nn.Module):
    """
    LSTM classifier with different time encoders for badge sequence prediction
    """
    def __init__(self, encoder_type='lete', embedding_dim=32, hidden_dim=128, 
                 num_unique_badges=22, max_badge_id=22, **encoder_kwargs):
        super(BadgePredictionModel, self).__init__()
        
        self.encoder_type = encoder_type
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_unique_badges = num_unique_badges
        self.max_badge_id = max_badge_id
        
        # Badge embedding for all encoders (except lstm_only which handles this internally)
        if encoder_type != 'lstm_only':
            # Vocabulary size includes all badge IDs + padding
            self.badge_embedding = nn.Embedding(max_badge_id + 1, embedding_dim // 2, padding_idx=0)
            time_dim = embedding_dim // 2  # Split embedding between badge and time
        else:
            time_dim = embedding_dim
        
        # Create time encoder
        self.time_encoder = self._create_time_encoder(encoder_type, time_dim, max_badge_id, **encoder_kwargs)
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
        # Classification head predicts among unique badges
        self.classifier = nn.Linear(hidden_dim, num_unique_badges)
        
    def _create_time_encoder(self, encoder_type, embedding_dim, max_badge_id, **kwargs):
        """Create the appropriate time encoder"""
        if encoder_type == 'lstm_only':
            return PlainLSTMEncoder(embedding_dim, max_badge_id)
        
        # Time-based encoders that use timestamps
        elif encoder_type == 'lete':
            if not LETE_AVAILABLE:
                raise ImportError("LeTE encoder not available")
            return LeTE(time_dim=embedding_dim)
        elif encoder_type in ['lete_abs', 'lete_absolute']:
            return LeTEAbsoluteTime(time_dim=embedding_dim, **kwargs)
        elif encoder_type in ['lete_rel', 'lete_relative']:
            return LeTERelativeTime(time_dim=embedding_dim, **kwargs)
        
        elif encoder_type == 'mercer':
            if not MERCER_AVAILABLE:
                raise ImportError("Mercer encoder not available")
            return MercerTimeEncoder(time_dim=embedding_dim)
        elif encoder_type in ['mercer_abs', 'mercer_absolute']:
            return MercerAbsoluteTime(time_dim=embedding_dim, **kwargs)
        elif encoder_type in ['mercer_rel', 'mercer_relative']:
            return MercerRelativeTime(time_dim=embedding_dim, **kwargs)
        
        elif encoder_type == 'time2vec':
            if not TIME2VEC_AVAILABLE:
                raise ImportError("Time2Vec encoder not available")
            return Time2VecEncoder(time_dim=embedding_dim)
        
        elif encoder_type == 'bochner':
            if not BOCHNER_AVAILABLE:
                raise ImportError("Bochner encoder not available")
            return BochnerTimeEncoder(time_dim=embedding_dim)
        
        # FTE (Fourier Time Encoder) variants
        elif encoder_type in ['fte_abs', 'fte_absolute']:
            return FTEAbsoluteTime(time_dim=embedding_dim, **kwargs)
        elif encoder_type in ['fte_rel', 'fte_relative']:
            return FTERelativeTime(time_dim=embedding_dim, **kwargs)
        
        # K-MOTE variants
        elif encoder_type == 'k_mote_abs':
            return KMOTEAbsOnly(
                time_dim=embedding_dim,
                expert_dim=kwargs.get('expert_dim', 64),
                wavelet_type=kwargs.get('wavelet_type', 'shock'),
                embedding_dim= embedding_dim
            )
        
        elif encoder_type == 'k_mote_rel':
            return KMOTERelOnly(
                time_dim=embedding_dim,
                expert_dim=kwargs.get('expert_dim', 64),
                wavelet_type=kwargs.get('wavelet_type', 'shock'),
                embedding_dim= embedding_dim
            )
        
        elif encoder_type == 'sm_kernel_only':
            return SMKernelOnly(
                time_dim=embedding_dim,
                num_mixtures=kwargs.get('num_mixtures', 16)
            )
        
        # KAN-MAMMOTE variants
        elif encoder_type == 'kan_mammote_lite':
            return KAN_MAMMOTE_Lite(
                time_dim=embedding_dim,
                expert_dim=kwargs.get('expert_dim', 64),
                mamba_d_state=kwargs.get('mamba_d_state', 16),
                mamba_d_conv=kwargs.get('mamba_d_conv', 4),
                mamba_expand=kwargs.get('mamba_expand', 2)
            )
        
        elif encoder_type == 'kan_mammote_full':
            return KAN_MAMMOTE(
                embedding_dim=embedding_dim,  # This is the output dimension (time component)
                expert_dim=kwargs.get('expert_dim', 64),  # Fixed: use smaller expert_dim to match time_dim
                mamba_d_state=kwargs.get('mamba_d_state', 256),
                mamba_d_conv=kwargs.get('mamba_d_conv', 4),
                mamba_expand=kwargs.get('mamba_expand', 4),
                mamba_headdim = kwargs.get('mamba_headdim', 32),  # Fixed: match expert_dim for proper division
                wavelet_type=kwargs.get('wavelet_type', 'shock'),
                dropout = 0.2
            )
        
        elif encoder_type == 'dual_stream_baseline':
            return DualStreamBaseline(
                time_dim=embedding_dim,
                expert_dim=kwargs.get('expert_dim', 64),
                num_mixtures=kwargs.get('num_mixtures', 16)
            )
        
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    def _needs_both_times(self, encoder_type):
        """Check if encoder needs both absolute and relative time inputs"""
        dual_time_encoders = [
            'kan_mammote_lite', 'kan_mammote_full', 'dual_stream_baseline',
            'k_mote_abs', 'k_mote_rel', 'sm_kernel_only',  # K-MOTE variants also need both times
            'lete_rel', 'lete_relative',  # LeTE relative needs both times
            'fte_rel', 'fte_relative',  # FTE relative needs both times
            'mercer_rel', 'mercer_relative'  # Mercer relative needs both times
        ]
        return encoder_type in dual_time_encoders
    
    def forward(self, badges, times, lengths):
        """
        Forward pass for badge prediction
        
        Args:
            badges: Badge sequences [batch, seq_len]
            times: Timestamp sequences [batch, seq_len]
            lengths: Actual sequence lengths [batch]
            
        Returns:
            logits: Next badge predictions [batch, num_badges]
        """
        batch_size, seq_len = badges.shape
        device = badges.device
        
        # Check encoder type and process accordingly
        if self.encoder_type == 'lstm_only':
            # LSTM baseline uses categorical embedding of badge IDs only
            embedded = self.time_encoder(badges)  # [batch, seq_len, embedding_dim]
        
        else:
            # All other encoders: combine badge embeddings + temporal embeddings
            
            # 1. Get badge embeddings
            badge_emb = self.badge_embedding(badges)  # [batch, seq_len, embedding_dim//2]
            
            # 2. Get temporal embedings
            if self._needs_both_times(self.encoder_type):
                # Dual-time encoders need both absolute and relative timestamps
                t_abs = times.unsqueeze(-1)  # [batch, seq_len, 1]
                
                # Generate relative time using GNN pattern, respecting actual sequence lengths
                t_rel = torch.zeros_like(t_abs)
                batch_size = times.shape[0]
                
                # Method: "time until sequence end" (GNN-style recency)
                # But use REAL last time, not padding!
                for i in range(batch_size):
                    real_length = lengths[i].item()
                    if real_length > 0:
                        real_last_time = times[i, real_length - 1]  # Get actual last timestamp
                        # Compute relative times only for valid positions
                        t_rel[i, :real_length, 0] = real_last_time - times[i, :real_length]
                        # Padding positions remain 0 (already initialized)
                
                # 🔍 DEBUG: Check if times are sorted before encoder
                if not hasattr(self, '_debug_printed_sorting'):
                    print(f"\n🔍 [STACK OVERFLOW] Time Sorting Debug (FIXED):")
                    print(f"   Encoder type: {self.encoder_type}")
                    print(f"   Batch size: {times.shape[0]}, Seq len: {times.shape[1]}")
                    
                    # Check first sequence for sorting (only actual length, not padding)
                    first_seq_length = lengths[0].item()  # Get actual sequence length
                    first_seq_times = times[0, :first_seq_length].cpu().numpy()  # Only actual times
                    is_sorted = all(first_seq_times[i] <= first_seq_times[i+1] for i in range(len(first_seq_times)-1))
                    print(f"   Actual sequence length: {first_seq_length}")
                    print(f"   First sequence times (actual): {first_seq_times}")
                    print(f"   Is chronologically sorted: {is_sorted}")
                    
                    # Show the REAL last time vs padding last time
                    real_last_time = times[0, first_seq_length - 1].item()
                    padded_last_time = times[0, -1].item()
                    print(f"   Real last time: {real_last_time:.6f}")
                    print(f"   Padded last time: {padded_last_time:.6f}")
                    
                    # Show corrected relative times
                    print(f"   t_rel computation: real_last_time - current_time (corrected)")
                    print(f"   t_rel sample (first 10, actual): {t_rel[0, :10, 0].cpu().numpy()}")
                    print(f"   t_rel sample (padding area): {t_rel[0, -5:, 0].cpu().numpy()}")
                    
                    self._debug_printed_sorting = True
                
                time_emb = self.time_encoder(t_abs, t_rel)  # [batch, seq_len, embedding_dim//2]
            
            else:
                # Single-time encoders use absolute timestamps
                t_input = times.unsqueeze(-1)  # [batch, seq_len, 1]
                time_emb = self.time_encoder(t_input)  # [batch, seq_len, embedding_dim//2]
            
            # 3. Combine badge and temporal embeddings
            embedded = torch.cat([badge_emb, time_emb], dim=-1)  # [batch, seq_len, embedding_dim]
        
        # Pack sequences for LSTM (handle variable lengths)
        packed_embedded = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # LSTM forward pass
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        # Use last hidden state for classification
        # hidden: [1, batch, hidden_dim] -> [batch, hidden_dim]
        final_hidden = hidden.squeeze(0)
        
        # Predict next badge
        logits = self.classifier(final_hidden)  # [batch, num_badges]
        
        return logits


def compute_ranking_metrics(outputs, targets, k=3):
    """
    Compute ranking metrics: Recall@k and MRR
    
    Args:
        outputs: Model predictions [batch_size, num_classes]
        targets: True class indices [batch_size]
        k: Top-k for Recall@k calculation
        
    Returns:
        recall_at_k: Recall@k score
        mrr: Mean Reciprocal Rank score
    """
    batch_size = outputs.size(0)
    
    # Get top-k predictions for each sample
    _, top_k_preds = torch.topk(outputs, k, dim=1)  # [batch_size, k]
    
    # Compute Recall@k
    targets_expanded = targets.unsqueeze(1).expand(-1, k)  # [batch_size, k]
    hits_at_k = (top_k_preds == targets_expanded).any(dim=1).float()  # [batch_size]
    recall_at_k = hits_at_k.mean().item()
    
    # Compute MRR
    # Get ranks of all classes, then find rank of true class
    _, all_ranks = torch.sort(outputs, dim=1, descending=True)  # [batch_size, num_classes]
    
    # Find position of true class in sorted predictions
    mrr_scores = []
    for i in range(batch_size):
        true_class = targets[i].item()
        # Find rank of true class (1-indexed)
        rank_pos = (all_ranks[i] == true_class).nonzero(as_tuple=True)[0].item() + 1
        mrr_scores.append(1.0 / rank_pos)
    
    mrr = sum(mrr_scores) / len(mrr_scores)
    
    return recall_at_k, mrr


def train_model(model, train_loader, val_loader, num_epochs, device, encoder_name, 
                models_dir='.', checkpoint_dir=None, resume_from_checkpoint=False, 
                use_amp: bool = False):
    """
    Train the badge prediction model
    """
    # Support optional AMP
    scaler = None
    if use_amp and device.type == 'cuda':
        from torch.cuda.amp import GradScaler, autocast
        scaler = GradScaler()
        print(f"🚀 Using Automatic Mixed Precision for training")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    
    # Early stopping parameters
    patience = EARLY_STOPPING_PATIENCE
    epochs_no_improve = 0
    
    # Initialize tracking
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_recall3': [], 'val_mrr': []
    }
    best_val_mrr = 0.0
    start_epoch = 0
    
    # Resume from checkpoint if requested
    if resume_from_checkpoint and checkpoint_dir:
        checkpoint_path = find_latest_checkpoint(checkpoint_dir, encoder_name)
        if checkpoint_path:
            checkpoint_data = load_checkpoint(checkpoint_path, model, optimizer)
            start_epoch = checkpoint_data['epoch'] + 1
            history = checkpoint_data['history']
            best_val_mrr = checkpoint_data.get('best_val_mrr', checkpoint_data.get('best_val_acc', 0.0))
            print(f"🔄 Resumed from epoch {start_epoch}")
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch_idx, (badges, times, lengths, targets) in enumerate(train_pbar):
            badges = badges.to(device)
            times = times.to(device)
            lengths = lengths.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            if use_amp and scaler:
                with autocast():
                    outputs = model(badges, times, lengths.cpu())
                    loss = criterion(outputs, targets)  # targets are already class indices
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(badges, times, lengths.cpu())
                loss = criterion(outputs, targets)  # targets are already class indices
                loss.backward()
                optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
            
            # Update progress bar
            train_acc = 100.0 * train_correct / train_total
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{train_acc:.2f}%'
            })
        
        # Validation phase
        val_loss, val_acc, val_recall3, val_mrr = evaluate_model(model, val_loader, device, criterion, use_amp)
        
        # Update history
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = 100.0 * train_correct / train_total
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_recall3'].append(val_recall3)
        history['val_mrr'].append(val_mrr)
        
        # Print epoch summary
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  Val Recall@3: {val_recall3:.4f}, Val MRR: {val_mrr:.4f}')
        
        # Save checkpoint and check for improvement
        if checkpoint_dir:
            save_checkpoint(model, optimizer, epoch, history, best_val_mrr, 
                          encoder_name, checkpoint_dir)
        
        # Use MRR as primary validation metric for early stopping
        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            epochs_no_improve = 0
            
            # Save best model
            if models_dir:
                os.makedirs(models_dir, exist_ok=True)
                best_model_path = os.path.join(models_dir, f'best_{encoder_name}.pth')
                torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1
        
        # Early stopping
        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    return history, best_val_mrr


def evaluate_model(model, data_loader, device, criterion, use_amp: bool = False):
    """Evaluate model on a dataset with ranking metrics"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_recall3 = 0.0
    total_mrr = 0.0
    
    val_pbar = tqdm(data_loader, desc='[Val]')
    
    with torch.no_grad():
        for badges, times, lengths, targets in val_pbar:
            badges = badges.to(device)
            times = times.to(device)
            lengths = lengths.to(device)
            targets = targets.to(device)
            
            if use_amp:
                from torch.cuda.amp import autocast
                with autocast():
                    outputs = model(badges, times, lengths)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(badges, times, lengths)
                loss = criterion(outputs, targets)
            
            # Basic metrics
            total_loss += loss.item() * badges.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            total_correct += (predicted == targets).sum().item()
            
            # Ranking metrics
            recall3, mrr = compute_ranking_metrics(outputs, targets, k=3)
            total_recall3 += recall3 * badges.size(0)
            total_mrr += mrr * badges.size(0)
            
            # Update progress bar
            val_acc = 100.0 * total_correct / total_samples
            val_pbar.set_postfix({
                'Acc': f'{val_acc:.2f}%',
                'R@3': f'{total_recall3/total_samples:.3f}',
                'MRR': f'{total_mrr/total_samples:.3f}'
            })
    
    avg_loss = total_loss / total_samples
    acc = 100.0 * total_correct / total_samples
    avg_recall3 = total_recall3 / total_samples
    avg_mrr = total_mrr / total_samples
    
    return avg_loss, acc, avg_recall3, avg_mrr


def get_available_encoders():
    """Get list of available encoders based on imports"""
    # Always available encoders (no external dependency)
    encoders = [
        'lstm_only',
        'k_mote_abs',
        'k_mote_rel', 
        #'sm_kernel_only',
        #'dual_stream_baseline',
        #'kan_mammote_lite',
        'kan_mammote_full',
        # FTE (Fourier Time Encoder) variants
        'fte_abs',  # FTE with absolute timestamps
        'fte_rel',  # FTE with relative time differences
    ]
    
    # Optional encoders (require imports)
    if LETE_AVAILABLE:
        encoders.extend(['lete_abs', 'lete_rel'])  # LeTE with both absolute and relative variants
    
    if MERCER_AVAILABLE:
        encoders.extend(['mercer_abs', 'mercer_rel'])  # Mercer with absolute and relative variants
    '''
    if TIME2VEC_AVAILABLE:
        pass
        encoders.append('time2vec')
    
    if BOCHNER_AVAILABLE:
        pass
        encoders.append('bochner')
    '''
    return encoders


def save_checkpoint(model, optimizer, epoch, history, best_val_mrr, encoder_name, checkpoint_dir):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'best_val_mrr': best_val_mrr,
        'best_val_acc': history['val_acc'][-1] if history['val_acc'] else 0.0,  # Keep for backward compatibility
        'encoder_name': encoder_name
    }
    
    os.makedirs(checkpoint_dir, exist_ok=True)
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
        'best_val_mrr': checkpoint.get('best_val_mrr', checkpoint.get('best_val_acc', 0.0)),
        'best_val_acc': checkpoint.get('best_val_acc', 0.0),  # Keep for backward compatibility
        'encoder_name': checkpoint['encoder_name']
    }


def find_latest_checkpoint(checkpoint_dir, encoder_name):
    """Find the latest checkpoint for a given encoder"""
    latest_path = os.path.join(checkpoint_dir, f'checkpoint_{encoder_name}_latest.pth')
    
    if os.path.exists(latest_path):
        return latest_path
    
    return None


def run_experiment(encoder_name, args, models_dir='.', checkpoint_dir=None):
    """
    Run experiment for a specific encoder
    
    Data Split Strategy:
    ==================
    The Stack Overflow dataset provides pre-defined train/test splits for each split_id (1-5).
    
    Current Implementation:
    - TRAIN: Uses train=True data from the specified split_id  (~5K sequences)
    - VALIDATION: Uses train=False data from the specified split_id (~1.3K sequences)
    - TEST: Currently not used (same as validation)
    
    Recommended 3-way Split Strategy:
    - TRAIN: 70% of train=True data for model training
    - VALIDATION: 30% of train=True data for hyperparameter tuning and early stopping
    - TEST: train=False data for final unbiased evaluation
    
    The current setup uses the pre-defined test set as validation, which is acceptable 
    for comparing different time encoders, but for final evaluation, a proper 3-way 
    split should be implemented.
    """
    print(f"\n{'='*60}")
    print(f"Running Stack Overflow Badge Prediction: {encoder_name}")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.use_proper_split:
        print(f"📊 Using proper 3-way split with {args.val_ratio:.1%} validation ratio")
        
        # Load full training data
        full_train_dataset = StackOverflowDataset(
            data_dir=args.data_dir,
            split_id=args.split,
            train=True,
            max_sequence_length=args.max_sequence_length,
            min_sequence_length=args.min_sequence_length,
            normalize_time=args.normalize_time
        )
        
        # Create train/val split
        train_indices, val_indices = create_train_val_split(
            full_train_dataset, val_ratio=args.val_ratio, random_seed=42
        )
        
        # Create subset datasets
        train_dataset = Subset(full_train_dataset, train_indices)
        val_dataset = Subset(full_train_dataset, val_indices)
        
        # Also load test dataset for final evaluation (not used in this function)
        test_dataset = StackOverflowDataset(
            data_dir=args.data_dir,
            split_id=args.split,
            train=False,
            max_sequence_length=args.max_sequence_length,
            min_sequence_length=1,
            normalize_time=args.normalize_time
        )
        print(f"📊 Test dataset loaded: {len(test_dataset)} sequences (for future final evaluation)")
        
    else:
        print(f"📊 Using original split: train data for training, test data for validation")
        
        # Create datasets (original approach)
        train_dataset = StackOverflowDataset(
            data_dir=args.data_dir,
            split_id=args.split,
            train=True,
            max_sequence_length=args.max_sequence_length,
            min_sequence_length=args.min_sequence_length,
            normalize_time=args.normalize_time
        )
        
        val_dataset = StackOverflowDataset(
            data_dir=args.data_dir,
            split_id=args.split,
            train=False,
            max_sequence_length=args.max_sequence_length,
            min_sequence_length=1,  # No minimum for validation
            normalize_time=args.normalize_time
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=0
    )
    
    # Create model
    try:
        # Get dataset info (handle both regular dataset and Subset)
        if hasattr(train_dataset, 'dataset'):  # Subset
            dataset_info = train_dataset.dataset
        else:  # Regular dataset
            dataset_info = train_dataset
            
        model = BadgePredictionModel(
            encoder_type=encoder_name,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_unique_badges=dataset_info.num_unique_badges,  # Use actual number of unique badges
            max_badge_id=dataset_info.max_badge_id,  # Use actual max badge ID
            expert_dim=args.expert_dim,
            mamba_d_state=args.mamba_d_state,
            mamba_d_conv=args.mamba_d_conv,
            mamba_expand=args.mamba_expand,
            wavelet_type=args.wavelet_type,
            num_mixtures=args.num_mixtures
        )
        
        model.to(device)
        
        print(f"🔧 Model created: {encoder_name}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Unique badges: {dataset_info.num_unique_badges}")
        print(f"   Max badge ID: {dataset_info.max_badge_id}")
        print(f"   Badge ID range: {dataset_info.min_badge_id} to {dataset_info.max_badge_id}")
        
        # Collect experiment parameters
        experiment_params = collect_experiment_parameters(args, encoder_name, dataset_info)
        experiment_params['model_architecture']['num_parameters'] = sum(p.numel() for p in model.parameters())
        
        # Train model
        history, best_val_mrr = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            device=device,
            encoder_name=encoder_name,
            models_dir=models_dir,
            checkpoint_dir=checkpoint_dir,
            resume_from_checkpoint=args.resume_training,
            use_amp=args.use_amp
        )
        
        # Add final training results to experiment parameters
        experiment_params['training_results'] = {
            'best_val_mrr': best_val_mrr,
            'best_val_acc': max(history['val_acc']) if history['val_acc'] else 0.0,
            'best_val_recall3': max(history['val_recall3']) if history['val_recall3'] else 0.0,
            'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None,
            'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None,
            'epochs_trained': len(history['train_loss']),
            'training_completed': True
        }
        
        # Save experiment parameters
        if models_dir:
            params_save_path = os.path.join(models_dir, f'{encoder_name}_experiment_params.json')
            save_experiment_parameters(experiment_params, params_save_path)
        
        return {
            'encoder': encoder_name,
            'best_val_mrr': best_val_mrr,
            'best_val_acc': max(history['val_acc']) if history['val_acc'] else 0.0,
            'best_val_recall3': max(history['val_recall3']) if history['val_recall3'] else 0.0,
            'history': history,
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'experiment_parameters': experiment_params  # Include in results for potential cross-validation aggregation
        }
        
    except Exception as e:
        print(f"❌ Error with encoder {encoder_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def plot_training_curves(results, save_path='so_training_curves.png'):
    """Plot training curves for all encoders"""
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 15))
    
    for result in results:
        if result is None:
            continue
        
        encoder = result['encoder']
        history = result['history']
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Training loss
        ax1.plot(epochs, history['train_loss'], label=encoder, alpha=0.8)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Validation loss
        ax2.plot(epochs, history['val_loss'], label=encoder, alpha=0.8)
        ax2.set_title('Validation Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Training accuracy
        ax3.plot(epochs, history['train_acc'], label=encoder, alpha=0.8)
        ax3.set_title('Training Accuracy')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Validation accuracy
        ax4.plot(epochs, history['val_acc'], label=encoder, alpha=0.8)
        ax4.set_title('Validation Accuracy')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Validation Recall@3
        ax5.plot(epochs, history['val_recall3'], label=encoder, alpha=0.8)
        ax5.set_title('Validation Recall@3')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Recall@3')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Validation MRR
        ax6.plot(epochs, history['val_mrr'], label=encoder, alpha=0.8)
        ax6.set_title('Validation MRR')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('MRR')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Training curves saved to: {save_path}")


def collect_experiment_parameters(args, encoder_name, dataset_info):
    """
    Collect all experiment parameters including training config and model-specific parameters
    
    Args:
        args: Command line arguments
        encoder_name: Name of the encoder being used
        dataset_info: Dataset information object
        
    Returns:
        Dictionary containing all experiment parameters
    """
    # Basic experiment info
    experiment_params = {
        'experiment_type': 'stackoverflow_badge_prediction',
        'timestamp': datetime.now().isoformat(),
        'encoder_name': encoder_name,
        
        # Dataset parameters
        'dataset': {
            'data_dir': args.data_dir,
            'split_id': args.split,
            'max_sequence_length': args.max_sequence_length,
            'min_sequence_length': args.min_sequence_length,
            'normalize_time': args.normalize_time,
            'use_proper_split': args.use_proper_split,
            'val_ratio': args.val_ratio if args.use_proper_split else None,
            'num_unique_badges': dataset_info.num_unique_badges,
            'max_badge_id': dataset_info.max_badge_id,
            'min_badge_id': dataset_info.min_badge_id
        },
        
        # Training parameters
        'training': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'embedding_dim': args.embedding_dim,
            'hidden_dim': args.hidden_dim,
            'use_amp': args.use_amp,
            'resume_training': args.resume_training,
            'optimizer': 'AdamW',
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'early_stopping_patience': 20,
            'criterion': 'CrossEntropyLoss'
        },
        
        # Model architecture parameters
        'model_architecture': {
            'encoder_type': encoder_name,
            'base_model': 'LSTM',
            'lstm_dropout': 0.1,
            'num_lstm_layers': 1
        }
    }
    
    # Add encoder-specific parameters
    if 'k_mote' in encoder_name or 'kan_mammote' in encoder_name:
        experiment_params['k_mote_parameters'] = {
            'expert_dim': args.expert_dim,
            'wavelet_type': args.wavelet_type
        }
    
    if 'kan_mammote' in encoder_name:
        experiment_params['kan_mammote_parameters'] = {
            'mamba_d_state': args.mamba_d_state,
            'mamba_d_conv': args.mamba_d_conv,
            'mamba_expand': args.mamba_expand,
            'mamba_headdim': getattr(args, 'mamba_headdim', 32),  # Default from model
            'fusion_strategy': 'mamba',
            'dropout': 0.2
        }
    
    if 'sm_kernel' in encoder_name or 'dual_stream' in encoder_name:
        experiment_params['kernel_parameters'] = {
            'num_mixtures': args.num_mixtures
        }
    
    if 'lete' in encoder_name:
        experiment_params['lete_parameters'] = {
            'time_encoding_type': 'lete'
        }
    
    if 'mercer' in encoder_name:
        experiment_params['mercer_parameters'] = {
            'time_encoding_type': 'mercer'
        }
    
    if 'fte' in encoder_name:
        experiment_params['fte_parameters'] = {
            'time_encoding_type': 'fourier',
            'parameter_requires_grad': True
        }
    
    # Add time handling type
    if encoder_name in ['lete_rel', 'lete_relative', 'mercer_rel', 'mercer_relative', 
                       'fte_rel', 'fte_relative', 'k_mote_rel', 'kan_mammote_lite', 
                       'kan_mammote_full', 'dual_stream_baseline']:
        experiment_params['time_handling'] = 'dual_time'  # Uses both absolute and relative
    elif encoder_name == 'lstm_only':
        experiment_params['time_handling'] = 'no_time'  # Only categorical embedding
    else:
        experiment_params['time_handling'] = 'absolute_time'  # Uses absolute timestamps only
    
    return experiment_params


def save_experiment_parameters(experiment_params, save_path):
    """
    Save experiment parameters to JSON file
    
    Args:
        experiment_params: Dictionary containing all experiment parameters
        save_path: Path to save the JSON file
    """
    try:
        with open(save_path, 'w') as f:
            json.dump(experiment_params, f, indent=2, default=str)
        print(f"📋 Experiment parameters saved to: {save_path}")
    except Exception as e:
        print(f"⚠️ Failed to save experiment parameters: {str(e)}")


def save_results_to_csv(results, save_path='so_results.csv'):
    """Save experiment results to CSV"""
    if not results or all(r is None for r in results):
        print("⚠️ No valid results to save")
        return
    
    # Prepare data for CSV
    data = []
    for result in results:
        if result is None:
            continue
        
        # Handle both single-split and cross-validation results
        if 'std_val_mrr' in result:  # Cross-validation results
            data.append({
                'encoder': result['encoder'],
                'avg_val_mrr': result['best_val_mrr'],
                'std_val_mrr': result['std_val_mrr'],
                'avg_val_acc': result['best_val_acc'],
                'std_val_acc': result['std_val_acc'],
                'avg_val_recall3': result['best_val_recall3'],
                'std_val_recall3': result['std_val_recall3'],
                'num_parameters': result['num_parameters'],
                'num_splits': result['num_splits']
            })
        else:  # Single-split results
            data.append({
                'encoder': result['encoder'],
                'best_val_mrr': result['best_val_mrr'],
                'best_val_acc': result['best_val_acc'],
                'best_val_recall3': result['best_val_recall3'],
                'num_parameters': result['num_parameters'],
                'final_train_loss': result['history']['train_loss'][-1] if result['history']['train_loss'] else None,
                'final_val_loss': result['history']['val_loss'][-1] if result['history']['val_loss'] else None,
                'epochs_trained': len(result['history']['train_loss'])
            })
    
    # Save to CSV
    df = pd.DataFrame(data)
    
    # Sort by appropriate metric
    sort_column = 'avg_val_mrr' if 'avg_val_mrr' in df.columns else 'best_val_mrr'
    df = df.sort_values(sort_column, ascending=False)
    
    df.to_csv(save_path, index=False)
    
    print(f"📄 Results saved to: {save_path}")
    print("\n🏆 RESULTS SUMMARY:")
    print(df.to_string(index=False, float_format='%.4f'))


def main():
    parser = argparse.ArgumentParser(description='Stack Overflow Badge Prediction Experiment')
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, 
                        default='NeuralPointProcess-master/data/real/so',
                        help='Path to Stack Overflow data directory')
    parser.add_argument('--split', type=int, default=1, choices=[1,2,3,4,5],
                        help='Data split to use (1-5)')
    parser.add_argument('--use_all_splits', action='store_true',
                        help='Use all 5 splits for cross-validation (averages results across splits)')
    parser.add_argument('--max_sequence_length', type=int, default=100,
                        help='Maximum sequence length (default: 100)')
    parser.add_argument('--min_sequence_length', type=int, default=3,
                        help='Minimum sequence length for training (default: 3)')
    parser.add_argument('--normalize_time', action='store_true', default=True,
                        help='Normalize timestamps to [0,1] within each sequence')
    
    # Training parameters (matching MNIST experiment)
    parser.add_argument('--epochs', type=int, default=MAX_EPOCHS,
                        help='Number of training epochs (default: 200)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Time embedding dimension (default: 128)')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='LSTM hidden dimension (default: 128)')
    
    # Mixed precision option
    parser.add_argument('--use_amp', action='store_true',
                        help='Enable CUDA Automatic Mixed Precision')
    
    # Model-specific parameters
    parser.add_argument('--expert_dim', type=int, default=128,
                        help='Expert dimension for K-MOTE (default: 128)')
    parser.add_argument('--mamba_d_state', type=int, default=256,
                        help='Mamba state dimension (default: 256)')
    parser.add_argument('--mamba_d_conv', type=int, default=4,
                        help='Mamba convolution dimension (default: 4)')
    parser.add_argument('--mamba_headdim', type=int, default=32,
                        help='Mamba head dimension (default: 32)')
    parser.add_argument('--mamba_expand', type=int, default=4,
                        help='Mamba expansion factor (default: 4)')
    parser.add_argument('--wavelet_type', type=str, default='shock',
                        help='Wavelet type for K-MOTE (default: shock)')
    parser.add_argument('--num_mixtures', type=int, default=16,
                        help='Number of mixtures for SM-Kernel (default: 16)')
    
    # Experiment control
    parser.add_argument('--encoders', nargs='+',
                        help='Specific encoders to test (default: all available)')
    parser.add_argument('--save_results', type=str, default='so_badge_prediction_results.json',
                        help='File to save results (default: so_badge_prediction_results.json)')
    parser.add_argument('--experiment_dir', type=str, default='so_experiments',
                        help='Directory to save all experiment outputs')
    
    # Checkpoint system
    parser.add_argument('--resume_training', action='store_true',
                        help='Resume training from latest checkpoint if available')
    parser.add_argument('--no_checkpoints', action='store_true',
                        help='Disable checkpoint saving')
    
    # Data split strategy
    parser.add_argument('--use_proper_split', action='store_true',
                        help='Use proper 3-way split: train/val from train data, test from test data')
    parser.add_argument('--val_ratio', type=float, default=0.3,
                        help='Validation ratio when using proper split (default: 0.3)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Get available encoders
    available_encoders = get_available_encoders()
    encoders_to_test = args.encoders if args.encoders else available_encoders
    
    print(f"🧪 Stack Overflow Badge Prediction Experiment")
    print(f"📁 Data directory: {args.data_dir}")
    print(f"📊 Using split: {args.split}")
    print(f"🔧 Available encoders: {available_encoders}")
    print(f"🎯 Testing encoders: {encoders_to_test}")
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(args.experiment_dir, f"run_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    models_dir = os.path.join(experiment_dir, "models")
    checkpoint_dir = None if args.no_checkpoints else os.path.join(experiment_dir, "checkpoints")
    
    print(f"📂 Experiment directory: {experiment_dir}")
    
    # Run experiments
    results = []
    
    if args.use_all_splits:
        print(f"🔄 Running experiments across all 5 splits for cross-validation...")
        
        # Store results for each split
        all_split_results = {}
        
        for encoder_name in encoders_to_test:
            if encoder_name not in available_encoders:
                print(f"⚠️ Encoder '{encoder_name}' not available, skipping...")
                continue
            
            print(f"\n{'='*60}")
            print(f"Testing {encoder_name} across all splits")
            print(f"{'='*60}")
            
            encoder_results = []
            
            for split_id in range(1, 6):  # Splits 1-5
                print(f"\n--- {encoder_name} on Split {split_id} ---")
                
                # Temporarily change args.split for this run
                original_split = args.split
                args.split = split_id
                
                result = run_experiment(
                    encoder_name=encoder_name,
                    args=args,
                    models_dir=models_dir,
                    checkpoint_dir=checkpoint_dir
                )
                
                # Restore original split
                args.split = original_split
                
                if result:
                    result['split_id'] = split_id
                    encoder_results.append(result)
                    print(f"Split {split_id} - MRR: {result['best_val_mrr']:.4f}, Acc: {result['best_val_acc']:.2f}%")
                else:
                    print(f"Split {split_id} - FAILED")
            
            # Calculate average results across splits
            if encoder_results:
                avg_result = {
                    'encoder': encoder_name,
                    'best_val_mrr': np.mean([r['best_val_mrr'] for r in encoder_results]),
                    'best_val_acc': np.mean([r['best_val_acc'] for r in encoder_results]),
                    'best_val_recall3': np.mean([r['best_val_recall3'] for r in encoder_results]),
                    'num_parameters': encoder_results[0]['num_parameters'],  # Same across splits
                    'std_val_mrr': np.std([r['best_val_mrr'] for r in encoder_results]),
                    'std_val_acc': np.std([r['best_val_acc'] for r in encoder_results]),
                    'std_val_recall3': np.std([r['best_val_recall3'] for r in encoder_results]),
                    'num_splits': len(encoder_results),
                    'split_results': encoder_results  # Store individual split results
                }
                
                results.append(avg_result)
                all_split_results[encoder_name] = encoder_results
                
                # Save aggregated experiment parameters for cross-validation
                if encoder_results and models_dir:
                    # Use the experiment parameters from the first split as base
                    base_params = encoder_results[0]['experiment_parameters'].copy()
                    
                    # Update with cross-validation specific info
                    base_params['cross_validation'] = {
                        'use_all_splits': True,
                        'num_splits': len(encoder_results),
                        'split_ids': [r['split_id'] for r in encoder_results],
                        'individual_results': {
                            'val_mrr_per_split': [r['best_val_mrr'] for r in encoder_results],
                            'val_acc_per_split': [r['best_val_acc'] for r in encoder_results],
                            'val_recall3_per_split': [r['best_val_recall3'] for r in encoder_results]
                        }
                    }
                    
                    # Update training results with averaged values
                    base_params['training_results'] = {
                        'best_val_mrr': avg_result['best_val_mrr'],
                        'std_val_mrr': avg_result['std_val_mrr'],
                        'best_val_acc': avg_result['best_val_acc'],
                        'std_val_acc': avg_result['std_val_acc'],
                        'best_val_recall3': avg_result['best_val_recall3'],
                        'std_val_recall3': avg_result['std_val_recall3'],
                        'num_splits': avg_result['num_splits'],
                        'training_completed': True,
                        'cross_validation': True
                    }
                    
                    # Save cross-validation experiment parameters
                    cv_params_path = os.path.join(models_dir, f'{encoder_name}_cv_experiment_params.json')
                    save_experiment_parameters(base_params, cv_params_path)
                
                print(f"\n🎯 {encoder_name} Average Results:")
                print(f"   MRR: {avg_result['best_val_mrr']:.4f} ± {avg_result['std_val_mrr']:.4f}")
                print(f"   Acc: {avg_result['best_val_acc']:.2f}% ± {avg_result['std_val_acc']:.2f}%")
                print(f"   Recall@3: {avg_result['best_val_recall3']:.4f} ± {avg_result['std_val_recall3']:.4f}")
        
        # Save detailed split results
        split_results_file = os.path.join(experiment_dir, 'all_splits_detailed_results.json')
        with open(split_results_file, 'w') as f:
            json.dump(all_split_results, f, indent=2)
        print(f"📁 Detailed split results saved to: {split_results_file}")
        
    else:
        # Original single-split approach
        for encoder_name in encoders_to_test:
            if encoder_name not in available_encoders:
                print(f"⚠️ Encoder '{encoder_name}' not available, skipping...")
                continue
            
            result = run_experiment(
                encoder_name=encoder_name,
                args=args,
                models_dir=models_dir,
                checkpoint_dir=checkpoint_dir
            )
            
            if result:
                results.append(result)
    
    # Save results
    if results:
        # Save detailed results
        results_file = os.path.join(experiment_dir, args.save_results)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save CSV summary
        csv_file = os.path.join(experiment_dir, 'results_summary.csv')
        save_results_to_csv(results, csv_file)
        
        # Save comprehensive experiment configuration
        experiment_config = {
            'experiment_info': {
                'experiment_type': 'stackoverflow_badge_prediction',
                'timestamp': timestamp,
                'experiment_dir': experiment_dir,
                'use_all_splits': args.use_all_splits,
                'encoders_tested': encoders_to_test,
                'total_experiments': len([r for r in results if r is not None])
            },
            "training_parameters": {
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            },
            'global_parameters': {
                'data_dir': args.data_dir,
                'split': args.split if not args.use_all_splits else 'all_splits_1_to_5',
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'embedding_dim': args.embedding_dim,
                'hidden_dim': args.hidden_dim,
                'use_amp': args.use_amp,
                'resume_training': args.resume_training,
                'max_sequence_length': args.max_sequence_length,
                'min_sequence_length': args.min_sequence_length,
                'normalize_time': args.normalize_time,
                'use_proper_split': args.use_proper_split,
                'val_ratio': args.val_ratio
            },
            'results_summary': [{
                'encoder': r['encoder'],
                'best_val_mrr': r['best_val_mrr'],
                'best_val_acc': r['best_val_acc'],
                'num_parameters': r['num_parameters']
            } for r in results if r is not None]
        }
        
        config_file = os.path.join(experiment_dir, 'experiment_configuration.json')
        save_experiment_parameters(experiment_config, config_file)
        
        # Plot training curves
        plot_file = os.path.join(experiment_dir, 'training_curves.png')
        plot_training_curves(results, plot_file)
        
        print(f"\n✅ Experiment completed successfully!")
        print(f"📁 All outputs saved to: {experiment_dir}")
        print(f"📋 Individual experiment parameters saved to: {models_dir}/*_experiment_params.json")
        print(f"📊 Experiment configuration saved to: {config_file}")
    else:
        print("❌ No successful experiments completed")


if __name__ == '__main__':
    main()