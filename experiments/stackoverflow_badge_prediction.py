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
- 21 badge types instead of 784 pixel positions  
- Variable sequence lengths
- Actual timestamps with meaningful temporal patterns
- Next badge prediction (21-class classification) instead of digit classification

Usage:
    python experiments/stackoverflow_badge_prediction.py [options]
    
Example:
    python experiments/stackoverflow_badge_prediction.py --epochs 200 --batch_size 256 --split 1
"""

# Global training configuration
MAX_EPOCHS = 200

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
    """Custom collate function to handle variable sequence lengths"""
    input_badges_list = []
    input_times_list = []
    target_classes_list = []
    lengths = []
    
    for input_badges, input_times, target_class in batch:
        input_badges_list.append(input_badges)
        input_times_list.append(input_times)
        target_classes_list.append(target_class)
        lengths.append(len(input_badges))
    
    # Pad sequences
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


class BadgePredictionModel(nn.Module):
    """
    LSTM classifier with different time encoders for badge sequence prediction
    """
    def __init__(self, encoder_type='lete', embedding_dim=32, hidden_dim=128, 
                 num_unique_badges=21, max_badge_id=21, **encoder_kwargs):
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
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, dropout=0.1)
        
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
        
        elif encoder_type == 'mercer':
            if not MERCER_AVAILABLE:
                raise ImportError("Mercer encoder not available")
            return MercerTimeEncoder(time_dim=embedding_dim)
        
        elif encoder_type == 'time2vec':
            if not TIME2VEC_AVAILABLE:
                raise ImportError("Time2Vec encoder not available")
            return Time2VecEncoder(time_dim=embedding_dim)
        
        elif encoder_type == 'bochner':
            if not BOCHNER_AVAILABLE:
                raise ImportError("Bochner encoder not available")
            return BochnerTimeEncoder(time_dim=embedding_dim)
        
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
                time_dim=embedding_dim,
                expert_dim=kwargs.get('expert_dim', 64),
                mamba_d_state=kwargs.get('mamba_d_state', 16),
                mamba_d_conv=kwargs.get('mamba_d_conv', 4),
                mamba_expand=kwargs.get('mamba_expand', 2),
                wavelet_type=kwargs.get('wavelet_type', 'shock'),
                embedding_dim= embedding_dim,
                dropout = 0.0
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
            'k_mote_abs', 'k_mote_rel', 'sm_kernel_only'  # K-MOTE variants also need both times
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
                
                # Generate relative time using GNN pattern: current_time - previous_time
                # This creates "time ago" values (recency), matching GNN semantics
                t_rel = torch.zeros_like(t_abs)
                # For each position i, compute how long ago the previous event happened
                t_rel[:, 1:, 0] = times[:, 1:] - times[:, :-1]  # current - previous = "time ago"
                t_rel[:, 0, 0] = 0  # First position has no previous event
                t_rel.reverse_()  # Reverse to get "time until next event"
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

    optimizer = AdamW8bit(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Early stopping parameters
    patience = 20
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
        'kan_mammote_full'
    ]
    
    # Optional encoders (require imports)
    if LETE_AVAILABLE:
        encoders.append('lete')
    
    if MERCER_AVAILABLE:
        encoders.append('mercer')
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
        
        return {
            'encoder': encoder_name,
            'best_val_mrr': best_val_mrr,
            'best_val_acc': max(history['val_acc']) if history['val_acc'] else 0.0,
            'best_val_recall3': max(history['val_recall3']) if history['val_recall3'] else 0.0,
            'history': history,
            'num_parameters': sum(p.numel() for p in model.parameters())
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
    df = df.sort_values('best_val_mrr', ascending=False)  # Sort by MRR instead of accuracy
    df.to_csv(save_path, index=False)
    
    print(f"📄 Results saved to: {save_path}")
    print("\n🏆 RESULTS SUMMARY:")
    print(df.to_string(index=False, float_format='%.3f'))


def main():
    parser = argparse.ArgumentParser(description='Stack Overflow Badge Prediction Experiment')
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, 
                        default='NeuralPointProcess-master/data/real/so',
                        help='Path to Stack Overflow data directory')
    parser.add_argument('--split', type=int, default=1, choices=[1,2,3,4,5],
                        help='Data split to use (1-5)')
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
    parser.add_argument('--embedding_dim', type=int, default=32,
                        help='Time embedding dimension (default: 32)')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='LSTM hidden dimension (default: 128)')
    
    # Mixed precision option
    parser.add_argument('--use_amp', action='store_true',
                        help='Enable CUDA Automatic Mixed Precision')
    
    # Model-specific parameters
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
        
        # Plot training curves
        plot_file = os.path.join(experiment_dir, 'training_curves.png')
        plot_training_curves(results, plot_file)
        
        print(f"\n✅ Experiment completed successfully!")
        print(f"📁 All outputs saved to: {experiment_dir}")
    else:
        print("❌ No successful experiments completed")


if __name__ == '__main__':
    main()