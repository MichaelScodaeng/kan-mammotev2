#!/usr/bin/env python3

import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from bitsandbytes.optim import AdamW8bit

# Tonic for neuromorphic datasets
import tonic
from tonic.datasets import NTIDIGITS18

# Global training configuration
MAX_EPOCHS = 200

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


class NTIDIGITS18Dataset(Dataset):
    """
    Wrapper for NTIDIGITS18 neuromorphic audio dataset.
    
    The dataset contains audio spikes from the TIDIGITS spoken digit dataset
    recorded by a 64-channel silicon cochlea sensor. Each sample is a sequence
    of (t, c) tuples where t represents time and c denotes the index of active
    frequency channel at time t.
    
    According to the paper, we use single digit samples for training and testing.
    """
    
    def __init__(self, save_to, train=True, max_events=None, time_window=None, 
                 normalize_time=True, channel_as_feature=True):
        """
        Args:
            save_to: Directory to save/load dataset
            train: Whether to load training or test split
            max_events: Maximum number of events per sequence (None = use all)
            time_window: Time window in microseconds to crop events (None = use all)
            normalize_time: Whether to normalize timestamps to [0, 1]
            channel_as_feature: Whether to use channel index as feature (True) or embed it
        """
        self.max_events = max_events
        self.time_window = time_window
        self.normalize_time = normalize_time
        self.channel_as_feature = channel_as_feature
        
        # Load NTIDIGITS18 dataset with single digits only (matching paper setup)
        self.dataset = NTIDIGITS18(
            save_to=save_to, 
            train=train, 
            single_digits=True  # Use single digit samples as mentioned in paper
        )
        
        print(f"📊 NTIDIGITS18 Dataset: {len(self.dataset)} samples ({'train' if train else 'test'})")
        
        # Get a sample to understand the data structure
        sample_events, sample_target = self.dataset[0]
        print(f"📋 Sample info:")
        print(f"  - Events shape: {sample_events.shape}")
        print(f"  - Events dtype: {sample_events.dtype}")
        print(f"  - Target: {sample_target} (digit)")
        print(f"  - Event fields: {sample_events.dtype.names}")
        
        # Create digit to class mapping (0-9 for digits, plus 'oh' = 10)
        unique_targets = set()
        for i in range(min(100, len(self.dataset))):  # Sample first 100 to get target types
            _, target = self.dataset[i]
            unique_targets.add(target)
        
        self.digit_to_class = {digit: idx for idx, digit in enumerate(sorted(unique_targets))}
        self.class_to_digit = {idx: digit for digit, idx in self.digit_to_class.items()}
        self.num_classes = len(self.digit_to_class)
        
        print(f"🎯 Target mapping: {self.digit_to_class}")
        print(f"📈 Number of classes: {self.num_classes}")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        events, target = self.dataset[idx]
        
        # Extract time and channel information
        # events.dtype = [('t', '<i8'), ('x', '<i8'), ('p', '<i8')]
        # In NTIDIGITS18: 't' = time, 'x' = channel, 'p' = polarity (but audio is typically all positive)
        times = events['t'].astype(np.float64)
        channels = events['x'].astype(np.int64)
        # Note: For audio data, polarity 'p' is often not meaningful, so we focus on time and channel
        
        # Apply time window if specified
        if self.time_window is not None:
            mask = times <= self.time_window
            times = times[mask]
            channels = channels[mask]
        
        # Limit number of events if specified
        if self.max_events is not None and len(times) > self.max_events:
            # Take first max_events (could also sample randomly)
            times = times[:self.max_events]
            channels = channels[:self.max_events]
        
        # Skip empty sequences
        if len(times) == 0:
            # Return a dummy sequence with one event
            times = np.array([0.0])
            channels = np.array([0])
        
        # Normalize timestamps to [0, 1] range if requested
        if self.normalize_time and len(times) > 1:
            times = (times - times.min()) / (times.max() - times.min())
        elif self.normalize_time:
            times = np.array([0.0])  # Single event gets time 0
        
        # Create sequence representation
        if self.channel_as_feature:
            # Use channel index directly as feature (like pixel position in MNIST)
            sequence = torch.tensor(channels, dtype=torch.long)
        else:
            # Could embed channels later, but for now use direct indices
            sequence = torch.tensor(channels, dtype=torch.long)
        
        # Convert target to class index
        target_class = self.digit_to_class[target]
        
        return sequence, target_class


def custom_collate_fn(batch):
    """Custom collate function to handle variable sequence lengths"""
    sequences_list = []
    labels_list = []
    lengths = []
    
    for sequence, label in batch:
        sequences_list.append(sequence)
        labels_list.append(label)
        lengths.append(len(sequence))
    
    # Pad sequences to same length
    padded_sequences = pad_sequence(sequences_list, batch_first=True, padding_value=0)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    
    return padded_sequences, labels_tensor, lengths_tensor


class PlainLSTMEncoder(nn.Module):
    """
    Plain embedding for LSTM baseline (no time encoding).
    Treats channel indices as categorical indices to be embedded.
    """
    def __init__(self, embedding_dim, max_channels=64):
        super().__init__()
        # NTIDIGITS18 has 64 frequency channels (0-63)
        self.embedding = nn.Embedding(max_channels + 1, embedding_dim, padding_idx=0)  # +1 for padding
        
    def forward(self, x):
        # x: (batch_size, seq_len) - channel indices
        return self.embedding(x)  # (batch_size, seq_len, embedding_dim)


class TimeEncoderClassifier(nn.Module):
    """
    LSTM classifier with different time encoders for neuromorphic audio sequences
    """
    def __init__(self, encoder_type='lete', embedding_dim=32, hidden_dim=64, 
                 num_classes=11, max_channels=64, **encoder_kwargs):
        super().__init__()
        self.encoder_type = encoder_type
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.max_channels = max_channels
        
        # Create time encoder
        self.time_encoder = self._create_time_encoder(encoder_type, embedding_dim, **encoder_kwargs)
        
        # LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_dim, 
            batch_first=True,
            dropout=0.1
        )
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.1)
        
    def _create_time_encoder(self, encoder_type, embedding_dim, **kwargs):
        """Create the specified time encoder"""
        
        if encoder_type == 'lstm_only':
            return PlainLSTMEncoder(embedding_dim, self.max_channels)
        
        elif encoder_type == 'lete':
            if not LETE_AVAILABLE:
                raise ImportError("LeTE encoder not available")
            return LeTE(
                time_dim=embedding_dim,
                max_pos=self.max_channels  # Use max channels as position range
            )
        
        elif encoder_type == 'time2vec':
            if not TIME2VEC_AVAILABLE:
                raise ImportError("Time2Vec encoder not available")
            return Time2VecEncoder(time_dim=embedding_dim)
        
        elif encoder_type == 'mercer':
            if not MERCER_AVAILABLE:
                raise ImportError("Mercer encoder not available")
            return MercerTimeEncoder(time_dim=embedding_dim)
        
        elif encoder_type == 'mercer_rel':
            if not MERCER_AVAILABLE:
                raise ImportError("Mercer encoder not available")
            return MercerTimeEncoder(time_dim=embedding_dim)  # Will use relative time via t_rel
        
        elif encoder_type == 'lete_rel':
            if not LETE_AVAILABLE:
                raise ImportError("LeTE encoder not available")
            return LeTE(
                time_dim=embedding_dim,
                max_pos=self.max_channels  # Use max channels as position range
            )  # Will use relative time via t_rel
        
        elif encoder_type == 'bochner':
            if not BOCHNER_AVAILABLE:
                raise ImportError("Bochner encoder not available")
            return BochnerTimeEncoder(time_dim=embedding_dim)
        
        # K-MOTE variants
        elif encoder_type == 'k_mote_abs':
            return KMOTEAbsOnly(
                time_dim=embedding_dim,
                expert_dim=kwargs.get('expert_dim', 64),
                mamba_d_state=kwargs.get('mamba_d_state', 16),
                mamba_d_conv=kwargs.get('mamba_d_conv', 4),
                mamba_expand=kwargs.get('mamba_expand', 2),
                wavelet_type=kwargs.get('wavelet_type', 'shock')
            )
        
        elif encoder_type == 'k_mote_rel':
            return KMOTERelOnly(
                time_dim=embedding_dim,
                expert_dim=kwargs.get('expert_dim', 64),
                mamba_d_state=kwargs.get('mamba_d_state', 16),
                mamba_d_conv=kwargs.get('mamba_d_conv', 4),
                mamba_expand=kwargs.get('mamba_expand', 2),
                wavelet_type=kwargs.get('wavelet_type', 'shock')
            )
        
        elif encoder_type == 'sm_kernel_only':
            return SMKernelOnly(time_dim=embedding_dim)
        
        elif encoder_type == 'dual_stream':
            return DualStreamBaseline(time_dim=embedding_dim)
        
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
                wavelet_type=kwargs.get('wavelet_type', 'shock')
            )
        
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    def _needs_both_times(self, encoder_type):
        """Check if encoder needs both absolute and relative time inputs"""
        dual_input_encoders = ['k_mote_abs', 'k_mote_rel', 'kan_mammote_lite', 'kan_mammote_full', 'mercer_rel', 'lete_rel']
        return encoder_type in dual_input_encoders
    
    def forward(self, x, lengths):
        # x: (batch_size, seq_len) - channel indices
        # lengths: (batch_size,) - actual sequence lengths
        
        batch_size, seq_len = x.shape
        
        if self.encoder_type == 'lstm_only':
            # Direct embedding without time encoding
            embedded = self.time_encoder(x)  # (batch_size, seq_len, embedding_dim)
        
        else:
            # For neuromorphic audio, we use channel indices as "positions"
            # This is analogous to pixel positions in Event-Based MNIST
            
            if self._needs_both_times(self.encoder_type):
                # Create time inputs for K-MOTE style encoders and relative encoders
                # t_abs: absolute channel positions (normalized)
                t_abs = x.float() / self.max_channels  # Normalize to [0, 1]
                
                # t_rel: relative differences between consecutive channels
                t_rel = torch.zeros_like(t_abs)
                
                #t_rel[:, 1:] = t_abs[:, 1:] - t_abs[:, :-1]
                t_rel[:, :, 0] = t_abs[:, -1:] - t_abs  # last_time - current_time = "time until end"
                if self.encoder_type in ['mercer_rel', 'lete_rel']:
                    # For relative-only encoders, only pass t_rel
                    embedded = self.time_encoder(t_rel=t_rel)
                else:
                    # For K-MOTE style encoders, pass both
                    embedded = self.time_encoder(t_abs=t_abs, t_rel=t_rel)
            else:
                # For other encoders, use channel positions as timestamps
                timestamps = x.float() / self.max_channels  # Normalize to [0, 1]
                embedded = self.time_encoder(timestamps=timestamps)
        
        # Pack padded sequences for LSTM
        packed_embedded = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # LSTM forward pass
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        # Use last hidden state for classification
        # hidden: (1, batch_size, hidden_dim)
        last_hidden = hidden.squeeze(0)  # (batch_size, hidden_dim)
        
        # Apply dropout and classify
        last_hidden = self.dropout(last_hidden)
        logits = self.classifier(last_hidden)  # (batch_size, num_classes)
        
        return logits


def train_model(model, train_loader, val_loader, num_epochs, device, encoder_name, 
                models_dir='.', checkpoint_dir=None, resume_from_checkpoint=False, use_amp=False):
    """Train the model with proper evaluation"""
    
    # Initialize GradScaler for AMP if using CUDA
    scaler = None
    if use_amp and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        print("🔧 Using Automatic Mixed Precision")
    
    # Optimizer and loss (matching paper configuration for NTIDIGITS18)
    if device.type == 'cuda':
        optimizer = AdamW8bit(model.parameters(), lr=1e-3, weight_decay=1e-4)
        print(f"🔧 Using AdamW8bit optimizer: lr=0.001, weight_decay=1e-4")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        print(f"🔧 Using AdamW optimizer: lr=0.001, weight_decay=1e-4")
    
    criterion = nn.CrossEntropyLoss()
    
    # Early stopping parameters
    patience = 20
    epochs_no_improve = 0
    
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
            history = checkpoint_info['history']
            best_val_acc = checkpoint_info['best_val_acc']
            print(f"📂 Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.2f}%")
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch_idx, (sequences, targets, lengths) in enumerate(train_pbar):
            sequences, targets, lengths = sequences.to(device), targets.to(device), lengths.to(device)
            
            optimizer.zero_grad()
            
            if use_amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(sequences, lengths)
                    loss = criterion(outputs, targets)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(sequences, lengths)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
            
            # Update progress bar
            if batch_idx % 10 == 0:
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.2f}%'
                })
        
        # Calculate epoch metrics
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100.0 * train_correct / train_total
        
        # Validation
        val_loss, val_acc = evaluate_model(model, val_loader, device, criterion, use_amp)
        
        # Save metrics
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train: Loss={avg_train_loss:.4f}, Acc={train_acc:.2f}%')
        print(f'  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%')
        
        # Save checkpoint and check for improvement
        if checkpoint_dir:
            save_checkpoint(model, optimizer, epoch, history, best_val_acc, encoder_name, checkpoint_dir)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            print(f'  💎 New best validation accuracy: {best_val_acc:.2f}%')
        else:
            epochs_no_improve += 1
        
        # Early stopping
        if epochs_no_improve >= patience:
            print(f'⏰ Early stopping after {patience} epochs without improvement')
            break
    
    return {
        'encoder_name': encoder_name,
        'best_val_acc': best_val_acc,
        'final_train_acc': history['train_acc'][-1] if history['train_acc'] else 0,
        'final_val_acc': history['val_acc'][-1] if history['val_acc'] else 0,
        'history': history,
        'success': True
    }


def evaluate_model(model, data_loader, device, criterion, use_amp=False):
    """Evaluate model on a dataset"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    val_pbar = tqdm(data_loader, desc='[Val]')
    
    with torch.no_grad():
        for sequences, targets, lengths in val_pbar:
            sequences, targets, lengths = sequences.to(device), targets.to(device), lengths.to(device)
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(sequences, lengths)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(sequences, lengths)
                loss = criterion(outputs, targets)
            
            total_loss += loss.item() * targets.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            total_correct += (predicted == targets).sum().item()
            
            # Update progress bar
            val_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*total_correct/total_samples:.2f}%'
            })
    
    avg_loss = total_loss / total_samples
    acc = 100.0 * total_correct / total_samples
    
    return avg_loss, acc


def get_available_encoders():
    """Get list of available encoders based on imports"""
    # Always available encoders
    encoders = [
        'lstm_only',
        #'sm_kernel_only',
        #'dual_stream', 
        'k_mote_abs',
        'k_mote_rel',
        #'kan_mammote_lite',
        'kan_mammote_full'
    ]
    
    # Optional encoders
    if LETE_AVAILABLE:
        encoders.append('lete')
        encoders.append('lete_rel')
    if MERCER_AVAILABLE:
        encoders.append('mercer')
        encoders.append('mercer_rel')
    '''
    if TIME2VEC_AVAILABLE:
        encoders.append('time2vec')
    if BOCHNER_AVAILABLE:
        encoders.append('bochner')
    '''
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
        # Return the checkpoint with highest epoch number
        checkpoint_files.sort(key=lambda x: x[0], reverse=True)
        return checkpoint_files[0][1]
    
    return None


def run_experiment(encoder_name, args, models_dir='.', checkpoint_dir=None):
    """Run experiment for a specific encoder"""
    print(f"\n{'='*60}")
    print(f"Running NTIDIGITS18 experiment: {encoder_name}")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = NTIDIGITS18Dataset(
        save_to=args.data_dir,
        train=True,
        max_events=args.max_events,
        time_window=args.time_window,
        normalize_time=args.normalize_time,
        channel_as_feature=args.channel_as_feature
    )
    
    val_dataset = NTIDIGITS18Dataset(
        save_to=args.data_dir,
        train=False,
        max_events=args.max_events,
        time_window=args.time_window,
        normalize_time=args.normalize_time,
        channel_as_feature=args.channel_as_feature
    )
    
    # Ensure same number of classes
    assert train_dataset.num_classes == val_dataset.num_classes, \
        f"Class mismatch: train={train_dataset.num_classes}, val={val_dataset.num_classes}"
    
    print(f"📊 Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}")
    print(f"🎯 Number of classes: {train_dataset.num_classes}")
    
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
            num_classes=train_dataset.num_classes,
            max_channels=64,  # NTIDIGITS18 has 64 frequency channels
            expert_dim=args.expert_dim,
            mamba_d_state=args.mamba_d_state,
            mamba_d_conv=args.mamba_d_conv,
            mamba_expand=args.mamba_expand,
            wavelet_type=args.wavelet_type
        ).to(device)
        
        print(f"🏗️  Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Train model
        result = train_model(
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
        
        return result
        
    except Exception as e:
        print(f"❌ Error running {encoder_name}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'encoder_name': encoder_name,
            'error': str(e),
            'success': False
        }


def plot_training_curves(results, save_path='ntidigits18_training_curves.png'):
    """Plot training curves"""
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("❌ No successful results to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Colors for different encoders
    colors = plt.cm.tab10(np.linspace(0, 1, len(successful_results)))
    
    for idx, result in enumerate(successful_results):
        history = result['history']
        epochs = range(1, len(history['val_acc']) + 1)
        
        # Plot validation accuracy and loss
        ax1.plot(epochs, [acc/100 for acc in history['val_acc']], 
                label=result['encoder_name'], color=colors[idx], linewidth=2)
        ax2.plot(epochs, history['val_loss'], 
                label=result['encoder_name'], color=colors[idx], linewidth=2)
    
    # Format plots
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('(a) NTIDIGITS18 Validation Accuracy', fontsize=14)
    ax1.legend(fontsize=10, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.0, 1.0])
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('(b) NTIDIGITS18 Validation Loss', fontsize=14)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Training curves saved to: {save_path}")
    plt.close()


def save_results_to_csv(results, save_path='ntidigits18_results.csv'):
    """Save results to CSV format"""
    csv_data = []
    
    for result in results:
        if result['success']:
            csv_data.append({
                'encoder': result['encoder_name'],
                'best_val_acc': result['best_val_acc'],
                'final_train_acc': result['final_train_acc'],
                'final_val_acc': result['final_val_acc'],
                'success': True
            })
        else:
            csv_data.append({
                'encoder': result['encoder_name'],
                'best_val_acc': 0.0,
                'final_train_acc': 0.0,
                'final_val_acc': 0.0,
                'success': False,
                'error': result.get('error', 'Unknown error')
            })
    
    # Write to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(save_path, index=False)
    print(f"📄 Results saved to CSV: {save_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='NTIDIGITS18 Time Encoder Comparison')
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory to save/load NTIDIGITS18 dataset (default: ./data)')
    parser.add_argument('--max_events', type=int, default=None,
                        help='Maximum events per sequence (default: None = use all)')
    parser.add_argument('--time_window', type=float, default=None,
                        help='Time window in microseconds (default: None = use all)')
    parser.add_argument('--normalize_time', action='store_true', default=True,
                        help='Normalize timestamps to [0, 1] (default: True)')
    parser.add_argument('--channel_as_feature', action='store_true', default=True,
                        help='Use channel index as feature (default: True)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=MAX_EPOCHS,
                        help=f'Number of training epochs (default: {MAX_EPOCHS})')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size (default: 128, matching paper for smaller dataset)')
    parser.add_argument('--embedding_dim', type=int, default=32,
                        help='Time embedding dimension (default: 32)')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='LSTM hidden dimension (default: 64, matching paper for smaller dataset)')
    
    # Mixed precision option
    parser.add_argument('--use_amp', action='store_true',
                        help='Enable CUDA Automatic Mixed Precision')
    
    # K-MOTE specific parameters
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
    parser.add_argument('--save_results', type=str, default='ntidigits18_time_encoder_results.json',
                        help='File to save results (default: ntidigits18_time_encoder_results.json)')
    parser.add_argument('--experiment_dir', type=str, default='ntidigits18_experiments',
                        help='Directory to save all experiment outputs (default: ntidigits18_experiments)')
    
    # Checkpoint system
    parser.add_argument('--resume_training', action='store_true',
                        help='Resume training from latest checkpoint if available')
    parser.add_argument('--resume_experiment', type=str,
                        help='Resume training from specific experiment directory')
    parser.add_argument('--resume_encoder', type=str,
                        help='Resume training for specific encoder only (use with --resume_experiment)')
    parser.add_argument('--additional_epochs', type=int, default=MAX_EPOCHS,
                        help=f'Additional epochs to train when resuming (default: {MAX_EPOCHS})')
    
    args = parser.parse_args()
    
    # Create experiment directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.resume_experiment:
        experiment_dir = args.resume_experiment
        print(f"📂 Resuming experiment from: {experiment_dir}")
    else:
        experiment_dir = os.path.join(args.experiment_dir, f"run_{timestamp}")
        os.makedirs(experiment_dir, exist_ok=True)
        print(f"📂 Created experiment directory: {experiment_dir}")
    
    # Create subdirectories
    models_dir = os.path.join(experiment_dir, "models")
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    results_dir = os.path.join(experiment_dir, "results")
    
    for dir_path in [models_dir, checkpoint_dir, results_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Determine which encoders to test
    if args.encoders:
        encoders_to_test = args.encoders
        print(f"🎯 Testing specified encoders: {encoders_to_test}")
    else:
        encoders_to_test = get_available_encoders()
        print(f"🎯 Testing all available encoders: {encoders_to_test}")
    
    # Handle resuming specific encoder
    if args.resume_experiment and args.resume_encoder:
        if args.resume_encoder not in encoders_to_test:
            print(f"❌ Encoder {args.resume_encoder} not in available encoders: {encoders_to_test}")
            return
        encoders_to_test = [args.resume_encoder]
        print(f"🔄 Resuming training for encoder: {args.resume_encoder}")
    
    # Run experiments
    results = []
    
    for encoder_name in encoders_to_test:
        print(f"\n{'='*80}")
        print(f"🚀 Starting experiment: {encoder_name}")
        print(f"{'='*80}")
        
        result = run_experiment(
            encoder_name=encoder_name,
            args=args,
            models_dir=models_dir,
            checkpoint_dir=checkpoint_dir
        )
        
        results.append(result)
        
        # Save intermediate results
        import json
        results_file = os.path.join(results_dir, args.save_results)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Intermediate results saved to: {results_file}")
    
    # Generate final analysis
    print(f"\n{'='*80}")
    print("📊 EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    if successful_results:
        print(f"✅ Successful experiments: {len(successful_results)}")
        print("\n🏆 RESULTS RANKING (by best validation accuracy):")
        successful_results.sort(key=lambda x: x['best_val_acc'], reverse=True)
        
        for i, result in enumerate(successful_results, 1):
            print(f"  {i:2d}. {result['encoder_name']:20s} - {result['best_val_acc']:6.2f}%")
        
        # Save plots and CSV
        plot_training_curves(results, os.path.join(results_dir, 'ntidigits18_training_curves.png'))
        save_results_to_csv(results, os.path.join(results_dir, 'ntidigits18_results.csv'))
    
    if failed_results:
        print(f"\n❌ Failed experiments: {len(failed_results)}")
        for result in failed_results:
            print(f"  - {result['encoder_name']}: {result.get('error', 'Unknown error')}")
    
    print(f"\n📁 All results saved in: {experiment_dir}")
    print("🎉 NTIDIGITS18 experiment completed!")


if __name__ == '__main__':
    main()