#!/usr/bin/env python3
"""
Event-Based MNIST Time Encoder Comparison
=========================================

This experiment tests different time encoders on Event-Based MNIST classification.
Each MNIST digit is converted to a sequence of pixel positions (events), and different
time encoders learn to represent these spatial-temporal sequences.

Encoders tested:
1. LeTE (baseline)
2. Mercer Time Encoder
3. Bochner Time Encoder
4. SM-Kernel Only (ablation)
5. K-MOTE Absolute Only (ablation)
6. K-MOTE Relative Only (ablation)
7. Dual Stream Baseline (ablation)
8. KAN-MAMMOTE Lite
9. Full KAN-MAMMOTE

Usage:
    python event_based_mnist_experiment.py --epochs 50 --encoders lete kan_mammote_full
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
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

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
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


class EventBasedMNIST(Dataset):
    """
    Convert MNIST images to event sequences based on pixel brightness threshold.
    Each event is a pixel position that exceeds the threshold.
    """
    def __init__(self, root, train=True, threshold=0.9, max_events=None, transform=None, download=True):
        super(EventBasedMNIST, self).__init__()
        
        # Load MNIST dataset
        self.mnist = datasets.MNIST(root=root, train=train, transform=transform, download=download)
        self.threshold = threshold
        self.max_events = max_events  # None = use all events (matching paper)
        
        # Convert images to event sequences
        self.event_sequences = []
        self.labels = []
        
        print(f"Converting MNIST to event sequences (threshold={threshold}, max_events={max_events})...")
        for idx in tqdm(range(len(self.mnist))):
            img, label = self.mnist[idx]
            
            # Convert to tensor if needed
            if not isinstance(img, torch.Tensor):
                img = transforms.ToTensor()(img)
            
            # Flatten image and find bright pixels
            img_flat = img.view(-1)  # 28*28 = 784 pixels
            bright_pixels = torch.nonzero(img_flat > threshold).squeeze()
            
            # Handle edge cases
            if bright_pixels.dim() == 0:
                bright_pixels = bright_pixels.unsqueeze(0)
            if len(bright_pixels) == 0:
                # If no pixels above threshold, add the brightest pixel
                bright_pixels = torch.tensor([torch.argmax(img_flat)])
            
            # Sort by pixel position and optionally limit to max_events
            bright_pixels = torch.sort(bright_pixels).values
            if self.max_events is not None and len(bright_pixels) > self.max_events:
                bright_pixels = bright_pixels[:self.max_events]
            
            self.event_sequences.append(bright_pixels)
            self.labels.append(label)
        
        print(f"Created {len(self.event_sequences)} event sequences")
    
    def __len__(self):
        return len(self.event_sequences)
    
    def __getitem__(self, idx):
        return self.event_sequences[idx], self.labels[idx]


def collate_fn(batch):
    """Custom collate function to handle variable sequence lengths (matching LeTE implementation)"""
    sequences, labels = zip(*batch)
    
    # Get sequence lengths
    lengths = [len(seq) for seq in sequences]
    
    # ✅ Use pad_sequence with padding_value=0 (matching LeTE exactly)
    from torch.nn.utils.rnn import pad_sequence as torch_pad_sequence
    padded_sequences = torch_pad_sequence(sequences, batch_first=True, padding_value=0)
    
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    
    return padded_sequences, labels_tensor, lengths_tensor


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
        super(TimeEncoderClassifier, self).__init__()
        
        self.encoder_type = encoder_type
        self.embedding_dim = embedding_dim
        
        # Create time encoder based on type
        self.time_encoder = self._create_time_encoder(encoder_type, embedding_dim, **encoder_kwargs)
        
        # ===== CRITICAL FIX: INITIALIZE SM-KERNEL PROPERLY =====
        if hasattr(self.time_encoder, 'initialize_sm_kernel'):
            # Create sample relative time data for initialization (RAW pixel difference range)
            # MNIST max pixel difference is ~784, but typical differences are much smaller
            sample_t_rel = torch.linspace(0, 100, 100).unsqueeze(0).unsqueeze(-1)  # (1, 100, 1)
            self.time_encoder.initialize_sm_kernel(sample_t_rel)
            print(f"✅ Initialized {encoder_type} SM-Kernel with RAW pixel difference range")
        # ===== END CRITICAL FIX =====
        
        # ✅ Simple LSTM without dropout (matching LeTE)
        self.lstm = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_dim, 
            batch_first=True
        )
        
        # ✅ Simple linear classifier (matching LeTE)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def _create_time_encoder(self, encoder_type, embedding_dim, **kwargs):
        """Create the appropriate time encoder"""
        
        if encoder_type == 'lstm_only':
            # ✅ Plain LSTM baseline (no time encoding)
            return PlainLSTMEncoder(embedding_dim=embedding_dim, max_position=784)
        
        elif encoder_type == 'lete':
            if not LETE_AVAILABLE:
                raise ImportError("LeTE encoder not available")
            return LeTE(time_dim=embedding_dim)
            
        elif encoder_type == 'mercer':
            if not MERCER_AVAILABLE:
                raise ImportError("Mercer encoder not available")
            return MercerTimeEncoder(time_dim=embedding_dim)
            
        elif encoder_type == 'bochner':
            if not BOCHNER_AVAILABLE:
                raise ImportError("Bochner encoder not available")
            return BochnerTimeEncoder(time_dim=embedding_dim)
            
        elif encoder_type == 'sm_kernel_only':
            return SMKernelOnly(
                embedding_dim=embedding_dim,
                num_mixtures=kwargs.get('num_mixtures', 12)
            )
            
        elif encoder_type == 'kmote_abs_only':
            return KMOTEAbsOnly(
                embedding_dim=embedding_dim,
                wavelet_type=kwargs.get('wavelet_type', 'shock')
            )
            
        elif encoder_type == 'kmote_rel_only':
            return KMOTERelOnly(
                embedding_dim=embedding_dim,
                wavelet_type=kwargs.get('wavelet_type', 'shock')
            )
            
        elif encoder_type == 'dual_stream_baseline':
            return DualStreamBaseline(
                embedding_dim=embedding_dim,
                num_mixtures=kwargs.get('num_mixtures', 12),
                wavelet_type=kwargs.get('wavelet_type', 'shock')
            )
            
        elif encoder_type == 'kan_mammote_lite':
            return KAN_MAMMOTE_Lite(
                embedding_dim=embedding_dim,
                num_mixtures=kwargs.get('num_mixtures', 12),
                wavelet_type=kwargs.get('wavelet_type', 'shock')
            )
            
        elif encoder_type == 'kan_mammote_full':
            return KAN_MAMMOTE(
                embedding_dim=embedding_dim,
                expert_dim=kwargs.get('expert_dim', 64),
                num_mixtures=kwargs.get('num_mixtures', 12),
                mamba_d_state=kwargs.get('mamba_d_state', 16),
                mamba_d_conv=kwargs.get('mamba_d_conv', 4),
                mamba_expand= 4 #kwargs.get('mamba_expand', 2)
            )
            
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    def _needs_both_times(self, encoder_type):
        """Check if encoder needs both absolute and relative time"""
        dual_time_encoders = [
            'sm_kernel_only', 'kmote_abs_only', 'kmote_rel_only', 
            'dual_stream_baseline', 'kan_mammote_lite', 'kan_mammote_full'
        ]
        return encoder_type in dual_time_encoders
    
    def forward(self, x, lengths):
        batch_size, seq_len = x.shape
        
        # Check encoder type and process accordingly
        if self.encoder_type == 'lstm_only':
            # ✅ Plain LSTM: simple embedding lookup (no time encoding)
            embedded = self.time_encoder(x)  # (batch, seq_len, embedding_dim)
            
            # Debug on first batch
            if not hasattr(self, '_debug_printed'):
                print(f"🔍 DEBUG - LSTM-Only Baseline (no time encoding):")
                print(f"  Input range: [{x.min():.1f}, {x.max():.1f}]")
                print(f"  Embedding output shape: {embedded.shape}")
                print(f"  Using categorical embedding table (784 positions)")
                self._debug_printed = True
        
        elif self._needs_both_times(self.encoder_type):
            # Generate both absolute and relative time
            t_abs = x.unsqueeze(-1).float()  # (batch, seq_len, 1) - RAW pixel positions (0-783)
            
            # Generate relative time (differences between consecutive positions)
            t_rel = torch.zeros_like(t_abs)
            t_rel[:, 1:, 0] = x[:, 1:] - x[:, :-1]  # Consecutive position differences
            t_rel[:, 0, 0] = 0  # First position has no predecessor
            
            # ===== PAPER-MATCHING: USE RAW VALUES (NO NORMALIZATION) =====
            # The paper (Kazemi et al., 2019) uses RAW pixel positions as time
            # No normalization applied - encoders must learn from raw temporal dynamics
            
            # Move to same device as model
            t_abs = t_abs.to(next(self.parameters()).device)
            t_rel = t_rel.to(next(self.parameters()).device)
            
            # Debug: Print input statistics on first batch
            if not hasattr(self, '_debug_printed'):
                print(f"🔍 DEBUG - Input Statistics (RAW - matching paper):")
                print(f"  t_abs range: [{t_abs.min():.1f}, {t_abs.max():.1f}], mean: {t_abs.mean():.1f}")
                print(f"  t_rel range: [{t_rel.min():.1f}, {t_rel.max():.1f}], mean: {t_rel.mean():.1f}")
                self._debug_printed = True
            
            # Forward through time encoder with RAW values
            embedded = self.time_encoder(t_abs, t_rel)  # (batch, seq_len, embedding_dim)
            # ===== END PAPER-MATCHING =====
            
        else:
            # Single input encoders (LeTE, Mercer, Bochner)
            # These encoders expect raw pixel positions (0-784), matching the paper
            x_float = x.float().to(next(self.parameters()).device)
            
            # Debug: Print input statistics on first batch
            if not hasattr(self, '_debug_printed'):
                print(f"🔍 DEBUG - Input Statistics (RAW for {self.encoder_type} - matching paper):")
                print(f"  x_float range: [{x_float.min():.1f}, {x_float.max():.1f}], mean: {x_float.mean():.1f}")
                self._debug_printed = True
            
            embedded = self.time_encoder(x_float)  # (batch, seq_len, embedding_dim)
        
        # Pack for LSTM
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # LSTM forward
        _, (h_n, c_n) = self.lstm(packed)
        h_n = h_n[-1]  # Get last layer hidden state
        
        # ✅ Simple classification (matching LeTE)
        output = self.fc(h_n)
        return output


def get_available_encoders():
    """Get list of available encoders based on imports"""
    # lstm_only is always available (no external dependency)
    encoders = ['lstm_only', 'sm_kernel_only', 'kmote_abs_only', 'kmote_rel_only', 
                'dual_stream_baseline', 'kan_mammote_lite', 'kan_mammote_full']
    
    if LETE_AVAILABLE:
        encoders.append('lete')
    if MERCER_AVAILABLE:
        encoders.append('mercer')
    if BOCHNER_AVAILABLE:
        encoders.append('bochner')
    
    return encoders


def train_model(model, train_loader, val_loader, num_epochs, device, encoder_name, models_dir='.'):
    """Train the model and return training history (matching LeTE implementation)"""
    
    criterion = nn.CrossEntropyLoss()
    
    # ✅ Simple Adam optimizer (matching LeTE exactly)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # lr=0.001, no weight decay
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    
    print(f"\nTraining {encoder_name} encoder...")
    print(f"🔧 Using Adam optimizer: lr=0.001 (matching LeTE)")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (sequences, labels, lengths) in enumerate(train_pbar):
            sequences, labels = sequences.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # ===== GRADIENT CLIPPING FOR STABILITY =====
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # ===== END GRADIENT CLIPPING =====
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
            
            # ===== DEBUG: Check gradients on first epoch =====
            if epoch == 0 and batch_idx == 0:
                total_grad_norm = 0
                param_count = 0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        total_grad_norm += grad_norm
                        param_count += 1
                        if 'time_encoder' in name and grad_norm < 1e-6:
                            print(f"⚠️  Very small gradient for {name}: {grad_norm:.2e}")
                
                avg_grad_norm = total_grad_norm / param_count if param_count > 0 else 0
                print(f"🔍 DEBUG - Average gradient norm: {avg_grad_norm:.6f}")
            # ===== END DEBUG =====
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for sequences, labels, lengths in val_pbar:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences, lengths)
                loss = criterion(outputs, labels)
                
                # ✅ Accumulate loss properly (multiply by batch size for averaging later)
                val_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        # ✅ Calculate epoch metrics (divide by total samples, matching LeTE)
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = 100. * train_correct / train_total
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = 100. * val_correct / val_total
        
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, '
              f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%')
        
        # ✅ No early stopping - just save best model (matching LeTE)
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            # Save best model in models directory
            model_save_path = os.path.join(models_dir, f'best_model_{encoder_name}.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"🔥 New best validation accuracy: {best_val_acc:.2f}%")
            print(f"💾 Model saved to: {model_save_path}")
    
    return history, best_val_acc


def run_experiment(encoder_name, args, models_dir='.'):
    """Run experiment for a specific encoder"""
    print(f"\n{'='*60}")
    print(f"Running experiment: {encoder_name}")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = EventBasedMNIST(
        root='./data', 
        train=True, 
        threshold=args.threshold,
        max_events=args.max_events,
        download=True
    )
    
    val_dataset = EventBasedMNIST(
        root='./data', 
        train=False, 
        threshold=args.threshold,
        max_events=args.max_events,
        download=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=2
    )
    
    # Create model
    try:
        model = TimeEncoderClassifier(
            encoder_type=encoder_name,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_classes=10,
            num_mixtures=args.num_mixtures,
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
            args.epochs, device, encoder_name, models_dir=models_dir
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
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size (default: 512)')
    parser.add_argument('--embedding_dim', type=int, default=32,
                        help='Time embedding dimension (default: 32)')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='LSTM hidden dimension (default: 128)')
    
    # Encoder parameters
    parser.add_argument('--num_mixtures', type=int, default=12,
                        help='Number of mixtures for SM-Kernel (default: 12)')
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
    
    args = parser.parse_args()
    
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
    timestamp = start_time.strftime('%Y%m%d_%H%M%S')
    
    # Create experiment directory structure
    experiment_folder = os.path.join(args.experiment_dir, f"run_{timestamp}")
    models_dir = os.path.join(experiment_folder, "models")
    history_dir = os.path.join(experiment_folder, "epoch_history")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)
    print(f"📁 Experiment folder: {experiment_folder}")
    
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
        result = run_experiment(encoder_name, args, models_dir=models_dir)
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