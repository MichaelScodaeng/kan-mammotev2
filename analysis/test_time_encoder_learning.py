"""
Time Encoder Learning Capability Test

This script evaluates how well different time encoders can learn temporal patterns
through simple supervised tasks, without using real datasets or link prediction.

Tests include:
1. Temporal Order Prediction: Can it learn that t1 < t2?
2. Temporal Distance Regression: Can it predict |t2 - t1|?
3. Periodicity Detection: Can it identify periodic patterns?
4. Temporal Pattern Classification: Can it classify different temporal patterns?
"""

# Global training configuration
MAX_EPOCHS = 50

import os
import sys
import argparse
import time
from typing import Tuple, Dict, List
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add repo root to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.time_encoders.kan_mammote import KAN_MAMMOTE
from models.time_encoders.lete_encoder import LeTE
from models.time_encoders.mercer_encoder import MercerTimeEncoder
from models.time_encoders.time2vec_encoder import Time2VecEncoder
from models.time_encoders.bochner_encoder import BochnerTimeEncoder
from models.time_encoders.original_encoder import OriginalTimeEncoder


# ============================================================================
# Task 1: Temporal Order Prediction (HARDER VERSION)
# ============================================================================

class TemporalOrderDataset(Dataset):
    """
    HARDER Dataset for learning temporal ordering: is t1 < t2?
    
    Challenge: Timestamps are CLOSE together (small gaps), making it harder
    to distinguish ordering compared to random pairs.
    """
    
    def __init__(self, n_samples: int = 1000, time_range: float = 100.0, min_gap: float = 0.5, max_gap: float = 5.0):
        self.n_samples = n_samples
        self.time_range = time_range
        
        # Generate base timestamps
        t1 = torch.rand(n_samples) * time_range
        
        # Generate SMALL gaps to make task harder
        gap = torch.rand(n_samples) * (max_gap - min_gap) + min_gap  # gaps in [0.5, 5.0]
        direction = (torch.rand(n_samples) > 0.5).float() * 2 - 1  # ±1 randomly
        
        t2 = t1 + direction * gap
        
        # Clip to valid range
        t2 = torch.clamp(t2, 0, time_range)
        
        # Labels: 1 if t1 < t2, 0 otherwise
        labels = (t1 < t2).float()
        
        self.t1 = t1.unsqueeze(-1)  # (N, 1)
        self.t2 = t2.unsqueeze(-1)  # (N, 1)
        self.labels = labels
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.t1[idx], self.t2[idx], self.labels[idx]


class TemporalOrderModel(nn.Module):
    """Simple classifier on top of time encoder."""
    
    def __init__(self, encoder: nn.Module, time_dim: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(time_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, t1, t2):
        # Encode both timestamps
        if hasattr(self.encoder, 'k_mote'):  # KAN-MAMMOTE
            enc1 = self.encoder.k_mote(t1).squeeze(1)
            enc2 = self.encoder.k_mote(t2).squeeze(1)
        else:  # Other encoders
            # Handle OriginalEncoder which requires both t_abs and t_rel
            if isinstance(self.encoder, OriginalTimeEncoder):
                enc1 = self.encoder(t_abs=t1, t_rel=t1)
                enc2 = self.encoder(t_abs=t2, t_rel=t2)
            else:
                enc1 = self.encoder(t_rel=t1)
                enc2 = self.encoder(t_rel=t2)
        
        # Concatenate and classify
        combined = torch.cat([enc1, enc2], dim=-1)
        return self.classifier(combined).squeeze(-1)


# ============================================================================
# Task 2: Temporal Distance Regression
# ============================================================================

class TemporalDistanceDataset(Dataset):
    """Dataset for learning temporal distances: predict |t2 - t1|."""
    
    def __init__(self, n_samples: int = 1000, time_range: float = 100.0):
        self.n_samples = n_samples
        self.time_range = time_range
        
        # Generate pairs of timestamps
        t1 = torch.rand(n_samples) * time_range
        t2 = torch.rand(n_samples) * time_range
        
        # Distance (normalized to [0, 1])
        distance = torch.abs(t2 - t1) / time_range
        
        self.t1 = t1.unsqueeze(-1)
        self.t2 = t2.unsqueeze(-1)
        self.distance = distance
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.t1[idx], self.t2[idx], self.distance[idx]


class TemporalDistanceModel(nn.Module):
    """Simple regressor on top of time encoder."""
    
    def __init__(self, encoder: nn.Module, time_dim: int):
        super().__init__()
        self.encoder = encoder
        self.regressor = nn.Sequential(
            nn.Linear(time_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, t1, t2):
        # Encode both timestamps
        if hasattr(self.encoder, 'k_mote'):  # KAN-MAMMOTE
            enc1 = self.encoder.k_mote(t1).squeeze(1)
            enc2 = self.encoder.k_mote(t2).squeeze(1)
        else:
            # Handle OriginalEncoder which requires both t_abs and t_rel
            if isinstance(self.encoder, OriginalTimeEncoder):
                enc1 = self.encoder(t_abs=t1, t_rel=t1)
                enc2 = self.encoder(t_abs=t2, t_rel=t2)
            else:
                enc1 = self.encoder(t_rel=t1)
                enc2 = self.encoder(t_rel=t2)
        
        # Concatenate and regress
        combined = torch.cat([enc1, enc2], dim=-1)
        return self.regressor(combined).squeeze(-1)


# ============================================================================
# Task 3: Periodicity Detection (HARDER VERSION)
# ============================================================================

class PeriodicityDataset(Dataset):
    """
    HARDER Dataset for detecting periodic patterns.
    
    Challenge: Periodic signals have HIGH NOISE (30-70%), variable amplitude/phase,
    and non-periodic class uses random walks (not pure noise) making distinction harder.
    """
    
    def __init__(self, n_samples: int = 1000, seq_len: int = 32):
        self.n_samples = n_samples
        self.seq_len = seq_len
        
        sequences = []
        labels = []
        
        for _ in range(n_samples):
            t = np.linspace(0, 10, seq_len)
            
            if np.random.rand() < 0.5:
                # Periodic but HEAVILY NOISY
                freq = np.random.uniform(0.5, 3.0)
                amplitude = np.random.uniform(0.5, 2.0)  # Variable amplitude
                phase = np.random.uniform(0, 2*np.pi)     # Random phase
                noise_level = np.random.uniform(0.4, 0.8)  # 40-80% noise!
                
                seq = amplitude * np.sin(2*np.pi*freq*t + phase)
                seq += np.random.randn(seq_len) * noise_level
                label = 1
            else:
                # Random walk (much harder than pure noise!)
                seq = np.cumsum(np.random.randn(seq_len) * 0.5)
                # Normalize to similar range as periodic
                seq = (seq - seq.mean()) / (seq.std() + 1e-8)
                label = 0
            
            sequences.append(seq)
            labels.append(label)
        
        self.sequences = torch.FloatTensor(sequences).unsqueeze(-1)  # (N, seq_len, 1)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class PeriodicityModel(nn.Module):
    """Sequence classifier on top of time encoder."""
    
    def __init__(self, encoder: nn.Module, time_dim: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(time_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, seq):
        # seq: (B, seq_len, 1)
        if hasattr(self.encoder, 'k_mote'):  # KAN-MAMMOTE
            enc = self.encoder.k_mote(seq)  # (B, seq_len, time_dim)
        else:
            # Handle OriginalEncoder which requires both t_abs and t_rel
            if isinstance(self.encoder, OriginalTimeEncoder):
                enc = self.encoder(t_abs=seq, t_rel=seq)
            else:
                enc = self.encoder(t_rel=seq)  # (B, seq_len, time_dim)
        
        # Mean pool over sequence
        pooled = enc.mean(dim=1)  # (B, time_dim)
        return self.classifier(pooled).squeeze(-1)


# ============================================================================
# Task 4: Temporal Pattern Classification (HARDER VERSION - Realistic Temporal Processes)
# ============================================================================

class TemporalPatternDataset(Dataset):
    """
    HARDER Dataset with 4 temporal patterns based on realistic point processes.
    
    Challenge: Classes are OVERLAPPING Poisson processes with similar characteristics,
    mimicking the UMAP task where Classes 0 & 2 were indistinguishable.
    
    Classes:
    0: Slow Poisson (λ=0.3)
    1: Bursty (alternating fast/slow)
    2: Fast Poisson (λ=2.5)
    3: Non-stationary sinusoidal rate
    """
    
    def __init__(self, n_samples: int = 1000, seq_len: int = 64):
        self.n_samples = n_samples
        self.seq_len = seq_len
        
        sequences = []
        labels = []
        
        for i in range(n_samples):
            pattern_type = i % 4
            t = np.linspace(0, 10, seq_len)
            noise_level = np.random.uniform(0.3, 0.5)  # High noise
            
            if pattern_type == 0:
                # Class 0: Slow constant rate Poisson (λ=0.3)
                seq = np.random.poisson(0.3, seq_len).astype(float)
                
            elif pattern_type == 1:
                # Class 1: Bursty pattern (alternating fast/slow)
                quarter = seq_len // 4
                seq = np.zeros(seq_len)
                for q in range(4):
                    start_idx = q * quarter
                    end_idx = min((q + 1) * quarter, seq_len)
                    if q % 2 == 0:
                        # Burst periods: fast rate
                        seq[start_idx:end_idx] = np.random.poisson(4.0, end_idx - start_idx)
                    else:
                        # Quiet periods: slow rate
                        seq[start_idx:end_idx] = np.random.poisson(0.5, end_idx - start_idx)
                        
            elif pattern_type == 2:
                # Class 2: Fast constant rate Poisson (λ=2.5)
                seq = np.random.poisson(2.5, seq_len).astype(float)
                
            else:
                # Class 3: Non-stationary sinusoidal rate
                rate = 1.5 + 1.2 * np.sin(2*np.pi*t/5)
                rate = np.clip(rate, 0.3, None)
                seq = np.random.poisson(rate).astype(float)
            
            # Add substantial noise to make it harder
            seq += np.random.randn(seq_len) * noise_level
            
            sequences.append(seq)
            labels.append(pattern_type)
        
        self.sequences = torch.FloatTensor(sequences).unsqueeze(-1)  # (N, seq_len, 1)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class TemporalPatternModel(nn.Module):
    """Multi-class classifier on top of time encoder."""
    
    def __init__(self, encoder: nn.Module, time_dim: int, n_classes: int = 4):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(time_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )
    
    def forward(self, seq):
        # seq: (B, seq_len, 1)
        if hasattr(self.encoder, 'k_mote'):  # KAN-MAMMOTE
            enc = self.encoder.k_mote(seq)  # (B, seq_len, time_dim)
        else:
            # Handle OriginalEncoder which requires both t_abs and t_rel
            if isinstance(self.encoder, OriginalTimeEncoder):
                enc = self.encoder(t_abs=seq, t_rel=seq)
            else:
                enc = self.encoder(t_rel=seq)  # (B, seq_len, time_dim)
        
        # Mean pool over sequence
        pooled = enc.mean(dim=1)  # (B, time_dim)
        return self.classifier(pooled)


# ============================================================================
# Training and Evaluation Functions
# ============================================================================

def train_classification(model, train_loader, val_loader, device, epochs=MAX_EPOCHS, lr=1e-3):
    """Train a classification model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            if len(batch) == 3:  # Pairwise tasks
                t1, t2, labels = batch
                t1, t2, labels = t1.to(device), t2.to(device), labels.to(device)
                outputs = model(t1, t2)
            else:  # Sequence tasks
                seq, labels = batch
                seq, labels = seq.to(device), labels.to(device)
                outputs = model(seq)
            
            # Ensure outputs and labels have compatible shapes
            outputs = outputs.squeeze()  # (B, 1) -> (B)
            labels = labels.float()  # Ensure float type for BCE
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = (outputs > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    t1, t2, labels = batch
                    t1, t2, labels = t1.to(device), t2.to(device), labels.to(device)
                    outputs = model(t1, t2)
                else:
                    seq, labels = batch
                    seq, labels = seq.to(device), labels.to(device)
                    outputs = model(seq)
                
                # Ensure outputs and labels have compatible shapes
                outputs = outputs.squeeze()  # (B, 1) -> (B)
                labels = labels.float()  # Ensure float type for BCE
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = (outputs > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
    
    return history, best_val_acc


def train_regression(model, train_loader, val_loader, device, epochs=MAX_EPOCHS, lr=1e-3):
    """Train a regression model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    best_val_mae = float('inf')
    history = {'train_loss': [], 'train_mae': [], 'val_loss': [], 'val_mae': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        
        for t1, t2, targets in train_loader:
            t1, t2, targets = t1.to(device), t2.to(device), targets.to(device)
            outputs = model(t1, t2)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_mae += torch.abs(outputs - targets).mean().item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        
        with torch.no_grad():
            for t1, t2, targets in val_loader:
                t1, t2, targets = t1.to(device), t2.to(device), targets.to(device)
                outputs = model(t1, t2)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_mae += torch.abs(outputs - targets).mean().item()
        
        avg_train_mae = train_mae / len(train_loader)
        avg_val_mae = val_mae / len(val_loader)
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_mae'].append(avg_train_mae)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_mae'].append(avg_val_mae)
        
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Train MAE={avg_train_mae:.4f}, Val MAE={avg_val_mae:.4f}")
    
    return history, best_val_mae


def train_multiclass(model, train_loader, val_loader, device, epochs=MAX_EPOCHS, lr=1e-3):
    """Train a multi-class classification model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device)
            outputs = model(seq)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for seq, labels in val_loader:
                seq, labels = seq.to(device), labels.to(device)
                outputs = model(seq)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
    
    return history, best_val_acc


# ============================================================================
# Main Evaluation Pipeline
# ============================================================================

def get_encoder(encoder_name: str, time_dim: int, device: str):
    """Initialize time encoder by name."""
    if encoder_name == 'kan_mammote':
        encoder = KAN_MAMMOTE(
            embedding_dim=time_dim,
            expert_dim=time_dim,
            num_mixtures=12,
            mamba_d_state=64,
            mamba_expand=2,
            wavelet_type='shock',
            mamba_headdim=16
        )
    elif encoder_name == 'lete':
        encoder = LeTE(time_dim=time_dim, device=device)
    elif encoder_name == 'mercer':
        encoder = MercerTimeEncoder(time_dim=time_dim, device=device)
    elif encoder_name == 'time2vec':
        encoder = Time2VecEncoder(time_dim=time_dim, device=device)
    elif encoder_name == 'bochner':
        encoder = BochnerTimeEncoder(time_dim=time_dim, device=device)
    elif encoder_name == 'original':
        encoder = OriginalTimeEncoder(time_dim=time_dim, device=device)
    else:
        raise ValueError(f"Unknown encoder: {encoder_name}")
    
    return encoder.to(device)


def evaluate_encoder(encoder_name: str, args):
    """Evaluate a single encoder on all tasks."""
    print(f"\n{'='*80}")
    print(f"Evaluating: {encoder_name.upper()}")
    print(f"{'='*80}")
    
    device = args.device
    results = {}
    
    # Task 1: Temporal Order Prediction
    print("\n[Task 1] Temporal Order Prediction (Binary Classification)")
    print("-" * 60)
    encoder = get_encoder(encoder_name, args.time_dim, device)
    model = TemporalOrderModel(encoder, args.time_dim).to(device)
    
    train_dataset = TemporalOrderDataset(n_samples=args.train_samples)
    val_dataset = TemporalOrderDataset(n_samples=args.val_samples)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    history, best_acc = train_classification(model, train_loader, val_loader, device, args.epochs, args.lr)
    results['temporal_order'] = {'history': history, 'best_acc': best_acc}
    print(f"  Best Val Accuracy: {best_acc:.2f}%")
    
    # Task 2: Temporal Distance Regression
    print("\n[Task 2] Temporal Distance Regression")
    print("-" * 60)
    encoder = get_encoder(encoder_name, args.time_dim, device)
    model = TemporalDistanceModel(encoder, args.time_dim).to(device)
    
    train_dataset = TemporalDistanceDataset(n_samples=args.train_samples)
    val_dataset = TemporalDistanceDataset(n_samples=args.val_samples)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    history, best_mae = train_regression(model, train_loader, val_loader, device, args.epochs, args.lr)
    results['temporal_distance'] = {'history': history, 'best_mae': best_mae}
    print(f"  Best Val MAE: {best_mae:.4f}")
    
    # Task 3: Periodicity Detection
    print("\n[Task 3] Periodicity Detection (Binary Classification)")
    print("-" * 60)
    encoder = get_encoder(encoder_name, args.time_dim, device)
    model = PeriodicityModel(encoder, args.time_dim).to(device)
    
    train_dataset = PeriodicityDataset(n_samples=args.train_samples)
    val_dataset = PeriodicityDataset(n_samples=args.val_samples)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    history, best_acc = train_classification(model, train_loader, val_loader, device, args.epochs, args.lr)
    results['periodicity'] = {'history': history, 'best_acc': best_acc}
    print(f"  Best Val Accuracy: {best_acc:.2f}%")
    
    # Task 4: Temporal Pattern Classification
    print("\n[Task 4] Temporal Pattern Classification (4-class)")
    print("-" * 60)
    encoder = get_encoder(encoder_name, args.time_dim, device)
    model = TemporalPatternModel(encoder, args.time_dim).to(device)
    
    train_dataset = TemporalPatternDataset(n_samples=args.train_samples)
    val_dataset = TemporalPatternDataset(n_samples=args.val_samples)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    history, best_acc = train_multiclass(model, train_loader, val_loader, device, args.epochs, args.lr)
    results['temporal_pattern'] = {'history': history, 'best_acc': best_acc}
    print(f"  Best Val Accuracy: {best_acc:.2f}%")
    
    return results


def plot_results(all_results: Dict, save_dir: str):
    """Plot comparison across all encoders and tasks."""
    encoders = list(all_results.keys())
    tasks = ['temporal_order', 'temporal_distance', 'periodicity', 'temporal_pattern']
    task_names = ['Temporal Order', 'Temporal Distance', 'Periodicity', 'Pattern Classification']
    
    # Summary bar plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (task, task_name) in enumerate(zip(tasks, task_names)):
        ax = axes[idx]
        
        if task == 'temporal_distance':
            # MAE (lower is better)
            values = [all_results[enc][task]['best_mae'] for enc in encoders]
            ax.bar(encoders, values, color='coral')
            ax.set_ylabel('MAE (lower is better)')
            ax.set_title(f'{task_name} - Best Val MAE')
        else:
            # Accuracy (higher is better)
            values = [all_results[enc][task]['best_acc'] for enc in encoders]
            ax.bar(encoders, values, color='skyblue')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title(f'{task_name} - Best Val Accuracy')
            ax.axhline(y=50 if 'temporal_order' in task or 'periodicity' in task else 25, 
                      color='red', linestyle='--', label='Random Baseline')
            ax.legend()
        
        ax.set_xlabel('Encoder')
        ax.grid(True, alpha=0.3)
        ax.set_xticklabels(encoders, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'time_encoder_comparison.png'), dpi=150)
    print(f"\nSaved comparison plot to: {os.path.join(save_dir, 'time_encoder_comparison.png')}")
    plt.close()
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Encoder':<15} {'Temp Order':<12} {'Temp Dist':<12} {'Periodicity':<12} {'Pattern':<12}")
    print("-"*80)
    for enc in encoders:
        print(f"{enc:<15} "
              f"{all_results[enc]['temporal_order']['best_acc']:>10.2f}% "
              f"{all_results[enc]['temporal_distance']['best_mae']:>11.4f} "
              f"{all_results[enc]['periodicity']['best_acc']:>11.2f}% "
              f"{all_results[enc]['temporal_pattern']['best_acc']:>11.2f}%")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Test time encoder learning capabilities')
    parser.add_argument('--encoders', nargs='+', 
                       default=['kan_mammote', 'lete', 'mercer', 'time2vec', 'bochner', 'original'],
                       help='List of encoders to test')
    parser.add_argument('--time-dim', type=int, default=128, help='Time encoding dimension')
    parser.add_argument('--train-samples', type=int, default=2000, help='Number of training samples per task')
    parser.add_argument('--val-samples', type=int, default=500, help='Number of validation samples per task')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs per task')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--save-dir', type=str, default=os.path.join(REPO_ROOT, 'analysis', 'figs'),
                       help='Directory to save results')
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("="*80)
    print("TIME ENCODER LEARNING CAPABILITY TEST")
    print("="*80)
    print(f"Encoders to test: {', '.join(args.encoders)}")
    print(f"Time dimension: {args.time_dim}")
    print(f"Training samples per task: {args.train_samples}")
    print(f"Validation samples per task: {args.val_samples}")
    print(f"Epochs per task: {args.epochs}")
    print(f"Device: {args.device}")
    print("="*80)
    
    # Evaluate all encoders
    all_results = {}
    for encoder_name in args.encoders:
        try:
            results = evaluate_encoder(encoder_name, args)
            all_results[encoder_name] = results
        except Exception as e:
            print(f"\n[ERROR] Failed to evaluate {encoder_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Plot and summarize results
    if all_results:
        plot_results(all_results, args.save_dir)
    else:
        print("\n[ERROR] No results to plot!")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
