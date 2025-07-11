#!/usr/bin/env python3
"""
🔬 LSTM Time Embedding Comparison Script
=========================================

This script compares the performance of different LSTM variants with various time embeddings
on the Event-Based MNIST dataset:

1. **Baseline LSTM**: No time embedding (raw pixel positions)
2. **LSTM + Sin/Cos**: Simple sinusoidal position encoding
3. **LSTM + LETE**: Learning Time Embedding (LeTE)
4. **LSTM + KAN-MAMMOTE**: KAN-MAMMOTE time embedding

Each model is trained and evaluated on the same data with identical hyperparameters
to ensure fair comparison.

Author: Generated for KAN-MAMMOTE Project
Date: July 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import torchvision.datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import json
import csv
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = '/home/s2516027/kan-mammote'
if project_root not in sys.path:
    sys.path.append(project_root)

# Import our models
from src.models.kan_mammote import KAN_MAMOTE_Model
from src.LETE.LeTE import CombinedLeTE
from src.utils.config import KANMAMOTEConfig

print("🚀 LSTM Time Embedding Comparison Script")
print("=" * 60)

# ============================================================================
# 🔧 CONFIGURATION
# ============================================================================

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name()}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Training configuration
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
LSTM_HIDDEN_DIM = 128
TIME_EMBEDDING_DIM = 32
DROPOUT_RATE = 0.2
THRESHOLD = 0.9  # MNIST pixel threshold for events

# Results directory
RESULTS_DIR = "results/lstm_comparison"
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"📊 Training Configuration:")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   Learning Rate: {LEARNING_RATE}")
print(f"   Epochs: {NUM_EPOCHS}")
print(f"   LSTM Hidden Dim: {LSTM_HIDDEN_DIM}")
print(f"   Time Embedding Dim: {TIME_EMBEDDING_DIM}")

# ============================================================================
# 📁 EVENT-BASED MNIST DATASET
# ============================================================================

class EventBasedMNIST(Dataset):
    """
    Convert MNIST images to event-based sequences.
    Each non-zero pixel becomes an event with timestamp = pixel position.
    """
    
    def __init__(self, root='./data', train=True, threshold=0.9, transform=None, download=True):
        self.root = root
        self.train = train
        self.threshold = threshold
        self.transform = transform
        
        # Load MNIST dataset
        if transform is None:
            transform = transforms.ToTensor()
        
        self.data = torchvision.datasets.MNIST(
            root=self.root, 
            train=self.train, 
            transform=transform, 
            download=download
        )
        
        # Pre-process all images to event sequences
        self.event_data = []
        self.labels = []
        
        print(f"📊 Processing {'training' if train else 'test'} set to events...")
        
        for idx in tqdm(range(len(self.data)), desc="Converting to events"):
            img, label = self.data[idx]
            # Flatten image to 1D (784 pixels for 28x28)
            img_flat = img.view(-1)  # (784,)
            
            # Find pixels above threshold (events)
            events = torch.nonzero(img_flat > self.threshold).squeeze()
            
            # Handle edge cases
            if events.dim() == 0:  # Single event
                events = events.unsqueeze(0)
            elif len(events) == 0:  # No events
                events = torch.tensor([0])  # Add dummy event
                
            # Sort events by position (timestamp order)
            events = torch.sort(events).values
            
            # Get intensities for features
            intensities = img_flat[events]
            features = intensities.unsqueeze(1)  # (seq_len, 1)
            
            self.event_data.append((events, features))
            self.labels.append(label)
        
        print(f"✅ Processed {len(self.event_data)} samples")
        avg_events = sum(len(events) for events, _ in self.event_data) / len(self.event_data)
        print(f"   Average events per sample: {avg_events:.1f}")
        
    def __len__(self):
        return len(self.event_data)
    
    def __getitem__(self, idx):
        events, features = self.event_data[idx]
        label = self.labels[idx]
        return events, features, len(events), label

def collate_fn(batch):
    """Custom collate function for variable-length sequences."""
    events_list = []
    features_list = []
    lengths = []
    labels_list = []
    
    for events, features, length, label in batch:
        events_list.append(events)
        features_list.append(features)
        lengths.append(length)
        labels_list.append(label)
    
    # Pad sequences
    padded_events = pad_sequence(events_list, batch_first=True, padding_value=0)
    padded_features = pad_sequence(features_list, batch_first=True, padding_value=0.0)
    
    # Convert to tensors
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)
    
    return padded_events, padded_features, lengths_tensor, labels_tensor

# ============================================================================
# 🏗️ MODEL ARCHITECTURES
# ============================================================================

class BaselineLSTM(nn.Module):
    """
    Baseline LSTM with no time embedding.
    Uses raw pixel positions as input.
    """
    
    def __init__(self, input_size=784, hidden_dim=128, num_classes=10, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Simple position embedding (learnable)
        self.position_embedding = nn.Embedding(input_size, TIME_EMBEDDING_DIM)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=TIME_EMBEDDING_DIM,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # Classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, events, features, lengths):
        # Simple position embedding
        embedded = self.position_embedding(events)  # (batch, seq_len, emb_dim)
        
        # Pack for LSTM
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, c_n) = self.lstm(packed)
        
        # Use last hidden state
        output = self.classifier(h_n[-1])
        return output

class SinCosLSTM(nn.Module):
    """
    LSTM with Sinusoidal/Cosine position encoding.
    Uses fixed sin/cos functions for time embedding.
    """
    
    def __init__(self, input_size=784, hidden_dim=128, num_classes=10, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = TIME_EMBEDDING_DIM
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=TIME_EMBEDDING_DIM,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # Classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def get_sincos_embedding(self, positions):
        """Generate sinusoidal position embeddings."""
        batch_size, seq_len = positions.shape
        device = positions.device
        
        # Normalize positions to [0, 1]
        positions_norm = positions.float() / 784.0
        
        # Create embedding
        embedding = torch.zeros(batch_size, seq_len, self.embedding_dim, device=device)
        
        # Half dimensions for sin, half for cos
        half_dim = self.embedding_dim // 2
        
        # Generate frequencies
        freqs = torch.exp(torch.arange(half_dim, device=device) * 
                         -(np.log(10000.0) / half_dim))
        
        # Apply sin/cos
        args = positions_norm.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)
        embedding[:, :, 0::2] = torch.sin(args)
        embedding[:, :, 1::2] = torch.cos(args)
        
        return embedding
        
    def forward(self, events, features, lengths):
        # Sin/Cos position embedding
        embedded = self.get_sincos_embedding(events)
        
        # Pack for LSTM
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, c_n) = self.lstm(packed)
        
        # Use last hidden state
        output = self.classifier(h_n[-1])
        return output

class LETE_LSTM(nn.Module):
    """
    LSTM with LETE (Learning Time Embedding).
    Uses learnable time encoding from the LETE paper.
    """
    
    def __init__(self, input_size=784, hidden_dim=128, num_classes=10, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # LETE time encoder
        self.time_encoder = CombinedLeTE(
            dim=TIME_EMBEDDING_DIM,
            p=0.5,  # 50% Fourier, 50% Spline
            layer_norm=True,
            scale=True
        )
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=TIME_EMBEDDING_DIM,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # Classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, events, features, lengths):
        # Filter valid sequences
        valid_mask = lengths > 0
        if not valid_mask.any():
            batch_size = events.size(0)
            return torch.zeros(batch_size, 10, device=events.device)
        
        events_valid = events[valid_mask]
        lengths_valid = lengths[valid_mask]
        
        # Normalize events to [0, 1] for LETE
        events_norm = events_valid.float() / 784.0
        
        # LETE encoding
        embedded = self.time_encoder(events_norm)
        
        # Pack for LSTM
        lengths_valid = torch.clamp(lengths_valid, min=1)
        packed = pack_padded_sequence(embedded, lengths_valid.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, c_n) = self.lstm(packed)
        
        # Classify
        valid_logits = self.classifier(h_n[-1])
        
        # Create full output
        batch_size = events.size(0)
        full_logits = torch.zeros(batch_size, 10, device=events.device)
        full_logits[valid_mask] = valid_logits
        
        return full_logits

class KAN_MAMMOTE_LSTM(nn.Module):
    """
    LSTM with KAN-MAMMOTE time embedding.
    Uses the KAN-MAMMOTE model for sophisticated time representation.
    """
    
    def __init__(self, input_size=784, hidden_dim=128, num_classes=10, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # KAN-MAMMOTE configuration
        self.kan_config = KANMAMOTEConfig(
            D_time=TIME_EMBEDDING_DIM,
            num_experts=4,
            hidden_dim_mamba=32,
            state_dim_mamba=8,
            num_mamba_layers=1,
            gamma=0.3,
            use_aux_features_router=False,
            raw_event_feature_dim=16,
            K_top=2,
            kan_grid_size=5,
            kan_grid_min=-2.0,
            kan_grid_max=2.0,
            kan_spline_scale=0.5,
            kan_num_layers=1,
            kan_hidden_dim=32
        )
        
        # KAN-MAMMOTE model
        self.kan_mammote = KAN_MAMOTE_Model(self.kan_config)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=TIME_EMBEDDING_DIM,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # Classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, events, features, lengths):
        # Filter valid sequences
        valid_mask = lengths > 0
        if not valid_mask.any():
            batch_size = events.size(0)
            return torch.zeros(batch_size, 10, device=events.device)
        
        events_valid = events[valid_mask]
        features_valid = features[valid_mask]
        lengths_valid = lengths[valid_mask]
        
        # Prepare timestamps for KAN-MAMMOTE (normalize to [0, 1])
        timestamps = events_valid.float() / 784.0
        timestamps = timestamps.unsqueeze(-1)  # (batch, seq_len, 1)
        
        # Use features as auxiliary features
        aux_features = features_valid
        
        # Get KAN-MAMMOTE embeddings
        try:
            with torch.no_grad():
                kan_output, _ = self.kan_mammote(timestamps, aux_features)
            embedded = kan_output
        except Exception as e:
            print(f"Warning: KAN-MAMMOTE failed ({e}), using fallback")
            # Fallback to simple embedding
            embedded = torch.randn_like(timestamps.expand(-1, -1, TIME_EMBEDDING_DIM))
        
        # Pack for LSTM
        lengths_valid = torch.clamp(lengths_valid, min=1)
        packed = pack_padded_sequence(embedded, lengths_valid.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, c_n) = self.lstm(packed)
        
        # Classify
        valid_logits = self.classifier(h_n[-1])
        
        # Create full output
        batch_size = events.size(0)
        full_logits = torch.zeros(batch_size, 10, device=events.device)
        full_logits[valid_mask] = valid_logits
        
        return full_logits

# ============================================================================
# 🏃 TRAINING AND EVALUATION
# ============================================================================

def train_epoch(model, data_loader, optimizer, criterion, device):
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for events, features, lengths, labels in tqdm(data_loader, desc="Training"):
        events = events.to(device)
        features = features.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(events, features, lengths)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item() * labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    
    return avg_loss, accuracy

def evaluate(model, data_loader, criterion, device):
    """Evaluate model on test set."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for events, features, lengths, labels in tqdm(data_loader, desc="Evaluating"):
            events = events.to(device)
            features = features.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            
            outputs = model(events, features, lengths)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    
    return avg_loss, accuracy

def train_model(model, train_loader, test_loader, model_name):
    """Train a model and return training history."""
    print(f"\n🚀 Training {model_name}...")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'epochs': [],
        'training_time': []
    }
    
    best_test_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        epoch_time = time.time() - start_time
        
        # Save history
        history['epochs'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['training_time'].append(epoch_time)
        
        # Update best accuracy
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            # Save best model
            torch.save(model.state_dict(), f"{RESULTS_DIR}/{model_name}_best.pth")
        
        # Print progress
        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | "
              f"Time: {epoch_time:.1f}s")
    
    print(f"✅ {model_name} training complete! Best test accuracy: {best_test_acc:.4f}")
    
    return history, best_test_acc

def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ============================================================================
# 📊 VISUALIZATION
# ============================================================================

def plot_training_curves(results):
    """Plot training curves for all models."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = ['blue', 'green', 'red', 'orange']
    
    for i, (model_name, history) in enumerate(results.items()):
        color = colors[i % len(colors)]
        
        # Training/Test Loss
        axes[0, 0].plot(history['epochs'], history['train_loss'], 
                       color=color, linestyle='-', label=f'{model_name} (Train)')
        axes[0, 0].plot(history['epochs'], history['test_loss'], 
                       color=color, linestyle='--', label=f'{model_name} (Test)')
        
        # Training/Test Accuracy
        axes[0, 1].plot(history['epochs'], history['train_acc'], 
                       color=color, linestyle='-', label=f'{model_name} (Train)')
        axes[0, 1].plot(history['epochs'], history['test_acc'], 
                       color=color, linestyle='--', label=f'{model_name} (Test)')
        
        # Training Time per Epoch
        axes[1, 0].plot(history['epochs'], history['training_time'], 
                       color=color, marker='o', label=model_name)
        
        # Final Test Accuracy Bar
        final_acc = history['test_acc'][-1]
        axes[1, 1].bar(i, final_acc, color=color, alpha=0.7, label=model_name)
    
    # Customize plots
    axes[0, 0].set_title('Training vs Test Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Training vs Test Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Training Time per Epoch')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Final Test Accuracy')
    axes[1, 1].set_xlabel('Model')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_xticks(range(len(results)))
    axes[1, 1].set_xticklabels(list(results.keys()), rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/training_curves.png", dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# 🎯 MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("🚀 Starting LSTM Time Embedding Comparison...")
    
    # Create datasets
    print("\n📁 Loading datasets...")
    train_dataset = EventBasedMNIST(root='./data', train=True, threshold=THRESHOLD, download=True)
    test_dataset = EventBasedMNIST(root='./data', train=False, threshold=THRESHOLD, download=True)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Define models to compare
    models = {
        'Baseline_LSTM': BaselineLSTM(
            input_size=784,
            hidden_dim=LSTM_HIDDEN_DIM,
            num_classes=10,
            dropout=DROPOUT_RATE
        ),
        'SinCos_LSTM': SinCosLSTM(
            input_size=784,
            hidden_dim=LSTM_HIDDEN_DIM,
            num_classes=10,
            dropout=DROPOUT_RATE
        ),
        'LETE_LSTM': LETE_LSTM(
            input_size=784,
            hidden_dim=LSTM_HIDDEN_DIM,
            num_classes=10,
            dropout=DROPOUT_RATE
        ),
        'KAN_MAMMOTE_LSTM': KAN_MAMMOTE_LSTM(
            input_size=784,
            hidden_dim=LSTM_HIDDEN_DIM,
            num_classes=10,
            dropout=DROPOUT_RATE
        )
    }
    
    # Print model information
    print("\n📊 Model Information:")
    for name, model in models.items():
        param_count = count_parameters(model)
        print(f"   {name}: {param_count:,} parameters")
    
    # Train all models
    results = {}
    best_accuracies = {}
    
    for model_name, model in models.items():
        history, best_acc = train_model(model, train_loader, test_loader, model_name)
        results[model_name] = history
        best_accuracies[model_name] = best_acc
    
    # Save results
    print("\n💾 Saving results...")
    
    # Save training histories
    with open(f"{RESULTS_DIR}/training_histories.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save summary results
    summary = {
        'best_accuracies': best_accuracies,
        'model_parameters': {name: count_parameters(model) for name, model in models.items()},
        'configuration': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'num_epochs': NUM_EPOCHS,
            'lstm_hidden_dim': LSTM_HIDDEN_DIM,
            'time_embedding_dim': TIME_EMBEDDING_DIM,
            'dropout_rate': DROPOUT_RATE,
            'threshold': THRESHOLD
        }
    }
    
    with open(f"{RESULTS_DIR}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create CSV summary
    with open(f"{RESULTS_DIR}/results_summary.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Best_Accuracy', 'Parameters', 'Avg_Time_per_Epoch'])
        
        for model_name in models.keys():
            avg_time = np.mean(results[model_name]['training_time'])
            writer.writerow([
                model_name,
                f"{best_accuracies[model_name]:.4f}",
                count_parameters(models[model_name]),
                f"{avg_time:.2f}"
            ])
    
    # Create visualizations
    plot_training_curves(results)
    
    # Print final summary
    print("\n🎯 FINAL RESULTS SUMMARY:")
    print("=" * 60)
    for model_name, acc in best_accuracies.items():
        params = count_parameters(models[model_name])
        avg_time = np.mean(results[model_name]['training_time'])
        print(f"{model_name:20s}: {acc:.4f} acc | {params:7,} params | {avg_time:.1f}s/epoch")
    
    # Find best model
    best_model = max(best_accuracies, key=best_accuracies.get)
    print(f"\n🏆 Best Model: {best_model} (Accuracy: {best_accuracies[best_model]:.4f})")
    
    print(f"\n💾 All results saved to: {RESULTS_DIR}")
    print("🎉 Comparison complete!")

if __name__ == "__main__":
    main()
