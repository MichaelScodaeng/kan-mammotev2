"""
UMAP & t-SNE analysis for KAN-MAMMOTE on synthetic event-time data.

This script generates simple point-process style sequences with class-dependent
inter-arrival distributions, extracts:
  - Pre-Mamba embeddings (KMOTE output, mean-pooled over time)
  - Post-Mamba embeddings (KAN-MAMMOTE forward output, mean-pooled over time)
and visualizes them via UMAP and t-SNE.

Outputs:
  - Saves figures to analysis/figs/*.png

Usage:
  python -m analysis.umap_tsne_mammote --samples 400 --seq-len 128 --classes 4

Notes:
  - Runs on CPU by default.
  - Requires: matplotlib, scikit-learn, umap-learn, torch, gpytorch
"""

from __future__ import annotations

import os
import sys
import math
import time
import argparse
import random
from typing import Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure repo root is on path so we can import models.* when executed as a script
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Soft dependency check to give a clearer message if missing
try:
    from models.time_encoders.kan_mammote import KAN_MAMMOTE
except Exception as e:
    print("[ERROR] Failed to import KAN_MAMMOTE. Ensure all dependencies are installed (gpytorch, etc.).")
    print(f"Exception: {e}")
    sys.exit(1)

# Plotting & DR libs (install via requirements_analysis.txt or pip)
try:
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    import umap
except Exception as e:
    print("[ERROR] Missing analysis dependencies. Please install: matplotlib, scikit-learn, umap-learn")
    print(f"Exception: {e}")
    sys.exit(1)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_event_time_sequences(
    n_samples: int,
    seq_len: int,
    n_classes: int = 4,
    max_time: float = 50.0,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Generate synthetic event-time sequences with different inter-arrival patterns per class.
    
    Enhanced patterns to make classes more distinguishable:
    - C0: Very slow Poisson (lambda=0.3) - sparse events
    - C1: Bursty pattern - alternating fast/slow periods
    - C2: Fast constant rate (lambda=2.5) - frequent events
    - C3: Non-stationary sinusoidal rate - varying intensity

    Returns:
      t_abs: (B, S, 1) absolute time stamps per sample
      t_rel: (B, S, 1) inter-arrival times (delta_t) per sample
      labels: (B,) numpy int labels
    """
    assert n_classes <= 4, "This generator supports up to 4 classes."

    # Class-specific rate patterns (enhanced for better separation)
    lambdas = [0.3, 1.0, 2.5]  # More spread out rates

    B = n_samples
    S = seq_len
    labels = []
    t_abs_list = []
    t_rel_list = []

    # base time grid to help shape a nonstationary rate for class 3
    t_idx = np.arange(S)

    for i in range(B):
        c = i % n_classes
        labels.append(c)
        if c == 0:
            # Class 0: Very slow Poisson
            lam = lambdas[0]
            delta = np.random.exponential(scale=1.0 / lam, size=S)
        elif c == 1:
            # Class 1: Bursty pattern - alternating fast/slow every S/4 steps
            quarter = S // 4
            delta = np.zeros(S)
            for q in range(4):
                start_idx = q * quarter
                end_idx = min((q + 1) * quarter, S)
                if q % 2 == 0:
                    # Burst periods: fast rate
                    delta[start_idx:end_idx] = np.random.exponential(scale=1.0 / 4.0, size=end_idx - start_idx)
                else:
                    # Quiet periods: slow rate
                    delta[start_idx:end_idx] = np.random.exponential(scale=1.0 / 0.5, size=end_idx - start_idx)
        elif c == 2:
            # Class 2: Fast constant Poisson
            lam = lambdas[2]
            delta = np.random.exponential(scale=1.0 / lam, size=S)
        else:
            # Class 3: Nonstationary sinusoidal (more pronounced variation)
            lam_t = 1.5 + 1.2 * np.sin(2 * math.pi * t_idx / S)
            lam_t = np.clip(lam_t, 0.3, None)
            delta = np.random.exponential(scale=1.0 / lam_t)

        t_abs = np.cumsum(delta)
        # Normalize time to a fixed window [0, max_time]
        t_abs = t_abs / t_abs[-1] * max_time
        t_rel = np.concatenate([[0.0], np.diff(t_abs)])  # ensure same length, first step 0

        t_abs_list.append(t_abs[:, None])
        t_rel_list.append(t_rel[:, None])

    t_abs_arr = np.stack(t_abs_list, axis=0)  # (B, S, 1)
    t_rel_arr = np.stack(t_rel_list, axis=0)  # (B, S, 1)

    t_abs_tensor = torch.from_numpy(t_abs_arr).float()
    t_rel_tensor = torch.from_numpy(t_rel_arr).float()
    labels_arr = np.array(labels, dtype=np.int64)
    return t_abs_tensor, t_rel_tensor, labels_arr


def train_model_on_sequences(
    model: KAN_MAMMOTE,
    t_abs_train: torch.Tensor,
    t_rel_train: torch.Tensor,
    labels_train: np.ndarray,
    t_abs_val: torch.Tensor,
    t_rel_val: torch.Tensor,
    labels_val: np.ndarray,
    device: str,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 10,
):
    """
    Train KAN-MAMMOTE to classify temporal sequences with early stopping.
    Uses a simple classification head on top of mean-pooled embeddings.
    
    Args:
        patience: Number of epochs to wait for improvement before early stopping
    
    Returns:
        model: Trained model (best checkpoint based on validation loss)
        classifier: Classification head (best checkpoint)
        history: Dict with train/val loss and accuracy history
    """
    from torch.utils.data import TensorDataset, DataLoader
    
    print(f"[Training] Starting training for up to {epochs} epochs (patience={patience})...")
    
    # Add classification head
    n_classes = int(labels_train.max()) + 1
    classifier = nn.Linear(model.embedding_dim, n_classes).to(device)
    
    # Prepare train data
    labels_train_tensor = torch.from_numpy(labels_train).long()
    train_dataset = TensorDataset(t_abs_train, t_rel_train, labels_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Prepare validation data
    labels_val_tensor = torch.from_numpy(labels_val).long()
    val_dataset = TensorDataset(t_abs_val, t_rel_val, labels_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Optimizer for both model and classifier with stronger regularization
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(classifier.parameters()),
        lr=lr,
        weight_decay=1e-2  # Increased from 1e-3 for stronger L2 regularization
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Early stopping setup
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_model_state = None
    best_classifier_state = None
    
    # History tracking
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(epochs):
        # ========== TRAINING ==========
        model.train()
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_t_abs, batch_t_rel, batch_labels in train_loader:
            batch_t_abs = batch_t_abs.to(device)
            batch_t_rel = batch_t_rel.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass through KAN-MAMMOTE
            embeddings = model(batch_t_abs, batch_t_rel)  # (B, S, D)
            pooled = embeddings.mean(dim=1)  # (B, D)
            
            # Add dropout for regularization during training
            pooled = F.dropout(pooled, p=0.3, training=True)
            
            # Classification
            logits = classifier(pooled)  # (B, n_classes)
            loss = criterion(logits, batch_labels)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Stats
            train_loss += loss.item()
            pred = logits.argmax(dim=1)
            train_correct += (pred == batch_labels).sum().item()
            train_total += batch_labels.size(0)
        
        train_acc = 100.0 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # ========== VALIDATION ==========
        model.eval()
        classifier.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_t_abs, batch_t_rel, batch_labels in val_loader:
                batch_t_abs = batch_t_abs.to(device)
                batch_t_rel = batch_t_rel.to(device)
                batch_labels = batch_labels.to(device)
                
                # Forward pass
                embeddings = model(batch_t_abs, batch_t_rel)
                pooled = embeddings.mean(dim=1)
                logits = classifier(pooled)
                loss = criterion(logits, batch_labels)
                
                # Stats
                val_loss += loss.item()
                pred = logits.argmax(dim=1)
                val_correct += (pred == batch_labels).sum().item()
                val_total += batch_labels.size(0)
        
        val_acc = 100.0 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            # Save best model
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_classifier_state = {k: v.cpu().clone() for k, v in classifier.state_dict().items()}
            improvement = "✓ (best)"
        else:
            patience_counter += 1
            improvement = ""
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0 or improvement:
            print(f"  Epoch {epoch+1}/{epochs}: "
                  f"Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}% | "
                  f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}% {improvement}")
        
        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if patience_counter >= patience:
            print(f"[Training] Early stopping at epoch {epoch+1}. Best epoch was {best_epoch}.")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        classifier.load_state_dict({k: v.to(device) for k, v in best_classifier_state.items()})
        print(f"[Training] Restored best model from epoch {best_epoch}")
    
    print("[Training] Complete!")
    model.eval()
    classifier.eval()
    return model, classifier, history


@torch.no_grad()
def extract_embeddings(
    model: KAN_MAMMOTE,
    t_abs: torch.Tensor,
    t_rel: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      pre_mamba: (B, D) mean pooled over time from KMOTE
      post_mamba: (B, D) mean pooled over time from full forward
    """
    device = next(model.parameters()).device
    t_abs = t_abs.to(device)
    t_rel = t_rel.to(device)

    # Pre-Mamba (KMOTE output)
    u_k = model.k_mote(t_abs)  # (B, S, D)
    pre_mamba = u_k.mean(dim=1)  # (B, D)

    # Full forward -> post-Mamba
    out = model(t_abs, t_rel)  # (B, S, D)
    post_mamba = out.mean(dim=1)
    return pre_mamba.cpu(), post_mamba.cpu()


def fit_umap(X: np.ndarray, n_neighbors: int = 20, min_dist: float = 0.1, seed: int = 42) -> np.ndarray:
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed)
    return reducer.fit_transform(X)


def fit_tsne(X: np.ndarray, perplexity: float = 30.0, seed: int = 42) -> np.ndarray:
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed, init="pca")
    return tsne.fit_transform(X)


def plot_scatter(ax, Z: np.ndarray, y: np.ndarray, title: str):
    cmap = plt.cm.get_cmap("tab10", int(y.max()) + 1)
    for cls in np.unique(y):
        idx = y == cls
        ax.scatter(Z[idx, 0], Z[idx, 1], s=14, alpha=0.8, label=f"class {cls}", color=cmap(cls))
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=1.5, fontsize=8, frameon=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=4000, help="More data to reduce overfitting")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--classes", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--num-mixtures", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-dir", type=str, default=os.path.join(REPO_ROOT, "analysis", "figs"))
    parser.add_argument("--epochs", type=int, default=200, help="Maximum number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--no-train", action="store_true", help="Skip training (use random init)")
    parser.add_argument("--val-split", type=float, default=0.15, help="Fraction of data for validation set")
    parser.add_argument("--test-split", type=float, default=0.15, help="Fraction of data for test set")
    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, but Mamba requires GPU. Exiting.")
        sys.exit(1)

    print("[1/7] Generating synthetic event-time sequences…")
    t_abs, t_rel, labels = generate_event_time_sequences(
        n_samples=args.samples, seq_len=args.seq_len, n_classes=args.classes
    )
    
    # ===== TRAIN/VAL/TEST SPLIT =====
    train_frac = 1.0 - args.val_split - args.test_split
    print(f"[2/7] Splitting data: {100*train_frac:.0f}% train, "
          f"{100*args.val_split:.0f}% val, {100*args.test_split:.0f}% test…")
    
    n_train = int(args.samples * train_frac)
    n_val = int(args.samples * args.val_split)
    
    # Shuffle indices
    indices = torch.randperm(args.samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Split data
    t_abs_train, t_abs_val, t_abs_test = t_abs[train_indices], t_abs[val_indices], t_abs[test_indices]
    t_rel_train, t_rel_val, t_rel_test = t_rel[train_indices], t_rel[val_indices], t_rel[test_indices]
    labels_train, labels_val, labels_test = labels[train_indices], labels[val_indices], labels[test_indices]
    
    print(f"  Train: {len(train_indices)} samples, Val: {len(val_indices)} samples, "
          f"Test: {len(test_indices)} samples")
    # ================================

    print("[3/7] Building KAN-MAMMOTE model…")
    # Note: expert_dim is unused internally but kept for API compatibility
    # Reduced capacity to prevent overfitting on simple synthetic task
    model = KAN_MAMMOTE(
        embedding_dim=args.embedding_dim,
        expert_dim=args.embedding_dim,
        num_mixtures=args.num_mixtures,
        mamba_d_state=256,      # Reduced from 128 to decrease model capacity
        mamba_d_conv=4,
        mamba_expand=4,        # Already reduced from default 4
        wavelet_type="shock",
        mamba_headdim=64,
    )
    model.to(args.device)
    model.eval()

    # Initialize SM kernel from a small sample of TRAIN delta_t
    print("[4/7] Initializing SM-Kernel from sample delta_t (TRAIN)…")
    sample_idx = torch.randperm(t_rel_train.shape[0])[: min(64, t_rel_train.shape[0])]
    model.initialize_sm_kernel(t_rel_train[sample_idx].to(args.device))

    # Train the model with early stopping
    classifier = None
    history = None
    if not args.no_train:
        print("[5/7] Training model with early stopping…")
        model, classifier, history = train_model_on_sequences(
            model,
            t_abs_train,
            t_rel_train,
            labels_train,
            t_abs_val,
            t_rel_val,
            labels_val,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience
        )
    else:
        print("[5/7] Skipping training (--no-train specified)")

    # Extract embeddings from ALL THREE SETS
    print("[6/7] Extracting embeddings from train/val/test sets…")
    pre_train, post_train = extract_embeddings(model, t_abs_train, t_rel_train)
    pre_val, post_val = extract_embeddings(model, t_abs_val, t_rel_val)
    pre_test, post_test = extract_embeddings(model, t_abs_test, t_rel_test)

    print("[7/7] Fitting UMAP and t-SNE and plotting all three sets…")
    
    # Combine all data for consistent UMAP/t-SNE fitting
    pre_all = torch.cat([pre_train, pre_val, pre_test], dim=0).numpy()
    post_all = torch.cat([post_train, post_val, post_test], dim=0).numpy()
    labels_all = np.concatenate([labels_train, labels_val, labels_test])
    
    # Fit dimensionality reduction on combined data
    pre_umap_all = fit_umap(pre_all)
    post_umap_all = fit_umap(post_all)
    
    perplexity = min(30.0, max(5.0, len(pre_all) / 20.0))
    pre_tsne_all = fit_tsne(pre_all, perplexity=perplexity)
    post_tsne_all = fit_tsne(post_all, perplexity=perplexity)
    
    # Split back into train/val/test
    n_train_pts = len(train_indices)
    n_val_pts = len(val_indices)
    
    pre_umap_train = pre_umap_all[:n_train_pts]
    pre_umap_val = pre_umap_all[n_train_pts:n_train_pts + n_val_pts]
    pre_umap_test = pre_umap_all[n_train_pts + n_val_pts:]
    
    post_umap_train = post_umap_all[:n_train_pts]
    post_umap_val = post_umap_all[n_train_pts:n_train_pts + n_val_pts]
    post_umap_test = post_umap_all[n_train_pts + n_val_pts:]
    
    pre_tsne_train = pre_tsne_all[:n_train_pts]
    pre_tsne_val = pre_tsne_all[n_train_pts:n_train_pts + n_val_pts]
    pre_tsne_test = pre_tsne_all[n_train_pts + n_val_pts:]
    
    post_tsne_train = post_tsne_all[:n_train_pts]
    post_tsne_val = post_tsne_all[n_train_pts:n_train_pts + n_val_pts]
    post_tsne_test = post_tsne_all[n_train_pts + n_val_pts:]

    ts = time.strftime("%Y%m%d_%H%M%S")

    # ========== UMAP Figure (2 rows x 3 cols) ==========
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    
    # Pre-Mamba row
    plot_scatter(axes[0, 0], pre_umap_train, labels_train, "UMAP: pre-Mamba (KMOTE) - TRAIN")
    plot_scatter(axes[0, 1], pre_umap_val, labels_val, "UMAP: pre-Mamba (KMOTE) - VAL")
    plot_scatter(axes[0, 2], pre_umap_test, labels_test, "UMAP: pre-Mamba (KMOTE) - TEST")
    
    # Post-Mamba row
    plot_scatter(axes[1, 0], post_umap_train, labels_train, "UMAP: post-Mamba - TRAIN")
    plot_scatter(axes[1, 1], post_umap_val, labels_val, "UMAP: post-Mamba - VAL")
    plot_scatter(axes[1, 2], post_umap_test, labels_test, "UMAP: post-Mamba - TEST")
    
    umap_path = os.path.join(args.save_dir, f"umap_train_val_test_{ts}.png")
    fig.tight_layout()
    fig.savefig(umap_path, dpi=150)
    plt.close(fig)

    # ========== t-SNE Figure (2 rows x 3 cols) ==========
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    
    # Pre-Mamba row
    plot_scatter(axes[0, 0], pre_tsne_train, labels_train, "t-SNE: pre-Mamba (KMOTE) - TRAIN")
    plot_scatter(axes[0, 1], pre_tsne_val, labels_val, "t-SNE: pre-Mamba (KMOTE) - VAL")
    plot_scatter(axes[0, 2], pre_tsne_test, labels_test, "t-SNE: pre-Mamba (KMOTE) - TEST")
    
    # Post-Mamba row
    plot_scatter(axes[1, 0], post_tsne_train, labels_train, "t-SNE: post-Mamba - TRAIN")
    plot_scatter(axes[1, 1], post_tsne_val, labels_val, "t-SNE: post-Mamba - VAL")
    plot_scatter(axes[1, 2], post_tsne_test, labels_test, "t-SNE: post-Mamba - TEST")
    
    tsne_path = os.path.join(args.save_dir, f"tsne_train_val_test_{ts}.png")
    fig.tight_layout()
    fig.savefig(tsne_path, dpi=150)
    plt.close(fig)

    # ========== Training History Plot (if trained) ==========
    if history is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs_range = range(1, len(history['train_loss']) + 1)
        
        # Loss plot
        axes[0].plot(epochs_range, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs_range, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training & Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(epochs_range, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
        axes[1].plot(epochs_range, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training & Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        history_path = os.path.join(args.save_dir, f"training_history_{ts}.png")
        fig.tight_layout()
        fig.savefig(history_path, dpi=150)
        plt.close(fig)
        
        print(f"Saved figures:\n  - {umap_path}\n  - {tsne_path}\n  - {history_path}")
    else:
        print(f"Saved figures:\n  - {umap_path}\n  - {tsne_path}")


if __name__ == "__main__":
    main()
