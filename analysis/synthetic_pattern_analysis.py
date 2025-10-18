"""
Comprehensive Synthetic Pattern Analysis for Time Encoders
=========================================================

This script performs a comprehensive analysis of different time encoders on 
synthetic periodic, non-periodic, and mixed data patterns using the proper
encode→decode methodology.

METHODOLOGY:
Time Input (1D) → Time Encoder → 64D Embedding → MLP Decoder → Output (1D)
     t      →      φ(t)     →   [e1,...,e64]  →     f()    →     y

This follows the standard approach in time encoding papers where:
1. Time encoders map timestamps to high-dimensional representations (64D)
2. Task-specific MLP decoders map representations to outputs (1D)
3. Training optimizes the entire encode→decode pipeline end-to-end

Models tested:
- KAN-MAMMOTE (main model) + MLP decoder
- K-MOTE (expert mixture) + MLP decoder  
- K-MOTE subcomponents (B-Spline, Fourier, Wavelet) + MLP decoders
- Baseline encoders (Original, Mercer, Time2Vec, LeTE) + MLP decoders

Data patterns:
- Synthetic Periodic Data (multiple harmonics)
- Synthetic Non-Periodic Data (exponential decay, steps, spikes)
- Synthetic Mixed Data (combination of periodic + non-periodic)

All models use 64-dimensional embeddings and are trained end-to-end with 
shared hyperparameters for fair comparison.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory for imports
# Global training configuration
MAX_EPOCHS = 3000

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all time encoders with graceful error handling
try:
    from models.time_encoders.k_mote import KMOTE, SplineKANLayer, FourierKANLayer, WaveletKANLayer
    KMOTE_AVAILABLE = True
except ImportError as e:
    print(f"K-MOTE not available: {e}")
    KMOTE_AVAILABLE = False

try:
    from models.time_encoders.kan_mammote import KAN_MAMMOTE as KANMAMMOTETimeEncoder
    KAN_MAMMOTE_AVAILABLE = True
except ImportError:
    try:
        # Try alternative import
        from models.time_encoders.factory import KAN_MAMMOTE as KANMAMMOTETimeEncoder
        KAN_MAMMOTE_AVAILABLE = True
    except ImportError as e:
        print(f"KAN-MAMMOTE not available: {e}")
        KAN_MAMMOTE_AVAILABLE = False

try:
    from models.gnn_backbones.modules import TimeEncoder as OriginalTimeEncoder
    ORIGINAL_AVAILABLE = True
except ImportError:
    try:
        from models.time_encoders.original_encoder import OriginalTimeEncoder
        ORIGINAL_AVAILABLE = True
    except ImportError as e:
        print(f"Original encoder not available: {e}")
        ORIGINAL_AVAILABLE = False

try:
    from models.time_encoders.mercer_encoder import MercerTimeEncoder
    MERCER_AVAILABLE = True
except ImportError as e:
    print(f"Mercer encoder not available: {e}")
    MERCER_AVAILABLE = False

try:
    from models.time_encoders.time2vec_encoder import Time2VecEncoder
    TIME2VEC_AVAILABLE = True
except ImportError as e:
    print(f"Time2Vec encoder not available: {e}")
    TIME2VEC_AVAILABLE = False

try:
    from models.time_encoders.lete_encoder import LeTE
    LETE_AVAILABLE = True
except ImportError as e:
    print(f"LeTE encoder not available: {e}")
    LETE_AVAILABLE = False

# Create output directories
os.makedirs('analysis_figures_synthetic', exist_ok=True)
os.makedirs('analysis_results_synthetic', exist_ok=True)

# Training configuration following paper's reconstruction task
SHARED_TRAINING_CONFIG = {
    'learning_rate': 1e-3,  # Standard learning rate for reconstruction tasks
    'patience': 200,  # Allow more patience for convergence
    'min_delta': 1e-6,
    'max_epochs': MAX_EPOCHS,  # Sufficient epochs for pattern learning
    'weight_decay': 1e-4,  # Standard weight decay
    'grad_clip_norm': 1.0,
}

# Device detection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Using device: {device}")

# Random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- Helper Classes ---

class SingleExpertModel(nn.Module):
    """Wrapper for individual K-MOTE experts"""
    def __init__(self, expert_class, **kwargs):
        super().__init__()
        self.expert = expert_class(input_dim=1, output_dim=1, **kwargs)
    
    def forward(self, x):
        return self.expert(x)

class TimeEncoderWrapper(nn.Module):
    """Wrapper to make time encoders compatible with our analysis"""
    def __init__(self, encoder_class, input_dim=1, output_dim=1, **kwargs):
        super().__init__()
        self.encoder = encoder_class(input_dim=input_dim, output_dim=output_dim, **kwargs)
    
    def forward(self, x):
        return self.encoder(x)


class PaperReconstructionModel(nn.Module):
    """
    Reconstruction model following the exact paper methodology (Section G.4)
    
    Architecture: Time Input → Time Encoder → d-dim Embedding → Linear Decoder → 1D Output
                     t     →      φ(t)     →   [e1,...,ed]  →   Linear   →    y
    
    Key differences from before:
    1. ALL models use the SAME simple linear decoder (fair comparison)
    2. Time encoders process INDIVIDUAL timestamps (not sequences)
    3. Same embedding dimension for all encoders
    """
    def __init__(self, time_encoder, embedding_dim=64):
        super().__init__()
        self.time_encoder = time_encoder
        self.embedding_dim = embedding_dim
        
        # Paper uses simple linear decoder for ALL models
        self.decoder = nn.Linear(embedding_dim, 1)
        
        # Initialize decoder weights
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)
    
    def forward(self, t):
        """
        Forward pass following paper methodology
        
        Args:
            t: Individual timestamps, shape (N,) or (N, 1)
            
        Returns:
            reconstruction: Output, shape (N, 1)
        """
        # Ensure timestamps are individual (not sequences)
        if t.dim() == 2 and t.shape[-1] == 1:
            t = t.squeeze(-1)  # (N,)
        
        # Encode individual timestamps
        embeddings = self.time_encoder(t)  # (N, embedding_dim)
        
        # Simple linear decoding (same for all models)
        reconstruction = self.decoder(embeddings)  # (N, 1)
        
        return reconstruction
    
    def forward(self, t):
        """
        Forward pass: t → embedding → decoded_output
        
        Args:
            t: Time input tensor, shape (B, S, 1) or (S, 1)
            
        Returns:
            decoded_output: Task output, shape (B, S, 1) or (S, 1)
        """
        # Step 1: Encode time to d-dimensional embedding
        temporal_embedding = self.time_encoder(t)  # (B, S, embedding_dim) or (S, embedding_dim)
        
        # Step 2: Decode embedding to task output
        decoded_output = self.decoder(temporal_embedding)  # (B, S, output_dim) or (S, output_dim)
        
        return decoded_output
    
    def get_temporal_embedding(self, t):
        """Get the intermediate temporal embedding for analysis"""
        with torch.no_grad():
            return self.time_encoder(t)


class KANMAMMOTEIndividualEncoder(nn.Module):
    """
    KAN-MAMMOTE wrapper for INDIVIDUAL timestamp processing (not sequences)
    
    This wrapper makes KAN-MAMMOTE compatible with the paper's methodology
    where each timestamp is processed independently.
    """
    def __init__(self, embedding_dim=64, expert_dim=16, num_mixtures=16, wavelet_type='shock'):
        super().__init__()
        # Create KAN-MAMMOTE encoder
        self.kan_mammote_encoder = KANMAMMOTETimeEncoder(
            embedding_dim=embedding_dim,
            expert_dim=expert_dim,
            num_mixtures=num_mixtures,
            wavelet_type=wavelet_type
        )
        self.embedding_dim = embedding_dim
    
    def forward(self, t):
        """
        Process individual timestamps (not sequences) for fair comparison
        
        Args:
            t: Individual timestamps, shape (N,)
        Returns:
            embeddings: Individual embeddings, shape (N, embedding_dim)
        """
        # Convert individual timestamps to batch format
        if t.dim() == 1:
            batch_size = t.shape[0]
            # Create single-timestep sequences for each timestamp
            t_abs = t.unsqueeze(-1).unsqueeze(-1)  # (N, 1, 1)
            # Use small constant relative time for individual timestamps
            t_rel = torch.ones_like(t_abs) * 0.1  # (N, 1, 1)
        else:
            raise ValueError("Expected 1D timestamps for individual processing")
        
        # Process each timestamp individually through KAN-MAMMOTE
        embeddings = []
        for i in range(batch_size):
            # Single timestamp as sequence of length 1
            single_t_abs = t_abs[i:i+1]  # (1, 1, 1)
            single_t_rel = t_rel[i:i+1]  # (1, 1, 1)
            
            # Get embedding from KAN-MAMMOTE
            emb = self.kan_mammote_encoder(t_abs=single_t_abs, t_rel=single_t_rel)  # (1, 1, embedding_dim)
            emb = emb.squeeze(0).squeeze(0)  # (embedding_dim,)
            embeddings.append(emb)
        
        # Stack to batch
        embeddings = torch.stack(embeddings)  # (N, embedding_dim)
        
        return embeddings
    
    def forward(self, t):
        """
        Standard forward interface: t → embedding → output
        
        Args:
            t: Time input, shape (seq_len, 1)
        Returns:
            output: Decoded output, shape (seq_len, 1)
        """
        # KAN-MAMMOTE expects (B, S, 1) and needs both t_abs and t_rel
        if t.dim() == 2:
            t_batch = t.unsqueeze(0)  # (1, seq_len, 1)
            batch_added = True
        else:
            t_batch = t
            batch_added = False
        
        # Generate proper relative time (time differences/deltas)
        t_abs = t_batch  # Absolute time
        
        # Create relative time as time differences
        if t_batch.shape[1] > 1:
            # Calculate differences: t[i] - t[i-1]
            t_flat = t_batch.squeeze(-1)  # (batch, seq_len)
            t_diffs = torch.diff(t_flat, dim=1)  # (batch, seq_len-1)
            
            # For the first point, use the first difference or a small value
            if t_diffs.shape[1] > 0:
                first_diff = t_diffs[:, :1]  # Use first actual difference
            else:
                first_diff = torch.ones_like(t_flat[:, :1]) * 0.3  # Default small delta
                
            # Concatenate to maintain sequence length
            t_rel_flat = torch.cat([first_diff, t_diffs], dim=1)  # (batch, seq_len)
            t_rel = t_rel_flat.unsqueeze(-1)  # (batch, seq_len, 1)
        else:
            # Single time point - use small positive delta
            t_rel = torch.ones_like(t_batch) * 0.3
        
        # Ensure positive relative times (add small epsilon for numerical stability)
        t_rel = torch.abs(t_rel) + 1e-6
        
        # Get embedding from KAN-MAMMOTE with proper abs/rel time
        embedding = self.kan_mammote_encoder(t_abs=t_abs, t_rel=t_rel)  # (1, seq_len, embedding_dim)
        
        # Remove batch dimension if it was added
        if batch_added:
            embedding = embedding.squeeze(0)  # (seq_len, embedding_dim)
        
        # Decode to output
        output = self.decoder(embedding)  # (seq_len, output_dim)
        
        return output
    
    def get_temporal_embedding(self, t):
        """Get the intermediate temporal embedding for analysis"""
        with torch.no_grad():
            # Use the same logic as forward() to generate proper t_abs and t_rel
            if t.dim() == 2:
                t_batch = t.unsqueeze(0)  # (1, seq_len, 1)
                batch_added = True
            else:
                t_batch = t
                batch_added = False
            
            # Generate relative time (same as in forward)
            t_abs = t_batch
            if t_batch.shape[1] > 1:
                t_flat = t_batch.squeeze(-1)
                t_diffs = torch.diff(t_flat, dim=1)
                if t_diffs.shape[1] > 0:
                    first_diff = t_diffs[:, :1]
                else:
                    first_diff = torch.ones_like(t_flat[:, :1]) * 0.3
                t_rel_flat = torch.cat([first_diff, t_diffs], dim=1)
                t_rel = t_rel_flat.unsqueeze(-1)
            else:
                t_rel = torch.ones_like(t_batch) * 0.3
            
            t_rel = torch.abs(t_rel) + 1e-6
            
            embedding = self.kan_mammote_encoder(t_abs=t_abs, t_rel=t_rel)
            
            if batch_added:
                return embedding.squeeze(0)
            else:
                return embedding


class SingleExpertDecoder(nn.Module):
    """Wrapper for K-MOTE single experts with proper decode methodology"""
    def __init__(self, expert_class, embedding_dim=64, output_dim=1, **kwargs):
        super().__init__()
        # Create expert that outputs embedding_dim
        self.expert = expert_class(input_dim=1, output_dim=embedding_dim, **kwargs)
        
        # Simplified decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.GELU(),
            nn.Linear(16, output_dim)
        )
        
        # Initialize decoder weights
        for m in self.decoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Encode: time → embedding
        embedding = self.expert(x)  # (B, S, embedding_dim)
        
        # Decode: embedding → output
        output = self.decoder(embedding)  # (B, S, output_dim)
        
        return output
    
    def get_temporal_embedding(self, x):
        """Get the intermediate temporal embedding for analysis"""
        with torch.no_grad():
            return self.expert(x)


class StandaloneBSplineEncoder(nn.Module):
    """
    Proper B-Spline time encoder (not K-MOTE internal component)
    
    This is a standalone time encoder that uses B-spline basis functions
    to encode timestamps, designed for direct use (not as K-MOTE component).
    """
    def __init__(self, input_dim=1, output_dim=64):
        super().__init__()
        self.output_dim = output_dim
        
        # B-spline basis functions
        self.num_basis = 16
        self.basis_centers = nn.Parameter(torch.linspace(-2, 2, self.num_basis))
        self.basis_widths = nn.Parameter(torch.ones(self.num_basis) * 0.5)
        
        # Projection to output dimension
        self.projection = nn.Sequential(
            nn.Linear(self.num_basis, output_dim // 2),
            nn.ReLU(),
            nn.Linear(output_dim // 2, output_dim)
        )
        
        # Proper initialization
        self._init_weights()
    
    def _init_weights(self):
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, t):
        # Handle input format
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (N, 1)
        
        # Normalize time to reasonable range for B-splines
        t_norm = torch.tanh(t * 0.01)  # Keep in [-1, 1], gentler scaling
        
        # Compute B-spline basis functions
        distances = torch.abs(t_norm - self.basis_centers.unsqueeze(0))  # (N, num_basis)
        basis_values = torch.exp(-distances / (torch.abs(self.basis_widths.unsqueeze(0)) + 1e-6))  # (N, num_basis)
        
        # Normalize basis values
        basis_values = basis_values / (basis_values.sum(dim=-1, keepdim=True) + 1e-6)
        
        # Project to output dimension
        output = self.projection(basis_values)  # (N, output_dim)
        
        return output


class StandaloneFourierEncoder(nn.Module):
    """
    Proper Fourier time encoder
    
    Uses learnable Fourier features to encode temporal patterns,
    designed as a standalone time encoder.
    """
    def __init__(self, input_dim=1, output_dim=64):
        super().__init__()
        self.output_dim = output_dim
        
        # Fourier frequencies - half for sin, half for cos
        self.num_freqs = output_dim // 2
        self.frequencies = nn.Parameter(torch.randn(self.num_freqs) * 0.01)  # Smaller initial frequencies
        self.phases = nn.Parameter(torch.randn(self.num_freqs) * 2 * torch.pi)
        
        # Learnable amplitude scaling
        self.amplitudes = nn.Parameter(torch.ones(self.num_freqs) * 0.1)  # Smaller initial amplitudes
        
    def forward(self, t):
        # Handle input format
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (N, 1)
        
        # Normalize time for better numerical stability
        t_norm = t * 0.01  # Scale down significantly
        
        # Compute Fourier features
        angles = t_norm * self.frequencies.unsqueeze(0) + self.phases.unsqueeze(0)  # (N, num_freqs)
        
        # Sin and cosine components with learnable amplitudes
        sin_features = torch.sin(angles) * torch.sigmoid(self.amplitudes.unsqueeze(0))  # (N, num_freqs)
        cos_features = torch.cos(angles) * torch.sigmoid(self.amplitudes.unsqueeze(0))  # (N, num_freqs)
        
        # Concatenate for full output dimension
        if self.output_dim % 2 == 0:
            output = torch.cat([sin_features, cos_features], dim=-1)  # (N, output_dim)
        else:
            # Handle odd output dimensions
            extra_feature = torch.tanh(t_norm.squeeze(-1))  # (N,)
            output = torch.cat([sin_features, cos_features, extra_feature.unsqueeze(-1)], dim=-1)
        
        return output


class StandaloneWaveletEncoder(nn.Module):
    """
    Proper Wavelet time encoder
    
    Uses Mexican Hat wavelets to capture localized temporal features,
    designed as a standalone time encoder.
    """
    def __init__(self, input_dim=1, output_dim=64):
        super().__init__()
        self.output_dim = output_dim
        
        # Wavelet parameters - distributed across different scales and locations
        self.num_wavelets = 16
        self.centers = nn.Parameter(torch.linspace(-3, 3, self.num_wavelets))
        self.scales = nn.Parameter(torch.ones(self.num_wavelets) * 0.5)  # Smaller initial scales
        
        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(self.num_wavelets, output_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),  # Add dropout for regularization
            nn.Linear(output_dim // 2, output_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)  # Smaller initial weights
                nn.init.zeros_(m.bias)
    
    def forward(self, t):
        # Handle input format
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (N, 1)
        
        # Normalize time for wavelets
        t_norm = t * 0.01  # Scale down for stability
        
        # Compute Mexican Hat wavelet features
        shifted_t = (t_norm - self.centers.unsqueeze(0)) / (torch.abs(self.scales.unsqueeze(0)) + 1e-6)
        
        # Mexican Hat wavelet: ψ(t) = (1 - t²) * exp(-t²/2)
        wavelet_values = (1 - shifted_t**2) * torch.exp(-0.5 * shifted_t**2)  # (N, num_wavelets)
        
        # Normalize wavelet responses
        wavelet_values = wavelet_values / (torch.abs(wavelet_values).sum(dim=-1, keepdim=True) + 1e-6)
        
        # Project to output dimension
        output = self.projection(wavelet_values)  # (N, output_dim)
        
        return output

# --- Synthetic Data Generators ---

def generate_periodic_data(t, noise_level=0.05):
    """
    Generate synthetic periodic data similar to paper's Figure 11
    
    Clear periodic patterns that FTE should handle well but others may struggle
    """
    # Ensure t is on the correct device
    t = t.to(device)
    
    # Normalize time to [0, 4π] for clear periodicity
    t_norm = (t - t.min()) / (t.max() - t.min()) * 4 * torch.pi
    
    # Main periodic components (similar to paper's periodic pattern)
    component1 = 3.0 * torch.sin(t_norm)  # Main frequency
    component2 = 1.5 * torch.sin(2 * t_norm + torch.pi/4)   # Second harmonic
    component3 = 0.8 * torch.cos(3 * t_norm)  # Third harmonic
    
    # Combine components
    signal = component1 + component2 + component3
    
    # Add controlled noise
    if noise_level > 0:
        noise = torch.randn_like(signal) * noise_level * signal.std()
        signal = signal + noise
    
    return signal

def generate_non_periodic_data(t, noise_level=0.05):
    """
    Generate synthetic non-periodic data similar to paper's Figure 11
    
    Complex non-periodic patterns that should challenge FTE but be handled by LeTE
    """
    # Ensure t is on the correct device
    t = t.to(device)
    
    # Normalize time to [0, 1] for consistent patterns
    t_norm = (t - t.min()) / (t.max() - t.min())
    
    # Non-periodic components (similar to paper's non-periodic pattern)
    # Exponential decay with oscillation
    decay_component = 4.0 * torch.exp(-3 * t_norm) * torch.sin(10 * t_norm)
    
    # Step function at different points
    step_component = torch.zeros_like(t_norm)
    step_component[t_norm > 0.3] += 2.0
    step_component[t_norm > 0.6] += -3.0
    step_component[t_norm > 0.8] += 1.5
    
    # Smooth non-linear trend
    trend_component = 2.0 * (t_norm ** 3 - 1.5 * t_norm ** 2 + 0.5 * t_norm)
    
    # Localized bursts
    burst_component = torch.zeros_like(t_norm)
    burst_locations = [0.2, 0.5, 0.7]
    for loc in burst_locations:
        mask = torch.abs(t_norm - loc) < 0.05
        burst_component[mask] += 3.0 * torch.exp(-50 * (t_norm[mask] - loc)**2)
    
    # Combine components
    signal = decay_component + step_component + trend_component + burst_component
    
    # Add controlled noise
    if noise_level > 0:
        noise = torch.randn_like(signal) * noise_level * signal.std()
        signal = signal + noise
    
    return signal

def generate_mixed_data(t, noise_level=0.1):
    """Generate synthetic mixed data combining periodic and non-periodic components"""
    # Periodic components (60% weight)
    periodic_component = 0.6 * generate_periodic_data(t, noise_level=0)
    
    # Non-periodic components (40% weight)
    non_periodic_component = 0.4 * generate_non_periodic_data(t, noise_level=0)
    
    # Combine
    signal = periodic_component + non_periodic_component
    
    # Add noise
    if noise_level > 0:
        noise = torch.randn_like(signal) * noise_level * signal.std()
        signal = signal + noise
    
    return signal

# --- Training Functions ---

def train_model_convergence(model, t_data, y_true, model_name="Model"):
    """Train model following paper methodology - individual timestamp processing"""
    config = SHARED_TRAINING_CONFIG
    optimizer = optim.Adam(model.parameters(), 
                          lr=config['learning_rate'], 
                          weight_decay=config['weight_decay'])
    loss_fn = nn.MSELoss()
    
    # Paper methodology: individual timestamps (not sequences)
    # Input: (N,) timestamps, Output: (N, 1) reconstructions
    if t_data.dim() == 1: 
        t_input = t_data  # Keep as (N,) for individual processing
    else:
        t_input = t_data.squeeze(-1)  # Convert to (N,)
        
    if y_true.dim() == 1: 
        y_target = y_true.unsqueeze(-1)  # (N, 1) for loss computation
    else:
        y_target = y_true

    best_loss = float('inf')
    patience_counter = 0
    loss_history = []
    training_info = {
        'converged': False,
        'final_loss': float('inf'),
        'convergence_epoch': config['max_epochs'],
        'loss_history': []
    }
    
    # Training loop with progress bar
    with tqdm(range(config['max_epochs']), desc=f"Training {model_name}", leave=False) as pbar:
        for epoch in pbar:
            model.train()
            
            # Forward pass: individual timestamps → reconstructions
            y_pred = model(t_input)  # (N,) → (N, 1)
            
            # Ensure y_pred matches y_target dimensions
            if y_pred.dim() == 2 and y_pred.shape[1] == 1:
                pass  # Already (N, 1)
            elif y_pred.dim() == 1:
                y_pred = y_pred.unsqueeze(-1)  # (N,) → (N, 1)
            
            # Debug: Check prediction statistics on first epoch
            if epoch == 0:
                y_pred_flat = y_pred.squeeze() if y_pred.dim() > 1 else y_pred
                y_target_flat = y_target.squeeze() if y_target.dim() > 1 else y_target
                
                y_pred_stats = {
                    'mean': y_pred_flat.mean().item(),
                    'std': y_pred_flat.std().item(),
                    'min': y_pred_flat.min().item(),
                    'max': y_pred_flat.max().item()
                }
                y_target_stats = {
                    'mean': y_target_flat.mean().item(),
                    'std': y_target_flat.std().item(),
                    'min': y_target_flat.min().item(),
                    'max': y_target_flat.max().item()
                }
                print(f"    🔍 {model_name} - Epoch 0 Stats:")
                print(f"      y_pred: mean={y_pred_stats['mean']:.4f}, std={y_pred_stats['std']:.4f}")
                print(f"      y_target: mean={y_target_stats['mean']:.4f}, std={y_target_stats['std']:.4f}")
            
            loss = loss_fn(y_pred, y_target)
            
            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"    WARNING: NaN/Inf loss in {model_name} at epoch {epoch+1}")
                training_info['final_loss'] = float('inf')
                return training_info
            
            optimizer.zero_grad()
            loss.backward()
            
            # Check for gradient flow on first few epochs
            if epoch < 3:
                total_grad_norm = 0.0
                num_params = 0
                for param in model.parameters():
                    if param.grad is not None:
                        total_grad_norm += param.grad.norm().item() ** 2
                        num_params += 1
                if num_params > 0:
                    avg_grad_norm = (total_grad_norm / num_params) ** 0.5
                    print(f"      Avg grad norm: {avg_grad_norm:.6f}")
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                         max_norm=config['grad_clip_norm'])
            optimizer.step()
            
            current_loss = loss.item()
            loss_history.append(current_loss)
            
            # Check for improvement
            if current_loss < best_loss - config['min_delta']:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{current_loss:.6f}',
                'Best': f'{best_loss:.6f}',
                'Patience': f'{patience_counter}/{config["patience"]}'
            })
            
            # Early stopping - patience
            if patience_counter >= config['patience']:
                training_info['converged'] = True
                training_info['convergence_epoch'] = epoch + 1
                break
                
            # Early stopping - stability
            if epoch > 100:
                recent_losses = loss_history[-50:]
                if max(recent_losses) - min(recent_losses) < config['min_delta']:
                    training_info['converged'] = True
                    training_info['convergence_epoch'] = epoch + 1
                    break
    
    training_info['final_loss'] = best_loss
    training_info['loss_history'] = loss_history
    
    return training_info

def evaluate_reconstruction(model, t_data, y_true, model_name="Model"):
    """Evaluate model reconstruction quality following paper methodology"""
    model.eval()
    with torch.no_grad():
        # Paper methodology: individual timestamps
        if t_data.dim() == 1:
            t_input = t_data  # Keep as (N,)
        else:
            t_input = t_data.squeeze(-1)  # Convert to (N,)
        
        # Forward pass: individual timestamps → reconstructions
        y_pred = model(t_input)  # (N,) → (N, 1)
        
        # Flatten for metrics calculation
        if y_pred.dim() > 1:
            y_pred = y_pred.squeeze()
        if y_pred.dim() == 0:
            y_pred = y_pred.unsqueeze(0)
        if y_true.dim() > 1:
            y_true = y_true.squeeze()
        if y_true.dim() == 0:
            y_true = y_true.unsqueeze(0)
        
        # Calculate metrics
        mse = nn.MSELoss()(y_pred, y_true).item()
        mae = nn.L1Loss()(y_pred, y_true).item()
        
        # R² score
        ss_res = torch.sum((y_true - y_pred) ** 2).item()
        ss_tot = torch.sum((y_true - y_true.mean()) ** 2).item()
        r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Debug R² calculation
        y_true_mean = y_true.mean().item()
        y_pred_mean = y_pred.mean().item()
        print(f"    📊 {model_name} R² Debug:")
        print(f"      ss_res: {ss_res:.6f}, ss_tot: {ss_tot:.6f}")
        print(f"      y_true_mean: {y_true_mean:.6f}, y_pred_mean: {y_pred_mean:.6f}")
        print(f"      R²: {r2_score:.6f}")
        
        # RMSE
        rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()
        
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2_score': r2_score,
        'prediction': y_pred.cpu().numpy()
    }

# --- Model Factory ---

def create_models():
    """
    Create models following the EXACT paper methodology (Section G.4)
    
    All models follow: Individual Timestamps → Time Encoder → d-dim → Linear Decoder → 1D
    - SAME linear decoder for all models (fair comparison)
    - Individual timestamp processing (not sequences)
    - Same embedding dimension for all encoders
    """
    models = {}
    EMBEDDING_DIM = 64  # Standard embedding dimension
    
    print(f"🔧 Creating models following paper methodology (Section G.4)...")
    print(f"📋 All models: Individual timestamps → {EMBEDDING_DIM}D encoder → Linear decoder")
    
    # Create proper standalone expert encoders (not K-MOTE internal components)
    print("📋 Creating standalone expert encoders (NOT K-MOTE internal components)")
    
    models['B-Spline Expert'] = lambda: PaperReconstructionModel(
        time_encoder=StandaloneBSplineEncoder(input_dim=1, output_dim=EMBEDDING_DIM),
        embedding_dim=EMBEDDING_DIM
    )
    models['Fourier Expert'] = lambda: PaperReconstructionModel(
        time_encoder=StandaloneFourierEncoder(input_dim=1, output_dim=EMBEDDING_DIM),
        embedding_dim=EMBEDDING_DIM
    )
    models['Wavelet Expert'] = lambda: PaperReconstructionModel(
        time_encoder=StandaloneWaveletEncoder(input_dim=1, output_dim=EMBEDDING_DIM),
        embedding_dim=EMBEDDING_DIM
    )
    
    # K-MOTE full system (only if available)
    if KMOTE_AVAILABLE:
        models['K-MOTE'] = lambda: PaperReconstructionModel(
            time_encoder=KMOTE(input_dim=1, output_dim=EMBEDDING_DIM, wavelet_type='shock'),
            embedding_dim=EMBEDDING_DIM
        )
    else:
        print("⚠️ K-MOTE not available, skipping K-MOTE full system...")
    
    # KAN-MAMMOTE with individual timestamp processing
    if KAN_MAMMOTE_AVAILABLE:
        if device.type == 'cuda':
            models['KAN-MAMMOTE'] = lambda: PaperReconstructionModel(
                time_encoder=KANMAMMOTEIndividualEncoder(
                    embedding_dim=EMBEDDING_DIM,
                    expert_dim=64,  # Must be multiple of 16
                    num_mixtures=16,
                    wavelet_type='shock'
                ),
                embedding_dim=EMBEDDING_DIM
            )
        else:
            print("⚠️ KAN-MAMMOTE requires CUDA, but only CPU available. Skipping...")
    else:
        print("⚠️ KAN-MAMMOTE not available, skipping...")
    
    # Baseline encoders - all use same methodology
    if ORIGINAL_AVAILABLE:
        models['Original'] = lambda: PaperReconstructionModel(
            time_encoder=OriginalTimeEncoder(time_dim=EMBEDDING_DIM),
            embedding_dim=EMBEDDING_DIM
        )
    else:
        print("⚠️ Original encoder not available, skipping...")
        
    if MERCER_AVAILABLE:
        models['Mercer'] = lambda: PaperReconstructionModel(
            time_encoder=MercerTimeEncoder(time_dim=EMBEDDING_DIM),
            embedding_dim=EMBEDDING_DIM
        )
    else:
        print("⚠️ Mercer encoder not available, skipping...")
        
    if TIME2VEC_AVAILABLE:
        models['Time2Vec'] = lambda: PaperReconstructionModel(
            time_encoder=Time2VecEncoder(time_dim=EMBEDDING_DIM),
            embedding_dim=EMBEDDING_DIM
        )
    else:
        print("⚠️ Time2Vec encoder not available, skipping...")
        
    if LETE_AVAILABLE:
        models['LeTE'] = lambda: PaperReconstructionModel(
            time_encoder=LeTE(time_dim=EMBEDDING_DIM),
            embedding_dim=EMBEDDING_DIM
        )
    else:
        print("⚠️ LeTE encoder not available, skipping...")
    
    if not models:
        print("❌ No models available! Check your imports.")
        sys.exit(1)
    
    print(f"✅ Created {len(models)} models following paper methodology")
    print(f"🔧 All models use SAME linear decoder for fair comparison")
    return models

# --- Main Analysis Function ---

def run_synthetic_pattern_analysis():
    """Run comprehensive synthetic pattern analysis"""
    print("🚀 Starting Comprehensive Synthetic Pattern Analysis")
    print("=" * 80)
    
    # Generate time points
    t = torch.linspace(0, 150, 500).to(device)
    
    # Generate synthetic datasets
    datasets = {
        'Periodic': generate_periodic_data(t, noise_level=0.05),
        'Non-Periodic': generate_non_periodic_data(t, noise_level=0.05),
        'Mixed': generate_mixed_data(t, noise_level=0.05)
    }
    
    # Get models
    models = create_models()
    print(models)
    
    print(f"📊 Testing {len(models)} models on {len(datasets)} synthetic patterns")
    print(f"🎯 Models: {list(models.keys())}")
    print(f"📈 Patterns: {list(datasets.keys())}")
    
    # Results storage
    all_results = []
    reconstruction_data = {}
    
    # Test each model on each dataset
    for dataset_name, y_data in datasets.items():
        print(f"\n{'='*20} {dataset_name} Data {'='*20}")
        
        # Normalize data for stable training
        y_data = y_data.to(device)
        y_mean = y_data.mean()
        y_std = y_data.std()
        y_norm = (y_data - y_mean) / y_std
        
        dataset_results = {}
        
        for model_name, model_factory in models.items():
            print(f"\n🔧 Testing {model_name} on {dataset_name} data...")
            
            try:
                # Create fresh model instance
                model = model_factory()
                
                # Move model to device
                model = model.to(device)
                
                # Move tensors to device
                t_device = t.to(device)
                y_norm_device = y_norm.to(device)
                
                # Train model
                training_info = train_model_convergence(model, t_device, y_norm_device, model_name)
                
                # Evaluate reconstruction
                eval_results = evaluate_reconstruction(model, t_device, y_norm_device, model_name)
                
                # Un-normalize prediction for visualization
                pred_unnorm = eval_results['prediction'] * y_std.item() + y_mean.item()
                
                # Store results
                result = {
                    'dataset': dataset_name,
                    'model': model_name,
                    'final_loss': training_info['final_loss'],
                    'converged': training_info['converged'],
                    'convergence_epoch': training_info['convergence_epoch'],
                    'mse': eval_results['mse'],
                    'mae': eval_results['mae'],
                    'rmse': eval_results['rmse'],
                    'r2_score': eval_results['r2_score'],
                    'training_config': SHARED_TRAINING_CONFIG
                }
                
                all_results.append(result)
                dataset_results[model_name] = {
                    'result': result,
                    'prediction': pred_unnorm,
                    'loss_history': training_info['loss_history']
                }
                
                print(f"  ✅ {model_name}: Loss={training_info['final_loss']:.6f}, "
                      f"R²={eval_results['r2_score']:.4f}, "
                      f"Epochs={training_info['convergence_epoch']}")
                      
            except Exception as e:
                print(f"  ❌ {model_name} failed: {str(e)}")
                # Store failed result
                failed_result = {
                    'dataset': dataset_name,
                    'model': model_name,
                    'final_loss': float('inf'),
                    'converged': False,
                    'convergence_epoch': SHARED_TRAINING_CONFIG['max_epochs'],
                    'mse': float('inf'),
                    'mae': float('inf'),
                    'rmse': float('inf'),
                    'r2_score': -1.0,
                    'training_config': SHARED_TRAINING_CONFIG
                }
                all_results.append(failed_result)
        
        reconstruction_data[dataset_name] = {
            'time': t.cpu().numpy() if hasattr(t, 'cpu') else t,
            'original': y_data.cpu().numpy() if hasattr(y_data, 'cpu') else y_data,
            'results': dataset_results
        }
    
    # Create visualizations
    create_comprehensive_visualizations(datasets, reconstruction_data, t)
    
    # Save results
    save_results(all_results, reconstruction_data)
    
    # Generate summary
    generate_analysis_summary(all_results)
    
    return all_results, reconstruction_data

def create_comprehensive_visualizations(datasets, reconstruction_data, t):
    """Create comprehensive visualizations matching the reference style"""
    print("\n🎨 Creating comprehensive visualizations...")
    
    # Ensure we can get numpy arrays from potentially CUDA tensors
    t_np = t.cpu().numpy() if hasattr(t, 'cpu') else t
    
    # Create main figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Top row: Original synthetic patterns
    for i, (dataset_name, y_data) in enumerate(datasets.items()):
        ax = plt.subplot(3, 3, i + 1)
        y_np = y_data.cpu().numpy() if hasattr(y_data, 'cpu') else y_data
        
        plt.plot(t_np, y_np, 'b-', linewidth=2, alpha=0.8)
        plt.title(f'Synthetic {dataset_name} Data', fontsize=12, fontweight='bold')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        # Add pattern description
        if dataset_name == 'Periodic':
            plt.text(0.02, 0.98, 'Multiple harmonics\nwith noise', 
                    transform=ax.transAxes, fontsize=9, 
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor="lightblue", alpha=0.5))
        elif dataset_name == 'Non-Periodic':
            plt.text(0.02, 0.98, 'Exponential decays\nsteps, spikes', 
                    transform=ax.transAxes, fontsize=9, 
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor="lightcoral", alpha=0.5))
        else:
            plt.text(0.02, 0.98, 'Combined periodic\n& non-periodic', 
                    transform=ax.transAxes, fontsize=9, 
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor="lightgreen", alpha=0.5))
    
    # Middle row: Best reconstruction for each pattern
    best_models = {}
    for i, dataset_name in enumerate(datasets.keys()):
        ax = plt.subplot(3, 3, i + 4)
        
        # Find best model (highest R²)
        best_r2 = -float('inf')
        best_model_name = None
        for model_name, model_data in reconstruction_data[dataset_name]['results'].items():
            if model_data['result']['r2_score'] > best_r2:
                best_r2 = model_data['result']['r2_score']
                best_model_name = model_name
        
        if best_model_name:
            best_models[dataset_name] = best_model_name
            original = reconstruction_data[dataset_name]['original']
            prediction = reconstruction_data[dataset_name]['results'][best_model_name]['prediction']
            
            plt.plot(t_np, original, 'b-', linewidth=2, alpha=0.7, label='Original')
            plt.plot(t_np, prediction, 'r--', linewidth=2, alpha=0.8, label=f'{best_model_name}')
            
            plt.title(f'Best Reconstruction: {dataset_name}\n{best_model_name} (R²={best_r2:.3f})', 
                     fontsize=11, fontweight='bold')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.legend(fontsize=9)
            plt.grid(True, alpha=0.3)
        
    # Bottom row: Performance comparison
    for i, dataset_name in enumerate(datasets.keys()):
        ax = plt.subplot(3, 3, i + 7)
        
        model_names = []
        r2_scores = []
        colors = []
        
        for model_name, model_data in reconstruction_data[dataset_name]['results'].items():
            model_names.append(model_name)
            r2_scores.append(model_data['result']['r2_score'])
            # Color coding
            if 'K-MOTE' in model_name or 'KAN-MAMMOTE' in model_name:
                colors.append('red')
            elif 'Expert' in model_name:
                colors.append('orange')
            else:
                colors.append('blue')
        
        bars = plt.barh(model_names, r2_scores, color=colors, alpha=0.7)
        plt.xlabel('R² Score')
        plt.title(f'Model Performance: {dataset_name}', fontsize=11, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for bar, score in zip(bars, r2_scores):
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('analysis_figures_synthetic/comprehensive_synthetic_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create individual detailed plots for each pattern
    create_detailed_pattern_plots(reconstruction_data, t)

def create_detailed_pattern_plots(reconstruction_data, t):
    """Create detailed plots for each pattern"""
    t_np = t.cpu().numpy() if hasattr(t, 'cpu') else t
    
    for dataset_name, data in reconstruction_data.items():
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Detailed Analysis: {dataset_name} Data', fontsize=16)
        
        # Plot 1: Original vs top 3 models
        ax1 = axes[0, 0]
        original = data['original']
        ax1.plot(t_np, original, 'k-', linewidth=3, alpha=0.8, label='Original')
        
        # Get top 3 models by R²
        sorted_models = sorted(data['results'].items(), 
                             key=lambda x: x[1]['result']['r2_score'], reverse=True)
        
        colors = ['red', 'blue', 'green']
        for i, (model_name, model_data) in enumerate(sorted_models[:3]):
            prediction = model_data['prediction']
            r2 = model_data['result']['r2_score']
            ax1.plot(t_np, prediction, '--', linewidth=2, alpha=0.7, 
                    color=colors[i], label=f'{model_name} (R²={r2:.3f})')
        
        ax1.set_title('Top 3 Model Reconstructions')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Amplitude')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Residuals for best model
        ax2 = axes[0, 1]
        if sorted_models:
            best_model_name, best_model_data = sorted_models[0]
            residuals = original - best_model_data['prediction']
            ax2.plot(t_np, residuals, 'r-', linewidth=1, alpha=0.7)
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax2.set_title(f'Residuals: {best_model_name}')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Residual')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Performance metrics
        ax3 = axes[1, 0]
        model_names = []
        metrics = {'R²': [], 'RMSE': []}
        
        for model_name, model_data in data['results'].items():
            model_names.append(model_name)
            metrics['R²'].append(model_data['result']['r2_score'])
            metrics['RMSE'].append(model_data['result']['rmse'])
        
        x = np.arange(len(model_names))
        width = 0.35
        
        ax3_twin = ax3.twinx()
        bars1 = ax3.bar(x - width/2, metrics['R²'], width, label='R²', alpha=0.7, color='blue')
        bars2 = ax3_twin.bar(x + width/2, metrics['RMSE'], width, label='RMSE', alpha=0.7, color='red')
        
        ax3.set_xlabel('Models')
        ax3.set_ylabel('R² Score', color='blue')
        ax3_twin.set_ylabel('RMSE', color='red')
        ax3.set_title('Performance Metrics Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(model_names, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Training convergence
        ax4 = axes[1, 1]
        for model_name, model_data in list(data['results'].items())[:5]:  # Show top 5
            if 'loss_history' in model_data and model_data['loss_history']:
                loss_history = model_data['loss_history']
                ax4.semilogy(loss_history, label=model_name, alpha=0.7)
        
        ax4.set_title('Training Convergence')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss (log scale)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        safe_filename = dataset_name.lower().replace(' ', '_').replace('-', '_')
        plt.savefig(f'analysis_figures_synthetic/detailed_{safe_filename}_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()

def save_results(all_results, reconstruction_data):
    """Save results to CSV files"""
    print("\n💾 Saving results to CSV files...")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Save main results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    main_file = f'analysis_results_synthetic/synthetic_analysis_{timestamp}.csv'
    df.to_csv(main_file, index=False)
    
    # Save summary by pattern
    summary_file = f'analysis_results_synthetic/synthetic_summary_{timestamp}.csv'
    summary_data = []
    
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        summary_data.append({
            'Pattern': dataset,
            'Best_Model': dataset_df.loc[dataset_df['r2_score'].idxmax(), 'model'],
            'Best_R2': dataset_df['r2_score'].max(),
            'Best_RMSE': dataset_df.loc[dataset_df['r2_score'].idxmax(), 'rmse'],
            'Avg_R2': dataset_df['r2_score'].mean(),
            'Avg_Convergence_Epochs': dataset_df['convergence_epoch'].mean()
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_file, index=False)
    
    # Save actual vs prediction CSV files for each model/dataset combination
    print("\n💾 Saving actual vs prediction data for each model...")
    prediction_files = []
    
    for dataset_name, data in reconstruction_data.items():
        time_data = data['time']
        actual_data = data['original']
        
        for model_name, model_data in data['results'].items():
            if 'prediction' in model_data:
                prediction_data = model_data['prediction']
                
                # Create prediction DataFrame
                pred_df = pd.DataFrame({
                    'time': time_data,
                    'actual': actual_data,
                    'prediction': prediction_data,
                    'residual': actual_data - prediction_data
                })
                
                # Safe filename
                safe_dataset = dataset_name.lower().replace(' ', '_').replace('-', '_')
                safe_model = model_name.lower().replace(' ', '_').replace('-', '_')
                pred_file = f'analysis_results_synthetic/actual_vs_pred_{safe_dataset}_{safe_model}_{timestamp}.csv'
                
                pred_df.to_csv(pred_file, index=False)
                prediction_files.append(pred_file)
                print(f"  ✅ {dataset_name} - {model_name}: {pred_file}")
    
    print(f"✅ Main results saved to: {main_file}")
    print(f"✅ Summary saved to: {summary_file}")
    print(f"✅ {len(prediction_files)} prediction files saved")
    
    return main_file, summary_file, prediction_files

def generate_analysis_summary(all_results):
    """Generate and print analysis summary"""
    print("\n" + "="*80)
    print("📊 SYNTHETIC PATTERN ANALYSIS SUMMARY")
    print("="*80)
    
    df = pd.DataFrame(all_results)
    
    # Overall best performers
    print(f"\n🏆 OVERALL BEST PERFORMERS:")
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        best_idx = dataset_df['r2_score'].idxmax()
        best_result = dataset_df.loc[best_idx]
        
        print(f"\n📈 {dataset} Data:")
        print(f"   🥇 Best Model: {best_result['model']}")
        print(f"   📊 R² Score: {best_result['r2_score']:.4f}")
        print(f"   📉 RMSE: {best_result['rmse']:.6f}")
        print(f"   ⏱️ Convergence: {best_result['convergence_epoch']} epochs")
        print(f"   ✅ Converged: {best_result['converged']}")
    
    # Model comparison across patterns
    print(f"\n🔄 MODEL PERFORMANCE ACROSS PATTERNS:")
    model_summary = df.groupby('model').agg({
        'r2_score': ['mean', 'std', 'max'],
        'rmse': ['mean', 'std', 'min'],
        'convergence_epoch': 'mean',
        'converged': 'sum'
    }).round(4)
    
    print(model_summary)
    
    # Pattern difficulty ranking
    print(f"\n📊 PATTERN DIFFICULTY RANKING (by average R²):")
    pattern_difficulty = df.groupby('dataset')['r2_score'].mean().sort_values(ascending=False)
    for i, (pattern, avg_r2) in enumerate(pattern_difficulty.items(), 1):
        difficulty = "Easy" if avg_r2 > 0.8 else "Medium" if avg_r2 > 0.5 else "Hard"
        print(f"   {i}. {pattern}: {avg_r2:.4f} ({difficulty})")
    
    print(f"\n✨ Analysis complete! Check 'analysis_figures_synthetic' for visualizations.")

# --- Main Execution ---

if __name__ == '__main__':
    print("🚀 Comprehensive Synthetic Pattern Analysis for Time Encoders")
    print("=" * 80)
    print("📊 METHODOLOGY: Time (1D) → Encoder (64D) → MLP Decoder → Output (1D)")
    print("🔬 Analyzing time encoder performance on synthetic patterns")
    print("⚙️ Using proper encode→decode methodology with shared hyperparameters")
    print("🎯 All models trained end-to-end with 64-dimensional embeddings")
    print("=" * 80)
    
    # Run analysis
    results, reconstruction_data = run_synthetic_pattern_analysis()
    
    print(f"\n🎉 Analysis completed successfully!")
    print(f"📁 Results saved in: analysis_results_synthetic/")
    print(f"🖼️ Plots saved in: analysis_figures_synthetic/")
    print(f"📋 Total experiments: {len(results)}")