# kan_mamote/src/models/c_mamba.py
from src.utils.config import KANMAMOTEConfig
import torch
import torch.nn as nn

import torch.nn.functional as F
from typing import Optional, Tuple
from src.utils.config import KANMAMOTEConfig
import math
import numpy as np


try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    from builtins import print
    print("Warning: mamba_ssm not available. Using fallback implementation.")
    MAMBA_AVAILABLE = False

class ContinuousMambaBlock(nn.Module):
    """
    Continuous-Time Mamba Block inspired by DyGMamba for KAN-MAMMOTE.

    This implementation takes time embeddings from K-MOTE as input and uses 
    mamba_ssm.Mamba for the core selective state space model with continuous-time
    modeling capabilities through time difference encoding.

    Key features:
    - Takes time embeddings from K-MOTE (not raw timestamps)
    - Uses standard Mamba from mamba_ssm for SSM operations
    - Time difference modeling for continuous sequences
    - Residual connections and layer normalization
    """

    def __init__(self, input_dim: int, config: KANMAMOTEConfig):
        super().__init__()
        self.input_dim = input_dim  # This should be D_time from K-MOTE
        self.hidden_dim = config.hidden_dim_mamba
        self.state_dim = config.state_dim_mamba
        self.num_layers = getattr(config, 'num_mamba_layers', 2)
        self.gamma = getattr(config, 'gamma', 0.5)  # For time difference scaling

        # Time difference encoder for modeling temporal gaps
        self.time_diff_encoder = TimeEncoder(time_dim=self.hidden_dim // 2)

        # Input projection to match Mamba's expected dimension
        # Note: input_dim is now D_time from K-MOTE time embeddings
        self.input_proj = nn.Linear(input_dim, self.hidden_dim)

        # Time difference projection layers
        # Note: time_diff_encoder outputs hidden_dim//2, so we need to project that
        self.time_diff_proj = nn.Linear(self.hidden_dim // 2, int(self.gamma * self.hidden_dim))
        self.time_diff_proj_up = nn.Linear(int(self.gamma * self.hidden_dim), self.hidden_dim)

        # Main Mamba blocks (using mamba_ssm)
        if MAMBA_AVAILABLE:
            from mamba_ssm import Mamba  # Ensure Mamba is imported only if available
            self.mamba_layers = nn.ModuleList([
                Mamba(
                    d_model=self.hidden_dim,    # Model dimension
                    d_state=16,                 # SSM state expansion factor
                    d_conv=4,                   # Local convolution width
                    expand=2,                   # Block expansion factor
                )
                for _ in range(self.num_layers)
            ])

            # Time difference Mamba (smaller dimension)
            self.mamba_time_diff = nn.ModuleList([
                Mamba(
                    d_model=int(self.gamma * self.hidden_dim),
                    d_state=16,
                    d_conv=4,
                    expand=2,
                )
                for _ in range(self.num_layers)
            ])
        else:
            # Fallback to simple transformer-like layers
            self.mamba_layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=self.hidden_dim,
                    nhead=8,
                    dim_feedforward=self.hidden_dim * 4,
                    dropout=0.1,
                    batch_first=True
                )
                for _ in range(self.num_layers)
            ])

            self.mamba_time_diff = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=int(self.gamma * self.hidden_dim),
                    nhead=4,
                    dim_feedforward=int(self.gamma * self.hidden_dim) * 4,
                    dropout=0.1,
                    batch_first=True
                )
                for _ in range(self.num_layers)
            ])

        # Layer normalization and feedforward
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.feedforward = FeedForwardNet(
            input_dim=self.hidden_dim,
            dim_expansion_factor=4,
            dropout=0.1
        )

        # Weight aggregation layer (for attention-like mechanism)
        self.weight_agg = nn.Linear(self.hidden_dim, 1)

        # Initialize parameters
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initialize parameters for stability"""
        with torch.no_grad():
            # Initialize projections
            nn.init.xavier_uniform_(self.input_proj.weight, gain=0.1)
            nn.init.zeros_(self.input_proj.bias)

            nn.init.xavier_uniform_(self.time_diff_proj.weight, gain=0.1)
            nn.init.zeros_(self.time_diff_proj.bias)

            nn.init.xavier_uniform_(self.time_diff_proj_up.weight, gain=0.1)
            nn.init.zeros_(self.time_diff_proj_up.bias)

    def compute_time_differences(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Compute time differences between consecutive events.
        Similar to DyGMamba's time modeling approach.
        
        Args:
            timestamps: (batch_size, seq_len, 1) - timestamps for each event
            
        Returns:
            time_diffs: (batch_size, seq_len, 1) - time differences
        """
        batch_size, seq_len, _ = timestamps.shape
        device = timestamps.device
        
        # Compute differences between consecutive timestamps
        # Pad with zeros for the first timestamp (no previous event)
        time_diffs = torch.zeros_like(timestamps)
        if seq_len > 1:
            time_diffs[:, 1:, :] = timestamps[:, 1:, :] - timestamps[:, :-1, :]
        
        # Apply exponential decay similar to DyGMamba
        shrink_coeff = 1e-8  # Similar to DyGMamba's shrink coefficient
        time_diffs = torch.exp(-time_diffs * shrink_coeff)
        
        return time_diffs

    def forward(self, 
                time_embeddings: torch.Tensor,      # (batch_size, seq_len, D_time) - from K-MOTE
                timestamps: torch.Tensor,           # (batch_size, seq_len, 1) - raw timestamps for time differences
                initial_state: Optional[torch.Tensor] = None  # Not used in this implementation
               ) -> torch.Tensor:
        """
        Forward pass through the continuous-time Mamba block.
        
        Args:
            time_embeddings: Input time embeddings from K-MOTE (batch_size, seq_len, D_time)
            timestamps: Raw timestamps for computing time differences (batch_size, seq_len, 1)
            initial_state: Initial hidden state (optional, not used)
            
        Returns:
            Output sequence with temporal dependencies modeled
        """
        batch_size, seq_len, _ = time_embeddings.shape
        
        # Project K-MOTE time embeddings to hidden dimension
        u_proj = self.input_proj(time_embeddings)  # (batch_size, seq_len, hidden_dim)
        
        # Compute time differences
        time_diffs = self.compute_time_differences(timestamps)  # (batch_size, seq_len, 1)
        
        # Encode time differences using the time encoder (fix attribute name)
        time_diff_emb = self.time_diff_encoder(time_diffs.squeeze(-1))  # (batch_size, seq_len, hidden_dim//2)
        
        # Project time differences to smaller dimension
        time_diff_projected = self.time_diff_proj(time_diff_emb)  # (batch_size, seq_len, gamma*hidden_dim)
        
        # Process time differences through Mamba layers
        time_diff_output = time_diff_projected
        for mamba_t in self.mamba_time_diff:
            if MAMBA_AVAILABLE:
                # For mamba_ssm, we need to handle the sequence processing
                time_diff_output = mamba_t(time_diff_output) + time_diff_output
            else:
                # For transformer fallback
                time_diff_output = mamba_t(time_diff_output) + time_diff_output
        
        # Project time difference output back up
        time_diff_final = self.time_diff_proj_up(time_diff_output)  # (batch_size, seq_len, hidden_dim)
        
        # Combine input embeddings with time difference information
        combined_input = u_proj + time_diff_final
        
        # Process through main Mamba layers
        mamba_output = combined_input
        for mamba_layer in self.mamba_layers:
            if MAMBA_AVAILABLE:
                # Mamba with residual connection
                mamba_output = mamba_layer(mamba_output) + mamba_output
            else:
                # Transformer fallback with residual connection
                mamba_output = mamba_layer(mamba_output) + mamba_output
            
            # Layer normalization
            mamba_output = self.layer_norm(mamba_output)
            
            # Feedforward with residual connection
            mamba_output = self.feedforward(mamba_output) + mamba_output
        
        # Apply attention-like weighting (similar to DyGMamba's weightagg)
        weights = self.weight_agg(mamba_output).transpose(1, 2)  # (batch_size, 1, seq_len)
        weights = F.softmax(weights, dim=-1)
        
        # Weighted combination of sequence elements
        weighted_output = torch.bmm(weights, mamba_output)  # (batch_size, 1, hidden_dim)
        
        # Broadcast back to sequence length for compatibility
        output_sequence = weighted_output.expand(-1, seq_len, -1)  # (batch_size, seq_len, hidden_dim)
        
        return output_sequence


class TimeEncoder(nn.Module):
    """
    Time encoder similar to DyGMamba's TimeEncoder.
    Encodes timestamps using learnable periodic functions.
    """
    
    def __init__(self, time_dim: int):
        super().__init__()
        self.time_dim = time_dim
        
        # Learnable frequency and phase parameters
        self.freq = nn.Parameter(torch.randn(time_dim // 2) * 0.1)
        self.phase = nn.Parameter(torch.randn(time_dim // 2) * 0.1)
        
        # Linear projection layer
        self.linear = nn.Linear(time_dim, time_dim)
        
    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Encode timestamps using periodic functions.
        
        Args:
            timestamps: (batch_size, seq_len) or (batch_size, seq_len, 1)
            
        Returns:
            time_features: (batch_size, seq_len, time_dim)
        """
        if timestamps.dim() == 3:
            timestamps = timestamps.squeeze(-1)  # Remove last dimension if present
        
        batch_size, seq_len = timestamps.shape
        
        # Expand timestamps for broadcasting
        t_expanded = timestamps.unsqueeze(-1)  # (batch_size, seq_len, 1)
        
        # Compute periodic features
        freq_expanded = self.freq.unsqueeze(0).unsqueeze(0)  # (1, 1, time_dim//2)
        phase_expanded = self.phase.unsqueeze(0).unsqueeze(0)  # (1, 1, time_dim//2)
        
        # Sine and cosine features
        sin_features = torch.sin(t_expanded * freq_expanded + phase_expanded)
        cos_features = torch.cos(t_expanded * freq_expanded + phase_expanded)
        
        # Concatenate sin and cos features
        periodic_features = torch.cat([sin_features, cos_features], dim=-1)  # (batch_size, seq_len, time_dim)
        
        # Apply linear transformation
        time_features = self.linear(periodic_features)
        
        return time_features


class FeedForwardNet(nn.Module):
    """
    Feed-forward network with GELU activation (from DyGMamba).
    """
    
    def __init__(self, input_dim: int, dim_expansion_factor: float, dropout: float = 0.0):
        super().__init__()
        
        self.input_dim = input_dim
        self.dim_expansion_factor = dim_expansion_factor
        self.dropout = dropout
        
        expanded_dim = int(dim_expansion_factor * input_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, expanded_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expanded_dim, input_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feed forward network forward pass.
        
        Args:
            x: Input tensor of shape (*, input_dim)
            
        Returns:
            Output tensor of the same shape as input
        """
        return self.ffn(x)


class SimplifiedContinuousMambaBlock(nn.Module):
    """
    Simplified Continuous-Time Mamba Block using standard Mamba from mamba_ssm.
    
    This version takes time embeddings from K-MOTE as input and provides a cleaner 
    interface while maintaining the core continuous-time modeling capabilities. 
    It uses the DyGMamba approach of combining standard Mamba blocks with time difference modeling.
    """
    
    def __init__(self, input_dim: int, config: KANMAMOTEConfig):
        super().__init__()
        self.input_dim = input_dim  # Should be D_time from K-MOTE
        self.hidden_dim = config.hidden_dim_mamba
        self.state_dim = config.state_dim_mamba
        
        # Input projection from K-MOTE embeddings to Mamba hidden dimension
        self.input_proj = nn.Linear(input_dim, self.hidden_dim)
        
        # Time encoder for continuous modeling (for time differences)
        self.time_encoder = TimeEncoder(time_dim=self.hidden_dim // 2)
        # Single Mamba layer (simpler than the full implementation)
        if MAMBA_AVAILABLE:
            from mamba_ssm import Mamba  # Ensure Mamba is imported only if available
            self.mamba = Mamba(
                d_model=self.hidden_dim,
                d_state=16,
                d_conv=4,
                expand=2,
            )
        else:
            # Fallback to LSTM
            self.mamba = nn.LSTM(
                input_size=self.hidden_dim,
                hidden_size=self.hidden_dim,
                batch_first=True,
                dropout=0.1
            )
            
        
        # Output normalization
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        
    def forward(self, 
                time_embeddings: torch.Tensor,      # (batch_size, seq_len, D_time) - from K-MOTE
                timestamps: torch.Tensor,           # (batch_size, seq_len, 1) - raw timestamps for time differences
                initial_state: Optional[torch.Tensor] = None
               ) -> torch.Tensor:
        """
        Simplified forward pass through the continuous-time Mamba block.
        
        Args:
            time_embeddings: Input time embeddings from K-MOTE (batch_size, seq_len, D_time)
            timestamps: Raw timestamps for computing time differences (batch_size, seq_len, 1) 
            initial_state: Initial hidden state (optional, not used)
            
        Returns:
            Output sequence with temporal dependencies modeled (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = time_embeddings.shape
        
        # Project K-MOTE time embeddings to hidden dimension
        u_proj = self.input_proj(time_embeddings)  # (batch_size, seq_len, hidden_dim)
        
        # Encode time differences for continuous modeling
        if timestamps.dim() == 3:
            timestamps = timestamps.squeeze(-1)  # Remove last dimension
        
        # Compute time differences (similar to DyGMamba approach)
        time_diffs = torch.zeros_like(timestamps)
        if seq_len > 1:
            time_diffs[:, 1:] = timestamps[:, 1:] - timestamps[:, :-1]
        
        # Encode time differences
        time_features = self.time_encoder(time_diffs)  # (batch_size, seq_len, hidden_dim//2)
        
        # Combine input features with time features
        # Pad time features to match hidden_dim if needed
        if time_features.shape[-1] == u_proj.shape[-1]:
            combined_input = u_proj + time_features
        else:
            # Pad time features to match hidden_dim
            time_features_padded = F.pad(time_features, (0, self.hidden_dim - time_features.shape[-1]))
            combined_input = u_proj + time_features_padded
        
        # Process through Mamba
        if MAMBA_AVAILABLE:
            # Standard Mamba forward pass
            mamba_output = self.mamba(combined_input)
            # Add residual connection
            output = mamba_output + combined_input
        else:
            # LSTM fallback
            mamba_output, _ = self.mamba(combined_input)
            output = mamba_output + combined_input
        
        # Apply layer normalization
        output = self.layer_norm(output)
        
        return output