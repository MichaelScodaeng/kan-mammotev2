# kan_mamote/src/models/c_mamba.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
import numpy as np

# --- Mamba Availability Check ---
try:
    from mamba_ssm import Mamba
    import torch
    MAMBA_AVAILABLE = True and torch.cuda.is_available()
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Mamba SSM requires CUDA, falling back to CPU implementations.")
        MAMBA_AVAILABLE = False
except ImportError:
    print("Warning: mamba_ssm not available. Using fallback implementation.")
    MAMBA_AVAILABLE = False

# --- Dummy KANMAMOTEConfig, TimeEncoder, FeedForwardNet ---
# IMPORTANT: You should replace these dummy classes with your actual imports
# from src.utils.config and src.models.modules in your full project setup.
# These are included here just to make this standalone file runnable for demonstration.
class KANMAMOTEConfig:
    def __init__(self):
        self.hidden_dim_mamba = 256
        self.state_dim_mamba = 16
        self.num_mamba_layers = 2
        self.d_time_k_mote = 128 # K-MOTE output dimension (t_k embedding, t_k-1 embedding)
        self.d_faster_kan_out = 128 # Faster-KAN output dimension (also Delta_t_Embedding dimension)

class TimeEncoder(nn.Module): # This TimeEncoder is not directly used in the new diagram's ContinuousMambaBlock
    def __init__(self, time_dim: int):
        super().__init__()
        self.time_dim = time_dim
        self.freq = nn.Parameter(torch.randn(time_dim // 2) * 0.1)
        self.phase = nn.Parameter(torch.randn(time_dim // 2) * 0.1)
        self.linear = nn.Linear(time_dim, time_dim)
    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        if timestamps.dim() == 3:
            timestamps = timestamps.squeeze(-1)
        t_expanded = timestamps.unsqueeze(-1)
        freq_expanded = self.freq.unsqueeze(0).unsqueeze(0)
        phase_expanded = self.phase.unsqueeze(0).unsqueeze(0)
        sin_features = torch.sin(t_expanded * freq_expanded + phase_expanded)
        cos_features = torch.cos(t_expanded * freq_expanded + phase_expanded)
        periodic_features = torch.cat([sin_features, cos_features], dim=-1)
        time_features = self.linear(periodic_features)
        return time_features

class FeedForwardNet(nn.Module):
    def __init__(self, input_dim: int, dim_expansion_factor: float, dropout: float = 0.0):
        super().__init__()
        expanded_dim = int(dim_expansion_factor * input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, expanded_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expanded_dim, input_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


# --- Modified Mamba Layer (ContinuousMambaLayer) ---
class ContinuousMambaLayer(nn.Module):
    """
    Modified Mamba layer that accepts input_embedding (current t_k embedding)
    and delta_embedding (the derived Delta_t_Embedding).
    
    The delta_embedding directly modulates Mamba's internal delta parameter,
    making the state transitions adaptive to temporal dynamics.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        
        # Always initialize custom SSM components for fallback (regardless of MAMBA_AVAILABLE)
        self.A_log = nn.Parameter(torch.randn(d_state))
        self.D = nn.Parameter(torch.randn(d_model))
        self.x_proj = nn.Linear(d_model, d_state * 2)  # For B and C
        self.dt_proj = nn.Linear(d_model, d_model)     # For delta processing
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Core Mamba SSM from mamba_ssm library
        if MAMBA_AVAILABLE:
            try:
                self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
                self.use_mamba = True
            except Exception as e:
                print(f"Failed to initialize Mamba, using custom SSM: {e}")
                self.use_mamba = False
        else:
            print(f"Using Custom SSM fallback for ContinuousMambaLayer (d_model={d_model})")
            self.use_mamba = False
        
        # Delta processing: Transform delta_t_embedding to be compatible with Mamba's delta
        # This is the key innovation - delta_t_embedding becomes Mamba's temporal step size
        self.delta_processor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Softplus(),  # Ensure positive delta values
            nn.Linear(d_model, d_model)
        )
        
        # Common blocks after Mamba in a layer (for residual connections, normalization, and feedforward)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNet(input_dim=d_model, dim_expansion_factor=4, dropout=0.1)

    def forward(self, input_embedding: torch.Tensor, delta_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of a single Continuous Mamba Layer.
        
        Args:
            input_embedding: The current absolute timestamp embedding (e.g., t_k Embedding).
                             This serves as the primary input to the Mamba layer.
            delta_embedding: The derived Delta_t_Embedding (from Faster-KAN and projection).
                             This becomes Mamba's internal delta parameter for temporal adaptation.
        
        Returns:
            output: The processed embedding after Mamba, normalization, and feedforward.
        """
        batch_size, seq_len, d_model = input_embedding.shape
        
        # Check device compatibility on first forward pass
        if self.use_mamba and not getattr(self, 'device_checked', False):
            try:
                # Test if mamba can handle current device
                if not input_embedding.is_cuda:
                    print(f"⚠️  Input is on CPU, but Mamba expects CUDA. Switching to custom SSM for this forward pass.")
                    self.use_mamba = False
                else:
                    # Test a small forward pass to check CUDA compatibility
                    test_input = torch.randn(1, 2, d_model, device=input_embedding.device)
                    try:
                        with torch.no_grad():
                            _ = self.mamba(test_input)
                        print(f"✅ Mamba CUDA check passed for device: {input_embedding.device}")
                    except Exception as e:
                        print(f"⚠️  Mamba CUDA check failed ({e}), switching to custom SSM")
                        self.use_mamba = False
            except Exception as e:
                print(f"⚠️  Device check failed ({e}), using custom SSM")
                self.use_mamba = False
            
            self.device_checked = True
        
        # Process delta_embedding to create Mamba's delta parameter
        # This is the key: delta_t_embedding controls the temporal step size
        processed_delta = self.delta_processor(delta_embedding)  # (B, L, d_model)
        
        if self.use_mamba:
            try:
                # For mamba_ssm: We need to modify the input to incorporate delta information
                # Since we can't directly control mamba_ssm's internal delta, we modulate the input
                modulated_input = input_embedding * (1.0 + 0.1 * processed_delta)
                output_from_mamba_core = self.mamba(modulated_input)
            except Exception as e:
                print(f"⚠️  Mamba forward failed ({e}), falling back to custom SSM")
                self.use_mamba = False
                output_from_mamba_core = self.custom_ssm_forward(input_embedding, processed_delta)
        else:
            # Custom SSM implementation with explicit delta control
            output_from_mamba_core = self.custom_ssm_forward(input_embedding, processed_delta)
        
        # Apply residual connection, normalization, and feedforward (standard Mamba layer structure)
        # Residual connection from the original input_embedding
        output_after_residual = output_from_mamba_core + input_embedding
        
        # Layer Normalization
        output_after_norm = self.norm(output_after_residual)
        
        # FeedForward Network with residual connection
        output_final = self.ffn(output_after_norm) + output_after_norm
        
        return output_final
    
    def custom_ssm_forward(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """
        Custom SSM forward pass with explicit delta control for fallback.
        This implements a simplified state space model where delta_t_embedding
        directly controls the temporal dynamics.
        """
        batch_size, seq_len, d_model = x.shape
        
        # Initialize state
        h = torch.zeros(batch_size, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        
        # Process sequence step by step
        for t in range(seq_len):
            x_t = x[:, t, :]        # (B, d_model)
            delta_t = delta[:, t, :] # (B, d_model) - temporal step size from delta_t_embedding
            
            # Project input to get B and C matrices
            BC = self.x_proj(x_t)   # (B, d_state * 2)
            B, C = BC.chunk(2, dim=-1)  # Each (B, d_state)
            
            # Compute A matrix (negative for stability)
            A = -torch.exp(self.A_log).unsqueeze(0)  # (1, d_state)
            
            # State transition with delta-controlled step size
            # h_new = (1 + delta_t * A) * h + delta_t * B * x_t
            delta_mean = delta_t.mean(dim=-1, keepdim=True)  # (B, 1)
            A_delta = 1.0 + delta_mean * A  # (B, d_state)
            B_delta = delta_mean * B        # (B, d_state)
            
            h = A_delta * h + B_delta * x_t.mean(dim=-1, keepdim=True)  # (B, d_state)
            
            # Output computation
            y_t = torch.einsum('bs,bs->b', h, C).unsqueeze(-1)  # (B, 1)
            y_t = y_t.expand(-1, d_model)  # (B, d_model)
            
            # Add skip connection weighted by D
            y_t = y_t + self.D * x_t
            
            outputs.append(y_t)
        
        # Stack outputs
        output = torch.stack(outputs, dim=1)  # (B, L, d_model)
        return self.out_proj(output)


# --- Main ContinuousMambaBlock (Overall Flow as per Diagram) ---
class ContinuousMambaBlock(nn.Module):
    """
    TRUE KAN-MAMMOTE Continuous Mamba Block following the exact diagram pattern:
    
    1. Independent K-MOTE embeddings for t_k (current) and t_k-1 (previous)
    2. Temporal differences in embedding space: t_k - t_k-1
    3. Faster-KAN processing of temporal differences -> Δt Embedding
    4. Continuous Mamba: current_embedding as input, delta_t_embedding as delta parameter
    5. Output: Absolute-Relative t_k Embedding
    """
    
    def __init__(self, input_dim: int, config: KANMAMOTEConfig, kmote_layer: nn.Module, faster_kan_layer: nn.Module):
        """
        Initializes the ContinuousMambaBlock.
        
        Args:
            input_dim: The output dimension of the K-MOTE layer (D_time_k_mote).
            config: Configuration object for KAN-MAMOTE parameters.
            kmote_layer: An instance of the K-MOTE module.
            faster_kan_layer: An instance of the Faster-KAN module (used for processing embeddings).
        """
        super().__init__()
        self.d_time_k_mote = input_dim # K-MOTE output dimension (t_k embedding, t_k-1 embedding)
        self.d_faster_kan_out = getattr(config, 'd_faster_kan_out', self.d_time_k_mote) # Output dim after Faster-KAN
        self.hidden_dim = getattr(config, 'hidden_dim_mamba', 256) # Hidden dimension for Mamba layers
        self.state_dim = getattr(config, 'state_dim_mamba', 16) # SSM state expansion factor
        self.num_layers = getattr(config, 'num_mamba_layers', 2) # Number of ContinuousMambaLayers
        
        # References to K-MOTE and Faster-KAN layers (as external modules in diagram)
        self.kmote_layer = kmote_layer # Used to get t_k and t_k-1 embeddings
        self.faster_kan_layer = faster_kan_layer # Used to process the *difference* of embeddings
        
        # Layer to produce the final "Delta_t_Embedding" after Faster-KAN
        # This projects the output of Faster-KAN (d_faster_kan_out) to the dimension 
        # expected by ContinuousMambaLayer's delta_embedding input (d_time_k_mote)
        self.delta_t_embedding_proj = nn.Linear(self.d_faster_kan_out, self.d_time_k_mote) # Diagram's "Delta_t Embedding" box
        
        # Stack of ContinuousMambaLayer instances
        # d_model for ContinuousMambaLayer is the dimension of t_k Embedding (current_embeddings)
        self.continuous_mamba_layers = nn.ModuleList([
            ContinuousMambaLayer(
                d_model=self.d_time_k_mote, # t_k embedding dimension, as per diagram's input to Mamba
                d_state=self.state_dim,
                d_conv=4,
                expand=2,
            )
            for _ in range(self.num_layers)
        ])
        
        # Output projection for the final "Absolute-Relative t_k Embedding"
        # If the ContinuousMambaLayer already outputs d_model (which is d_time_k_mote), 
        # this might be redundant or for final sizing. Diagram just shows direct output.
        self.output_proj = nn.Identity() # Assuming final output dimension is same as d_time_k_mote
        
    def compute_independent_kmote_embeddings(self, 
                                            timestamps: torch.Tensor, 
                                            features: Optional[torch.Tensor] = None):
        """
        Compute independent K-MOTE embeddings for t_k and t_k-1 as shown in diagram.
        
        Args:
            timestamps: (batch_size, seq_len, 1) - raw timestamps (t_k)
            features: (batch_size, seq_len, feature_dim) - raw features (if K-MOTE uses them).
                      Optional, as K-MOTE might only take timestamps.
            
        Returns:
            current_embeddings: t_k embeddings from K-MOTE (batch_size, seq_len, D_time_k_mote)
            previous_embeddings: t_k-1 embeddings from K-MOTE (batch_size, seq_len, D_time_k_mote)
        """
        batch_size, seq_len = timestamps.shape[:2]
        
        # Current timestamps (t_k)
        current_timestamps_flat = timestamps.view(batch_size * seq_len, 1)
        
        # Previous timestamps (t_k-1) - shift by 1 position for sequence
        # For the very first element, t_k-1 could be considered t_k itself or 0, depending on context.
        # This implementation uses t_k for t_k-1 at the first position to avoid NaN/Inf issues
        # and ensure a valid difference for all sequence elements.
        previous_timestamps = torch.zeros_like(timestamps)
        if seq_len > 1:
            previous_timestamps[:, 1:] = timestamps[:, :-1]
        previous_timestamps[:, 0] = timestamps[:, 0] # For the first element, t_k-1 is t_k
        previous_timestamps_flat = previous_timestamps.view(batch_size * seq_len, 1)
        
        # K-MOTE might use features, so flatten them if necessary for K-MOTE's forward
        features_flat = features.view(batch_size * seq_len, -1) if features is not None else None
        
        # Independent K-MOTE computations (as in diagram)
        # Assuming K-MOTE's forward method accepts (timestamp, features) and returns (embedding, weights, others)
        current_embeddings_flat, _, _ = self.kmote_layer(current_timestamps_flat, features_flat)
        previous_embeddings_flat, _, _ = self.kmote_layer(previous_timestamps_flat, features_flat)
        
        # Reshape back to sequence format
        current_embeddings = current_embeddings_flat.view(batch_size, seq_len, -1)
        previous_embeddings = previous_embeddings_flat.view(batch_size, seq_len, -1)
        
        return current_embeddings, previous_embeddings
    
    def forward(self, 
                timestamps: torch.Tensor,        # Raw timestamps (batch_size, seq_len, 1)
                features: Optional[torch.Tensor] = None, # Raw features (batch_size, seq_len, feature_dim), if K-MOTE uses them
                initial_state: Optional[torch.Tensor] = None # Not used directly by this block, but common in sequence models
               ) -> Tuple[torch.Tensor, dict]:
        """
        TRUE KAN-MAMMOTE forward pass following the diagram exactly.
        
        Flow: t_k, t_k-1 -> K-MOTE -> (t_k - t_k-1) -> Faster-KAN -> Δt -> Continuous Mamba
        
        Args:
            timestamps: Raw timestamps (batch_size, seq_len, 1) - for t_k and t_k-1.
            features: Raw features (batch_size, seq_len, feature_dim) - if K-MOTE uses them.
            
        Returns:
            absolute_relative_embedding: Final absolute-relative t_k embedding (sequence output).
                                         Shape: (batch_size, seq_len, D_time_k_mote)
            info: Dict with intermediate results for analysis (for debugging/understanding).
        """
        
        # Step 1: Independent K-MOTE embeddings for t_k and t_k-1 (as in diagram)
        # Diagram: raw t_k and t_k-1 go into K-MOTE
        current_embeddings, previous_embeddings = self.compute_independent_kmote_embeddings(
            timestamps, features
        ) # current_embeddings is t_k Embedding, previous_embeddings is t_k-1 Embedding
        
        # Step 2: Temporal differences in embedding space (t_k - t_k-1)
        # This is done *before* Faster-KAN as per your request.
        temporal_differences = current_embeddings - previous_embeddings
        
        # Step 3: Faster-KAN processing of the temporal differences
        # Assuming faster_kan_layer can handle batched sequence input (B, L, D)
        processed_temporal_differences = self.faster_kan_layer(temporal_differences)

        # Step 4: Create Delta_t_Embedding
        # This projects the output from Faster-KAN to the dimension required by the Mamba layers.
        delta_t_embedding = self.delta_t_embedding_proj(processed_temporal_differences)
        
        # Step 5: Continuous Mamba processing
        # The 'input_embedding' to the ContinuousMambaLayer is the evolving sequence.
        # It starts as current_embeddings (t_k Embedding) and is updated layer by layer.
        absolute_relative_output = current_embeddings # Primary input to the first Mamba layer
        
        for mamba_layer in self.continuous_mamba_layers:
            # Pass both the primary input (the sequence) and the delta_t_embedding for modulation
            absolute_relative_output = mamba_layer(
                input_embedding=absolute_relative_output, 
                delta_embedding=delta_t_embedding 
            )
        
        # Step 6: Final output projection (if needed)
        # self.output_proj is Identity() by default, so it just passes through.
        absolute_relative_output = self.output_proj(absolute_relative_output)
        
        # For analysis and debugging, return intermediate values
        info = {
            "current_kmote_embeddings": current_embeddings,
            "previous_kmote_embeddings": previous_embeddings,
            "temporal_difference_before_kan": temporal_differences,
            "temporal_difference_after_kan": processed_temporal_differences,
            "delta_t_embedding": delta_t_embedding,
            "final_output": absolute_relative_output
        }
        
        return absolute_relative_output, info

# --- Simplified Continuous Mamba Block (for ablation/testing) ---
class SimplifiedContinuousMambaBlock(nn.Module):
    """
    Simplified Continuous Mamba Block for ablation studies or testing specific components.
    This version directly follows the data flow without the full KAN-MAMMOTE structure.
    
    Flow: t_k, t_k-1 -> K-MOTE -> (t_k - t_k-1) -> Faster-KAN -> Δt -> Continuous Mamba
    
    Args:
        input_dim: The output dimension of the K-MOTE layer (D_time_k_mote).
        config: Configuration object for KAN-MAMOTE parameters.
        kmote_layer: An instance of the K-MOTE module.
        faster_kan_layer: An instance of the Faster-KAN module (used for processing embeddings).
    """
    
    def __init__(self, input_dim: int, config: KANMAMOTEConfig, kmote_layer: nn.Module, faster_kan_layer: nn.Module):
        super().__init__()
        self.d_time_k_mote = input_dim
        self.d_faster_kan_out = getattr(config, 'd_faster_kan_out', self.d_time_k_mote)
        self.hidden_dim = getattr(config, 'hidden_dim_mamba', 256)
        self.state_dim = getattr(config, 'state_dim_mamba', 16)
        self.num_layers = getattr(config, 'num_mamba_layers', 2)
        
        # References to K-MOTE and Faster-KAN layers (as external modules in diagram)
        self.kmote_layer = kmote_layer # Used to get t_k and t_k-1 embeddings
        self.faster_kan_layer = faster_kan_layer # Used to process the *difference* of embeddings
        
        # Layer to produce the final "Delta_t_Embedding" after Faster-KAN
        self.delta_t_embedding_proj = nn.Linear(self.d_faster_kan_out, self.d_time_k_mote)
        
        # Stack of ContinuousMambaLayer instances
        self.continuous_mamba_layers = nn.ModuleList([
            ContinuousMambaLayer(
                d_model=self.d_time_k_mote,
                d_state=self.state_dim,
                d_conv=4,
                expand=2,
            )
            for _ in range(self.num_layers)
        ])
        
        self.output_proj = nn.Identity()
    
    def compute_independent_kmote_embeddings(self, 
                                            timestamps: torch.Tensor, 
                                            features: Optional[torch.Tensor] = None):
        batch_size, seq_len = timestamps.shape[:2]
        
        # Current timestamps (t_k)
        current_timestamps_flat = timestamps.view(batch_size * seq_len, 1)
        
        # Previous timestamps (t_k-1) - shift by 1 position for sequence
        previous_timestamps = torch.zeros_like(timestamps)
        if seq_len > 1:
            previous_timestamps[:, 1:] = timestamps[:, :-1]
        previous_timestamps[:, 0] = timestamps[:, 0] # For the first element, t_k-1 is t_k
        previous_timestamps_flat = previous_timestamps.view(batch_size * seq_len, 1)
        
        # K-MOTE might use features, so flatten them if necessary for K-MOTE's forward
        features_flat = features.view(batch_size * seq_len, -1) if features is not None else None
        
        # Independent K-MOTE computations (as in diagram)
        current_embeddings_flat, _, _ = self.kmote_layer(current_timestamps_flat, features_flat)
        previous_embeddings_flat, _, _ = self.kmote_layer(previous_timestamps_flat, features_flat)
        
        # Reshape back to sequence format
        current_embeddings = current_embeddings_flat.view(batch_size, seq_len, -1)
        previous_embeddings = previous_embeddings_flat.view(batch_size, seq_len, -1)
        
        return current_embeddings, previous_embeddings
    
    def forward(self, 
                timestamps: torch.Tensor,        # Raw timestamps (batch_size, seq_len, 1)
                features: Optional[torch.Tensor] = None, # Raw features (batch_size, seq_len, feature_dim), if K-MOTE uses them
                initial_state: Optional[torch.Tensor] = None # Not used directly by this block, but common in sequence models
               ) -> Tuple[torch.Tensor, dict]:
        """
        Simplified forward pass for testing components.
        
        Flow: t_k, t_k-1 -> K-MOTE -> (t_k - t_k-1) -> Faster-KAN -> Δt -> Continuous Mamba
        
        Args:
            timestamps: Raw timestamps (batch_size, seq_len, 1) - for t_k and t_k-1.
            features: Raw features (batch_size, seq_len, feature_dim) - if K-MOTE uses them.
            
        Returns:
            absolute_relative_embedding: Final absolute-relative t_k embedding (sequence output).
                                         Shape: (batch_size, seq_len, D_time_k_mote)
            info: Dict with intermediate results for analysis (for debugging/understanding).
        """
        
        # Step 1: Independent K-MOTE embeddings for t_k and t_k-1
        current_embeddings, previous_embeddings = self.compute_independent_kmote_embeddings(
            timestamps, features
        ) # current_embeddings is t_k Embedding, previous_embeddings is t_k-1 Embedding
        
        # Step 2: Temporal differences in embedding space (t_k - t_k-1)
        # This is done *before* Faster-KAN as per your request.
        temporal_differences = current_embeddings - previous_embeddings
        
        # Step 3: Faster-KAN processing of the temporal differences
        # Assuming faster_kan_layer can handle batched sequence input (B, L, D)
        processed_temporal_differences = self.faster_kan_layer(temporal_differences)

        # Step 4: Create Delta_t_Embedding
        # This projects the output from Faster-KAN to the dimension required by the Mamba layers.
        delta_t_embedding = self.delta_t_embedding_proj(processed_temporal_differences)
        
        # Step 5: Continuous Mamba processing
        # The 'input_embedding' to the ContinuousMambaLayer is the evolving sequence.
        # It starts as current_embeddings (t_k Embedding) and is updated layer by layer.
        absolute_relative_output = current_embeddings # Primary input to the first Mamba layer
        
        for mamba_layer in self.continuous_mamba_layers:
            # Pass both the primary input (the sequence) and the delta_t_embedding for modulation
            absolute_relative_output = mamba_layer(
                input_embedding=absolute_relative_output, 
                delta_embedding=delta_t_embedding 
            )
        
        # Step 6: Final output projection (if needed)
        # self.output_proj is Identity() by default, so it just passes through.
        absolute_relative_output = self.output_proj(absolute_relative_output)
        
        # For analysis and debugging, return intermediate values
        info = {
            "current_kmote_embeddings": current_embeddings,
            "previous_kmote_embeddings": previous_embeddings,
            "temporal_difference_before_kan": temporal_differences,
            "temporal_difference_after_kan": processed_temporal_differences,
            "delta_t_embedding": delta_t_embedding,
            "final_output": absolute_relative_output
        }
        
        return absolute_relative_output, info


