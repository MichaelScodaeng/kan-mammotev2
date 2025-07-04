# kan_mamote/src/models/c_mamba.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from src.utils.config import KANMAMOTEConfig

class ContinuousMambaBlock(nn.Module):
    """
    Implements a Continuous-Time Mamba Block by dynamically adapting SSM parameters
    based on the time difference between events (delta_t).

    This block processes a sequence of input vectors (u_k) and corresponding
    time differences (delta_t_k) to maintain a continuous hidden state.

    Conceptual adaptation of LTI (Linear Time-Invariant) SSMs to continuous-time.
    For the full efficiency of Mamba's selective scan, custom CUDA kernels are usually
    required. This implementation provides the PyTorch-native conceptual logic.
    """
    def __init__(self, input_dim: int, config: KANMAMOTEConfig):
        super().__init__()
        # `input_dim` is the dimension of the raw input to this Mamba block
        # (e.g., K-MOTE embeddings + raw_event_features)
        self.input_dim = input_dim 
        self.hidden_dim = config.hidden_dim_mamba # Mamba's internal embedding dimension (D)
        self.state_dim = config.state_dim_mamba # Dimension of the hidden state (N)
        self.config = config

        # --- Input Projection ---
        # Project the incoming `input_dim` to Mamba's internal `hidden_dim`.
        self.in_proj = nn.Linear(self.input_dim, self.hidden_dim)

        # --- SSM Parameters (A, B, C, D) ---
        # A: State matrix (NxN). Often initialized to be stable.
        self.A = nn.Parameter(torch.randn(self.state_dim, self.state_dim)) # N x N
        nn.init.xavier_uniform_(self.A) 

        # B: Input matrix (NxD). Maps projected input (D-dim) to state (N-dim).
        self.B = nn.Parameter(torch.randn(self.state_dim, self.hidden_dim)) # N x D
        nn.init.xavier_uniform_(self.B)

        # C: Output matrix (DxN). Maps state (N-dim) to output (D-dim).
        self.C = nn.Parameter(torch.randn(self.hidden_dim, self.state_dim)) # D x N
        nn.init.xavier_uniform_(self.C)

        # D: Direct skip connection (DxD).
        # This parameter directly connects the projected input to the output.
        self.D_param = nn.Parameter(torch.randn(self.hidden_dim)) # D-dimensional vector (element-wise multiplication)
        nn.init.ones_(self.D_param) # Common initialization for skip/residual

        # --- Delta (Discretization Step) Modulation ---
        # Project scalar delta_t_k to `state_dim` for per-state delta values.
        self.delta_t_proj = nn.Linear(1, self.state_dim) 

    def _discretize_ssm(self, A: torch.Tensor, B: torch.Tensor, delta_t_k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dynamically discretizes the continuous SSM parameters (A, B) using delta_t_k.
        This uses a differentiable approximation for the discretization.
        
        Args:
            A: Continuous state matrix (state_dim, state_dim).
            B: Continuous input matrix (state_dim, hidden_dim).
            delta_t_k: Time difference for current step, shape (batch_size, 1).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - A_k_bar: Discretized state matrix (batch_size, state_dim, state_dim).
                - B_k_bar: Discretized input matrix (batch_size, state_dim, hidden_dim).
        """
        # Proposal: "Mamba,k = softplus(Linear(Δtk))"
        # This `delta` parameter controls the discretization strength per state dimension.
        # Shape: (batch_size, state_dim)
        delta_param = F.softplus(self.delta_t_proj(delta_t_k)) 

        # Reshape for broadcasting with A and B matrices: (batch_size, state_dim, 1)
        delta_expanded = delta_param.unsqueeze(-1) 

        # Reshape A, B for broadcasting across batch:
        A_expanded = A.unsqueeze(0) # (1, state_dim, state_dim)
        B_expanded = B.unsqueeze(0) # (1, state_dim, hidden_dim)

        # Apply discretization approximation:
        # A_k_bar: element-wise exponential of A scaled by delta
        A_k_bar = torch.exp(A_expanded * delta_expanded) # (batch_size, state_dim, state_dim)
        
        # B_k_bar: B scaled by delta (a simplified form)
        B_k_bar = B_expanded * delta_expanded # (batch_size, state_dim, hidden_dim)

        return A_k_bar, B_k_bar

    def forward(self, 
                u_k_sequence: torch.Tensor, 
                delta_t_sequence: torch.Tensor, 
                initial_hidden_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Performs the forward pass of the Continuous-Time Mamba Block over a sequence.

        Args:
            u_k_sequence: Input sequence of concatenated embeddings,
                          shape (batch_size, seq_len, input_dim).
            delta_t_sequence: Sequence of time differences,
                              shape (batch_size, seq_len, 1).
            initial_hidden_state: Optional initial hidden state h_0,
                                  shape (batch_size, state_dim). Defaults to zeros.

        Returns:
            torch.Tensor: Sequence of final hidden states,
                          shape (batch_size, seq_len, hidden_dim).
        """
        batch_size, seq_len, _ = u_k_sequence.shape
        device = u_k_sequence.device

        # Initialize hidden state h_k-1 (state_dim)
        if initial_hidden_state is None:
            h_state = torch.zeros(batch_size, self.state_dim, device=device) # (batch_size, state_dim)
        else:
            h_state = initial_hidden_state.to(device) # Use provided initial state

        output_hidden_states = []

        # Iterate through the sequence (Scan operation)
        for k in range(seq_len):
            raw_u_k = u_k_sequence[:, k, :] # (batch_size, input_dim)
            delta_t_k = delta_t_sequence[:, k, :] # (batch_size, 1)

            # 1. Project raw input to Mamba's internal `hidden_dim`
            u_k_projected = self.in_proj(raw_u_k) # (batch_size, hidden_dim)

            # 2. Dynamically discretize A and B based on delta_t_k
            A_k_bar, B_k_bar = self._discretize_ssm(self.A, self.B, delta_t_k)
            # A_k_bar: (batch_size, state_dim, state_dim)
            # B_k_bar: (batch_size, state_dim, hidden_dim)

            # 3. Update hidden state: h_k = A_k_bar @ h_{k-1} + B_k_bar @ u_k_projected
            # h_state_new: (batch_size, state_dim) after operations
            h_state_new = torch.bmm(A_k_bar, h_state.unsqueeze(-1)).squeeze(-1) # (batch_size, state_dim)

            h_state_new = h_state_new + torch.bmm(B_k_bar, u_k_projected.unsqueeze(-1)).squeeze(-1)

            h_state = h_state_new # Update state for next iteration

            # 4. Project current hidden state to output (hidden_dim) using C and D_param
            # output_h_k = C @ h_state + D_param * u_k_projected (element-wise multiplication for D)
            output_h_k = torch.matmul(h_state, self.C.T) + (self.D_param * u_k_projected) # (batch_size, hidden_dim)
            output_hidden_states.append(output_h_k)

        # Stack the collected hidden states to form the output sequence
        # Result: (batch_size, seq_len, hidden_dim)
        return torch.stack(output_hidden_states, dim=1)