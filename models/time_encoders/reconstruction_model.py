# file: analysis/reconstruction_model.py

import torch
import torch.nn as nn
# Make sure kan_mammote.py and k_mote.py are in the correct path
from models.time_encoders.kan_mammote import KAN_MAMMOTE

class ReconstructionAnalysisModel(nn.Module):
    """
    An Encoder-Decoder model to analyze the expressive power of KAN_MAMMOTE.
    Encoder: KAN_MAMMOTE (in dual K-MOTE mode)
    Decoder: A simple nn.Linear layer
    """
    def __init__(self, embedding_dim=64, expert_dim=32, **kan_mammote_kwargs):
        super().__init__()
        
        # Ensure we are in the specified dual K-MOTE mode for this analysis
        kan_mammote_kwargs['use_kmote_for_relative'] = True
        
        # The ENCODER: Our KAN_MAMMOTE model
        self.encoder = KAN_MAMMOTE(
            embedding_dim=embedding_dim,
            expert_dim=expert_dim,
            num_mixtures=expert_dim, # As per KAN_MAMMOTE's internal logic
            **kan_mammote_kwargs
        )

        # The DECODER: Maps the learned time embedding back to a single value
        self.decoder = nn.Linear(embedding_dim, 1)

    def forward(self, t_abs, t_rel, return_gating_weights=False):
        """
        Performs a forward pass for reconstruction and optionally returns gating weights.
        
        Note: We manually step through the KAN_MAMMOTE logic to intercept the 
        gating weights from the absolute time K-MOTE module.
        """
        # --- ENCODER FORWARD PASS (MANUAL) ---

        # 1. Get absolute time features & gating weights from k_mote_abs
        k_mote_abs = self.encoder.k_mote_abs
        u_k, abs_gating_weights = k_mote_abs(t_abs, return_weights=True)

        # 2. Get relative time features (we are in dual K-MOTE mode)
        v_k = self.encoder.k_mote_rel(t_rel)

        # 3. Fuse relative features
        fusion_features = self.encoder.fusion_mlp_base(v_k)
        
        # 4. Combine with absolute features (residual connection)
        combined_input = u_k + fusion_features
        
        # 5. Pass through Mamba2
        if self.encoder.use_controllable_mamba:
            modulator_logits = self.encoder.modulator_head(fusion_features)
            gamma_logits, beta = modulator_logits.chunk(2, dim=-1)
            gamma = torch.sigmoid(gamma_logits) + 0.5
            temporal_modulators = (gamma, beta)
            mamba_output = self.encoder.mamba2(combined_input, temporal_modulators=temporal_modulators)
        else:
            mamba_output = self.encoder.mamba2(combined_input)
            
        # 6. Final projection to embedding dimension
        final_embedding = self.encoder.output_projection(mamba_output)

        # --- DECODER FORWARD PASS ---
        reconstructed_signal = self.decoder(final_embedding)

        if return_gating_weights:
            return reconstructed_signal, abs_gating_weights
        
        return reconstructed_signal