

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import sys
from .k_mote import KMOTE
from .controllable_mamba2 import ControllableMamba2
from mamba_ssm.modules.mamba2 import Mamba2  # Import vanilla Mamba2

class KMM(nn.Module):
    def __init__(self, embedding_dim: int, expert_dim: int, num_mixtures: int = None,
                 mamba_d_state: int = 256, mamba_d_conv: int = 4, mamba_expand: int = 4, 
                 wavelet_type: str = 'shock', mamba_headdim: int = 32, 
                 use_controllable_mamba: bool = True,  # Only for fusion_strategy='mamba'
                 use_kmote_for_relative: bool = True,  # Default: K-MOTE
                 fusion_strategy: str = 'mamba',  # NEW: 'mamba', 'concat', 'weighted', 'attention'
                 separate_modulation_pathway: bool = True,dropout: float = 0.1,  # NEW: For ControllableMamba2 only
                 use_amp: bool = False,
                 **kwargs):
        super().__init__()
        # whether to run internal expert/mamba ops under CUDA AMP autocast
        self.use_amp = use_amp
        # Extract optional gradient checkpointing flag if provided via kwargs
        use_gc = kwargs.pop('use_gradient_checkpointing', True)
        
        # Validate fusion strategy
        valid_strategies = ['mamba', 'concat', 'weighted', 'attention']
        if fusion_strategy not in valid_strategies:
            raise ValueError(f"fusion_strategy must be one of {valid_strategies}, got: {fusion_strategy}")
        
        # Enforce that dimensions are multiples of 16 for hardware compatibility (for Mamba)
        if fusion_strategy == 'mamba':
            if expert_dim % 16 != 0:
                raise ValueError(f"expert_dim ({expert_dim}) must be a multiple of 16 for Mamba2 compatibility.")
            if mamba_d_state % 16 != 0:
                raise ValueError(f"mamba_d_state ({mamba_d_state}) must be a multiple of 16 for Mamba2 compatibility.")
        
        self.embedding_dim = embedding_dim
        self.wavelet_type = wavelet_type
        self.expert_dim = expert_dim
        self.use_controllable_mamba = use_controllable_mamba
        self.use_kmote_for_relative = use_kmote_for_relative
        self.fusion_strategy = fusion_strategy
        self.separate_modulation_pathway = separate_modulation_pathway  # Store the config
        self.dropout_rate = dropout  # Store for logging
        self.num_mixtures = expert_dim  # Store for logging
        #self.rel_to_content_alpha = nn.Parameter(torch.tensor(0.0))
        # Enhanced K-MOTE for absolute time with configurable wavelet type
        self.k_mote_abs = KMOTE(
            input_dim=1, 
            output_dim=expert_dim, 
            wavelet_type=wavelet_type,
            use_scale=True,       # Enable scale parameter like LeTE
            use_gradient_checkpointing=use_gc,
            use_amp=self.use_amp
        )
        
        # ===== Choose between K-MOTE (default) =====
        if use_kmote_for_relative:
            print("Using K-MOTE for relative time encoding (default, dual K-MOTE mode)")
            self.k_mote_rel = KMOTE(
                input_dim=1, 
                output_dim=expert_dim, 
                wavelet_type=wavelet_type,
                use_scale=True,        # Enable scale parameter like LeTE
                use_gradient_checkpointing=use_gc,
                use_amp=self.use_amp
            )
            self.sm_kernel = None
            rel_time_dim = expert_dim  # K-MOTE outputs expert_dim
        
        # ===== Build Fusion Architecture based on strategy =====
        print(f"Building fusion architecture: {fusion_strategy}")
        
        if fusion_strategy == 'mamba':
            # ===== MAMBA FUSION (Original KAN-MAMMOTE) =====
            if use_controllable_mamba:
                print("   ├─ Using ControllableMamba2 (with FiLM modulation)")
                self.mamba2 = ControllableMamba2(
                    d_model=self.expert_dim,
                    d_state=mamba_d_state,
                    d_conv=mamba_d_conv,
                    expand=mamba_expand, 
                    headdim=mamba_headdim
                )
                print("Controllable Mamba2 hyperparameters:")
                print(f"   ├─ d_model: {self.expert_dim}")
                print(f"   ├─ d_state: {mamba_d_state}")
                print(f"   ├─ d_conv: {mamba_d_conv}")
                print(f"   ├─ expand: {mamba_expand}")
                print(f"   ├─ headdim: {mamba_headdim}")
                
                # Separate MLPs for gamma and beta modulation
                # Clean separation: relative time directly generates control signals
                # Ensure exact dimension match to prevent repeat() operations
                mamba_dt_dim = self.mamba2.nheads  # This should match ControllableMamba2's dt_content
                
                # Separate MLPs for better expressiveness
                self.gamma_head = nn.Sequential(
                    nn.Linear(rel_time_dim, expert_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(expert_dim // 2, mamba_dt_dim)  # Only gamma dimensions
                )
                
                self.beta_head = nn.Sequential(
                    nn.Linear(rel_time_dim, expert_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(expert_dim // 2, mamba_dt_dim)  # Only beta dimensions
                )
                

                self.fusion_mlp_base = None
                self.fusion_norm = None
                
                print(f"   ├─ Separate modulation heads: gamma_head & beta_head")
                print(f"   ├─ Each head: {rel_time_dim} → {expert_dim//2} → {mamba_dt_dim}")
                print(f"   └─ Better expressiveness with independent parameter learning")
                
                       
            print(f"   ├─ Mamba2 parameters:")
            print(f"   │  ├─ nheads: {self.mamba2.nheads}")
            print(f"   │  ├─ d_state: {self.mamba2.d_state}")
            print(f"   │  ├─ d_conv: {self.mamba2.d_conv}")
            print(f"   │  ├─ expand: {self.mamba2.expand}")
            print(f"   │  └─ headdim: {self.mamba2.headdim}")
            
            self.mamba_dropout = nn.Dropout(dropout)
            
            
            self.output_projection = nn.Sequential(
                nn.Linear(expert_dim, embedding_dim),
                nn.LayerNorm(embedding_dim)
            )

        print(f"Initialized KAN-MAMMOTE with proper dropout:")
        print(f"   ├─ Dropout rate: {dropout}")
        print(f"   ├─ Dropout locations: after activations, before final norms")
        print(f"   ├─ Residual + post-LN pattern enabled")
        print(f"   ├─ Wavelet type: {wavelet_type}")
        print(f"   ├─ Fusion strategy: {fusion_strategy}")
        print(f"   ├─ Relative encoder: {'K-MOTE'}")
        if fusion_strategy == 'mamba' and use_controllable_mamba:
            print(f"   ├─ Modulation pathway: {'Separate (u_k only)' if separate_modulation_pathway else 'Combined (u_k + fusion_features)'}")
        print(f"   ├─ Expert dim: {expert_dim}")
        print(f"   └─ Embedding dim: {embedding_dim}")
    
    
    def warmup(self, device='cuda', num_iterations=3):
        
        """
        Warm up the model by running a few forward passes.
        This compiles the CUDA kernels (especially for Mamba2) and caches them 
        for the entire training session, avoiding ~5-40 second compilation delays.
        
        Args:
            device: Device to run warm-up on (default: 'cuda')
            num_iterations: Number of warm-up iterations (default: 3)
        """
        if self.fusion_strategy == 'mamba':
            mamba_type = "ControllableMamba2" if self.use_controllable_mamba else "Vanilla Mamba2"
            print(f"\n{'='*60}")
            print(f"Warming up KAN-MAMMOTE ({self.fusion_strategy} fusion with {mamba_type})...")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print(f"Warming up KAN-MAMMOTE ({self.fusion_strategy} fusion)...")
            print(f"{'='*60}")
        
        # Ensure model is on the correct device
        self.to(device)
        
        # Create dummy input matching typical batch size
        batch_size = 2
        seq_len = 64  # Shorter sequence for faster warm-up
        
        dummy_t_abs = torch.randn(batch_size, seq_len, 1).to(device)
        dummy_t_rel = torch.randn(batch_size, seq_len, 1).abs().to(device)
        
        # Put model in eval mode for warm-up
        was_training = self.training
        self.eval()
        
        # Run warm-up iterations
        import time
        with torch.no_grad():
            for i in range(num_iterations):
                start = time.time()
                _ = self.forward(dummy_t_abs, dummy_t_rel)
                torch.cuda.synchronize()
                elapsed = time.time() - start
                
                if i == 0:
                    print(f"  Iteration {i+1}/{num_iterations}: {elapsed:.3f}s (compilation)")
                else:
                    print(f"  Iteration {i+1}/{num_iterations}: {elapsed:.3f}s (cached)")
        
        # Restore training mode
        if was_training:
            self.train()
        
        print(f"Warm-up complete! CUDA kernels cached for this session.")
        print(f"{'='*60}\n")
        
        
    def forward(self, t_abs: torch.Tensor, t_rel: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """
        Forward pass with flexible fusion strategies and relative time encoding.
        
        Fusion Strategies:
        1. 'mamba': Uses Mamba2 for temporal fusion (original KAN-MAMMOTE)
           - ControllableMamba2: With FiLM temporal modulation
           - Vanilla Mamba2: Without temporal modulation
        2. 'concat': Concatenates absolute + relative, then MLP projection
        3. 'weighted': Learnable weighted sum of absolute + relative streams
        4. 'attention': Cross-attention between absolute and relative streams
        
        Args:
            t_abs: Absolute time tensor (B, S, 1)
            t_rel: Relative time tensor (B, S, 1)
            debug: Enable detailed debugging output
            
        Returns:
            final_embedding: Output embeddings (B, S, embedding_dim)
        """
        from .factory import DEBUG_TIME_ENCODER_FACTORY, DEBUG_TIME_ENCODER_CALLS, DEBUG_TIME_ENCODER_VALUES, DEBUG_TIME_ENCODER_SHAPES
        if DEBUG_TIME_ENCODER_FACTORY and DEBUG_TIME_ENCODER_CALLS:
            print(f"[KMM] Forward pass called!")
        if debug or hasattr(self, '_debug_mode'):
            print(f"\n{'='*60}")
            print(f"KAN-MAMMOTE DEBUG - Forward Pass")
            print(f"{'='*60}")
            print(f"INPUT SHAPES:")
            print(f"   t_abs shape: {t_abs.shape}")
            print(f"   t_rel shape: {t_rel.shape}")
            print(f"   t_abs dtype: {t_abs.dtype}, device: {t_abs.device}")
            print(f"   t_rel dtype: {t_rel.dtype}, device: {t_rel.device}")
            
            # Sample a few values for inspection
            print(f"📈 SAMPLE VALUES:")
            print(f"   t_abs sample: {t_abs.flatten()[:5].detach().cpu().numpy()}")
            print(f"   t_rel sample: {t_rel.flatten()[:5].detach().cpu().numpy()}")
            print(f"   t_abs range: [{t_abs.min().item():.3f}, {t_abs.max().item():.3f}]")
            print(f"   t_rel range: [{t_rel.min().item():.3f}, {t_rel.max().item():.3f}]")
        
        # Get absolute time features using K-MOTE
        u_k = self.k_mote_abs(t_abs)  # (B, S, expert_dim)
        
        if debug or hasattr(self, '_debug_mode'):
            print(f"K-MOTE ABS OUTPUT:")
            print(f"   u_k shape: {u_k.shape}")
            print(f"   u_k sample: {u_k.flatten()[:5].detach().cpu().numpy()}")
        
        # ===== Get relative time features based on mode =====
        if self.use_kmote_for_relative:
            # K-MOTE mode (default): Use K-MOTE for relative time
            v_k = self.k_mote_rel(t_rel)  # (B, S, expert_dim or rel_time_dim)
            if debug or hasattr(self, '_debug_mode'):
                print(f"K-MOTE REL OUTPUT (dual K-MOTE mode):")
                print(f"   v_k shape: {v_k.shape}")
        # ===== Apply fusion strategy =====
        if self.fusion_strategy == 'mamba':
            # ===== MAMBA FUSION (Original KAN-MAMMOTE) =====
            
            if self.use_controllable_mamba:
                # Use separate MLPs for better expressiveness
                gamma_logits = self.gamma_head(v_k)  # (B, S, nheads)
                beta = self.beta_head(v_k)           # (B, S, nheads)
                
                gamma = F.softplus(gamma_logits)  # Clean softplus (no bias needed)
                temporal_modulators = (gamma, beta)
                
                # Pure absolute time as content (separate pathway)
                combined_input = u_k
                #Test swap
                #combined_input = v_k  # (B, S, expert_dim)
 

                if debug or hasattr(self, '_debug_mode'):
                    print(f"CONTROLLABLE MAMBA2 (SEPARATE MLPS):")
                    print(f"   Separate gamma_head & beta_head for better expressiveness")
                    print(f"   gamma range: [{gamma.min().item():.3f}, {gamma.max().item():.3f}]")
                    print(f"   beta range: [{beta.min().item():.3f}, {beta.max().item():.3f}]")
                    print(f"   combined_input = u_k (pure absolute)")
                    print(f"   combined_input shape: {combined_input.shape}")
                
                # Run Mamba2 under autocast when AMP is enabled to save memory/compute
                with amp.autocast(enabled=(self.use_amp and torch.cuda.is_available())):
                    mamba_output = self.mamba2(u=combined_input, temporal_modulators=temporal_modulators)
                
                       
            # Apply dropout after Mamba2
            mamba_output = self.mamba_dropout(mamba_output)
            
            if debug or hasattr(self, '_debug_mode'):
                
                print(f"   mamba_output shape: {mamba_output.shape}")
            
            final_embedding = self.output_projection(mamba_output)
            

        if debug or hasattr(self, '_debug_mode'):
            print(f"FINAL OUTPUT:")
            print(f"   final_embedding shape: {final_embedding.shape}")
            print(f"   final_embedding sample: {final_embedding.flatten()[:5].detach().cpu().numpy()}")
            print(f"{'='*60}\n")

        return final_embedding
    
    # --- AMP convenience helpers (delegate to internal KMOTE) ---
    def make_grad_scaler(self):
        """Return a GradScaler if AMP is enabled and CUDA is available, else None."""
        if self.use_amp and torch.cuda.is_available():
            return torch.cuda.amp.GradScaler()
        return None

    def scaled_backward(self, loss: torch.Tensor, scaler):
        """Backward using scaler when available (delegates to k_mote_abs helper)."""
        # Prefer delegating to underlying KMOTE which implements scaled_backward
        if hasattr(self.k_mote_abs, 'scaled_backward'):
            return self.k_mote_abs.scaled_backward(loss, scaler)
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

    def scaler_step(self, scaler, optimizer):
        """Step optimizer using scaler when available."""
        if hasattr(self.k_mote_abs, 'scaler_step'):
            return self.k_mote_abs.scaler_step(scaler, optimizer)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
    
    def enable_debug_mode(self):
        """Enable persistent debug mode"""
        self._debug_mode = True
        print("KAN-MAMMOTE Debug Mode ENABLED")
    
    def disable_debug_mode(self):
        """Disable persistent debug mode"""
        if hasattr(self, '_debug_mode'):
            delattr(self, '_debug_mode')
        print("KAN-MAMMOTE Debug Mode DISABLED")