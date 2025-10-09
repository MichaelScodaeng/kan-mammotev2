# file: models/time_encoders/kan_mammote.py (Fair comparison between variants)

import torch
import torch.nn as nn
import torch.nn.functional as F

from .k_mote import KMOTE
from .sm_kernel import SMKernelLayer
from .controllable_mamba2 import ControllableMamba2
from mamba_ssm.modules.mamba2 import Mamba2  # Import vanilla Mamba2

class KAN_MAMMOTE(nn.Module):
    """Enhanced KAN-MAMMOTE with Custom Shock Wavelet for abrupt change detection."""
    def __init__(self, embedding_dim: int, expert_dim: int, num_mixtures: int, 
                 mamba_d_state: int = 256, mamba_d_conv: int = 4, mamba_expand: int = 4, 
                 wavelet_type: str = 'shock', mamba_headdim: int = 16, 
                 use_controllable_mamba: bool = True,  # Option to use vanilla Mamba2
                 use_kmote_for_relative: bool = False,  # NEW: Option to use K-MOTE instead of SM-kernel for relative time
                 **kwargs):
        super().__init__()
        
        # Enforce that dimensions are multiples of 16 for hardware compatibility.
        if expert_dim % 16 != 0:
            raise ValueError(f"expert_dim ({expert_dim}) must be a multiple of 16 for Mamba2 compatibility.")
        if mamba_d_state % 16 != 0:
            raise ValueError(f"mamba_d_state ({mamba_d_state}) must be a multiple of 16 for Mamba2 compatibility.")
         # ===== NEW: Enforce num_mixtures = expert_dim for fair comparison =====
        if num_mixtures != expert_dim:
            print(f"⚠️  WARNING: num_mixtures ({num_mixtures}) != expert_dim ({expert_dim})")
            print(f"🔧 Setting num_mixtures = expert_dim = {expert_dim} for architectural consistency")
            num_mixtures = expert_dim
            
        self.embedding_dim = embedding_dim
        self.wavelet_type = wavelet_type
        self.expert_dim = expert_dim
        self.use_controllable_mamba = use_controllable_mamba
        self.use_kmote_for_relative = use_kmote_for_relative
        
        # Enhanced K-MOTE for absolute time with configurable wavelet type
        self.k_mote_abs = KMOTE(input_dim=1, output_dim=expert_dim, wavelet_type=wavelet_type)
        
        # ===== NEW: Choose between SM-kernel and K-MOTE for relative time =====
        if use_kmote_for_relative:
            print("🔧 Using K-MOTE for relative time encoding (dual K-MOTE mode)")
            self.k_mote_rel = KMOTE(input_dim=1, output_dim=expert_dim, wavelet_type=wavelet_type)
            self.sm_kernel = None
            fusion_input_dim = expert_dim  # K-MOTE outputs expert_dim
        else:
            print("🔧 Using SM-Kernel for relative time encoding (default mode)")
            self.k_mote_rel = None
            self.sm_kernel = SMKernelLayer(num_mixtures=num_mixtures, input_dim=1)
            fusion_input_dim = num_mixtures  # SM-kernel outputs num_mixtures
        
        # ===== Mamba2 and Fusion Architecture =====
        if use_controllable_mamba:
            print("🔧 Using ControllableMamba2 (with FiLM modulation)")
            self.mamba2 = ControllableMamba2(
                d_model=self.expert_dim,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand, 
                headdim=mamba_headdim
            )
            
            # Fusion architecture: relative features → expert_dim (for residual addition)
            self.fusion_mlp_base = nn.Sequential(
                nn.Linear(fusion_input_dim, expert_dim),
                nn.LayerNorm(expert_dim),
                nn.GELU(),
                nn.Linear(expert_dim, expert_dim)
            )
            
            # Path 2: Additional head for temporal modulators
            self.modulator_head = nn.Sequential(
                nn.Linear(expert_dim, expert_dim // 2),
                nn.GELU(),
                nn.Linear(expert_dim // 2, self.mamba2.nheads * 2)  # gamma and beta
            )
            
        else:
            print("🔧 Using vanilla Mamba2 (no FiLM modulation)")
            self.mamba2 = Mamba2(
                d_model=self.expert_dim,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand, 
                headdim=mamba_headdim
            )
            
            # Same fusion architecture as ControllableMamba2 base
            self.fusion_mlp_base = nn.Sequential(
                nn.Linear(fusion_input_dim, expert_dim),
                nn.LayerNorm(expert_dim),
                nn.GELU(),
                nn.Linear(expert_dim, expert_dim)
            )
            # No modulator head for vanilla
        # ===== END MAMBA ARCHITECTURE =====
        
        print(f"Mamba2 parameters:")
        print(f"  nheads: {self.mamba2.nheads}")
        print(f"  d_state: {self.mamba2.d_state}")
        print(f"  d_conv: {self.mamba2.d_conv}")
        print(f"  expand: {self.mamba2.expand}")
        print(f"  headdim: {self.mamba2.headdim}")
        print(f"  embedding_dim: {self.embedding_dim}")
        print(f"  use_kmote_for_relative: {use_kmote_for_relative}")

        print(f"Initialized Enhanced KAN-MAMMOTE Framework with {wavelet_type} wavelet.")
        
        # Output projection to match embedding_dim
        if expert_dim != embedding_dim:
            self.output_projection = nn.Sequential(
                nn.Linear(expert_dim, embedding_dim),
                nn.LayerNorm(embedding_dim)
            )
        else:
            self.output_projection = nn.Identity()
    
    def initialize_sm_kernel(self, delta_t_sample: torch.Tensor):
        """Passes the initialization call to the SM-Kernel module (if using SM-kernel mode)."""
        if self.sm_kernel is not None:
            self.sm_kernel.initialize_from_data(delta_t_sample)
        else:
            print("INFO: Skipping SM-kernel initialization (using dual K-MOTE mode)")
    
    def warmup(self, device='cuda', num_iterations=3):
        """
        Warm up the model by running a few forward passes.
        This compiles the CUDA kernels (especially for Mamba2) and caches them 
        for the entire training session, avoiding ~5-40 second compilation delays.
        
        Args:
            device: Device to run warm-up on (default: 'cuda')
            num_iterations: Number of warm-up iterations (default: 3)
        """
        mamba_type = "ControllableMamba2" if self.use_controllable_mamba else "Vanilla Mamba2"
        print(f"\n{'='*60}")
        print(f"🔥 Warming up KAN-MAMMOTE ({mamba_type})...")
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
        
        print(f"✅ Warm-up complete! CUDA kernels cached for this session.")
        print(f"{'='*60}\n")
        
    def forward(self, t_abs: torch.Tensor, t_rel: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """
        Forward pass with flexible relative time encoding and FAIR comparison between ControllableMamba2 and vanilla Mamba2.
        
        Two modes supported:
        1. SM-kernel mode (default): Uses SM-kernel for relative time
        2. Dual K-MOTE mode: Uses K-MOTE for both absolute and relative time
        
        Both modes:
        1. Use the same fusion_mlp_base architecture
        2. Add fusion features to u_k (residual connection)
        3. Pass combined input to Mamba2
        
        The ONLY difference:
        - ControllableMamba2: Also modulates dt using temporal_modulators
        - Vanilla Mamba2: No dt modulation
        
        Args:
            t_abs: Absolute time tensor (B, S, 1)
            t_rel: Relative time tensor (B, S, 1)
            debug: Enable detailed debugging output
            
        Returns:
            final_embedding: Output embeddings (B, S, embedding_dim)
        """
        if debug or hasattr(self, '_debug_mode'):
            print(f"\n{'='*60}")
            print(f"🔍 KAN-MAMMOTE DEBUG - Forward Pass")
            print(f"{'='*60}")
            print(f"📊 INPUT SHAPES:")
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
            print(f"🎯 K-MOTE ABS OUTPUT:")
            print(f"   u_k shape: {u_k.shape}")
            print(f"   u_k sample: {u_k.flatten()[:5].detach().cpu().numpy()}")
        
        # ===== Get relative time features based on mode =====
        if self.use_kmote_for_relative:
            # Dual K-MOTE mode: Use K-MOTE for relative time
            v_k = self.k_mote_rel(t_rel)  # (B, S, expert_dim)
            if debug or hasattr(self, '_debug_mode'):
                print(f"🎯 K-MOTE REL OUTPUT (dual mode):")
                print(f"   v_k shape: {v_k.shape}")
        else:
            # SM-kernel mode: Use SM-kernel for relative time
            v_k = self.sm_kernel(t_rel)  # (B, S, num_mixtures)
            if debug or hasattr(self, '_debug_mode'):
                print(f"🎯 SM-KERNEL OUTPUT:")
                print(f"   v_k shape: {v_k.shape}")
        
        # ===== Fuse relative time features to expert_dim =====
        fusion_features = self.fusion_mlp_base(v_k)  # (B, S, expert_dim)
        
        if debug or hasattr(self, '_debug_mode'):
            print(f"🔧 FUSION OUTPUT:")
            print(f"   fusion_features shape: {fusion_features.shape}")
        
        # ===== Combine with absolute time features (residual connection) =====
        combined_input = u_k + fusion_features  # (B, S, expert_dim)
        
        if debug or hasattr(self, '_debug_mode'):
            print(f"🔗 COMBINED INPUT TO MAMBA:")
            print(f"   combined_input shape: {combined_input.shape}")
            print(f"   Sequence length being processed: {combined_input.shape[1]}")
        
        if self.use_controllable_mamba:
            # ===== ControllableMamba2 path: Add temporal modulation =====
            # Generate modulators from fusion features
            modulator_logits = self.modulator_head(fusion_features)  # (B, S, nheads * 2)
            
            # Split into gamma and beta
            gamma_logits, beta = modulator_logits.chunk(2, dim=-1)
            gamma = torch.sigmoid(gamma_logits) + 0.5  # Range: [0.5, 1.5]
            temporal_modulators = (gamma, beta)
            
            if debug or hasattr(self, '_debug_mode'):
                print(f"⚙️  CONTROLLABLE MAMBA2 MODULATION:")
                print(f"   modulator_logits shape: {modulator_logits.shape}")
                print(f"   gamma range: [{gamma.min().item():.3f}, {gamma.max().item():.3f}]")
            
            # Forward through ControllableMamba2 with temporal modulation
            # Input: combined_input (u_k + fusion)
            # Modulation: gamma, beta modify dt inside Mamba2
            mamba_output = self.mamba2(u=combined_input, temporal_modulators=temporal_modulators)
            
        else:
            # ===== Vanilla Mamba2 path: No modulation =====
            if debug or hasattr(self, '_debug_mode'):
                print(f"⚙️  VANILLA MAMBA2 (no modulation)")
            
            # Input: same combined_input (u_k + fusion)
            # No temporal modulation
            mamba_output = self.mamba2(combined_input)
        
        if debug or hasattr(self, '_debug_mode'):
            print(f"🐍 MAMBA OUTPUT:")
            print(f"   mamba_output shape: {mamba_output.shape}")
        
        # Project to embedding dimension
        final_embedding = self.output_projection(mamba_output)

        if debug or hasattr(self, '_debug_mode'):
            print(f"🎯 FINAL OUTPUT:")
            print(f"   final_embedding shape: {final_embedding.shape}")
            print(f"   final_embedding sample: {final_embedding.flatten()[:5].detach().cpu().numpy()}")
            print(f"{'='*60}\n")

        return final_embedding
    
    def enable_debug_mode(self):
        """Enable persistent debug mode"""
        self._debug_mode = True
        print("🔍 KAN-MAMMOTE Debug Mode ENABLED")
    
    def disable_debug_mode(self):
        """Disable persistent debug mode"""
        if hasattr(self, '_debug_mode'):
            delattr(self, '_debug_mode')
        print("🔍 KAN-MAMMOTE Debug Mode DISABLED")