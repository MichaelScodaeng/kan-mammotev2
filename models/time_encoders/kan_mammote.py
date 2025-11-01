# file: models/time_encoders/kan_mammote.py (Fair comparison between variants)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import sys
from .k_mote import KMOTE
from .sm_kernel import SMKernelLayer
from .controllable_mamba2 import ControllableMamba2
from mamba_ssm.modules.mamba2 import Mamba2  # Import vanilla Mamba2

class KAN_MAMMOTE(nn.Module):
    """
    Enhanced KAN-MAMMOTE with flexible fusion strategies and Mamba variants.
    
    Fusion Strategies:
        - 'mamba' (default): Uses Mamba2 for temporal fusion (original KAN-MAMMOTE)
        - 'concat': Concatenates absolute + relative, then MLP projection
        - 'weighted': Learnable weighted sum of absolute + relative streams
        - 'attention': Cross-attention between absolute and relative streams
    
    Relative Time Encoding:
        - K-MOTE (default): Uses K-MOTE for relative time encoding
        - SM-Kernel (legacy): Uses SM-Kernel for ablation studies
    
    Mamba Variants (only for fusion_strategy='mamba'):
        - ControllableMamba2 (default): With FiLM temporal modulation
        - Vanilla Mamba2: Standard Mamba2 without modulation
    
    Modulation Pathway (only for ControllableMamba2):
        - separate_modulation_pathway=True (default): 
          * Content pathway: pure absolute time (u_k)
          * Modulation pathway: relative time controls dynamics via FiLM gates
          * Cleaner separation of "what" (absolute) vs "how" (relative)
        - separate_modulation_pathway=False (legacy):
          * Combined pathway: u_k + fusion_features
          * Relative time flows through both gates and input
          * May provide richer information but less architectural clarity
    """
    def __init__(self, embedding_dim: int, expert_dim: int, num_mixtures: int = None,
                 mamba_d_state: int = 256, mamba_d_conv: int = 4, mamba_expand: int = 4, 
                 wavelet_type: str = 'shock', mamba_headdim: int = 16, 
                 use_controllable_mamba: bool = True,  # Only for fusion_strategy='mamba'
                 use_kmote_for_relative: bool = True,  # Default: K-MOTE (SM-kernel is legacy)
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
        
        # Handle num_mixtures for backward compatibility with SM-kernel
        if not use_kmote_for_relative and num_mixtures is None:
            num_mixtures = expert_dim
            print(f"🔧 Setting num_mixtures = expert_dim = {expert_dim} for SM-kernel mode")
        
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
        
        # ===== Choose between K-MOTE (default) and SM-kernel (legacy) for relative time =====
        if use_kmote_for_relative:
            print("🔧 Using K-MOTE for relative time encoding (default, dual K-MOTE mode)")
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
        else:
            print("🔧 Using SM-Kernel for relative time encoding (legacy, for ablation)")
            self.k_mote_rel = None
            self.sm_kernel = SMKernelLayer(num_mixtures=num_mixtures, input_dim=1)
            rel_time_dim = num_mixtures  # SM-kernel outputs num_mixtures
        
        # ===== Build Fusion Architecture based on strategy =====
        print(f"🏗️  Building fusion architecture: {fusion_strategy}")
        
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
                
                # ✅ SIMPLIFIED: Direct v_k → modulator_head (no intermediate fusion_mlp_base)
                # Clean separation: relative time directly generates control signals
                self.modulator_head = nn.Sequential(
                    nn.Linear(rel_time_dim, expert_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),  # ✅ Dropout before final linear
                    nn.Linear(expert_dim // 2, self.mamba2.nheads * 2)  # Direct to gamma/beta
                )
                
                # ✅ Remove redundant components for ControllableMamba2
                self.fusion_mlp_base = None
                self.fusion_norm = None
                
            else:
                print("   ├─ Using vanilla Mamba2 (no FiLM modulation)")
                self.mamba2 = Mamba2(
                    d_model=self.expert_dim,
                    d_state=mamba_d_state,
                    d_conv=mamba_d_conv,
                    expand=mamba_expand, 
                    headdim=mamba_headdim
                )
                
                # ✅ FIXED: Same clean structure as ControllableMamba2
                self.fusion_mlp_base = nn.Sequential(
                    nn.Linear(rel_time_dim, expert_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(expert_dim, expert_dim)
                )
                self.fusion_norm = nn.LayerNorm(expert_dim)
            
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


        elif fusion_strategy == 'concat':
            # ===== CONCAT FUSION (Lite variant) =====
            print("   ├─ Concatenation + MLP fusion")
            self.mamba2 = None
            concat_dim = expert_dim + rel_time_dim
            
            # ✅ FIXED: Cleaner MLP without internal LayerNorm
            self.fusion_mlp = nn.Sequential(
                nn.Linear(concat_dim, expert_dim),
                nn.GELU(),
                nn.Dropout(dropout),  # ✅ After activation
                nn.Linear(expert_dim, embedding_dim),
                nn.Dropout(dropout),  # ✅ Before final norm
                nn.LayerNorm(embedding_dim)  # ✅ Norm is last
            )
            self.output_projection = nn.Identity()
            print(f"   └─ Concat dimension: {concat_dim} → {embedding_dim}")
            
        elif fusion_strategy == 'weighted':
            # ===== WEIGHTED SUM FUSION (Lite variant) =====
            print("   ├─ Learnable weighted sum fusion")
            self.mamba2 = None
            
            # ✅ FIXED: Clear projection with dropout
            if rel_time_dim != expert_dim:
                self.rel_projection = nn.Sequential(
                    nn.Linear(rel_time_dim, expert_dim),
                    nn.Dropout(dropout)
                )
            else:
                self.rel_projection = nn.Identity()  # ✅ Clear, not bare dropout
            
            # Learnable weights
            self.weight_abs = nn.Parameter(torch.tensor(0.5))
            self.weight_rel = nn.Parameter(torch.tensor(0.5))
            
            self.output_projection = nn.Sequential(
                nn.Linear(expert_dim, embedding_dim),
                nn.Dropout(dropout),
                nn.LayerNorm(embedding_dim)
            )
            print(f"   └─ Weighted sum: {expert_dim} → {embedding_dim}")
            
        elif fusion_strategy == 'attention':
            # ===== CROSS-ATTENTION FUSION (Lite variant) =====
            print("   ├─ Cross-attention fusion")
            self.mamba2 = None
            
            # ✅ FIXED: Clear projection with dropout
            if rel_time_dim != expert_dim:
                self.rel_projection = nn.Sequential(
                    nn.Linear(rel_time_dim, expert_dim),
                    nn.Dropout(dropout)
                )
            else:
                self.rel_projection = nn.Identity()  # ✅ Clear, not bare dropout
            
            # ✅ Cross-attention with configurable dropout
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=expert_dim,
                num_heads=4,
                dropout=dropout,  # ✅ Use configurable dropout
                batch_first=True
            )
            self.norm_after_attn = nn.LayerNorm(expert_dim)
            
            self.output_projection = nn.Sequential(
                nn.Linear(expert_dim, embedding_dim),
                nn.Dropout(dropout),
                nn.LayerNorm(embedding_dim)
            )
            print(f"   └─ Cross-attention: {expert_dim} → {embedding_dim}")

        print(f"✅ Initialized KAN-MAMMOTE with proper dropout:")
        print(f"   ├─ Dropout rate: {dropout}")
        print(f"   ├─ Dropout locations: after activations, before final norms")
        print(f"   ├─ Residual + post-LN pattern enabled")
        print(f"   ├─ Wavelet type: {wavelet_type}")
        print(f"   ├─ Fusion strategy: {fusion_strategy}")
        print(f"   ├─ Relative encoder: {'K-MOTE' if use_kmote_for_relative else 'SM-Kernel (legacy)'}")
        if fusion_strategy == 'mamba' and use_controllable_mamba:
            print(f"   ├─ Modulation pathway: {'Separate (u_k only)' if separate_modulation_pathway else 'Combined (u_k + fusion_features)'}")
        print(f"   ├─ Expert dim: {expert_dim}")
        print(f"   └─ Embedding dim: {embedding_dim}")
    
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
        if self.fusion_strategy == 'mamba':
            mamba_type = "ControllableMamba2" if self.use_controllable_mamba else "Vanilla Mamba2"
            print(f"\n{'='*60}")
            print(f"🔥 Warming up KAN-MAMMOTE ({self.fusion_strategy} fusion with {mamba_type})...")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print(f"🔥 Warming up KAN-MAMMOTE ({self.fusion_strategy} fusion)...")
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
        Forward pass with flexible fusion strategies and relative time encoding.
        
        Relative Time Encoding:
        1. K-MOTE mode (default): Uses K-MOTE for both absolute and relative time
        2. SM-kernel mode (legacy): Uses SM-kernel for relative time (ablation)
        
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
        # 🔍 DEBUG: Add factory debug integration
        from .factory import DEBUG_TIME_ENCODER_FACTORY, DEBUG_TIME_ENCODER_CALLS, DEBUG_TIME_ENCODER_VALUES, DEBUG_TIME_ENCODER_SHAPES
        cnt = 0
        if DEBUG_TIME_ENCODER_FACTORY:
            if DEBUG_TIME_ENCODER_CALLS:
                print(f"🔍 [KAN_MAMMOTE] Forward pass called!", flush=True)
            if DEBUG_TIME_ENCODER_VALUES:
                
                print(f"🔍 [KAN_MAMMOTE] Input shapes - t_abs: {t_abs.shape}, t_rel: {t_rel.shape}", flush=True)
                print(f"🔍 [KAN_MAMMOTE] Input ranges - t_abs: [{t_abs.min().item():.3f}, {t_abs.max().item():.3f}]", flush=True)
                print(f"🔍 [KAN_MAMMOTE] Input ranges - t_rel: [{t_rel.min().item():.3f}, {t_rel.max().item():.3f}]", flush=True)
                # 🔍 DEBUG: Analyze the actual values to understand preprocessing
                print(f"🔍 [VALUE ANALYSIS]", flush=True)
                print(f"   ├─ t_abs stats: mean={t_abs.mean().item():.4f}, std={t_abs.std().item():.4f}", flush=True)
                print(f"   ├─ t_rel stats: mean={t_rel.mean().item():.4f}, std={t_rel.std().item():.4f}", flush=True)
                print(f"   ├─ t_abs negative count: {(t_abs < 0).sum().item()}/{t_abs.numel()}", flush=True)
                print(f"   ├─ t_rel negative count: {(t_rel < 0).sum().item()}/{t_rel.numel()}", flush=True)
                print(f"   └─ 💡 Negative t_abs suggests normalization/standardization", flush=True)
            if DEBUG_TIME_ENCODER_SHAPES:
                print(f"🔍 [KAN_MAMMOTE] Batch size: {t_abs.shape[0]}, Sequence length: {t_abs.shape[1] if len(t_abs.shape) > 1 else 'N/A'}", flush=True)
                
                # 🎯 SEQUENCE ANALYSIS: Understand the data pattern
                print(f"🎯 [SEQUENCE ANALYSIS]", flush=True)
                print(f"   ├─ Expected TGN shape: (num_events*2, 1, 1) with seq_len=1", flush=True)
                print(f"   ├─ Actual shape: t_abs{t_abs.shape}, t_rel{t_rel.shape}", flush=True)
                
                if len(t_abs.shape) >= 2:
                    seq_len = t_abs.shape[1] if len(t_abs.shape) > 1 else 1
                    print(f"   ├─ Sequence length: {seq_len}", flush=True)
                    
                    if seq_len == 1:
                        print(f"   ├─ ✅ CORRECT: seq_len=1 matches TGN expectation (single Δt per event)", flush=True)
                        print(f"   ├─ 🤔 BUT: Mamba2 works better with seq_len > 1 for temporal patterns", flush=True)
                        print(f"   └─ 💡 This explains potential architectural mismatch!", flush=True)
                    else:
                        print(f"   ├─ ❓ UNEXPECTED: seq_len={seq_len} > 1", flush=True)
                        print(f"   ├─ 💭 Possible reasons:", flush=True)
                        print(f"   │  ├─ Memory module providing historical embeddings", flush=True)
                        print(f"   │  ├─ Multi-hop neighbor aggregation", flush=True)
                        print(f"   │  └─ Batched temporal sequences", flush=True)
                        print(f"   └─ 🎯 This could justify Mamba2 usage!", flush=True)
                    
                    # Analyze the actual time values to understand the pattern
                    if DEBUG_TIME_ENCODER_VALUES and seq_len > 1:
                        print(f"   🔍 [TIME PATTERN ANALYSIS]", flush=True)
                        print(f"      ├─ t_abs[0]: {t_abs[0].flatten()[:min(10, seq_len)].tolist()}", flush=True)
                        print(f"      ├─ t_rel[0]: {t_rel[0].flatten()[:min(10, seq_len)].tolist()}", flush=True)
                        
                        # Check if values are identical (broadcasting) or truly sequential
                        if seq_len > 1:
                            abs_all_same = torch.all(t_abs[0] == t_abs[0, 0]).item()
                            rel_all_same = torch.all(t_rel[0] == t_rel[0, 0]).item()
                            print(f"      ├─ t_abs all same: {abs_all_same}", flush=True)
                            print(f"      ├─ t_rel all same: {rel_all_same}", flush=True)
                            
                            if abs_all_same and rel_all_same:
                                print(f"      └─ 🎯 PATTERN: Values repeated → Broadcasting/Memory module", flush=True)
                            else:
                                print(f"      └─ 🎯 PATTERN: Values differ → True temporal sequence", flush=True)
            cnt += 200
            if cnt == 1:
                sys.exit()
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
            # K-MOTE mode (default): Use K-MOTE for relative time
            v_k = self.k_mote_rel(t_rel)  # (B, S, expert_dim or rel_time_dim)
            if debug or hasattr(self, '_debug_mode'):
                print(f"🎯 K-MOTE REL OUTPUT (dual K-MOTE mode):")
                print(f"   v_k shape: {v_k.shape}")
        else:
            # SM-kernel mode (legacy): Use SM-kernel for relative time
            v_k = self.sm_kernel(t_rel)  # (B, S, num_mixtures)
            if debug or hasattr(self, '_debug_mode'):
                print(f"🎯 SM-KERNEL OUTPUT (legacy mode):")
                print(f"   v_k shape: {v_k.shape}")
        
        # ===== Apply fusion strategy =====
        if self.fusion_strategy == 'mamba':
            # ===== MAMBA FUSION (Original KAN-MAMMOTE) =====
            
            if self.use_controllable_mamba:
                # 🔍 DEBUG: Track Mamba2 processing
                if DEBUG_TIME_ENCODER_FACTORY and DEBUG_TIME_ENCODER_SHAPES:
                    print(f"🔍 [MAMBA2 PROCESSING]", flush=True)
                    print(f"   ├─ Input to Mamba2: u_k.shape={u_k.shape}", flush=True)
                    print(f"   ├─ Modulation: v_k.shape={v_k.shape}", flush=True)
                    print(f"   ├─ Mamba2 expects: (batch, seq_len, d_model)", flush=True)
                    
                    if u_k.shape[1] == 1:
                        print(f"   ├─ ⚠️  WARNING: seq_len=1 may not utilize Mamba2's temporal modeling", flush=True)
                        print(f"   └─ 💡 Mamba2 shines with longer sequences for temporal dependencies", flush=True)
                    else:
                        print(f"   └─ ✅ Good: seq_len={u_k.shape[1]} allows Mamba2 to model temporal patterns", flush=True)
                
                # ✅ SIMPLIFIED: Direct v_k → modulator_head (no fusion_mlp_base)
                modulator_logits = self.modulator_head(v_k)  # (B, S, nheads * 2)
                
                # Split into gamma and beta with improved parameterization
                gamma_logits, beta = modulator_logits.chunk(2, dim=-1)
                gamma = F.softplus(gamma_logits)  # Clean softplus (no bias needed)
                temporal_modulators = (gamma, beta)
                
                # Pure absolute time as content (separate pathway)
                combined_input = u_k
                
                if debug or hasattr(self, '_debug_mode'):
                    print(f"🔧 CONTROLLABLE MAMBA2 (SIMPLIFIED):")
                    print(f"   Direct v_k → modulator_head (no fusion_mlp_base)")
                    print(f"   gamma range: [{gamma.min().item():.3f}, {gamma.max().item():.3f}]")
                    print(f"   beta range: [{beta.min().item():.3f}, {beta.max().item():.3f}]")
                    print(f"   combined_input = u_k (pure absolute)")
                    print(f"   combined_input shape: {combined_input.shape}")
                
                # Run Mamba2 under autocast when AMP is enabled to save memory/compute
                with amp.autocast(enabled=(self.use_amp and torch.cuda.is_available())):
                    mamba_output = self.mamba2(u=combined_input, temporal_modulators=temporal_modulators)
                
            else:
                # Vanilla Mamba2: No temporal modulation
                #fusion_features = self.fusion_mlp_base(v_k)  # (B, S, expert_dim)
                fusion_features = v_k
                if debug or hasattr(self, '_debug_mode'):
                    print(f"� VANILLA MAMBA2:")
                    print(f"   fusion_features shape: {fusion_features.shape}")
                
                # ✅ FIXED: Residual + Post-LN pattern (Transformer best practice)
                combined_input = u_k + fusion_features  # Residual connection
                combined_input = self.fusion_norm(combined_input)  # ✅ LayerNorm AFTER residual
                
                if debug or hasattr(self, '_debug_mode'):
                    print(f"🔗 COMBINED INPUT PATHWAY (vanilla Mamba2) + Post-LN:")
                    print(f"   combined_input = fusion_norm(u_k + fusion_features)")
                    print(f"   combined_input shape: {combined_input.shape}")
                
                with amp.autocast(enabled=(self.use_amp and torch.cuda.is_available())):
                    mamba_output = self.mamba2(combined_input)
            
            # ✅ FIXED: Apply dropout after Mamba2 (Transformer best practice)
            mamba_output = self.mamba_dropout(mamba_output)
            
            if debug or hasattr(self, '_debug_mode'):
                print(f"🐍 MAMBA OUTPUT (after dropout):")
                print(f"   mamba_output shape: {mamba_output.shape}")
            
            final_embedding = self.output_projection(mamba_output)
            
        elif self.fusion_strategy == 'concat':
            # ===== CONCAT FUSION =====
            if debug or hasattr(self, '_debug_mode'):
                print(f"🔧 CONCAT FUSION:")
                print(f"   u_k (abs) shape: {u_k.shape}")
                print(f"   v_k (rel) shape: {v_k.shape}")
            
            # Concatenate absolute and relative features
            concat_features = torch.cat([u_k, v_k], dim=-1)  # (B, S, expert_dim + rel_time_dim)
            
            if debug or hasattr(self, '_debug_mode'):
                print(f"   concat_features shape: {concat_features.shape}")
            
            # Project through MLP
            final_embedding = self.fusion_mlp(concat_features)
            
        elif self.fusion_strategy == 'weighted':
            # ===== WEIGHTED SUM FUSION =====
            # Project relative to same dimension as absolute if needed
            v_k_proj = self.rel_projection(v_k)  # (B, S, expert_dim)
            
            if debug or hasattr(self, '_debug_mode'):
                print(f"🔧 WEIGHTED FUSION:")
                print(f"   u_k (abs) shape: {u_k.shape}")
                print(f"   v_k_proj (rel) shape: {v_k_proj.shape}")
                print(f"   weight_abs: {self.weight_abs.item():.3f}")
                print(f"   weight_rel: {self.weight_rel.item():.3f}")
            
            # Normalize weights
            w_abs = torch.sigmoid(self.weight_abs)
            w_rel = torch.sigmoid(self.weight_rel)
            w_sum = w_abs + w_rel
            w_abs_norm = w_abs / w_sum
            w_rel_norm = w_rel / w_sum
            
            # Weighted sum
            fused = w_abs_norm * u_k + w_rel_norm * v_k_proj  # (B, S, expert_dim)
            
            if debug or hasattr(self, '_debug_mode'):
                print(f"   normalized weights: abs={w_abs_norm:.3f}, rel={w_rel_norm:.3f}")
            
            # Project to embedding dimension
            final_embedding = self.output_projection(fused)
            
        elif self.fusion_strategy == 'attention':
            # ===== CROSS-ATTENTION FUSION =====
            # Project relative to same dimension as absolute if needed
            v_k_proj = self.rel_projection(v_k)  # (B, S, expert_dim)
            
            if debug or hasattr(self, '_debug_mode'):
                print(f"� ATTENTION FUSION:")
                print(f"   u_k (abs, query) shape: {u_k.shape}")
                print(f"   v_k_proj (rel, key/value) shape: {v_k_proj.shape}")
            
            # Cross-attention: absolute features attend to relative features
            # query: u_k (absolute), key/value: v_k_proj (relative)
            attn_output, _ = self.cross_attention(
                query=u_k,
                key=v_k_proj,
                value=v_k_proj
            )  # (B, S, expert_dim)
            
            # Residual connection + layer norm
            fused = self.norm_after_attn(u_k + attn_output)
            
            if debug or hasattr(self, '_debug_mode'):
                print(f"   attn_output shape: {attn_output.shape}")
                print(f"   fused shape: {fused.shape}")
            
            # Project to embedding dimension
            final_embedding = self.output_projection(fused)

        if debug or hasattr(self, '_debug_mode'):
            print(f"🎯 FINAL OUTPUT:")
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
        print("🔍 KAN-MAMMOTE Debug Mode ENABLED")
    
    def disable_debug_mode(self):
        """Disable persistent debug mode"""
        if hasattr(self, '_debug_mode'):
            delattr(self, '_debug_mode')
        print("🔍 KAN-MAMMOTE Debug Mode DISABLED")