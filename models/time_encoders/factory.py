import torch
import numpy as np
import sys
import inspect

# ===== GLOBAL DEBUG VARIABLES =====
DEBUG_TIME_ENCODER_FACTORY = False  # Set to True to enable detailed debugging
DEBUG_TIME_ENCODER_CALLS = False    # Set to True to debug every forward call
DEBUG_TIME_ENCODER_VALUES = False   # Set to True to print actual time values
DEBUG_TIME_ENCODER_SHAPES = False   # Set to True to debug tensor shapes

# 🔍 Global Model Debug Controls (consolidated from root factory.py)
DEBUG_MODEL = False  # Master switch for all model debugging output
DEBUG_TIME_SORTING = False      # Neighbor time sorting verification
DEBUG_TIME_COMPUTATION = False  # Time computation pattern verification
DEBUG_ENCODER_INTERFACE = False # KAN_MAMMOTE encoder interface debugging

def enable_factory_debug(enable_calls=True, enable_values=True, enable_shapes=True):
    """Enable debug mode for time encoder factory"""
    global DEBUG_TIME_ENCODER_FACTORY, DEBUG_TIME_ENCODER_CALLS, DEBUG_TIME_ENCODER_VALUES, DEBUG_TIME_ENCODER_SHAPES
    DEBUG_TIME_ENCODER_FACTORY = True
    DEBUG_TIME_ENCODER_CALLS = enable_calls
    DEBUG_TIME_ENCODER_VALUES = enable_values
    DEBUG_TIME_ENCODER_SHAPES = enable_shapes
    print("🔍 TIME ENCODER FACTORY DEBUG MODE ENABLED")
    print(f"   - Calls: {DEBUG_TIME_ENCODER_CALLS}")
    print(f"   - Values: {DEBUG_TIME_ENCODER_VALUES}")
    print(f"   - Shapes: {DEBUG_TIME_ENCODER_SHAPES}")

def disable_factory_debug():
    """Disable debug mode for time encoder factory"""
    global DEBUG_TIME_ENCODER_FACTORY, DEBUG_TIME_ENCODER_CALLS, DEBUG_TIME_ENCODER_VALUES, DEBUG_TIME_ENCODER_SHAPES
    DEBUG_TIME_ENCODER_FACTORY = False
    DEBUG_TIME_ENCODER_CALLS = False
    DEBUG_TIME_ENCODER_VALUES = False
    DEBUG_TIME_ENCODER_SHAPES = False
    print("🔍 TIME ENCODER FACTORY DEBUG MODE DISABLED")

# 🔍 Consolidated Model Debug Functions (from root factory.py)
def should_debug_model():
    """Check if model debugging is enabled."""
    #print(f"🔍 [FACTORY] should_debug_model() called, returning: {DEBUG_MODEL}")
    return DEBUG_MODEL

def should_debug_time_sorting():
    """Check if time sorting debugging is enabled."""
    return DEBUG_MODEL and DEBUG_TIME_SORTING

def should_debug_time_computation():
    """Check if time computation debugging is enabled."""
    return DEBUG_MODEL and DEBUG_TIME_COMPUTATION

def should_debug_encoder_interface():
    """Check if encoder interface debugging is enabled."""
    return DEBUG_MODEL and DEBUG_ENCODER_INTERFACE

# Quick toggle functions
def enable_all_debug():
    """Enable all debugging."""
    global DEBUG_MODEL, DEBUG_TIME_SORTING, DEBUG_TIME_COMPUTATION, DEBUG_ENCODER_INTERFACE
    DEBUG_MODEL = True
    DEBUG_TIME_SORTING = True
    DEBUG_TIME_COMPUTATION = True
    DEBUG_ENCODER_INTERFACE = True

def disable_all_debug():
    """Disable all debugging."""
    global DEBUG_MODEL, DEBUG_TIME_SORTING, DEBUG_TIME_COMPUTATION, DEBUG_ENCODER_INTERFACE
    DEBUG_MODEL = False
    DEBUG_TIME_SORTING = False
    DEBUG_TIME_COMPUTATION = False
    DEBUG_ENCODER_INTERFACE = False

def set_debug_mode(enabled: bool):
    """Set main debug mode on/off."""
    global DEBUG_MODEL
    DEBUG_MODEL = enabled

def debug_print(message, debug_type="general"):
    """Conditional debug printing"""
    if DEBUG_TIME_ENCODER_FACTORY:
        # Use independent if statements instead of elif to allow multiple debug types
        if debug_type == "calls" and DEBUG_TIME_ENCODER_CALLS:
            print(f"🔍 [CALLS] {message}", flush=True)
        if debug_type == "values" and DEBUG_TIME_ENCODER_VALUES:
            print(f"🔍 [VALUES] {message}", flush=True)
        if debug_type == "shapes" and DEBUG_TIME_ENCODER_SHAPES:
            print(f"🔍 [SHAPES] {message}", flush=True)
        if debug_type == "general":
            print(f"🔍 [DEBUG] {message}", flush=True)
        
        # Force flush stdout
        import sys
        sys.stdout.flush()

# Debug mode is DISABLED by default
# Call enable_factory_debug() explicitly if you need debugging

# Import standard encoder
from ..gnn_backbones.modules import TimeEncoder as OriginalTimeEncoder

# Import Mamba-based encoders with optional fallback
try:
    from .kan_mammote import KAN_MAMMOTE
    print("🔍 [FACTORY] ✅ Successfully imported KAN_MAMMOTE", flush=True)
except ImportError as e:
    KAN_MAMMOTE = None
    print(f"🔍 [FACTORY] ❌ Failed to import KAN_MAMMOTE: {e}", flush=True)

try:
    from .kan_mammote_lite import KAN_MAMMOTE_Lite
    print("🔍 [FACTORY] ✅ Successfully imported KAN_MAMMOTE_Lite", flush=True)
except ImportError as e:
    KAN_MAMMOTE_Lite = None
    print(f"🔍 [FACTORY] ❌ Failed to import KAN_MAMMOTE_Lite: {e}", flush=True)

# Import additional encoders
try:
    from .mercer_encoder import MercerTimeEncoder
except ImportError:
    MercerTimeEncoder = None

try:
    from .lete_encoder import LeTE
except ImportError:
    LeTE = None

try:
    from .bochner_encoder import BochnerTimeEncoder
except ImportError:
    BochnerTimeEncoder = None

try:
    from .time2vec_encoder import Time2VecEncoder
except ImportError:
    Time2VecEncoder = None

# Import ablation study encoders
try:
    from .ablation_encoders import SMKernelOnly, KMOTEAbsOnly, KMOTERelOnly, DualStreamBaseline
    print("🔍 [FACTORY] ✅ Successfully imported ablation encoders", flush=True)
except ImportError as e:
    SMKernelOnly = None
    KMOTEAbsOnly = None
    KMOTERelOnly = None
    DualStreamBaseline = None
    print(f"🔍 [FACTORY] ❌ Failed to import ablation encoders: {e}", flush=True)

print("🔍 [FACTORY] ✅ ALL IMPORTS COMPLETED - Factory ready!", flush=True)
sys.stdout.flush()

class TimeEncoderWrapper(torch.nn.Module):
    """
    Adapter to make various time encoders compatible with different interfaces.
    Ensures consistent output dimensions while preserving KAN-MAMMOTE functionality.
    
    Key Insight:
    - KAN-MAMMOTE uses: (t_abs, t_rel) - dual stream
    - OriginalTimeEncoder uses: timestamps (which is actually t_rel/delta_t)
    - TGAT passes: neighbor_time_features = time_encoder(t_abs=..., t_rel=...)
    """
    def __init__(self, encoder):
        super(TimeEncoderWrapper, self).__init__()
        self.encoder = encoder
        self.encoder_name = encoder.__class__.__name__
        self._call_count = 0
        self._debug_mode = False
        
    def enable_debug_mode(self):
        """Enable debug mode for this wrapper and the underlying encoder"""
        self._debug_mode = True
        if hasattr(self.encoder, 'enable_debug_mode'):
            self.encoder.enable_debug_mode()
        print(f"🔍 TimeEncoderWrapper Debug Mode ENABLED for {self.encoder_name}")
    
    def disable_debug_mode(self):
        """Disable debug mode"""
        self._debug_mode = False
        if hasattr(self.encoder, 'disable_debug_mode'):
            self.encoder.disable_debug_mode()
        print(f"🔍 TimeEncoderWrapper Debug Mode DISABLED for {self.encoder_name}")
        
    def forward(self, t_abs=None, t_rel=None, timestamps=None):
        self._call_count += 1
        
        # Global debug information (always enabled when global flag is on)
        if DEBUG_TIME_ENCODER_FACTORY:
            debug_print(f"TimeEncoderWrapper.forward() call #{self._call_count} for {self.encoder_name}", "calls")
            if DEBUG_TIME_ENCODER_VALUES:
                debug_print(f"Input parameters:", "values")
                if t_abs is not None:
                    debug_print(f"  t_abs: shape={t_abs.shape if hasattr(t_abs, 'shape') else 'scalar'}, "
                              f"min={t_abs.min().item() if hasattr(t_abs, 'min') else t_abs}, "
                              f"max={t_abs.max().item() if hasattr(t_abs, 'max') else t_abs}", "values")
                if t_rel is not None:
                    debug_print(f"  t_rel: shape={t_rel.shape if hasattr(t_rel, 'shape') else 'scalar'}, "
                              f"min={t_rel.min().item() if hasattr(t_rel, 'min') else t_rel}, "
                              f"max={t_rel.max().item() if hasattr(t_rel, 'max') else t_rel}", "values")
                if timestamps is not None:
                    debug_print(f"  timestamps: shape={timestamps.shape if hasattr(timestamps, 'shape') else 'scalar'}, "
                              f"min={timestamps.min().item() if hasattr(timestamps, 'min') else timestamps}, "
                              f"max={timestamps.max().item() if hasattr(timestamps, 'max') else timestamps}", "values")
        
        if self._debug_mode:
            print(f"\n🔧 TimeEncoderWrapper Call #{self._call_count}")
            print(f"   Encoder: {self.encoder_name}")
            print(f"   Inputs provided:")
            if t_abs is not None:
                print(f"     t_abs: {t_abs.shape} | range: [{t_abs.min().item():.3f}, {t_abs.max().item():.3f}]")
            if t_rel is not None:
                print(f"     t_rel: {t_rel.shape} | range: [{t_rel.min().item():.3f}, {t_rel.max().item():.3f}]")
            if timestamps is not None:
                print(f"     timestamps: {timestamps.shape} | range: [{timestamps.min().item():.3f}, {timestamps.max().item():.3f}]")
        
        import inspect
        
        result = None
        sig = inspect.signature(self.encoder.forward)
        params = list(sig.parameters.keys())
        
        # Global debug for function signature
        if DEBUG_TIME_ENCODER_FACTORY:
            debug_print(f"Encoder forward signature: {params}", "calls")
            
        print(params)
        # Strategy 1: Dual-stream interface (KAN-MAMMOTE)
        if 't_abs' in params and 't_rel' in params:
            if t_abs is not None and t_rel is not None:
                if DEBUG_TIME_ENCODER_FACTORY:
                    debug_print("Using dual-stream interface: t_abs + t_rel", "calls")
                    debug_print(f"t_abs: {t_abs.shape} | range: [{t_abs.min().item():.3f}, {t_abs.max().item():.3f}]", "values")
                    debug_print(f"t_rel: {t_rel.shape} | range: [{t_rel.min().item():.3f}, {t_rel.max().item():.3f}]", "values")
                
                print("Using both t_abs and t_rel")
                #debug how t_abs and t_rel is used with easy to read while print all
                print(f"   🎯 Using dual-stream interface: t_abs + t_rel")
                print(f"     t_abs: {t_abs.shape} | range: [{t_abs.min().item():.3f}, {t_abs.max().item():.3f}]")
                print(f"     t_rel: {t_rel.shape} | range: [{t_rel.min().item():.3f}, {t_rel.max().item():.3f}]")
                
                # TEMPORARY: Exit to debug values - REMOVE THIS WHEN DONE DEBUGGING
                if DEBUG_TIME_ENCODER_FACTORY and DEBUG_TIME_ENCODER_VALUES:
                    debug_print("TEMPORARY EXIT FOR DEBUGGING - REMOVE WHEN DONE", "general")
                    import sys
                    sys.exit()
                    
                if self._debug_mode:
                    print(f"   🎯 Using dual-stream interface: t_abs + t_rel")
                result = self.encoder.forward(t_abs=t_abs, t_rel=t_rel)
            elif timestamps is not None:
                # timestamps is usually delta_t, so treat as t_rel
                t_rel_default = timestamps
                t_abs_default = torch.zeros_like(timestamps)
                
                if DEBUG_TIME_ENCODER_FACTORY:
                    debug_print("Using dual-stream interface: dummy t_abs + timestamps as t_rel", "calls")
                    debug_print(f"t_abs_default (zeros): {t_abs_default.shape}", "values")
                    debug_print(f"t_rel_default (timestamps): {t_rel_default.shape} | range: [{t_rel_default.min().item():.3f}, {t_rel_default.max().item():.3f}]", "values")
                
                print("kuay I sus this case")
                # TEMPORARY: Exit to debug values - REMOVE THIS WHEN DONE DEBUGGING  
                if DEBUG_TIME_ENCODER_FACTORY:
                    debug_print("TEMPORARY EXIT FOR DEBUGGING - REMOVE WHEN DONE", "general")
                    import sys
                    sys.exit()
                    
                if self._debug_mode:
                    print(f"   🎯 Using dual-stream interface: dummy t_abs + timestamps as t_rel")
                result = self.encoder.forward(t_abs=t_abs_default, t_rel=t_rel_default)
        
        # Strategy 2: OriginalTimeEncoder uses relative time (delta_t)
        # It expects: forward(timestamps) where timestamps = delta_t (relative time)
        elif self.encoder_name == 'TimeEncoder' or 'timestamps' in params:
            # OriginalTimeEncoder uses RELATIVE time (t_rel), not absolute!
            # In TGAT context: delta_t = current_time - neighbor_time
            if t_rel is not None:
                if self._debug_mode:
                    print(f"   🎯 Using single-stream interface: t_rel")
                # Use t_rel (this is the correct input for OriginalTimeEncoder)
                result = self.encoder.forward(t_rel)
            elif timestamps is not None:
                if self._debug_mode:
                    print(f"   🎯 Using single-stream interface: timestamps")
                # timestamps in TGAT is actually delta_t (relative time)
                result = self.encoder.forward(timestamps)
            elif t_abs is not None:
                if self._debug_mode:
                    print(f"   ⚠️  Fallback: Using t_abs as timestamps (unexpected)")
                # Fallback: if only t_abs provided, assume it's delta_t
                # (this shouldn't happen in proper usage)
                print(f"WARNING: OriginalTimeEncoder expects t_rel but got t_abs. Using t_abs as fallback.")
                result = self.encoder.forward(t_abs)
        
        # Strategy 3: Generic positional argument
        elif result is None:
            # Determine which input to use
            # Priority: t_rel (for time encoders) > timestamps > t_abs
            if t_rel is not None:
                input_tensor = t_rel
            elif timestamps is not None:
                input_tensor = timestamps
            elif t_abs is not None:
                input_tensor = t_abs
            else:
                raise ValueError("Must provide at least one time input")
            
            if self._debug_mode:
                print(f"   🎯 Using generic interface: {input_tensor.shape}")
            result = self.encoder.forward(input_tensor)
        
        if self._debug_mode and result is not None:
            print(f"   📤 Output shape: {result.shape}")
        
        # Dimension fixing (keep existing logic)
        if result is not None:
            # Determine expected shape from input
            input_for_shape = t_rel if t_rel is not None else (t_abs if t_abs is not None else timestamps)
            
            if input_for_shape.dim() == 3:  # (batch, neighbors, 1)
                expected_shape = (input_for_shape.shape[0], input_for_shape.shape[1], result.shape[-1])
            elif input_for_shape.dim() == 2:  # (batch, 1)
                expected_shape = (input_for_shape.shape[0], result.shape[-1])
            else:
                expected_shape = result.shape
            
            # Handle different dimension cases
            if result.dim() == 4:  # [batch, seq, 1, feat] -> [batch, seq, feat]
                result = result.squeeze(2)
            elif result.dim() == 3 and result.shape[-2] == 1:  # [batch, 1, feat] -> [batch, feat]
                pass
                #result = result.squeeze(-2)
            elif result.dim() == 2 and input_for_shape.dim() == 3:
                # Need to expand for neighbors: [batch, feat] -> [batch, neighbors, feat]
                num_neighbors = input_for_shape.shape[1]
                result = result.unsqueeze(1).expand(-1, num_neighbors, -1)
            elif result.dim() == 1:  # [feat] -> proper shape
                if input_for_shape.dim() == 3:
                    result = result.unsqueeze(0).unsqueeze(0).expand(
                        input_for_shape.shape[0], input_for_shape.shape[1], -1
                    )
                else:
                    result = result.unsqueeze(0)
        
        return result

def get_available_encoders():
    """
    Return a list of available time encoder types.
    """
    encoders = [
        'original',
        'time_encoder',
        'default'
    ]
    
    # Add Mamba-based encoders only if available
    if KAN_MAMMOTE is not None:
        encoders.extend([
            'kan_mammote',
            'kan_mammote_dual_kmote'  # NEW: Dual K-MOTE variant
            ,'dual_stream_baseline',
            "kan_mammote_dual_kmote_tgat"
        ])
    
    if KAN_MAMMOTE_Lite is not None:
        encoders.append('kan_mammote_lite')
    
    # Add available optional encoders
    if MercerTimeEncoder is not None:
        encoders.append('mercer')
    if LeTE is not None:
        encoders.append('lete')
    if BochnerTimeEncoder is not None:
        encoders.append('bochner')
    if Time2VecEncoder is not None:
        encoders.append('time2vec')
    
    # Add ablation study encoders (always available since they're local)
    encoders.extend([
        'sm_kernel_only',
        'kmote_abs_only',
        'kmote_rel_only',
        'k_mote',  # ✅ Standalone K-MOTE (without Mamba, for MNIST experiments)
    ])
    
    return encoders

def get_encoder_config(encoder_type: str):
    """
    Return the configuration parameters for a specific encoder type.
    """
    encoder_type = encoder_type.lower()
    
    configs = {
        'kan_mammote': {
            'required_params': ['embedding_dim', 'expert_dim', 'num_mixtures'],
            'optional_params': {
                'mamba_d_state': 16,
                'mamba_d_conv': 4, 
                'mamba_expand': 2,
                'mamba_headdim': 64,
                'use_kmote_for_relative': False
            },
            'description': 'KAN-MAMMOTE: Advanced time encoder with Mamba2 and SM-kernel for relative time'
        },
        'kan_mammote_dual_kmote': {
            'required_params': ['embedding_dim', 'expert_dim'],
            'optional_params': {
                'mamba_d_state': 16,
                'mamba_d_conv': 4, 
                'mamba_expand': 2,
                'mamba_headdim': 64,
                'use_kmote_for_relative': True,
                'num_mixtures': 16  # Still needed for fusion architecture
            },
            'description': 'KAN-MAMMOTE Dual K-MOTE: Uses K-MOTE for both absolute and relative time encoding'
        },
        'kan_mammote_dual_kmote_tgat': {
            'required_params': ['embedding_dim', 'expert_dim'],
            'optional_params': {
                'mamba_d_state': 16,
                'mamba_d_conv': 4, 
                'mamba_expand': 2,
                'mamba_headdim': 64,
                'use_kmote_for_relative': True,
                'num_mixtures': 16  # Still needed for fusion architecture
            },
            'description': 'KAN-MAMMOTE Dual K-MOTE: Uses K-MOTE for both absolute and relative time encoding for TGAT'
        },
        'kan_mammote_lite': {
            'required_params': ['embedding_dim', 'num_mixtures'],
            'optional_params': {
                'wavelet_type': 'shock',
                'use_dual_stream': True
            },
            'description': 'KAN-MAMMOTE Lite: Lightweight stateless version without Mamba (for TGAT/attention models)'
        },
        'mercer': {
            'required_params': ['time_dim'],
            'optional_params': {},
            'description': 'Mercer Time Encoder: Kernel-based temporal encoding'
        },
        'lete': {
            'required_params': ['time_dim'],
            'optional_params': {},
            'description': 'LeTE: Learnable Time Encoder'
        },
        'bochner': {
            'required_params': ['time_dim'],
            'optional_params': {},
            'description': 'Bochner Time Encoder: Fourier feature based encoding'
        },
        'time2vec': {
            'required_params': ['time_dim'],
            'optional_params': {},
            'description': 'Time2Vec: Time encoding with periodic and linear components'
        },
        'original': {
            'required_params': ['time_dim'],
            'optional_params': {},
            'description': 'Original TimeEncoder wrapped for compatibility'
        },
        'time_encoder': {
            'required_params': ['time_dim'],
            'optional_params': {},
            'description': 'Alias for original TimeEncoder'
        },
        'default': {
            'required_params': ['time_dim'],
            'optional_params': {},
            'description': 'Default TimeEncoder'
        }
    }
    
    return configs.get(encoder_type, configs['default'])

def create_time_encoder(encoder_type: str, time_dim: int, train_data=None, train_neighbor_sampler=None, args=None, device='cpu', **kwargs):
    """
    Factory function to create and initialize the correct time encoder.
    """
    print(f"🔍 [CREATE_ENCODER] CALLED with encoder_type='{encoder_type}', time_dim={time_dim}", flush=True)
    sys.stdout.flush()
    
    encoder_type = encoder_type.lower()
    
    # Global debug information for encoder creation
    if DEBUG_TIME_ENCODER_FACTORY:
        debug_print(f"Creating time encoder: {encoder_type} with time_dim={time_dim}", "general")
        debug_print(f"Device: {device}", "general")
        debug_print(f"Train data provided: {train_data is not None}", "general")
        debug_print(f"Train neighbor sampler provided: {train_neighbor_sampler is not None}", "general")
        if args is not None:
            debug_print(f"Args provided with keys: {vars(args).keys()}", "general")
        if kwargs:
            debug_print(f"Kwargs provided: {kwargs}", "general")
    
    # Show available encoders for debugging
    available = get_available_encoders()
    print(f"Available encoders: {available}")
    print(f"Requested encoder: {encoder_type}")
    
    if DEBUG_TIME_ENCODER_FACTORY:
        debug_print(f"Available encoders: {available}", "general")
        debug_print(f"Creating encoder type: {encoder_type}", "general")
    
    if encoder_type == 'kan_mammote':
        if KAN_MAMMOTE is None:
            raise ImportError(f"KAN_MAMMOTE encoder is not available. Please install Mamba dependencies or use a different encoder.")
        
        print("INFO: Creating KAN-MAMMOTE time encoder.")
        
        # Handle args gracefully - try args first, then kwargs, then defaults
        if args is not None:
            print("INFO: Extracting KAN-MAMMOTE parameters from args.")
            expert_dim = getattr(args, 'expert_dim', kwargs.get('expert_dim', 128))
            num_mixtures = getattr(args, 'num_mixtures', kwargs.get('num_mixtures', 16))
            mamba_d_state = getattr(args, 'mamba_d_state', kwargs.get('mamba_d_state', 256))
            mamba_d_conv = getattr(args, 'mamba_d_conv', kwargs.get('mamba_d_conv', 4))
            mamba_expand = getattr(args, 'mamba_expand', kwargs.get('mamba_expand', 2))
            mamba_headdim = getattr(args, 'mamba_headdim', kwargs.get('mamba_headdim', 64))
            batch_size = getattr(args, 'batch_size', kwargs.get('batch_size', 200))
            num_neighbors = getattr(args, 'num_neighbors', kwargs.get('num_neighbors', 20))
            # NEW: Support for dual K-MOTE mode
            use_kmote_for_relative = getattr(args, 'use_kmote_for_relative', kwargs.get('use_kmote_for_relative', False))
        else:
            # Get from kwargs or use defaults
            print("INFO: Extracting KAN-MAMMOTE parameters from kwargs or using defaults.")
            expert_dim = kwargs.get('expert_dim', 64)
            num_mixtures = kwargs.get('num_mixtures', 4)
            mamba_d_state = kwargs.get('mamba_d_state', 16)
            mamba_d_conv = kwargs.get('mamba_d_conv', 4)
            mamba_expand = kwargs.get('mamba_expand', 2)
            mamba_headdim = kwargs.get('mamba_headdim', 64)
            batch_size = kwargs.get('batch_size', 200)
            num_neighbors = kwargs.get('num_neighbors', 20)
            # NEW: Support for dual K-MOTE mode
            use_kmote_for_relative = kwargs.get('use_kmote_for_relative', False)
        
        print(f"KAN-MAMMOTE parameters:")
        print(f"  - embedding_dim: {time_dim}")
        print(f"  - expert_dim: {expert_dim}")
        print(f"  - num_mixtures: {num_mixtures}")
        print(f"  - mamba_d_state: {mamba_d_state}")
        print(f"  - mamba_d_conv: {mamba_d_conv}")
        print(f"  - mamba_expand: {mamba_expand}")
        print(f"  - use_kmote_for_relative: {use_kmote_for_relative}")
        
        time_encoder = KAN_MAMMOTE(
            embedding_dim=time_dim,
            expert_dim=expert_dim,
            num_mixtures=num_mixtures,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            use_kmote_for_relative=use_kmote_for_relative
        )
        
        # SM-Kernel Initialization (if data is provided)
        if train_data is not None and train_neighbor_sampler is not None:
            try:
                print("INFO: Performing SM-Kernel initialization...")
                # Get a single batch of indices to create a sample
                sample_batch_indices = np.arange(min(batch_size, len(train_data.src_node_ids)))
                sample_src = train_data.src_node_ids[sample_batch_indices]
                sample_ts = train_data.node_interact_times[sample_batch_indices]
                
                _, _, sample_neighbor_ts = train_neighbor_sampler.get_historical_neighbors(
                    sample_src, sample_ts, num_neighbors
                )
                
                sample_delta_t = sample_ts[:, np.newaxis] - sample_neighbor_ts
                
                # Ensure sample is not empty
                if sample_delta_t.size > 0:
                    sample_delta_t_tensor = torch.from_numpy(sample_delta_t).float().unsqueeze(-1)
                    time_encoder.initialize_sm_kernel(sample_delta_t_tensor.to(device))
                    print("INFO: SM-Kernel initialization complete!")
                else:
                    print("WARNING: Could not generate a sample for SM-Kernel initialization (dataset might be small). Skipping.")
            except Exception as e:
                print(f"WARNING: SM-Kernel initialization failed: {e}. Using default initialization.")
        else:
            print("INFO: No training data provided. SM-Kernel will use default initialization.")
    
    elif encoder_type == 'kan_mammote_dual_kmote':
        if KAN_MAMMOTE is None:
            raise ImportError(f"KAN_MAMMOTE encoder is not available. Please install Mamba dependencies or use a different encoder.")
        
        print("INFO: Creating KAN-MAMMOTE time encoder with dual K-MOTE (no SM-kernel).")
        
        # Handle args gracefully - try args first, then kwargs, then defaults
        if args is not None:
            print("INFO: Extracting KAN-MAMMOTE Dual K-MOTE parameters from args.")
            expert_dim = getattr(args, 'expert_dim', kwargs.get('expert_dim', 128))
            num_mixtures = getattr(args, 'num_mixtures', kwargs.get('num_mixtures', 16))
            mamba_d_state = getattr(args, 'mamba_d_state', kwargs.get('mamba_d_state', 256))
            mamba_d_conv = getattr(args, 'mamba_d_conv', kwargs.get('mamba_d_conv', 4))
            mamba_expand = getattr(args, 'mamba_expand', kwargs.get('mamba_expand', 2))
            mamba_headdim = getattr(args, 'mamba_headdim', kwargs.get('mamba_headdim', 64))
            encoder_dropout = getattr(args, 'encoder_dropout', getattr(args, 'dropout', 0.1))  # Use encoder_dropout, fallback to dropout
        else:
            # Get from kwargs or use defaults
            print("INFO: Extracting KAN-MAMMOTE Dual K-MOTE parameters from kwargs or using defaults.")
            expert_dim = kwargs.get('expert_dim', 64)
            num_mixtures = kwargs.get('num_mixtures', 4)
            mamba_d_state = kwargs.get('mamba_d_state', 16)
            mamba_d_conv = kwargs.get('mamba_d_conv', 4)
            mamba_expand = kwargs.get('mamba_expand', 2)
            mamba_headdim = kwargs.get('mamba_headdim', 64)
            encoder_dropout = kwargs.get('encoder_dropout', kwargs.get('dropout', 0.1))
        
        print(f"KAN-MAMMOTE Dual K-MOTE parameters:")
        print(f"  - embedding_dim: {time_dim}")
        print(f"  - expert_dim: {expert_dim}")
        print(f"  - num_mixtures: {num_mixtures} (for fusion architecture)")
        print(f"  - mamba_d_state: {mamba_d_state}")
        print(f"  - mamba_d_conv: {mamba_d_conv}")
        print(f"  - mamba_expand: {mamba_expand}")
        print(f"  - use_kmote_for_relative: True (dual K-MOTE mode)")
        
        time_encoder = KAN_MAMMOTE(
            embedding_dim=time_dim,
            expert_dim=expert_dim,
            num_mixtures=num_mixtures,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            mamba_headdim=mamba_headdim,
            dropout=encoder_dropout,  # Use encoder-specific dropout
            use_kmote_for_relative=True  # Force dual K-MOTE mode
        )
        
        print("INFO: No SM-Kernel initialization needed (using dual K-MOTE mode).")
    elif encoder_type == 'kan_mammote_dual_kmote_tgat':
        if KAN_MAMMOTE is None:
            raise ImportError(f"KAN_MAMMOTE encoder is not available. Please install Mamba dependencies or use a different encoder.")
        
        print("INFO: Creating kan_mammote_dual_kmote_tgat")
        
        # Handle args gracefully - try args first, then kwargs, then defaults
        if args is not None:
            print("INFO: Extracting KAN-MAMMOTE Dual K-MOTE parameters from args.")
            expert_dim = 64
            num_mixtures = getattr(args, 'num_mixtures', kwargs.get('num_mixtures', 16))
            mamba_d_state = getattr(args, 'mamba_d_state', kwargs.get('mamba_d_state', 256))
            mamba_d_conv = getattr(args, 'mamba_d_conv', kwargs.get('mamba_d_conv', 4))
            mamba_expand = getattr(args, 'mamba_expand', kwargs.get('mamba_expand', 2))
            mamba_headdim = getattr(args, 'mamba_headdim', kwargs.get('mamba_headdim', 64))
        else:
            # Get from kwargs or use defaults
            print("INFO: Extracting KAN-MAMMOTE Dual K-MOTE parameters from kwargs or using defaults.")
            expert_dim = kwargs.get('expert_dim', 64)
            num_mixtures = kwargs.get('num_mixtures', 4)
            mamba_d_state = kwargs.get('mamba_d_state', 16)
            mamba_d_conv = kwargs.get('mamba_d_conv', 4)
            mamba_expand = kwargs.get('mamba_expand', 2)
            mamba_headdim = kwargs.get('mamba_headdim', 64)
            encoder_dropout = kwargs.get('encoder_dropout', kwargs.get('dropout', 0.1))
        
        print(f"KAN-MAMMOTE Dual K-MOTE parameters:")
        print(f"  - embedding_dim: {time_dim}")
        print(f"  - expert_dim: {expert_dim}")
        print(f"  - num_mixtures: {num_mixtures} (for fusion architecture)")
        print(f"  - mamba_d_state: {mamba_d_state}")
        print(f"  - mamba_d_conv: {mamba_d_conv}")
        print(f"  - mamba_expand: {mamba_expand}")
        print(f"  - encoder_dropout: {encoder_dropout}")
        print(f"  - use_kmote_for_relative: True (dual K-MOTE mode)")
        
        time_encoder = KAN_MAMMOTE(
            embedding_dim=time_dim,
            expert_dim=expert_dim,
            num_mixtures=num_mixtures,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            mamba_headdim=mamba_headdim,
            dropout=encoder_dropout,  # Use encoder-specific dropout
            use_kmote_for_relative=True  # Force dual K-MOTE mode
        )
        
        print("INFO: No SM-Kernel initialization needed (using dual K-MOTE mode).")
    
    elif encoder_type == 'kan_mammote_lite':
        if KAN_MAMMOTE_Lite is None:
            raise ImportError(f"KAN_MAMMOTE_Lite encoder is not available. Please install Mamba dependencies or use a different encoder.")
        
        print("INFO: Creating KAN-MAMMOTE Lite time encoder (stateless version).")
        
        # Handle args gracefully
        if args is not None:
            embedding_dim = getattr(args, 'time_feat_dim', time_dim)
            num_mixtures = getattr(args, 'num_mixtures', 12)
            wavelet_type = getattr(args, 'wavelet_type', 'shock')
            use_dual_stream = getattr(args, 'use_dual_stream', True)
        else:
            embedding_dim = kwargs.get('embedding_dim', time_dim)
            num_mixtures = kwargs.get('num_mixtures', 12)
            wavelet_type = kwargs.get('wavelet_type', 'shock')
            use_dual_stream = kwargs.get('use_dual_stream', True)
        
        time_encoder = KAN_MAMMOTE_Lite(
            embedding_dim=embedding_dim,
            num_mixtures=num_mixtures,
            wavelet_type=wavelet_type,
            use_dual_stream=use_dual_stream
        )
        
        # Initialize SM-Kernel if training data is available
        if train_data is not None and train_neighbor_sampler is not None:
            print("Initializing SM-Kernel for KAN-MAMMOTE Lite from training data...")
            batch_size = 200
            node_interact_times = train_data.node_interact_times
            node_idx_array = np.arange(train_data.num_unique_nodes)  # Fixed: use num_unique_nodes
            
            sampled_indices = np.random.choice(len(node_idx_array), size=min(batch_size, len(node_idx_array)), replace=False)
            sampled_node_ids = node_idx_array[sampled_indices]
            sampled_edge_times = node_interact_times[sampled_indices]
            
            neighbor_node_ids, neighbor_edge_ids, neighbor_times = train_neighbor_sampler.get_all_first_hop_neighbors(
                node_ids=sampled_node_ids, node_interact_times=sampled_edge_times
            )
            
            delta_t_samples = []
            for idx in range(len(sampled_node_ids)):
                current_time = sampled_edge_times[idx]
                neighbor_t = neighbor_times[idx]
                if len(neighbor_t) > 0:
                    delta = current_time - neighbor_t
                    delta_t_samples.append(delta)
            
            if delta_t_samples:
                delta_t_tensor = torch.from_numpy(np.concatenate(delta_t_samples)).float().unsqueeze(-1)
                time_encoder.initialize_sm_kernel(delta_t_tensor)
                print(f"SM-Kernel initialized with {len(delta_t_tensor)} delta_t samples.")
            else:
                print("Warning: No delta_t samples found; SM-Kernel not initialized.")
        
    elif encoder_type == 'mercer' and MercerTimeEncoder is not None:
        print("INFO: Creating Mercer Time Encoder.")
        print("Time Embedding dim:", time_dim)
        encoder = MercerTimeEncoder(time_dim=time_dim)
        time_encoder = TimeEncoderWrapper(encoder)
        
    elif encoder_type == 'lete' and LeTE is not None:
        print("INFO: Creating LeTE Time Encoder.")
        print("Time Embedding dim:", time_dim)
        encoder = LeTE(time_dim=time_dim)
        time_encoder = TimeEncoderWrapper(encoder)
        
    elif encoder_type == 'bochner' and BochnerTimeEncoder is not None:
        print("INFO: Creating Bochner Time Encoder.")
        print("Time Embedding dim:", time_dim)
        encoder = BochnerTimeEncoder(time_dim=time_dim)
        time_encoder = TimeEncoderWrapper(encoder)
        
    elif encoder_type == 'time2vec' and Time2VecEncoder is not None:
        print("INFO: Creating Time2Vec Time Encoder.")
        print("Time Embedding dim:", time_dim)
        encoder = Time2VecEncoder(time_dim=time_dim)
        time_encoder = TimeEncoderWrapper(encoder)
        
    elif encoder_type == 'sm_kernel_only':
        print("INFO: Creating SM-Kernel Only encoder (ablation study).")
        print("Time Embedding dim:", time_dim)
        
        # Get parameters
        if args is not None:
            num_mixtures = getattr(args, 'num_mixtures', kwargs.get('num_mixtures', 16))
        else:
            num_mixtures = kwargs.get('num_mixtures', 12)
            
        time_encoder = SMKernelOnly(
            embedding_dim=time_dim,
            num_mixtures=num_mixtures
        )
        
        # Initialize SM-Kernel if training data is available
        if train_data is not None and train_neighbor_sampler is not None:
            try:
                print("INFO: Initializing SM-Kernel for ablation study...")
                batch_size = getattr(args, 'batch_size', kwargs.get('batch_size', 200))
                num_neighbors = getattr(args, 'num_neighbors', kwargs.get('num_neighbors', 20))
                
                sample_batch_indices = np.arange(min(batch_size, len(train_data.src_node_ids)))
                sample_src = train_data.src_node_ids[sample_batch_indices]
                sample_ts = train_data.node_interact_times[sample_batch_indices]
                
                _, _, sample_neighbor_ts = train_neighbor_sampler.get_historical_neighbors(
                    sample_src, sample_ts, num_neighbors
                )
                
                sample_delta_t = sample_ts[:, np.newaxis] - sample_neighbor_ts
                
                if sample_delta_t.size > 0:
                    sample_delta_t_tensor = torch.from_numpy(sample_delta_t).float().unsqueeze(-1)
                    time_encoder.initialize_sm_kernel(sample_delta_t_tensor.to(device))
                    print("INFO: SM-Kernel initialization complete for ablation!")
                else:
                    print("WARNING: Could not generate SM-Kernel sample for ablation. Using defaults.")
            except Exception as e:
                print(f"WARNING: SM-Kernel initialization failed for ablation: {e}")
        else:
            print("INFO: No training data provided. SM-Kernel will use default initialization.")
    
    elif encoder_type == 'kmote_abs_only':
        print("INFO: Creating K-MOTE Absolute Only encoder (ablation study).")
        print("Time Embedding dim:", time_dim)
        
        # Get K-MOTE parameters
        if args is not None:
            wavelet_type = getattr(args, 'wavelet_type', kwargs.get('wavelet_type', 'shock'))
        else:
            wavelet_type = kwargs.get('wavelet_type', 'shock')
            
        time_encoder = KMOTEAbsOnly(
            embedding_dim=time_dim,
            wavelet_type=wavelet_type
        )
    
    elif encoder_type == 'kmote_rel_only':
        print("INFO: Creating K-MOTE Relative Only encoder (ablation study).")
        print("Time Embedding dim:", time_dim)
        
        # Get K-MOTE parameters
        if args is not None:
            wavelet_type = getattr(args, 'wavelet_type', kwargs.get('wavelet_type', 'shock'))
        else:
            wavelet_type = kwargs.get('wavelet_type', 'shock')
            
        time_encoder = KMOTERelOnly(
            embedding_dim=time_dim,
            wavelet_type=wavelet_type
        )
    
    elif encoder_type == 'dual_stream_baseline':
        print("INFO: Creating Dual Stream Baseline encoder (ablation study).")
        print("Time Embedding dim:", time_dim)
        
        # Get parameters
        if args is not None:
            num_mixtures = getattr(args, 'num_mixtures', kwargs.get('num_mixtures', 16))
            wavelet_type = getattr(args, 'wavelet_type', kwargs.get('wavelet_type', 'shock'))
        else:
            num_mixtures = kwargs.get('num_mixtures', 4)
            wavelet_type = kwargs.get('wavelet_type', 'shock')
            
        time_encoder = DualStreamBaseline(
            embedding_dim=time_dim,
            num_mixtures=num_mixtures,
            wavelet_type=wavelet_type
        )
        
        # Initialize SM-Kernel if training data is available
        if train_data is not None and train_neighbor_sampler is not None:
            try:
                print("INFO: Initializing SM-Kernel for dual stream baseline...")
                batch_size = getattr(args, 'batch_size', kwargs.get('batch_size', 200))
                num_neighbors = getattr(args, 'num_neighbors', kwargs.get('num_neighbors', 20))
                
                sample_batch_indices = np.arange(min(batch_size, len(train_data.src_node_ids)))
                sample_src = train_data.src_node_ids[sample_batch_indices]
                sample_ts = train_data.node_interact_times[sample_batch_indices]
                
                _, _, sample_neighbor_ts = train_neighbor_sampler.get_historical_neighbors(
                    sample_src, sample_ts, num_neighbors
                )
                
                sample_delta_t = sample_ts[:, np.newaxis] - sample_neighbor_ts
                
                if sample_delta_t.size > 0:
                    sample_delta_t_tensor = torch.from_numpy(sample_delta_t).float().unsqueeze(-1)
                    time_encoder.initialize_sm_kernel(sample_delta_t_tensor.to(device))
                    print("INFO: SM-Kernel initialization complete for dual stream baseline!")
                else:
                    print("WARNING: Could not generate SM-Kernel sample for dual stream baseline.")
            except Exception as e:
                print(f"WARNING: SM-Kernel initialization failed for dual stream baseline: {e}")
    
    elif encoder_type == 'k_mote':
        # ✅ Standalone K-MOTE (without Mamba) for MNIST-style experiments
        from .k_mote import KMOTE
        
        print("INFO: Creating standalone K-MOTE encoder.")
        print("Time Embedding dim:", time_dim)
        
        # Get K-MOTE parameters
        if args is not None:
            wavelet_type = getattr(args, 'wavelet_type', kwargs.get('wavelet_type', 'shock'))
            transform_mode = getattr(args, 'transform_mode', kwargs.get('transform_mode', 'adapter'))
            adapter_type = getattr(args, 'adapter_type', kwargs.get('adapter_type', 'affine'))
        else:
            wavelet_type = kwargs.get('wavelet_type', 'shock')
            transform_mode = kwargs.get('transform_mode', 'adapter')
            adapter_type = kwargs.get('adapter_type', 'affine')
        
        print(f"K-MOTE parameters:")
        print(f"  - output_dim: {time_dim}")
        print(f"  - wavelet_type: {wavelet_type}")
        print(f"  - transform_mode: {transform_mode}")
        print(f"  - adapter_type: {adapter_type if transform_mode == 'adapter' else 'N/A'}")
        
        time_encoder = KMOTE(
            input_dim=1,
            output_dim=time_dim,
            wavelet_type=wavelet_type,
            transform_mode=transform_mode,
            adapter_type=adapter_type if transform_mode == 'adapter' else None,
            use_scale=True,
            use_layernorm=True
        )
    
    elif encoder_type in ['original', 'time_encoder', 'default']:
        print("INFO: Creating original TimeEncoder (wrapped for compatibility).")
        print("Time Embedding dim:", time_dim)
        encoder = OriginalTimeEncoder(time_dim=time_dim)
        time_encoder = TimeEncoderWrapper(encoder)
        
    else:
        print(f"WARNING: Unknown encoder type '{encoder_type}' or encoder not available. Using default TimeEncoder.")
        print("Time Embedding dim:", time_dim)
        encoder = OriginalTimeEncoder(time_dim=time_dim)
        time_encoder = TimeEncoderWrapper(encoder)
        
    return time_encoder

def list_encoders():
    """
    Print information about available encoders.
    """
    print("Available Time Encoders:")
    print("=" * 50)
    
    for encoder_type in get_available_encoders():
        config = get_encoder_config(encoder_type)
        print(f"\n{encoder_type.upper()}:")
        print(f"  Description: {config['description']}")
        print(f"  Required params: {config['required_params']}")
        if config['optional_params']:
            print(f"  Optional params: {config['optional_params']}")

# Convenience aliases
create_encoder = create_time_encoder
get_encoders = get_available_encoders