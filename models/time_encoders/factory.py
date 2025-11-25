import torch
import numpy as np
import sys
import inspect


DEBUG_TIME_ENCODER_FACTORY = False
DEBUG_TIME_ENCODER_CALLS = False
DEBUG_TIME_ENCODER_VALUES = False
DEBUG_TIME_ENCODER_SHAPES = False

DEBUG_MODEL = False
DEBUG_TIME_SORTING = False
DEBUG_TIME_COMPUTATION = False
DEBUG_ENCODER_INTERFACE = False

def enable_factory_debug(enable_calls=True, enable_values=True, enable_shapes=True):
    global DEBUG_TIME_ENCODER_FACTORY, DEBUG_TIME_ENCODER_CALLS, DEBUG_TIME_ENCODER_VALUES, DEBUG_TIME_ENCODER_SHAPES
    DEBUG_TIME_ENCODER_FACTORY = True
    DEBUG_TIME_ENCODER_CALLS = enable_calls
    DEBUG_TIME_ENCODER_VALUES = enable_values
    DEBUG_TIME_ENCODER_SHAPES = enable_shapes

def disable_factory_debug():
    global DEBUG_TIME_ENCODER_FACTORY, DEBUG_TIME_ENCODER_CALLS, DEBUG_TIME_ENCODER_VALUES, DEBUG_TIME_ENCODER_SHAPES
    DEBUG_TIME_ENCODER_FACTORY = False
    DEBUG_TIME_ENCODER_CALLS = False
    DEBUG_TIME_ENCODER_VALUES = False
    DEBUG_TIME_ENCODER_SHAPES = False


def should_debug_model():
    return DEBUG_MODEL

def should_debug_time_sorting():
    return DEBUG_MODEL and DEBUG_TIME_SORTING

def should_debug_time_computation():
    return DEBUG_MODEL and DEBUG_TIME_COMPUTATION

def should_debug_encoder_interface():
    return DEBUG_MODEL and DEBUG_ENCODER_INTERFACE

def enable_all_debug():
    global DEBUG_MODEL, DEBUG_TIME_SORTING, DEBUG_TIME_COMPUTATION, DEBUG_ENCODER_INTERFACE
    DEBUG_MODEL = True
    DEBUG_TIME_SORTING = True
    DEBUG_TIME_COMPUTATION = True
    DEBUG_ENCODER_INTERFACE = True

def disable_all_debug():
    global DEBUG_MODEL, DEBUG_TIME_SORTING, DEBUG_TIME_COMPUTATION, DEBUG_ENCODER_INTERFACE
    DEBUG_MODEL = False
    DEBUG_TIME_SORTING = False
    DEBUG_TIME_COMPUTATION = False
    DEBUG_ENCODER_INTERFACE = False

def set_debug_mode(enabled: bool):
    global DEBUG_MODEL
    DEBUG_MODEL = enabled

def debug_print(message, debug_type="general"):
    if DEBUG_TIME_ENCODER_FACTORY:
        if debug_type == "calls" and DEBUG_TIME_ENCODER_CALLS:
            print(f"[CALLS] {message}")
        if debug_type == "values" and DEBUG_TIME_ENCODER_VALUES:
            print(f"[VALUES] {message}")
        if debug_type == "shapes" and DEBUG_TIME_ENCODER_SHAPES:
            print(f"[SHAPES] {message}")
        if debug_type == "general":
            print(f"[DEBUG] {message}")



# Import standard encoder
from ..gnn_backbones.modules import TimeEncoder as OriginalTimeEncoder

# Import Mamba-based encoders with optional fallback
try:
    from .kmm import KMM
except ImportError as e:
    KMM = None

# Import additional encoders
try:
    from .lete_encoder import LeTE
except ImportError:
    LeTE = None

try:
    from .time2vec_encoder import Time2VecEncoder
except ImportError:
    Time2VecEncoder = None

# Import optional encoders that may not be available
SMKernelOnly = None
KMOTEAbsOnly = None
KMOTERelOnly = None
DualStreamBaseline = None




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
        self._debug_mode = True
        if hasattr(self.encoder, 'enable_debug_mode'):
            self.encoder.enable_debug_mode()
    
    def disable_debug_mode(self):
        self._debug_mode = False
        if hasattr(self.encoder, 'disable_debug_mode'):
            self.encoder.disable_debug_mode()
        
    def forward(self, t_abs=None, t_rel=None, timestamps=None):
        self._call_count += 1
        
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
            print(f"TimeEncoderWrapper Call #{self._call_count}")
            print(f"   Encoder: {self.encoder_name}")
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
                
                if self._debug_mode:
                    print(f"   Using dual-stream interface: t_abs + t_rel")
                result = self.encoder.forward(t_abs=t_abs, t_rel=t_rel)
            elif timestamps is not None:
                t_rel_default = timestamps
                t_abs_default = torch.zeros_like(timestamps)
                
                if DEBUG_TIME_ENCODER_FACTORY:
                    debug_print("Using dual-stream interface: dummy t_abs + timestamps as t_rel", "calls")
                    debug_print(f"t_abs_default (zeros): {t_abs_default.shape}", "values")
                    debug_print(f"t_rel_default (timestamps): {t_rel_default.shape} | range: [{t_rel_default.min().item():.3f}, {t_rel_default.max().item():.3f}]", "values")
                    
                if self._debug_mode:
                    print(f"   Using dual-stream interface: dummy t_abs + timestamps as t_rel")
                result = self.encoder.forward(t_abs=t_abs_default, t_rel=t_rel_default)
        
        # Strategy 2: OriginalTimeEncoder uses relative time (delta_t)
        elif self.encoder_name == 'TimeEncoder' or 'timestamps' in params:
            if t_rel is not None:
                if self._debug_mode:
                    print(f"   Using single-stream interface: t_rel")
                result = self.encoder.forward(t_rel)
            elif timestamps is not None:
                if self._debug_mode:
                    print(f"   Using single-stream interface: timestamps")
                result = self.encoder.forward(timestamps)
            elif t_abs is not None:
                if self._debug_mode:
                    print(f"   Fallback: Using t_abs as timestamps (unexpected)")
                print(f"WARNING: OriginalTimeEncoder expects t_rel but got t_abs. Using t_abs as fallback.")
                result = self.encoder.forward(t_abs)
        
        # Strategy 3: Generic positional argument
        elif result is None:
            if t_rel is not None:
                input_tensor = t_rel
            elif timestamps is not None:
                input_tensor = timestamps
            elif t_abs is not None:
                input_tensor = t_abs
            else:
                raise ValueError("Must provide at least one time input")
            
            if self._debug_mode:
                print(f"   Using generic interface: {input_tensor.shape}")
            result = self.encoder.forward(input_tensor)
        
        if self._debug_mode and result is not None:
            print(f"   Output shape: {result.shape}")
        
        # Handle dimension compatibility
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
    if KMM is not None:
        encoders.extend([
            'KMM',
            'KMM'  # NEW: Dual K-MOTE variant
            ,'dual_stream_baseline',
            "KMM_tgat"
        ])
    
    # Add available optional encoders
    if LeTE is not None:
        encoders.append('lete')
    if Time2VecEncoder is not None:
        encoders.append('time2vec')
    
    # Add ablation study encoders (always available since they're local)
    encoders.extend([
        'sm_kernel_only',
        'kmote_abs_only',
        'kmote_rel_only',
        'k_mote',  # Standalone K-MOTE (without Mamba, for MNIST experiments)
    ])
    
    return encoders

def get_encoder_config(encoder_type: str):
    """
    Return the configuration parameters for a specific encoder type.
    """
    encoder_type = encoder_type.lower()
    
    configs = {
        'KMM': {
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
        'KMM_tgat': {
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
        'lete': {
            'required_params': ['time_dim'],
            'optional_params': {},
            'description': 'LeTE: Learnable Time Encoder'
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
    """Factory function to create and initialize the correct time encoder."""
    
    encoder_type = encoder_type.lower()
    
    if DEBUG_TIME_ENCODER_FACTORY:
        debug_print(f"Creating time encoder: {encoder_type} with time_dim={time_dim}", "general")
        debug_print(f"Device: {device}", "general")
    
    available = get_available_encoders()
    if DEBUG_TIME_ENCODER_FACTORY:
        debug_print(f"Available encoders: {available}", "general")
    
    if encoder_type == 'KMM':
        if KMM is None:
            raise ImportError(f"KMM encoder is not available. Please install Mamba dependencies or use a different encoder.")
        
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
        
        time_encoder = KMM(
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
    
    elif encoder_type == 'KMM':
        if KMM is None:
            raise ImportError(f"KMM encoder is not available. Please install Mamba dependencies or use a different encoder.")
        
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
        
        time_encoder = KMM(
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
    elif encoder_type == 'KMM_tgat':
        if KMM is None:
            raise ImportError(f"KMM encoder is not available. Please install Mamba dependencies or use a different encoder.")
        
        print("INFO: Creating KMM_tgat")
        
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
        
        time_encoder = KMM(
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
    
    elif encoder_type == 'lete' and LeTE is not None:
        print("INFO: Creating LeTE Time Encoder.")
        print("Time Embedding dim:", time_dim)
        encoder = LeTE(time_dim=time_dim)
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
        # Standalone K-MOTE (without Mamba) for MNIST-style experiments
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