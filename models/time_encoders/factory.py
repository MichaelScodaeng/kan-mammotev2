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



# Import optional encoders that may not be available
SMKernelOnly = None
KMOTEAbsOnly = None
KMOTERelOnly = None
DualStreamBaseline = None




class TimeEncoderWrapper(torch.nn.Module):
    """
    Adapter to make various time encoders compatible with different interfaces.
    Ensures consistent output dimensions while preserving KMM functionality.
    
    Key Insight:
    - KMM uses: (t_abs, t_rel) - dual stream
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
            
        # Strategy 1: Dual-stream interface (KMM)
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
   
    
    # Add ablation study encoders (always available since they're local)
    encoders.extend([
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
            'description': 'KMM Dual K-MOTE: Uses K-MOTE for both absolute and relative time encoding'
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
            'description': 'KMM Dual K-MOTE: Uses K-MOTE for both absolute and relative time encoding for TGAT'
        },
        'lete': {
            'required_params': ['time_dim'],
            'optional_params': {},
            'description': 'LeTE: Learnable Time Encoder'
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
    
    
    if encoder_type == 'KMM' or encoder_type == 'kmm':
        if KMM is None:
            raise ImportError(f"KMM encoder is not available. Please install Mamba dependencies or use a different encoder.")
        
        print("INFO: Creating KMM time encoder with dual K-MOTE (no SM-kernel).")
        
        # Handle args gracefully - try args first, then kwargs, then defaults
        if args is not None:
            print("INFO: Extracting KMM Dual K-MOTE parameters from args.")
            expert_dim = getattr(args, 'expert_dim', kwargs.get('expert_dim', 128))
            num_mixtures = getattr(args, 'num_mixtures', kwargs.get('num_mixtures', 16))
            mamba_d_state = getattr(args, 'mamba_d_state', kwargs.get('mamba_d_state', 256))
            mamba_d_conv = getattr(args, 'mamba_d_conv', kwargs.get('mamba_d_conv', 4))
            mamba_expand = getattr(args, 'mamba_expand', kwargs.get('mamba_expand', 2))
            mamba_headdim = getattr(args, 'mamba_headdim', kwargs.get('mamba_headdim', 64))
            encoder_dropout = getattr(args, 'encoder_dropout', getattr(args, 'dropout', 0.1))  # Use encoder_dropout, fallback to dropout
        else:
            # Get from kwargs or use defaults
            print("INFO: Extracting KMM Dual K-MOTE parameters from kwargs or using defaults.")
            expert_dim = kwargs.get('expert_dim', 64)
            num_mixtures = kwargs.get('num_mixtures', 4)
            mamba_d_state = kwargs.get('mamba_d_state', 16)
            mamba_d_conv = kwargs.get('mamba_d_conv', 4)
            mamba_expand = kwargs.get('mamba_expand', 2)
            mamba_headdim = kwargs.get('mamba_headdim', 64)
            encoder_dropout = kwargs.get('encoder_dropout', kwargs.get('dropout', 0.1))
        
        print(f"KMM Dual K-MOTE parameters:")
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
    elif encoder_type == 'lete' and LeTE is not None:
        print("INFO: Creating LeTE Time Encoder.")
        print("Time Embedding dim:", time_dim)
        encoder = LeTE(time_dim=time_dim)
        time_encoder = TimeEncoderWrapper(encoder)
        
    
        
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