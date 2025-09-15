import torch
import numpy as np

# Import all encoder classes that will be used
from .kan_mammote import KAN_MAMMOTE
from ..gnn_backbones.modules import TimeEncoder as OriginalTimeEncoder

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

class TimeEncoderWrapper(torch.nn.Module):
    """
    Adapter to make various time encoders compatible with different interfaces.
    Ensures consistent output dimensions while preserving KAN-MAMMOTE functionality.
    """
    def __init__(self, encoder):
        super(TimeEncoderWrapper, self).__init__()
        self.encoder = encoder
        
    def forward(self, t_abs=None, t_rel=None, timestamps=None):
        # Check what interface the encoder supports
        result = None
        
        # Debug input shapes
        input_tensor = timestamps if timestamps is not None else t_abs
        if input_tensor is not None:
            #print(f"DEBUG TimeEncoderWrapper: Input shape: {input_tensor.shape}")
            pass
        
        if hasattr(self.encoder, 'forward'):
            # Try the dual-stream interface first (t_abs, t_rel)
            try:
                if t_abs is not None and t_rel is not None:
                    print(f"DEBUG: Using dual-stream interface (t_abs: {t_abs}, t_rel: {t_rel})")
                    result = self.encoder.forward(t_abs=t_abs, t_rel=t_rel)
                elif timestamps is not None:
                    # Check if encoder supports timestamps parameter
                    import inspect
                    sig = inspect.signature(self.encoder.forward)
                    if 'timestamps' in sig.parameters:
                        #print(f"DEBUG: Using timestamps parameter")
                        result = self.encoder.forward(timestamps=timestamps)
                    elif 't_abs' in sig.parameters:
                        # Convert timestamps to t_abs/t_rel format
                        #print(f"DEBUG: Converting timestamps to t_abs/t_rel")
                        t_rel_default = torch.zeros_like(timestamps)
                        result = self.encoder.forward(t_abs=timestamps, t_rel=t_rel_default)
                    else:
                        # Fallback: call with positional argument
                        #print(f"DEBUG: Using positional argument")
                        result = self.encoder.forward(timestamps)
                elif t_abs is not None:
                    # Try with just t_abs
                    print(f"DEBUG: Using t_abs only")
                    try:
                        result = self.encoder.forward(t_abs=t_abs)
                    except:
                        result = self.encoder.forward(timestamps=t_abs)
            except Exception as e:
                print(f"DEBUG: Interface attempt failed: {e}")
                # If interface doesn't match, try alternative
                if input_tensor is not None:
                    try:
                        print(f"DEBUG: Trying direct call")
                        result = self.encoder(input_tensor)
                    except Exception as e2:
                        print(f"DEBUG: Direct call failed: {e2}")
                        try:
                            print(f"DEBUG: Trying forward call")
                            result = self.encoder.forward(input_tensor)
                        except Exception as e3:
                            print(f"DEBUG: All attempts failed: {e3}")
                            raise e3
        
        # Last resort
        if result is None:
            print(f"DEBUG: Last resort call")
            result = self.encoder(input_tensor)
        
        # DIMENSION FIXING: Ensure consistent output dimensions
        if result is not None:
            #print(f"DEBUG: Raw output shape: {result.shape}")
            
            # Determine expected output shape based on input
            if t_abs is not None:
                batch_size = t_abs.shape[0]
                if t_abs.dim() == 3:  # (batch_size, num_neighbors, 1)
                    num_neighbors = t_abs.shape[1]
                    expected_neighbors = True
                else:  # (batch_size, 1) 
                    expected_neighbors = False
            else:
                expected_neighbors = False
            
            # Handle different dimension cases
            if result.dim() == 4:  # [batch, seq, 1, feat_dim] -> [batch, seq, feat_dim]
                result = result.squeeze(2)
                #print(f"DEBUG: After squeeze(2): {result.shape}")
            elif result.dim() == 3 and result.shape[1] == 1:  # [batch, 1, feat_dim] -> [batch, feat_dim]
                result = result.squeeze(1)
                #print(f"DEBUG: After squeeze(1): {result.shape}")
            elif result.dim() == 2:  # [batch, feat_dim]
                #print(f"DEBUG: 2D tensor: {result.shape}")
                # Check if we need to expand for neighbors
                if expected_neighbors and t_abs is not None:
                    num_neighbors = t_abs.shape[1]
                    result = result.unsqueeze(1).expand(-1, num_neighbors, -1)  # [batch, neighbors, feat_dim]
                    #print(f"DEBUG: Expanded for neighbors: {result.shape}")
            elif result.dim() == 1:  # [feat_dim] -> [1, feat_dim] or [1, neighbors, feat_dim]
                if expected_neighbors and t_abs is not None:
                    num_neighbors = t_abs.shape[1]
                    result = result.unsqueeze(0).unsqueeze(0).expand(1, num_neighbors, -1)
                    #print(f"DEBUG: Expanded 1D for neighbors: {result.shape}")
                else:
                    result = result.unsqueeze(0)
                    #print(f"DEBUG: After unsqueeze(0): {result.shape}")
            
            # Final check: ensure we have proper dimensions for TGAT
            if result.dim() == 3 and result.shape[1] > 1 and not expected_neighbors:
                # If we have sequences but input was 2D, take the mean or last timestep
                #print(f"DEBUG: Converting sequence output to single timestep")
                result = result.mean(dim=1)  # or result[:, -1, :] for last timestep
                #print(f"DEBUG: After mean(dim=1): {result.shape}")
            
            #print(f"DEBUG: Final output shape: {result.shape}")
        
        return result

def get_available_encoders():
    """
    Return a list of available time encoder types.
    """
    encoders = [
        'kan_mammote',
        'original',
        'time_encoder',
        'default'
    ]
    
    # Add available optional encoders
    if MercerTimeEncoder is not None:
        encoders.append('mercer')
    if LeTE is not None:
        encoders.append('lete')
    if BochnerTimeEncoder is not None:
        encoders.append('bochner')
    if Time2VecEncoder is not None:
        encoders.append('time2vec')
    
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
                'mamba_headdim': 64
            },
            'description': 'KAN-MAMMOTE: Advanced time encoder with Mamba2 and spectral kernels'
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
    encoder_type = encoder_type.lower()
    
    # Show available encoders for debugging
    available = get_available_encoders()
    print(f"Available encoders: {available}")
    print(f"Requested encoder: {encoder_type}")
    
    if encoder_type == 'kan_mammote':
        print("INFO: Creating KAN-MAMMOTE time encoder.")
        
        # Handle args gracefully - try args first, then kwargs, then defaults
        if args is not None:
            expert_dim = getattr(args, 'expert_dim', kwargs.get('expert_dim', 64))
            num_mixtures = getattr(args, 'num_mixtures', kwargs.get('num_mixtures', 4))
            mamba_d_state = getattr(args, 'mamba_d_state', kwargs.get('mamba_d_state', 16))
            mamba_d_conv = getattr(args, 'mamba_d_conv', kwargs.get('mamba_d_conv', 4))
            mamba_expand = getattr(args, 'mamba_expand', kwargs.get('mamba_expand', 2))
            mamba_headdim = getattr(args, 'mamba_headdim', kwargs.get('mamba_headdim', 64))
            batch_size = getattr(args, 'batch_size', kwargs.get('batch_size', 200))
            num_neighbors = getattr(args, 'num_neighbors', kwargs.get('num_neighbors', 20))
        else:
            # Get from kwargs or use defaults
            expert_dim = kwargs.get('expert_dim', 64)
            num_mixtures = kwargs.get('num_mixtures', 4)
            mamba_d_state = kwargs.get('mamba_d_state', 16)
            mamba_d_conv = kwargs.get('mamba_d_conv', 4)
            mamba_expand = kwargs.get('mamba_expand', 2)
            mamba_headdim = kwargs.get('mamba_headdim', 64)
            batch_size = kwargs.get('batch_size', 200)
            num_neighbors = kwargs.get('num_neighbors', 20)
        
        print(f"KAN-MAMMOTE parameters:")
        print(f"  - embedding_dim: {time_dim}")
        print(f"  - expert_dim: {expert_dim}")
        print(f"  - num_mixtures: {num_mixtures}")
        print(f"  - mamba_d_state: {mamba_d_state}")
        print(f"  - mamba_d_conv: {mamba_d_conv}")
        print(f"  - mamba_expand: {mamba_expand}")
        
        time_encoder = KAN_MAMMOTE(
            embedding_dim=time_dim,
            expert_dim=expert_dim,
            num_mixtures=num_mixtures,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand
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
        
    elif encoder_type == 'mercer' and MercerTimeEncoder is not None:
        print("INFO: Creating Mercer Time Encoder.")
        encoder = MercerTimeEncoder(time_dim=time_dim)
        time_encoder = TimeEncoderWrapper(encoder)
        
    elif encoder_type == 'lete' and LeTE is not None:
        print("INFO: Creating LeTE Time Encoder.")
        encoder = LeTE(time_dim=time_dim)
        time_encoder = TimeEncoderWrapper(encoder)
        
    elif encoder_type == 'bochner' and BochnerTimeEncoder is not None:
        print("INFO: Creating Bochner Time Encoder.")
        encoder = BochnerTimeEncoder(time_dim=time_dim)
        time_encoder = TimeEncoderWrapper(encoder)
        
    elif encoder_type == 'time2vec' and Time2VecEncoder is not None:
        print("INFO: Creating Time2Vec Time Encoder.")
        encoder = Time2VecEncoder(time_dim=time_dim)
        time_encoder = TimeEncoderWrapper(encoder)
        
    elif encoder_type in ['original', 'time_encoder', 'default']:
        print("INFO: Creating original TimeEncoder (wrapped for compatibility).")
        encoder = OriginalTimeEncoder(time_dim=time_dim)
        time_encoder = TimeEncoderWrapper(encoder)
        
    else:
        print(f"WARNING: Unknown encoder type '{encoder_type}' or encoder not available. Using default TimeEncoder.")
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