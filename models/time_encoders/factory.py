"""
Time Encoder Factory

Factory function to create different types of time encoders for experiments.
"""

from typing import Dict, Any


def create_time_encoder(encoder_type: str, time_dim: int, device: str = 'cpu', **kwargs):
    """
    Factory function to create time encoders.
    
    Args:
        encoder_type: Type of encoder ('original', 'lete', 'kan_mammote')
        time_dim: Output dimension of time encoding
        device: Device to place the encoder on
        **kwargs: Additional arguments for specific encoders
    
    Returns:
        Time encoder instance
    
    Raises:
        ValueError: If encoder_type is not recognized
    """
    encoder_type = encoder_type.lower()
    
    if encoder_type == 'original':
        from .original_encoder import OriginalTimeEncoder
        return OriginalTimeEncoder(
            time_dim=time_dim,
            device=device,
            **kwargs
        )
    
    elif encoder_type == 'lete':
        from .lete_baseline import CombinedLeTE
        return CombinedLeTE(
            dim=time_dim,
            **kwargs
        )
    
    elif encoder_type == 'kan_mammote':
        from .kan_mammote import KAN_MAMMOTE
        return KAN_MAMMOTE(
            embedding_dim=time_dim,
            **kwargs
        )
    
    elif encoder_type == 'bochner':
        from .bochner_encoder import BochnerTimeEncoder
        return BochnerTimeEncoder(
            time_dim=time_dim,
            device=device,
            **kwargs
        )
    
    elif encoder_type == 'mercer':
        from .mercer_encoder import MercerTimeEncoder
        return MercerTimeEncoder(
            time_dim=time_dim,
            device=device,
            **kwargs
        )
    
    elif encoder_type == 'time2vec':
        from .time2vec_encoder import Time2VecEncoder
        return Time2VecEncoder(
            time_dim=time_dim,
            device=device,
            **kwargs
        )
    elif encoder_type == 'kan_mammote':
        from .kan_mammote import KAN_MAMMOTE
        # This will correctly catch 'expert_dim' and 'num_mixtures' from kwargs
        return KAN_MAMMOTE(
            embedding_dim=time_dim,
            device=device,
            **kwargs 
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}. "
                        f"Available types: 'original', 'lete', 'kan_mammote', 'bochner', 'mercer', 'time2vec'")


def get_available_encoders() -> Dict[str, str]:
    """
    Get dictionary of available encoder types and descriptions.
    
    Returns:
        Dictionary mapping encoder names to descriptions
    """
    return {
        'original': 'Traditional cosine-based time encoding',
        'lete': 'Learnable Time Encoding with Fourier and spline components',
        'kan_mammote': 'KAN-MAMMOTE dual-stream encoding with K-MOTE and SM-Kernel',
        'bochner': 'Bochner theorem-based encoding with random Fourier features',
        'mercer': 'Mercer theorem-based encoding with eigenfunction expansion',
        'time2vec': 'Time2Vec: Learning a Vector Representation of Time'
    }


def get_encoder_config(encoder_type: str) -> Dict[str, Any]:
    """
    Get default configuration for a specific encoder type.
    
    Args:
        encoder_type: Type of encoder
        
    Returns:
        Dictionary of default configuration parameters
    """
    encoder_type = encoder_type.lower()
    
    configs = {
        'original': {
            'parameter_requires_grad': True
        },
        'lete': {
            'p': 0.5,
            'layer_norm': True,
            'scale': True,
            'parameter_requires_grad': True
        },
        'kan_mammote': {
            'mamba_d_state': 16,
            'mamba_d_conv': 4,
            'mamba_expand': 2
        },
        'bochner': {
            'sigma': 1.0
        },
        'mercer': {
            'expand_dim': 8,
            'time_range': 10.0
        },
        'time2vec': {
            'activation': 'sin'
        }
    }
    
    return configs.get(encoder_type, {})
