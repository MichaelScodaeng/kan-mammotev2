"""
Time Encoders Module

This module provides various time encoding methods for temporal graph learning.

Available Encoders:
- KAN_MAMMOTE: Full version with Mamba2 for sequence modeling
- KAN_MAMMOTE_Lite: Lightweight stateless version for attention models
- LeTE: Learnable Time Encoder
- MercerTimeEncoder: Mercer kernel-based encoding
- BochnerTimeEncoder: Bochner random Fourier features
- Time2VecEncoder: Time2Vec encoding
- OriginalTimeEncoder: Standard cosine-based encoding

Factory Functions:
- create_time_encoder: Main factory function
- get_available_encoders: List all available encoders
"""

from .kan_mammote import KAN_MAMMOTE
from .kan_mammote_lite import KAN_MAMMOTE_Lite
from .factory import (
    create_time_encoder, 
    get_available_encoders, 
    get_encoder_config,
    TimeEncoderWrapper,
    list_encoders,
    create_encoder,  # Alias
    get_encoders     # Alias
)

# Optional encoders (import only if available)
try:
    from .time2vec_encoder import Time2VecEncoder
    __all__ = ['KAN_MAMMOTE', 'KAN_MAMMOTE_Lite', 'Time2VecEncoder', 'create_time_encoder', 
               'get_available_encoders', 'get_encoder_config', 'TimeEncoderWrapper', 'list_encoders']
except ImportError:
    __all__ = ['KAN_MAMMOTE', 'KAN_MAMMOTE_Lite', 'create_time_encoder', 'get_available_encoders', 
               'get_encoder_config', 'TimeEncoderWrapper', 'list_encoders']

try:
    from .lete_encoder import LeTE
    __all__.append('LeTE')
except ImportError:
    pass

try:
    from .bochner_encoder import BochnerTimeEncoder
    __all__.append('BochnerTimeEncoder')
except ImportError:
    pass

try:
    from .mercer_encoder import MercerTimeEncoder
    __all__.append('MercerTimeEncoder')
except ImportError:
    pass

try:
    from .original_encoder import OriginalTimeEncoder
    __all__.append('OriginalTimeEncoder')
except ImportError:
    pass

# Version info
__version__ = "1.0.0"