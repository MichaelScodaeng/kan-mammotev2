"""
Time Encoders Module

This module provides various time encoding methods for temporal graph neural networks.
"""

from .kan_mammote import KAN_MAMMOTE
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
    __all__ = ['KAN_MAMMOTE', 'Time2VecEncoder', 'create_time_encoder', 'get_available_encoders', 
               'get_encoder_config', 'TimeEncoderWrapper', 'list_encoders']
except ImportError:
    __all__ = ['KAN_MAMMOTE', 'create_time_encoder', 'get_available_encoders', 
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

# Version info
__version__ = "1.0.0"