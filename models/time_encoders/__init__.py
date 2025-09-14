"""
Time Encoders for KAN-MAMMOTE Framework

This module provides various time encoding strategies for continuous-time dynamic graphs:
- OriginalTimeEncoder: Traditional cosine-based time encoding
- LeTE: Learnable Time Encoding with Fourier and spline components  
- KANMammote: Novel dual-stream encoding with K-MOTE and SM-Kernel
"""

from .factory import create_time_encoder, get_available_encoders, get_encoder_config

# Import your existing implementations
try:
    from .original_encoder import OriginalTimeEncoder
except ImportError:
    OriginalTimeEncoder = None

try:
    from .lete_baseline import CombinedLeTE
except ImportError:
    CombinedLeTE = None

try:
    from .kan_mammote import KAN_MAMMOTE
except ImportError:
    KAN_MAMMOTE = None

__all__ = [
    'create_time_encoder',
    'get_available_encoders', 
    'get_encoder_config',
    'OriginalTimeEncoder',
    'CombinedLeTE',
    'KAN_MAMMOTE'
]
