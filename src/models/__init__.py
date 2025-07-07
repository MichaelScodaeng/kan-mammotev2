# KAN-MAMMOTE Models
from .immediate_fasterkan_layer import ImprovedKANMAMOTE, ImmediateFasterKANLayer
from .kan_mammote import KAN_MAMOTE_Model as LegacyKANMAMOTE
from .k_mote import K_MOTE
from .c_mamba import ContinuousMambaBlock, SimplifiedContinuousMambaBlock
from .moe_router import MoERouter

# Set Improved KAN-MAMMOTE as the default
KAN_MAMMOTE_Model = ImprovedKANMAMOTE

# Add deprecation warning for legacy model
import warnings

def get_legacy_model():
    warnings.warn(
        "LegacyKANMAMOTE (original KAN_MAMOTE_Model) is deprecated. "
        "Use ImprovedKANMAMOTE for better performance and features.",
        DeprecationWarning,
        stacklevel=2
    )
    return LegacyKANMAMOTE

__all__ = [
    'KAN_MAMMOTE_Model',  # Default: Improved version
    'ImprovedKANMAMOTE',  # Explicit improved version
    'LegacyKANMAMOTE',    # Legacy version
    'ImmediateFasterKANLayer',
    'K_MOTE',
    'ContinuousMambaBlock',
    'SimplifiedContinuousMambaBlock',
    'MoERouter',
    'get_legacy_model'
]