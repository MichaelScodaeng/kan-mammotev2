# KAN-MAMMOTE Models
from .immediate_fasterkan_layer import ImprovedKANMAMOTE, ImmediateFasterKANLayer
from .k_mote import K_MOTE
from .moe_router import MoERouter
from .kan_mammote import KANMAMMOTE



__all__ = [
    'KANMAMMOTE',  # Default: Improved version
    'ImprovedKANMAMOTE',  # Explicit improved version
    'ImmediateFasterKANLayer',
    'K_MOTE',
    'MoERouter',
]