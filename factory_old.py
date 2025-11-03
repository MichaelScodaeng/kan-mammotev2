"""
Global configuration factory for KAN-MAMMOTE experiments.

This module provides centralized control over debugging and other global settings.
"""

# 🔍 Global Debug Controls
DEBUG_MODEL = True  # Set to False to disable all model debugging output

# Additional debug flags for fine-grained control
DEBUG_TIME_SORTING = True      # Neighbor time sorting verification
DEBUG_TIME_COMPUTATION = True  # Time computation pattern verification
DEBUG_ENCODER_INTERFACE = True # KAN_MAMMOTE encoder interface debugging

def should_debug_model():
    """Check if model debugging is enabled."""
    print(f"🔍 [FACTORY] should_debug_model() called, returning: {DEBUG_MODEL}")
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