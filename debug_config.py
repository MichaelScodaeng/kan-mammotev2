"""
Simple debug configuration for KAN-MAMMOTE project.
This module provides a centralized debug control that can be easily imported from anywhere.
"""

# 🔍 Global Debug Controls
DEBUG_MODEL = True  # Master switch for all model debugging output
DEBUG_TIME_SORTING = True      # Neighbor time sorting verification
DEBUG_TIME_COMPUTATION = True  # Time computation pattern verification
DEBUG_ENCODER_INTERFACE = True # KAN_MAMMOTE encoder interface debugging

def should_debug_model():
    """Check if model debugging is enabled."""
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
    print("🔍 All debugging enabled!")

def disable_all_debug():
    """Disable all debugging."""
    global DEBUG_MODEL, DEBUG_TIME_SORTING, DEBUG_TIME_COMPUTATION, DEBUG_ENCODER_INTERFACE
    DEBUG_MODEL = False
    DEBUG_TIME_SORTING = False
    DEBUG_TIME_COMPUTATION = False
    DEBUG_ENCODER_INTERFACE = False
    print("🔍 All debugging disabled!")

def set_debug_mode(enabled: bool):
    """Set main debug mode on/off."""
    global DEBUG_MODEL
    DEBUG_MODEL = enabled
    print(f"🔍 Debug mode set to: {enabled}")