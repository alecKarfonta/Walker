#!/usr/bin/env python3
"""
Utility functions for the Walker training system.
Handles data conversion between numpy/Box2D types and JSON-serializable types.
"""

import numpy as np


def safe_convert_numeric(value):
    """Convert numpy numeric types to JSON-serializable types without recursion."""
    if isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, (int, float, str, bool)) or value is None:
        return value
    else:
        # For other types, try to convert to float if possible, otherwise return as-is
        try:
            return float(value)
        except (ValueError, TypeError):
            return value


def safe_convert_list(lst):
    """Convert a list of potentially numpy values efficiently."""
    if not lst:
        return lst
    return [safe_convert_numeric(item) for item in lst]


def safe_convert_position(pos):
    """Convert a position tuple/list safely."""
    if hasattr(pos, 'x') and hasattr(pos, 'y'):
        # Box2D vector
        return (float(pos.x), float(pos.y))
    elif isinstance(pos, (tuple, list)) and len(pos) >= 2:
        return (safe_convert_numeric(pos[0]), safe_convert_numeric(pos[1]))
    return pos 