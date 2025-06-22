#!/usr/bin/env python3
"""
Simple test script for the Walker physics system.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from main import main
    main()
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install the required dependencies:")
    print("pip install -r requirements.txt")
except Exception as e:
    print(f"Error running simulation: {e}") 