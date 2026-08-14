#!/usr/bin/env python3

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from synthetic_data.lifetime_generator import _run_validation_tests
    print("Testing lifetime generator...")
    _run_validation_tests()
    print("Lifetime generator tests completed successfully!")
except Exception as e:
    print(f"Error testing lifetime generator: {e}")
    import traceback
    traceback.print_exc()