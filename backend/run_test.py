#!/usr/bin/env python
"""
Run the comprehensive test directly with Python
"""
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

# Import and run the comprehensive test
from comprehensive_test import comprehensive_test

if __name__ == '__main__':
    print("Running Comprehensive CP-VTON Test...")
    print("=" * 50)
    
    success = comprehensive_test()
    
    if success:
        print("\n[SUCCESS] All tests completed!")
        print("\nGenerated files:")
        print("  Inputs:")
        print("    - comp_person.jpg (person image)")
        print("    - comp_cloth.jpg (cloth image)")
        print("  Intermediate:")
        print("    - comp_pose.jpg (pose keypoints)")
        print("    - comp_parse.jpg (parsing mask)")
        print("    - comp_agnostic.jpg (agnostic representation)")
        print("    - comp_warped.jpg (warped cloth)")
        print("    - comp_rendered.jpg (TOM rendered)")
        print("    - comp_mask.jpg (composite mask)")
        print("  Final:")
        print("    - comp_result.jpg (final try-on result)")
    else:
        print("\n[FAILED] Tests failed!")
    
    print("\nTest completed! Check the generated images.")
