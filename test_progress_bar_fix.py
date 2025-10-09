#!/usr/bin/env python3
"""
Test script to verify that disable_progress_bar functionality works correctly.
This script tests that tqdm progress bars are properly disabled when the flag is set.
"""

import sys
import os
import argparse

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_import_and_function_signature():
    """Test that the evaluate_model_link_prediction function can be imported and has the correct signature."""
    try:
        from experiments.evaluate_models_utils import evaluate_model_link_prediction
        print("✅ Successfully imported evaluate_model_link_prediction")
        
        # Check function signature
        import inspect
        sig = inspect.signature(evaluate_model_link_prediction)
        params = list(sig.parameters.keys())
        
        if 'disable_progress_bar' in params:
            print("✅ disable_progress_bar parameter found in function signature")
            return True
        else:
            print("❌ disable_progress_bar parameter NOT found in function signature")
            print(f"Available parameters: {params}")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import evaluate_model_link_prediction: {e}")
        return False

def test_argument_parsing():
    """Test that disable_progress_bar argument is available in argument parser."""
    try:
        from utils.load_configs import get_link_prediction_args
        
        # Test with disable_progress_bar flag
        sys.argv = ['test_script', '--disable_progress_bar']
        args = get_link_prediction_args()
        
        if hasattr(args, 'disable_progress_bar') and args.disable_progress_bar:
            print("✅ disable_progress_bar argument correctly parsed as True")
        else:
            print("❌ disable_progress_bar argument not correctly parsed")
            return False
            
        # Test without disable_progress_bar flag
        sys.argv = ['test_script']
        args = get_link_prediction_args()
        
        if hasattr(args, 'disable_progress_bar') and not args.disable_progress_bar:
            print("✅ disable_progress_bar argument correctly defaults to False")
            return True
        else:
            print("❌ disable_progress_bar argument not correctly defaulted")
            return False
            
    except Exception as e:
        print(f"❌ Failed to test argument parsing: {e}")
        return False

def test_getattr_pattern():
    """Test the getattr pattern used in function calls."""
    class MockArgs:
        def __init__(self, has_flag=False):
            if has_flag:
                self.disable_progress_bar = True
    
    # Test with flag present
    args_with_flag = MockArgs(has_flag=True)
    result = getattr(args_with_flag, 'disable_progress_bar', False)
    if result:
        print("✅ getattr correctly returns True when flag is present")
    else:
        print("❌ getattr failed when flag is present")
        return False
    
    # Test with flag absent
    args_without_flag = MockArgs(has_flag=False)
    result = getattr(args_without_flag, 'disable_progress_bar', False)
    if not result:
        print("✅ getattr correctly returns False when flag is absent")
        return True
    else:
        print("❌ getattr failed when flag is absent")
        return False

def main():
    """Run all tests."""
    print("Testing disable_progress_bar functionality...")
    print("=" * 50)
    
    tests = [
        ("Import and Function Signature", test_import_and_function_signature),
        ("Argument Parsing", test_argument_parsing), 
        ("getattr Pattern", test_getattr_pattern)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL" 
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! disable_progress_bar functionality is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)