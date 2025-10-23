#!/usr/bin/env python3
"""
Test script for Optuna integration with KAN-MAMMOTE.
Runs a small validation test to ensure everything works correctly.
"""

import sys
import os
import tempfile
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tune_kan_mammote_optuna import run_single_dataset_tuning

def test_optuna_integration():
    """
    Run a minimal test to validate the Optuna integration works.
    """
    print("🧪 Testing Optuna integration...")
    
    # Test parameters - very small for quick validation
    test_params = {
        'dataset': 'wikipedia',      # Small, fast dataset
        'model': 'TGAT',             # Simple model
        'n_trials': 3,               # Just a few trials
        'num_epochs': 2,             # Very short training
        'study_name': f'test_optuna_{datetime.now().strftime("%H%M%S")}',
        'storage': None,             # Will use default SQLite
        'resume': False              # Fresh start
    }
    
    print(f"📋 Test configuration:")
    for key, value in test_params.items():
        print(f"  ├─ {key}: {value}")
    
    try:
        # Run the test
        print(f"\n🚀 Starting test run...")
        study = run_single_dataset_tuning(**test_params)
        
        # Validate results
        if study and len(study.trials) > 0:
            print(f"✅ Test PASSED!")
            print(f"  ├─ Completed trials: {len([t for t in study.trials if t.state.name == 'COMPLETE'])}")
            print(f"  ├─ Pruned trials: {len([t for t in study.trials if t.state.name == 'PRUNED'])}")
            print(f"  └─ Failed trials: {len([t for t in study.trials if t.state.name == 'FAIL'])}")
            
            if study.best_trial:
                print(f"\n🏆 Best trial (trial {study.best_trial.number}):")
                print(f"  ├─ Validation AP: {study.best_trial.value:.4f}")
                print(f"  └─ Parameters: {study.best_trial.params}")
            
            return True
        else:
            print(f"❌ Test FAILED: No trials completed")
            return False
            
    except Exception as e:
        print(f"❌ Test FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_constraint_validation():
    """
    Test the Mamba2 architecture constraint validation.
    """
    print(f"\n🔧 Testing constraint validation...")
    
    # Import the constraint function
    from tune_kan_mammote_optuna import is_valid_mamba_config
    
    # Test cases: (expert_dim, mamba_expand, mamba_headdim) -> expected_result
    test_cases = [
        (64, 2, 64, True),    # 64*2=128, 128%64=0, 128//64=2, 2%8!=0 -> False
        (128, 2, 64, True),   # 128*2=256, 256%64=0, 256//64=4, 4%8!=0 -> False  
        (256, 2, 64, True),   # 256*2=512, 512%64=0, 512//64=8, 8%8=0 -> True
        (256, 4, 64, True),   # 256*4=1024, 1024%64=0, 1024//64=16, 16%8=0 -> True
        (64, 4, 64, True),    # 64*4=256, 256%64=0, 256//64=4, 4%8!=0 -> False
    ]
    
    print(f"  Running {len(test_cases)} constraint validation tests...")
    
    passed = 0
    for i, (expert_dim, mamba_expand, mamba_headdim, expected) in enumerate(test_cases):
        result = is_valid_mamba_config(expert_dim, mamba_expand, mamba_headdim)
        inner_dim = expert_dim * mamba_expand
        ngroups = inner_dim // mamba_headdim
        actual_valid = (inner_dim % mamba_headdim == 0) and (ngroups % 8 == 0)
        
        status = "✅" if result == actual_valid else "❌"
        print(f"  {status} Test {i+1}: expert_dim={expert_dim}, expand={mamba_expand} -> "
              f"inner_dim={inner_dim}, ngroups={ngroups}, valid={result}")
        
        if result == actual_valid:
            passed += 1
    
    print(f"  └─ Constraint tests: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)

def main():
    """
    Run all tests to validate the Optuna integration.
    """
    print("="*80)
    print("🧪 KAN-MAMMOTE OPTUNA INTEGRATION TEST")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test 1: Constraint validation
    constraint_ok = test_constraint_validation()
    
    # Test 2: Optuna integration (only if constraints work)
    if constraint_ok:
        integration_ok = test_optuna_integration()
    else:
        print(f"❌ Skipping integration test due to constraint validation failures")
        integration_ok = False
    
    # Summary
    print(f"\n{'='*80}")
    print(f"TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Constraint validation: {'✅ PASS' if constraint_ok else '❌ FAIL'}")
    print(f"Optuna integration: {'✅ PASS' if integration_ok else '❌ FAIL'}")
    
    if constraint_ok and integration_ok:
        print(f"\n🎉 ALL TESTS PASSED! Ready for production tuning.")
        print(f"\n💡 Next steps:")
        print(f"  1. Run single dataset: python tune_kan_mammote_optuna.py --dataset wikipedia --model TGAT --n_trials 50")
        print(f"  2. Run multi-dataset: python tune_kan_mammote_optuna.py --multi_dataset --trials_per_combo 30")
        print(f"  3. Monitor with dashboard: optuna-dashboard sqlite:///optuna_results/<study_name>.db")
        return True
    else:
        print(f"\n❌ TESTS FAILED! Please fix issues before running production tuning.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)