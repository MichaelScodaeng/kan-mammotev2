#!/usr/bin/env python3
"""
Test script to verify the data ratio fix is working correctly.
This validates that data_ratio is applied BEFORE splitting and all splits are proportional.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.DataLoader import get_link_prediction_data

def test_data_ratio_fix():
    """Test that data_ratio is applied correctly."""
    
    print("="*80)
    print("TESTING DATA RATIO FIX")
    print("="*80)
    
    # Test 1: data_ratio = 1.0 (full data)
    print("\n" + "="*80)
    print("TEST 1: data_ratio=1.0 (Full Dataset)")
    print("="*80)
    
    _, _, full_data_1, train_data_1, val_data_1, test_data_1, _, _ = \
        get_link_prediction_data(
            dataset_name='wikipedia',
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
            data_ratio=1.0
        )
    
    print(f"\nResults:")
    print(f"  Full:  {len(full_data_1.src_node_ids):,} edges")
    print(f"  Train: {len(train_data_1.src_node_ids):,} edges ({len(train_data_1.src_node_ids)/len(full_data_1.src_node_ids)*100:.1f}%)")
    print(f"  Val:   {len(val_data_1.src_node_ids):,} edges ({len(val_data_1.src_node_ids)/len(full_data_1.src_node_ids)*100:.1f}%)")
    print(f"  Test:  {len(test_data_1.src_node_ids):,} edges ({len(test_data_1.src_node_ids)/len(full_data_1.src_node_ids)*100:.1f}%)")
    
    ratio_train_1 = len(train_data_1.src_node_ids) / len(full_data_1.src_node_ids)
    ratio_val_1 = len(val_data_1.src_node_ids) / len(full_data_1.src_node_ids)
    ratio_test_1 = len(test_data_1.src_node_ids) / len(full_data_1.src_node_ids)
    
    print(f"\n  Ratio: {ratio_train_1:.3f}:{ratio_val_1:.3f}:{ratio_test_1:.3f}")
    
    # Test 2: data_ratio = 0.1 (10% of data)
    print("\n" + "="*80)
    print("TEST 2: data_ratio=0.1 (10% of Dataset)")
    print("="*80)
    
    _, _, full_data_2, train_data_2, val_data_2, test_data_2, _, _ = \
        get_link_prediction_data(
            dataset_name='wikipedia',
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
            data_ratio=0.1
        )
    
    print(f"\nResults:")
    print(f"  Full:  {len(full_data_2.src_node_ids):,} edges")
    print(f"  Train: {len(train_data_2.src_node_ids):,} edges ({len(train_data_2.src_node_ids)/len(full_data_2.src_node_ids)*100:.1f}%)")
    print(f"  Val:   {len(val_data_2.src_node_ids):,} edges ({len(val_data_2.src_node_ids)/len(full_data_2.src_node_ids)*100:.1f}%)")
    print(f"  Test:  {len(test_data_2.src_node_ids):,} edges ({len(test_data_2.src_node_ids)/len(full_data_2.src_node_ids)*100:.1f}%)")
    
    ratio_train_2 = len(train_data_2.src_node_ids) / len(full_data_2.src_node_ids)
    ratio_val_2 = len(val_data_2.src_node_ids) / len(full_data_2.src_node_ids)
    ratio_test_2 = len(test_data_2.src_node_ids) / len(full_data_2.src_node_ids)
    
    print(f"\n  Ratio: {ratio_train_2:.3f}:{ratio_val_2:.3f}:{ratio_test_2:.3f}")
    
    # Test 3: Reproducibility check
    print("\n" + "="*80)
    print("TEST 3: Reproducibility (same seed should give same results)")
    print("="*80)
    
    _, _, full_data_3a, train_data_3a, val_data_3a, test_data_3a, _, _ = \
        get_link_prediction_data(
            dataset_name='wikipedia',
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
            data_ratio=0.1
        )
    
    _, _, full_data_3b, train_data_3b, val_data_3b, test_data_3b, _, _ = \
        get_link_prediction_data(
            dataset_name='wikipedia',
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
            data_ratio=0.1
        )
    
    same_full = len(full_data_3a.src_node_ids) == len(full_data_3b.src_node_ids)
    same_train = len(train_data_3a.src_node_ids) == len(train_data_3b.src_node_ids)
    same_val = len(val_data_3a.src_node_ids) == len(val_data_3b.src_node_ids)
    same_test = len(test_data_3a.src_node_ids) == len(test_data_3b.src_node_ids)
    
    print(f"\n  Full data identical:  {same_full} ✅" if same_full else f"  Full data identical:  {same_full} ❌")
    print(f"  Train data identical: {same_train} ✅" if same_train else f"  Train data identical: {same_train} ❌")
    print(f"  Val data identical:   {same_val} ✅" if same_val else f"  Val data identical:   {same_val} ❌")
    print(f"  Test data identical:  {same_test} ✅" if same_test else f"  Test data identical:  {same_test} ❌")
    
    # Validation checks
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    checks = []
    
    # Check 1: Full dataset ratio is consistent
    ratio_diff_1 = abs(ratio_train_1 - ratio_train_2) + abs(ratio_val_1 - ratio_val_2) + abs(ratio_test_1 - ratio_test_2)
    check_1 = ratio_diff_1 < 0.05  # Allow 5% tolerance
    checks.append(("Train/Val/Test ratios are consistent", check_1))
    
    # Check 2: 10% dataset is approximately 10% of full dataset
    size_ratio = len(full_data_2.src_node_ids) / len(full_data_1.src_node_ids)
    check_2 = 0.09 < size_ratio < 0.11  # 10% ± 1%
    checks.append((f"10% subset is actually ~10% of full data ({size_ratio*100:.1f}%)", check_2))
    
    # Check 3: Proportional splits maintained (val and test)
    # Note: Training data is reduced due to inductive setting (removing edges with new test nodes)
    # So we only check that val and test ratios are correct (~15% each)
    expected_val_ratio = 0.15
    expected_test_ratio = 0.15
    
    # Training ratio will be lower due to inductive setting, but val/test should match
    check_3a = abs(ratio_val_2 - expected_val_ratio) < 0.02
    check_3b = abs(ratio_test_2 - expected_test_ratio) < 0.02
    # Also check that ratios are consistent between full and subset data
    check_3c = abs(ratio_val_1 - ratio_val_2) < 0.01
    check_3d = abs(ratio_test_1 - ratio_test_2) < 0.01
    check_3 = check_3a and check_3b and check_3c and check_3d
    checks.append((f"Val/Test ratios match expected ~15%:15% and are consistent", check_3))
    
    # Check 4: Reproducibility
    check_4 = same_full and same_train and same_val and same_test
    checks.append(("Results are reproducible with same seed", check_4))
    
    print()
    for check_desc, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {check_desc}")
    
    all_passed = all(passed for _, passed in checks)
    
    print("\n" + "="*80)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Data ratio fix is working correctly.")
    else:
        print("⚠️  SOME TESTS FAILED! Please review the implementation.")
    print("="*80)
    
    return all_passed

if __name__ == '__main__':
    success = test_data_ratio_fix()
    sys.exit(0 if success else 1)
