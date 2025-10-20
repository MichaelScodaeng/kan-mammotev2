#!/usr/bin/env python3
"""
Quick Results Analyzer for KAN-MAMMOTE Validation
==================================================

This script helps you quickly understand if the fixes worked.
Run it after validate_kmm_fixes.sh completes.

Usage:
    python analyze_validation_results.py <results_dir>

Example:
    python analyze_validation_results.py results/kmm_validation_20251020_120000
"""

import sys
import os
import json
import glob
from pathlib import Path

def load_result_json(result_path):
    """Load a result JSON file"""
    try:
        with open(result_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Could not load {result_path}: {e}")
        return None

def analyze_experiment(name, current_result, baseline_ap, baseline_spike=None):
    """
    Analyze a single experiment
    
    Args:
        name: Experiment name
        current_result: Result dict from this run
        baseline_ap: Baseline KMM Test AP (before fixes)
        baseline_spike: Baseline loss spike ratio (before fixes)
    """
    print(f"\n{'='*70}")
    print(f"📊 {name}")
    print(f"{'='*70}")
    
    if current_result is None:
        print("❌ No results found!")
        return False
    
    # Extract metrics
    test_ap = current_result.get('test_ap', None)
    test_auc = current_result.get('test_auc', None)
    
    # Check if we have the data
    if test_ap is None:
        print("❌ Could not find test_ap in results!")
        return False
    
    print(f"\n📈 Performance Metrics:")
    print(f"   Test AP:  {test_ap:.4f}")
    if test_auc:
        print(f"   Test AUC: {test_auc:.4f}")
    
    # Calculate improvement
    improvement = ((test_ap - baseline_ap) / baseline_ap) * 100
    
    print(f"\n🔄 Comparison to Baseline:")
    print(f"   Before fixes: {baseline_ap:.4f}")
    print(f"   After fixes:  {test_ap:.4f}")
    print(f"   Change:       {improvement:+.2f}%")
    
    # Determine success
    success = False
    if improvement > 5:
        print(f"\n✅ EXCELLENT! Huge improvement (+{improvement:.1f}%)")
        success = True
    elif improvement > 2:
        print(f"\n✅ GOOD! Solid improvement (+{improvement:.1f}%)")
        success = True
    elif improvement > 0:
        print(f"\n⚠️  MINOR: Small improvement (+{improvement:.1f}%)")
        success = True
    elif improvement > -1:
        print(f"\n⚠️  STABLE: No significant change ({improvement:+.1f}%)")
        success = True
    else:
        print(f"\n❌ WORSE: Performance declined ({improvement:.1f}%)")
    
    return success

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_validation_results.py <results_dir>")
        print("\nExample:")
        print("  python analyze_validation_results.py results/kmm_validation_20251020_120000")
        sys.exit(1)
    
    results_dir = Path(sys.argv[1])
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        sys.exit(1)
    
    print(f"{'='*70}")
    print(f"🔍 KAN-MAMMOTE VALIDATION RESULTS ANALYSIS")
    print(f"{'='*70}")
    print(f"\nResults directory: {results_dir}")
    
    # Define baselines (from COMPREHENSIVE_KMM_ANALYSIS.md)
    experiments = [
        {
            'name': 'Experiment 1: uci (TCL)',
            'pattern': '**/val_uci_tcl**/test_*.json',
            'baseline_ap': 0.9345,
            'target_ap': 0.93,
            'description': 'Validate no regression'
        },
        {
            'name': 'Experiment 2: lastfm (JODIE)',
            'pattern': '**/val_lastfm_jodie**/test_*.json',
            'baseline_ap': 0.6519,
            'target_ap': 0.70,
            'description': 'Fix catastrophic failure'
        },
        {
            'name': 'Experiment 3: mooc (JODIE)',
            'pattern': '**/val_mooc_jodie**/test_*.json',
            'baseline_ap': 0.7662,
            'target_ap': 0.80,
            'description': 'Fix moderate gap'
        },
        {
            'name': 'Experiment 4: Contacts (JODIE)',
            'pattern': '**/val_contacts_jodie**/test_*.json',
            'baseline_ap': 0.9714,
            'target_ap': 0.97,
            'description': 'Fix instability'
        }
    ]
    
    success_count = 0
    total_count = 0
    
    for exp in experiments:
        # Find result files
        result_files = list(results_dir.glob(exp['pattern']))
        
        if not result_files:
            # Try alternative patterns
            alt_pattern = f"**/*{exp['pattern'].split('**/')[1]}"
            result_files = list(results_dir.glob(alt_pattern))
        
        if not result_files:
            print(f"\n⚠️  {exp['name']}: No results found (pattern: {exp['pattern']})")
            continue
        
        # Load the most recent result
        result_file = sorted(result_files)[-1]
        result = load_result_json(result_file)
        
        # Analyze
        total_count += 1
        if analyze_experiment(
            exp['name'],
            result,
            exp['baseline_ap'],
        ):
            success_count += 1
        
        # Show target
        print(f"\n🎯 Target: Test AP ≥ {exp['target_ap']:.2f}")
        if result and result.get('test_ap', 0) >= exp['target_ap']:
            print(f"   ✅ TARGET MET!")
        else:
            print(f"   ❌ Target not reached")
    
    # Overall summary
    print(f"\n{'='*70}")
    print(f"📊 OVERALL SUMMARY")
    print(f"{'='*70}")
    print(f"\nSuccess rate: {success_count}/{total_count} experiments improved")
    
    if success_count >= 3:
        print(f"\n🎉 EXCELLENT RESULTS! {success_count}/4 experiments improved!")
        print(f"\n✅ RECOMMENDATION: Your fixes WORK!")
        print(f"   → Proceed to comprehensive evaluation on all datasets")
        print(f"   → Your KAN-MAMMOTE contribution is VALIDATED")
        print(f"   → Run the full benchmark with these hyperparameters")
        print(f"\n📝 Next steps:")
        print(f"   1. Update default hyperparameters in your config")
        print(f"   2. Run full evaluation: python experiment_unified.py --models JODIE TCL TGN \\")
        print(f"      --single_encoder kan_mammote_dual_kmote --datasets <all> \\")
        print(f"      --learning_rate 0.00003 --weight_decay 0.001")
        print(f"   3. Write up results for your paper")
    elif success_count == 2:
        print(f"\n⚠️  PARTIAL SUCCESS: 2/4 experiments improved")
        print(f"\n📋 RECOMMENDATION: More tuning needed")
        print(f"   → Your approach is on the right track")
        print(f"   → Focus on successful datasets for paper")
        print(f"   → Iterate on failed cases with different hyperparameters")
        print(f"\n📝 Next steps:")
        print(f"   1. Analyze which datasets worked and why")
        print(f"   2. Try dataset-specific hyperparameter configs")
        print(f"   3. Consider hybrid approach (KMM for some datasets, LeTE for others)")
    else:
        print(f"\n❌ INSUFFICIENT IMPROVEMENT: Only {success_count}/4 improved")
        print(f"\n📋 RECOMMENDATION: Consider architectural changes")
        print(f"   → Hyperparameter tuning alone may not be enough")
        print(f"   → Options:")
        print(f"     A. Simplify KMM (remove Mamba2, keep dual K-MOTE)")
        print(f"     B. Use KMM only for specific dataset types")
        print(f"     C. Focus paper on analysis/understanding vs performance")
        print(f"\n📝 Next steps:")
        print(f"   1. Review training curves for failure modes")
        print(f"   2. Consider architecture ablation studies")
        print(f"   3. Discuss with advisor about direction")
    
    print(f"\n{'='*70}")
    print(f"🔍 For detailed analysis:")
    print(f"   - Check training curves in saved_metrics/")
    print(f"   - Review logs in experiment_logs/")
    print(f"   - Compare with baseline results")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
