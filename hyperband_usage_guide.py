#!/usr/bin/env python3
"""
KAN-MAMMOTE Hyperband Usage Guide
================================

This script demonstrates how to use the new Hyperband hyperparameter tuning
to dramatically reduce tuning time for KAN-MAMMOTE.

QUICK START:
-----------

1. Test with one dataset/model (recommended first):
   python tune_kan_mammote_hyperband.py --datasets wikipedia --models TGAT

2. Preview the schedule without running:
   python tune_kan_mammote_hyperband.py --datasets wikipedia --models TGAT --dry_run

3. Run multiple datasets/models:
   python tune_kan_mammote_hyperband.py --datasets wikipedia reddit --models TGAT TGN

4. Resume interrupted tuning:
   python tune_kan_mammote_hyperband.py --resume

COMPARISON WITH GRID SEARCH:
----------------------------

Grid Search (Original):
- 36 configs × 13 datasets × 6 models = 2,808 experiments
- Each experiment: 15 epochs 
- Total cost: 42,120 epoch-experiments
- Estimated time: ~2-4 weeks on GPU cluster

Hyperband (New):
- Same search space, but with successive halving
- Schedule per dataset/model: [(36, 3), (12, 9), (8, 15)]
- Total cost: 26,208 epoch-experiments  
- Time savings: 37.8% (1.6x speedup)
- Estimated time: ~1-2 weeks on GPU cluster

HYPERBAND ALGORITHM:
-------------------

Round 1: 36 configs × 3 epochs  (Quick screening)
Round 2: 12 best  × 9 epochs   (Medium validation) 
Round 3: 8 best   × 15 epochs  (Final evaluation)

- Eliminates poor configs early (after 3 epochs)
- Focuses computational budget on promising configs
- Maintains same final evaluation quality as grid search

ADVANCED OPTIONS:
----------------

--max_configs_per_round 24    # Use fewer configs per round (faster)
--seed 42                     # Set random seed for reproducible sampling
--verbose                     # Show detailed progress
--progress_file .hb_prog.pkl  # Custom progress file name

INTERPRETING RESULTS:
--------------------

The script will show:
1. Round-by-round progress with config rankings
2. Best configuration per dataset/model combination  
3. Final summary with success rates and time savings

Results are saved in: ./hyperband_results/{dataset}/{model}/

For questions or issues, see tune_kan_mammote_hyperband.py source code.
"""

def main():
    print(__doc__)

if __name__ == '__main__':
    main()