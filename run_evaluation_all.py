#!/usr/bin/env python3
"""
Comprehensive Evaluation Runner for Time Encoder System

This script runs evaluation on all combinations of:
- Time encoders
- Models  
- Datasets
- Negative sampling strategies

It only evaluates combinations where BOTH trained models AND existing results exist.
This allows re-running evaluations or additional analysis on completed experiments.
Results are saved as CSV files with detailed completion tracking.

Usage:
    python run_evaluation_all.py                    # Run all evaluations
    python run_evaluation_all.py --quick            # Quick test (fewer combinations)
    python run_evaluation_all.py --models TGAT TGN  # Test specific models only
    python run_evaluation_all.py --dry_run          # Show what would be evaluated
    python run_evaluation_all.py --timeout 15       # Increase timeout per evaluation
"""

import subprocess
import itertools
import os
import time
import argparse
import sys
import logging
import glob
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# All configurations - same as test_integration.py
ALL_TIME_ENCODERS = ['mercer','time2vec', 'lete', 'original'] #, 'kan_mammote_dual_kmote',
ALL_MODELS = ['TGAT', 'JODIE', 'TGN',  'DyGFormer', 'DyGMamba', 'TCL']  # Exclude CAWN
ALL_DATASETS = ['wikipedia', 'reddit', 'mooc', 'lastfm', 'enron', 'SocialEvo', 'uci',
                'CanParl', 'Contacts', 'Flights', 'UNtrade', 'UNvote', 'USLegis']
ALL_NEG_STRATEGIES = ['historical', 'inductive'] #'random', 

# Quick test configurations
QUICK_DATASETS = ['wikipedia', 'reddit', 'mooc']
QUICK_MODELS = ['TGAT', 'TGN', 'DyGMamba']
QUICK_ENCODERS = ['mercer', 'kan_mammote_dual_kmote']

class EvaluationStatus:
    """Track evaluation status and results"""
    def __init__(self):
        self.completed = []
        self.skipped = []
        self.failed = []
        self.results = []
        self.start_time = time.time()
    
    def add_completed(self, model: str, dataset: str, encoder: str, neg_strategy: str, 
                     metrics: Dict[str, float], result_files: List[str]):
        """Add a successfully completed evaluation"""
        self.completed.append({
            'model': model,
            'dataset': dataset,
            'encoder': encoder,
            'neg_strategy': neg_strategy,
            'status': 'completed',
            'result_files': result_files,
            'timestamp': datetime.now().isoformat()
        })
        
        # Add to results for CSV export
        self.results.append({
            'model': model,
            'dataset': dataset,
            'encoder': encoder,
            'neg_strategy': neg_strategy,
            'status': 'completed',
            'timestamp': datetime.now().isoformat(),
            **metrics  # Add all metrics as columns
        })
    
    def add_skipped(self, model: str, dataset: str, encoder: str, neg_strategy: str, reason: str):
        """Add a skipped evaluation"""
        self.skipped.append({
            'model': model,
            'dataset': dataset,
            'encoder': encoder,
            'neg_strategy': neg_strategy,
            'status': 'skipped',
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        })
        
        # Add to results for CSV export
        self.results.append({
            'model': model,
            'dataset': dataset,
            'encoder': encoder,
            'neg_strategy': neg_strategy,
            'status': 'skipped',
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        })
    
    def add_failed(self, model: str, dataset: str, encoder: str, neg_strategy: str, error: str):
        """Add a failed evaluation"""
        self.failed.append({
            'model': model,
            'dataset': dataset,
            'encoder': encoder,
            'neg_strategy': neg_strategy,
            'status': 'failed',
            'error': error,
            'timestamp': datetime.now().isoformat()
        })
        
        # Add to results for CSV export
        self.results.append({
            'model': model,
            'dataset': dataset,
            'encoder': encoder,
            'neg_strategy': neg_strategy,
            'status': 'failed',
            'error': error,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_summary(self) -> str:
        """Get a summary of evaluation status"""
        duration = time.time() - self.start_time
        total = len(self.completed) + len(self.skipped) + len(self.failed)
        
        summary = f"""
{'='*80}
EVALUATION SUMMARY
{'='*80}
Total Combinations: {total}
Completed: {len(self.completed)}
Skipped: {len(self.skipped)} (no trained models available)
Failed: {len(self.failed)}
Duration: {duration:.1f} seconds
{'='*80}
"""
        
        if self.failed:
            summary += f"\nFAILED EVALUATIONS ({len(self.failed)}):\n"
            for item in self.failed:
                summary += f"❌ {item['model']}-{item['dataset']}-{item['encoder']}-{item['neg_strategy']}: {item['error']}\n"
        
        if self.skipped:
            summary += f"\nSKIPPED EVALUATIONS ({len(self.skipped)}):\n"
            # Group by reason
            skip_reasons = {}
            for item in self.skipped:
                reason = item['reason']
                if reason not in skip_reasons:
                    skip_reasons[reason] = []
                skip_reasons[reason].append(f"{item['model']}-{item['dataset']}-{item['encoder']}")
            
            for reason, combinations in skip_reasons.items():
                summary += f"⏭️  {reason}: {len(combinations)} combinations\n"
                if len(combinations) <= 10:  # Show first few if not too many
                    for combo in combinations[:5]:
                        summary += f"    - {combo}\n"
                    if len(combinations) > 5:
                        summary += f"    ... and {len(combinations) - 5} more\n"
        
        return summary
    
    def save_csv(self, output_dir: str = "evaluation_results"):
        """Save results to CSV files"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Convert results to DataFrame
        if self.results:
            df = pd.DataFrame(self.results)
            
            # Save complete results
            csv_file = os.path.join(output_dir, f"evaluation_results_{timestamp}.csv")
            df.to_csv(csv_file, index=False)
            print(f"📊 Complete results saved to: {csv_file}")
            
            # Save completed evaluations only (with metrics)
            completed_df = df[df['status'] == 'completed']
            if not completed_df.empty:
                completed_csv = os.path.join(output_dir, f"evaluation_completed_{timestamp}.csv")
                completed_df.to_csv(completed_csv, index=False)
                print(f"📊 Completed evaluations saved to: {completed_csv}")
            
            # Save status summary
            status_summary = df.groupby(['model', 'dataset', 'encoder']).agg({
                'status': lambda x: '/'.join(x.unique()),
                'neg_strategy': 'count'
            }).rename(columns={'neg_strategy': 'total_combinations'})
            
            summary_csv = os.path.join(output_dir, f"evaluation_status_summary_{timestamp}.csv")
            status_summary.to_csv(summary_csv)
            print(f"📊 Status summary saved to: {summary_csv}")
            
            return csv_file, completed_csv if not completed_df.empty else None, summary_csv
        else:
            print("⚠️  No results to save")
            return None, None, None

def setup_logging(log_dir: str = "eval_logs"):
    """Setup logging for evaluation runs"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"evaluation_all_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return log_file

def check_model_availability(model: str, dataset: str, encoder: str) -> Tuple[bool, List[str]]:
    """Check if trained models exist for the given combination"""
    import glob
    
    found_models = []
    possible_patterns = [
        f"./saved_models/{model}/{dataset}/*{encoder}*seed*/*.pth",
        f"./saved_models/{model}/{dataset}/*{encoder}*seed*/*.pkl",
        f"./saved_models/{model}/{dataset}/{model}_{encoder}_seed*/*.pth",
        f"./saved_models/{model}/{dataset}/{model}_{encoder}_seed*/*.pkl"
    ]
    
    for pattern in possible_patterns:
        model_files = glob.glob(pattern)
        if model_files:
            found_models.extend(model_files)
    
    # Remove duplicates and sort
    found_models = sorted(list(set(found_models)))
    
    return len(found_models) > 0, found_models

def check_evaluation_results_exist(model: str, dataset: str, encoder: str, neg_strategy: str) -> Tuple[bool, List[str]]:
    """
    Check if evaluation results already exist for the given combination.
    
    Handles multiple file naming patterns:
    1. Strategy-specific files: {strategy}_negative_sampling_{model}_{encoder}_seed*.json
    2. Main result file (random): {model}_{encoder}_seed*_{timestamp}.json (not comprehensive)
    3. Comprehensive file: {model}_{encoder}_seed*_comprehensive_{timestamp}.json (contains all strategies)
    
    Returns:
        Tuple[bool, List[str]]: (exists, list_of_matching_files)
    """
    import glob
    base_dir = f"./saved_results/{model}/{dataset}"
    
    if not os.path.exists(base_dir):
        return False, []
    
    existing_results = []
    
    # Pattern 1: Check for strategy-specific file (historical/inductive)
    if neg_strategy in ['historical', 'inductive']:
        strategy_pattern = f"{base_dir}/{neg_strategy}_negative_sampling_{model}_{encoder}_seed*.json"
        strategy_files = glob.glob(strategy_pattern)
        existing_results.extend(strategy_files)
    
    # Pattern 2: Check for main result file (random strategy)
    elif neg_strategy == 'random':
        # Match files like: JODIE_time2vec_seed0_1760583545.480092.json
        # But NOT: JODIE_time2vec_seed0_comprehensive_1760583545.480092.json
        main_pattern = f"{base_dir}/{model}_{encoder}_seed*.json"
        main_files = glob.glob(main_pattern)
        # Filter out comprehensive and strategy-specific files
        main_files = [f for f in main_files 
                     if 'comprehensive' not in f 
                     and 'historical' not in f 
                     and 'inductive' not in f]
        existing_results.extend(main_files)
    
    # Pattern 3: Also check comprehensive file (contains all strategies)
    comprehensive_pattern = f"{base_dir}/{model}_{encoder}_seed*_comprehensive_*.json"
    comprehensive_files = glob.glob(comprehensive_pattern)
    
    if comprehensive_files:
        # If comprehensive file exists, verify it contains the requested strategy
        for comp_file in comprehensive_files:
            try:
                with open(comp_file, 'r') as f:
                    data = json.load(f)
                    # Check if this strategy's results exist in comprehensive file
                    if 'strategies' in data and neg_strategy in data['strategies']:
                        existing_results.append(comp_file)
            except (json.JSONDecodeError, IOError):
                # Skip corrupted files
                continue
    
    return len(existing_results) > 0, existing_results

def check_all_strategies_complete(model: str, dataset: str, encoder: str) -> Tuple[bool, Dict[str, bool]]:
    """
    Check if ALL negative sampling strategies are complete for a model/dataset/encoder combination.
    
    A combination is considered complete if:
    - ALL THREE individual strategy files exist (random + historical + inductive), OR
    - A comprehensive file exists that contains all three strategies
    
    Returns:
        Tuple[bool, Dict[str, bool]]: (all_complete, strategy_status_dict)
    """
    base_dir = f"./saved_results/{model}/{dataset}"
    
    if not os.path.exists(base_dir):
        return False, {'random': False, 'historical': False, 'inductive': False}
    
    strategies = ['random', 'historical', 'inductive']
    strategy_status = {}
    
    # First, check if comprehensive file exists with all strategies
    comprehensive_pattern = f"{base_dir}/{model}_{encoder}_seed*_comprehensive_*.json"
    comprehensive_files = glob.glob(comprehensive_pattern)
    
    if comprehensive_files:
        # Check if comprehensive file contains all three strategies
        for comp_file in comprehensive_files:
            try:
                with open(comp_file, 'r') as f:
                    data = json.load(f)
                    if 'strategies' in data:
                        # Check if all three strategies are present
                        has_all = all(strat in data['strategies'] for strat in strategies)
                        if has_all:
                            # Comprehensive file has everything - mark all as complete
                            return True, {strat: True for strat in strategies}
            except (json.JSONDecodeError, IOError):
                continue
    
    # If no comprehensive file, check individual strategy files
    for strategy in strategies:
        exists, files = check_evaluation_results_exist(model, dataset, encoder, strategy)
        strategy_status[strategy] = exists
    
    # All complete only if ALL THREE individual files exist
    all_complete = all(strategy_status.values())
    
    return all_complete, strategy_status

def run_evaluation(model: str, dataset: str, encoder: str, neg_strategy: str,
                  timeout_minutes: int = 15, data_ratio: float = 1.0, 
                  num_runs: int = 1, verbose: bool = False) -> Tuple[bool, str, Optional[Dict]]:
    """Run evaluation for specific combination"""
    
    test_name = f"Eval-{model}-{dataset}-{encoder}-{neg_strategy}"
    
    try:
        command = [
            'python', 'experiments/evaluate_link_prediction.py',
            '--model_name', model,
            '--dataset_name', dataset,
            '--time_encoder_type', encoder,
            '--negative_sample_strategy', neg_strategy,
            '--num_runs', str(num_runs),
            '--data_ratio', str(data_ratio),
            '--load_best_configs'
        ]
        
        # Add disable_progress_bar flag when not in verbose mode
        if not verbose:
            command.append('--disable_progress_bar')
        
        if verbose:
            print(f"   🚀 Running: {' '.join(command)}")
        
        # Track timing
        eval_start_time = time.time()
        
        # Run the evaluation
        if verbose:
            # Show real-time output when verbose
            print(f"   📝 Running evaluation with real-time output...")
            result = subprocess.run(
                command,
                text=True,
                timeout=timeout_minutes * 60,
                check=True
            )
        else:
            # Capture output for silent execution but show progress
            print(f"   🔄 Running evaluation (timeout: {timeout_minutes}min)...")
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_minutes * 60,
                check=True
            )
        
        eval_duration = time.time() - eval_start_time
        if verbose:
            print(f"   ⏱️  Evaluation completed in {eval_duration:.1f} seconds")
        
        # Check if results were created
        result_patterns = [
            f"./saved_results/{model}/{dataset}/*{neg_strategy}*{encoder}*seed*.json",
            f"./saved_results/{model}/{dataset}/*{encoder}*{neg_strategy}*seed*.json"
        ]
        
        found_results = []
        for pattern in result_patterns:
            result_files = glob.glob(pattern)
            found_results.extend(result_files)
        
        if not found_results:
            return False, "No result files generated", None
        
        # Parse metrics from result files
        all_metrics = {}
        for result_file in found_results:
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    
                # Extract key metrics
                for section_name, section_data in data.items():
                    if isinstance(section_data, dict):
                        for metric_name, metric_value in section_data.items():
                            if isinstance(metric_value, (int, float)):
                                key = f"{section_name}_{metric_name}".replace(' ', '_')
                                all_metrics[key] = float(metric_value)
                
            except Exception as e:
                print(f"   ⚠️  Warning: Could not parse {result_file}: {e}")
        
        if verbose:
            print(f"   ✅ Evaluation completed. Found {len(found_results)} result files")
        
        return True, f"Completed successfully", all_metrics
        
    except subprocess.TimeoutExpired:
        return False, f"Timeout after {timeout_minutes} minutes", None
    except subprocess.CalledProcessError as e:
        return False, f"Process failed with code {e.returncode}", None
    except Exception as e:
        return False, f"Unexpected error: {str(e)}", None

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run evaluation on all combinations')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick test with reduced combinations')
    parser.add_argument('--models', nargs='+', choices=ALL_MODELS,
                        help='Evaluate specific models only')
    parser.add_argument('--datasets', nargs='+', choices=ALL_DATASETS,
                        help='Evaluate specific datasets only')
    parser.add_argument('--encoders', nargs='+', choices=ALL_TIME_ENCODERS,
                        help='Evaluate specific encoders only')
    parser.add_argument('--neg_strategies', nargs='+', choices=ALL_NEG_STRATEGIES,
                        help='Evaluate specific negative sampling strategies only')
    parser.add_argument('--timeout', type=int, default=10000,
                        help='Timeout per evaluation in minutes (default: 15)')
    parser.add_argument('--data_ratio', type=float, default=1.0,
                        help='Fraction of data to use (default: 1.0 = full data)')
    parser.add_argument('--num_runs', type=int, default=1,
                        help='Number of runs to evaluate (default: 1, uses existing trained runs)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save results (default: evaluation_results)')
    parser.add_argument('--log_dir', type=str, default='eval_logs',
                        help='Directory to save log files (default: eval_logs)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Show what would be evaluated without running')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed output')
    parser.add_argument('--continue_on_error', action='store_true', default=True,
                        help='Continue evaluation even if some fail (default: True)')
    
    return parser.parse_args()

def get_combinations(args):
    """Get evaluation combinations based on arguments"""
    if args.quick:
        models = args.models or QUICK_MODELS
        datasets = args.datasets or QUICK_DATASETS
        encoders = args.encoders or QUICK_ENCODERS
        neg_strategies = args.neg_strategies or ALL_NEG_STRATEGIES
    else:
        models = args.models or ALL_MODELS
        datasets = args.datasets or ALL_DATASETS
        encoders = args.encoders or ALL_TIME_ENCODERS
        neg_strategies = args.neg_strategies or ALL_NEG_STRATEGIES
    
    return models, datasets, encoders, neg_strategies

def main():
    args = parse_arguments()
    status = EvaluationStatus()
    
    # Setup logging
    main_log_file = setup_logging(args.log_dir)
    
    print("🧪 Starting Comprehensive Evaluation for Time Encoder System")
    print(f"Evaluation started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📝 Main log file: {main_log_file}")
    print(f"📂 Output directory: {args.output_dir}")
    
    models, datasets, encoders, neg_strategies = get_combinations(args)
    
    print(f"\nEvaluation Configuration:")
    print(f"  Models: {models}")
    print(f"  Datasets: {datasets}")
    print(f"  Encoders: {encoders}")
    print(f"  Negative Strategies: {neg_strategies}")
    print(f"  Data Ratio: {args.data_ratio}")
    print(f"  Timeout: {args.timeout} minutes per evaluation")
    print(f"  Quick Mode: {args.quick}")
    
    # Calculate total combinations
    total_combinations = len(models) * len(datasets) * len(encoders) * len(neg_strategies)
    print(f"\nTotal Combinations: {total_combinations}")
    print(f"Estimated Duration: {total_combinations * args.timeout / 6:.1f} - {total_combinations * args.timeout / 3:.1f} minutes")
    
    if args.dry_run:
        print("\n🔍 DRY RUN MODE - Checking what models exist and what would be evaluated")
        
        combinations_to_run = []
        combinations_to_skip = []
        
        for model, dataset, encoder, neg_strategy in itertools.product(models, datasets, encoders, neg_strategies):
            model_exists, found_models = check_model_availability(model, dataset, encoder)
            if not model_exists:
                combinations_to_skip.append((model, dataset, encoder, neg_strategy))
                print(f"❌ Would skip: {model}-{dataset}-{encoder}-{neg_strategy} (No trained model)")
                continue
            
            # Check if evaluation results already exist
            results_exist, existing_results = check_evaluation_results_exist(model, dataset, encoder, neg_strategy)
            if not results_exist:
                combinations_to_skip.append((model, dataset, encoder, neg_strategy))
                print(f"❌ Would skip: {model}-{dataset}-{encoder}-{neg_strategy} (No existing results)")
                continue
                
            # Only evaluate if BOTH model AND results exist
            combinations_to_run.append((model, dataset, encoder, neg_strategy))
            print(f"✅ Would evaluate: {model}-{dataset}-{encoder}-{neg_strategy} (Both model and results exist)")
        
        print(f"\nDRY RUN SUMMARY:")
        print(f"  Would evaluate: {len(combinations_to_run)} (have both model and results)")
        print(f"  Would skip: {len(combinations_to_skip)} (missing model or results)")
        print(f"  Total: {len(combinations_to_run) + len(combinations_to_skip)}")
        return

    print(f"\n{'='*80}")
    print("RUNNING EVALUATIONS")
    print(f"{'='*80}")
    
    # Run evaluations
    count = 0
    # Track which combinations we've already determined are complete (to skip all their strategies)
    skip_all_strategies_for = set()
    
    for model, dataset, encoder, neg_strategy in itertools.product(models, datasets, encoders, neg_strategies):
        count += 1
        combo_name = f"{model}-{dataset}-{encoder}-{neg_strategy}"
        combo_key = f"{model}|{dataset}|{encoder}"  # Key without strategy
        
        print(f"\n[{count}/{total_combinations}] Evaluating: {combo_name}")
        
        # If we've already determined all strategies are complete for this combo, skip
        if combo_key in skip_all_strategies_for:
            print(f"   ⏭️  Skipping: All strategies already complete for {model}-{dataset}-{encoder}")
            status.add_skipped(model, dataset, encoder, neg_strategy, 
                             "All strategies already evaluated (checked earlier)")
            continue
        
        # Check if ALL strategies are already complete for this model/dataset/encoder
        all_complete, strategy_status = check_all_strategies_complete(model, dataset, encoder)
        
        if all_complete:
            print(f"   ✅ All strategies already complete for {model}-{dataset}-{encoder}:")
            for strat, status_val in strategy_status.items():
                print(f"      - {strat}: {'✓' if status_val else '✗'}")
            
            # Mark this combination to skip all strategies
            skip_all_strategies_for.add(combo_key)
            
            # Record skips for all strategies
            for strat in ['random', 'historical', 'inductive']:
                if strat in neg_strategies:
                    status.add_skipped(model, dataset, encoder, strat, 
                                     "All strategies already evaluated (found existing results)")
            
            print(f"   ⏭️  Skipping all remaining strategies for this combination")
            continue
        
        # Check if trained model exists
        model_exists, found_models = check_model_availability(model, dataset, encoder)
        
        if not model_exists:
            print(f"   ⏭️  Skipping: No trained model found")
            status.add_skipped(model, dataset, encoder, neg_strategy, "No trained model found")
            continue
        
        # Check if THIS SPECIFIC strategy's results already exist
        results_exist, existing_results = check_evaluation_results_exist(model, dataset, encoder, "random")
        
        print(existing_results)
        print(found_models)
        if not (results_exist and model_exists):
            print(f"   🔄 Strategy '{neg_strategy}' not yet evaluated - so it means training is not completed even we have models")
            status.add_skipped(model, dataset, encoder, neg_strategy, 
                             "No existing results found - training may not be completed")
            continue
        print(f"   ✅ Found trained models for evaluation and found results so we can start evaluating")
        
        if args.verbose:
            print(f"   📁 Found models: {[os.path.basename(f) for f in found_models[:3]]}")
        
        # Run evaluation
        try:
            success, message, metrics = run_evaluation(
                model, dataset, encoder, neg_strategy,
                args.timeout, args.data_ratio, args.num_runs, args.verbose
            )
            
            if success:
                print(f"   ✅ Completed: {message}")
                # Find result files
                result_patterns = [
                    f"./saved_results/{model}/{dataset}/*{neg_strategy}*{encoder}*seed*.json",
                    f"./saved_results/{model}/{dataset}/*{encoder}*{neg_strategy}*seed*.json"
                ]
                result_files = []
                for pattern in result_patterns:
                    result_files.extend(glob.glob(pattern))
                
                status.add_completed(model, dataset, encoder, neg_strategy, 
                                   metrics or {}, result_files)
            else:
                print(f"   ❌ Failed: {message}")
                status.add_failed(model, dataset, encoder, neg_strategy, message)
                if not args.continue_on_error:
                    break
                    
        except KeyboardInterrupt:
            print(f"\n⚠️  Interrupted by user")
            break
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"   ❌ Error: {error_msg}")
            status.add_failed(model, dataset, encoder, neg_strategy, error_msg)
            if not args.continue_on_error:
                break
    
    # Save results and print summary
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")
    
    csv_files = status.save_csv(args.output_dir)
    
    # Print final summary
    print(status.get_summary())
    
    # Print file locations
    if csv_files[0]:
        print(f"\n📄 Results Files:")
        print(f"  Complete results: {csv_files[0]}")
        if csv_files[1]:
            print(f"  Completed only: {csv_files[1]}")
        if csv_files[2]:
            print(f"  Status summary: {csv_files[2]}")
    
    # Exit with appropriate code
    if len(status.failed) > 0 and not args.continue_on_error:
        print(f"\n❌ Some evaluations failed.")
        sys.exit(1)
    else:
        print(f"\n✅ Evaluation run completed!")
        print(f"📊 Results saved as CSV files in: {args.output_dir}")
        sys.exit(0)

if __name__ == "__main__":
    main()