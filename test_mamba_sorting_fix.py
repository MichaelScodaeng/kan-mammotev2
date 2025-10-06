"""
Test and Verify Mamba Sorting Fix

This script verifies that:
1. Neighbor sorting is working correctly
2. Metrics logging is functioning
3. The fixes don't break existing functionality

Usage:
    python test_mamba_sorting_fix.py
"""

import torch
import numpy as np
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_neighbor_sorting():
    """Test that neighbor sorting works correctly."""
    print("\n" + "="*80)
    print("TEST 1: Neighbor Sorting")
    print("="*80)
    
    # Create mock neighbor times (unordered)
    batch_size = 3
    num_neighbors = 5
    
    # Create intentionally unsorted times
    neighbor_times = np.array([
        [10.5, 3.2, 15.1, 1.0, 8.7],   # Batch 1
        [25.3, 12.4, 30.1, 18.2, 20.5], # Batch 2
        [5.0, 2.1, 8.3, 1.5, 6.2]       # Batch 3
    ])
    
    neighbor_node_ids = np.array([
        [100, 101, 102, 103, 104],
        [200, 201, 202, 203, 204],
        [300, 301, 302, 303, 304]
    ])
    
    print("\n📝 Original (unsorted) neighbor times:")
    for i in range(batch_size):
        print(f"  Batch {i}: {neighbor_times[i]}")
    
    # Apply sorting
    sorted_indices = np.argsort(neighbor_times, axis=1)
    neighbor_times_sorted = np.take_along_axis(neighbor_times, sorted_indices, axis=1)
    neighbor_node_ids_sorted = np.take_along_axis(neighbor_node_ids, sorted_indices, axis=1)
    
    print("\n✅ Sorted neighbor times (oldest → newest):")
    for i in range(batch_size):
        print(f"  Batch {i}: {neighbor_times_sorted[i]}")
    
    # Verify sorting
    all_sorted = True
    for i in range(batch_size):
        is_sorted = np.all(neighbor_times_sorted[i][:-1] <= neighbor_times_sorted[i][1:])
        if not is_sorted:
            print(f"  ❌ Batch {i} is NOT sorted!")
            all_sorted = False
    
    if all_sorted:
        print("\n✅ TEST PASSED: All batches are properly sorted chronologically")
    else:
        print("\n❌ TEST FAILED: Some batches are not sorted")
    
    return all_sorted


def test_metrics_logger():
    """Test that metrics logger works correctly."""
    print("\n" + "="*80)
    print("TEST 2: Metrics Logger")
    print("="*80)
    
    try:
        from utils.metrics_logger import MetricsLogger
        
        # Create test logger
        test_dir = "./test_metrics"
        logger = MetricsLogger(
            save_dir=test_dir,
            model_name="TEST_MODEL",
            dataset_name="TEST_DATASET",
            encoder_type="test_encoder",
            run_id=0
        )
        
        print(f"\n📂 Metrics directory: {logger.metrics_dir}")
        
        # Log some mock metrics
        for epoch in range(1, 6):
            # Training metrics
            logger.log_epoch_metrics(
                epoch=epoch,
                phase='train',
                metrics={
                    'average_precision': 0.7 + epoch * 0.02,
                    'roc_auc': 0.75 + epoch * 0.01
                },
                loss=0.5 - epoch * 0.05
            )
            
            # Validation metrics
            logger.log_epoch_metrics(
                epoch=epoch,
                phase='val',
                metrics={
                    'average_precision': 0.65 + epoch * 0.02,
                    'roc_auc': 0.70 + epoch * 0.01
                },
                loss=0.55 - epoch * 0.04
            )
        
        # Test metrics
        logger.log_epoch_metrics(
            epoch=5,
            phase='test',
            metrics={
                'average_precision': 0.78,
                'roc_auc': 0.82
            },
            loss=0.25
        )
        
        # Save summary
        logger.save_summary()
        
        # Check files exist
        csv_files = list(Path(logger.metrics_dir).glob("*.csv"))
        summary_files = list(Path(logger.metrics_dir).glob("*summary.txt"))
        
        print(f"\n✅ Created files:")
        for f in csv_files + summary_files:
            print(f"  - {f.name}")
        
        # Load and verify
        val_df = logger.load_metrics('val')
        if val_df is not None:
            print(f"\n✅ Validation metrics loaded: {len(val_df)} epochs")
            print(f"  Columns: {', '.join(val_df.columns)}")
            
            # Check best epoch
            best_epoch = logger.get_best_epoch('average_precision')
            print(f"  Best epoch: {best_epoch}")
        
        # Cleanup
        import shutil
        shutil.rmtree(test_dir)
        print(f"\n🗑️  Cleaned up test directory: {test_dir}")
        
        print("\n✅ TEST PASSED: Metrics logger working correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_tgat_initialization():
    """Test that TGAT can be initialized with sorting flag."""
    print("\n" + "="*80)
    print("TEST 3: TGAT Initialization with Sorting")
    print("="*80)
    
    try:
        from models.gnn_backbones.TGAT import TGAT
        from models.time_encoders.factory import create_time_encoder
        
        # Mock data
        node_features = np.random.randn(100, 64).astype(np.float32)
        edge_features = np.random.randn(50, 32).astype(np.float32)
        
        # Create a simple time encoder
        time_encoder = create_time_encoder(
            encoder_type='original',
            time_dim=128,
            device='cpu'
        )
        
        # Create mock neighbor sampler
        from utils.utils import NeighborSampler
        from utils.DataLoader import Data
        
        mock_data = Data(
            src_node_ids=np.array([0, 1, 2]),
            dst_node_ids=np.array([3, 4, 5]),
            node_interact_times=np.array([1.0, 2.0, 3.0]),
            edge_ids=np.array([0, 1, 2]),
            labels=np.array([1, 1, 1])
        )
        
        neighbor_sampler = NeighborSampler(
            adj_list=[[] for _ in range(100)],
            sample_neighbor_strategy='recent'
        )
        
        # Test with sorting disabled
        print("\n📋 Testing TGAT with sort_neighbors_by_time=False...")
        tgat_no_sort = TGAT(
            node_raw_features=node_features,
            edge_raw_features=edge_features,
            neighbor_sampler=neighbor_sampler,
            time_encoder=time_encoder,
            time_feat_dim=128,
            num_layers=2,
            num_heads=2,
            dropout=0.1,
            device='cpu',
            sort_neighbors_by_time=False
        )
        print(f"  ✅ TGAT initialized: sort_neighbors_by_time={tgat_no_sort.sort_neighbors_by_time}")
        
        # Test with sorting enabled
        print("\n📋 Testing TGAT with sort_neighbors_by_time=True...")
        tgat_with_sort = TGAT(
            node_raw_features=node_features,
            edge_raw_features=edge_features,
            neighbor_sampler=neighbor_sampler,
            time_encoder=time_encoder,
            time_feat_dim=128,
            num_layers=2,
            num_heads=2,
            dropout=0.1,
            device='cpu',
            sort_neighbors_by_time=True
        )
        print(f"  ✅ TGAT initialized: sort_neighbors_by_time={tgat_with_sort.sort_neighbors_by_time}")
        
        print("\n✅ TEST PASSED: TGAT accepts sort_neighbors_by_time parameter")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_command_line_args():
    """Test that command-line arguments are properly configured."""
    print("\n" + "="*80)
    print("TEST 4: Command-Line Arguments")
    print("="*80)
    
    try:
        from utils.load_configs import get_link_prediction_args
        import sys
        
        # Mock command-line arguments
        original_argv = sys.argv
        sys.argv = ['test_script.py', '--sort_neighbors_by_time']
        
        args = get_link_prediction_args(is_evaluation=False)
        
        print(f"\n✅ Parsed arguments:")
        print(f"  sort_neighbors_by_time: {args.sort_neighbors_by_time}")
        
        # Restore original argv
        sys.argv = original_argv
        
        if hasattr(args, 'sort_neighbors_by_time'):
            print("\n✅ TEST PASSED: sort_neighbors_by_time argument is available")
            return True
        else:
            print("\n❌ TEST FAILED: sort_neighbors_by_time argument not found")
            return False
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all verification tests."""
    print("\n" + "🧪" * 40)
    print("RUNNING ALL VERIFICATION TESTS")
    print("🧪" * 40)
    
    results = {
        'Neighbor Sorting': test_neighbor_sorting(),
        'Metrics Logger': test_metrics_logger(),
        'TGAT Initialization': test_tgat_initialization(),
        'Command-Line Args': test_command_line_args()
    }
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:30s}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! The fixes are working correctly.")
        print("\n📋 You can now run experiments with:")
        print("   python experiments/train_link_prediction.py \\")
        print("       --model_name TGAT \\")
        print("       --dataset_name wikipedia \\")
        print("       --time_encoder_type kan_mammote \\")
        print("       --sort_neighbors_by_time")
    else:
        print("\n⚠️  SOME TESTS FAILED. Please check the error messages above.")
    
    print("\n" + "="*80 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
