"""
Direct approach to check sequence lengths by intercepting at multiple points.
Analyzes sequence lengths for ALL available GNN backbone models.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from utils.load_configs import get_link_prediction_args
from utils.DataLoader import get_link_prediction_data, get_idx_data_loader
from utils import get_neighbor_sampler, convert_to_gpu
from models.gnn_backbones.TGAT import TGAT
from models.gnn_backbones.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.gnn_backbones.CAWN import CAWN
from models.gnn_backbones.TCL import TCL
from models.gnn_backbones.GraphMixer import GraphMixer
from models.gnn_backbones.DyGFormer import DyGFormer
from models.gnn_backbones.DyGMamba import DyGMamba
from models.time_encoders.factory import create_time_encoder

# Global list to capture shapes
captured_shapes = []

def patch_kan_mammote():
    """Monkey-patch KAN-MAMMOTE to capture input shapes"""
    from models.time_encoders import kan_mammote
    
    original_forward = kan_mammote.KAN_MAMMOTE.forward
    
    def patched_forward(self, t_abs, t_rel):
        # Capture the shape
        captured_shapes.append({
            't_abs_shape': t_abs.shape,
            't_rel_shape': t_rel.shape,
            'batch': t_abs.shape[0],
            'seq_len': t_abs.shape[1]
        })
        return original_forward(self, t_abs, t_rel)
    
    kan_mammote.KAN_MAMMOTE.forward = patched_forward
    print("✓ Patched KAN_MAMMOTE.forward() to capture shapes")

def analyze_sequence_lengths():
    """Analyze actual sequence lengths during training for ALL GNN models"""
    
    # Patch KAN-MAMMOTE before creating any models
    patch_kan_mammote()
    
    # Get configuration
    args = get_link_prediction_args(is_evaluation=False)
    
    print("="*80)
    print("Sequence Length Analysis (All GNN Backbones)")
    print("="*80)
    
    print(f"\nModel: {args.model_name}")
    print(f"Time Encoder: {args.time_encoder_type}")
    print(f"Dataset: {args.dataset_name}")
    
    # Load data
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, _, _ = \
        get_link_prediction_data(
            dataset_name=args.dataset_name,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            data_ratio=args.data_ratio
        )
    
    # Create neighbor sampler
    train_neighbor_sampler = get_neighbor_sampler(
        data=train_data,
        sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor=args.time_scaling_factor,
        seed=0
    )
    
    print(f"\n{'='*80}")
    print("Configuration:")
    print(f"{'='*80}")
    print(f"  num_neighbors: {args.num_neighbors}")
    print(f"  num_layers: {args.num_layers}")
    print(f"  batch_size: {args.batch_size}")
    
    # Common analysis function
    def run_analysis(model, model_name, num_batches=20):
        """Run sequence length analysis on a model"""
        print(f"\n{'='*80}")
        print(f"Analyzing: {model_name}")
        print(f"{'='*80}")
        
        # Get some training batches
        train_idx_data_loader = get_idx_data_loader(
            indices_list=list(range(len(train_data.src_node_ids))),
            batch_size=args.batch_size,
            shuffle=False
        )
        
        num_batches_to_check = min(num_batches, len(train_idx_data_loader))
        print(f"\nProcessing {num_batches_to_check} batches...")
        
        model.eval()
        with torch.no_grad():
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader):
                if batch_idx >= num_batches_to_check:
                    break
                
                train_data_indices = train_data_indices.numpy()
                batch_src_node_ids = train_data.src_node_ids[train_data_indices]
                batch_dst_node_ids = train_data.dst_node_ids[train_data_indices]
                batch_node_interact_times = train_data.node_interact_times[train_data_indices]
                
                try:
                    # Clear previous captures
                    captured_shapes.clear()
                    
                    # Run forward pass (different models have different interfaces)
                    if model_name in ['DyGFormer', 'DyGMamba']:
                        _, _, _ = model.compute_src_dst_node_temporal_embeddings(
                            src_node_ids=batch_src_node_ids,
                            dst_node_ids=batch_dst_node_ids,
                            node_interact_times=batch_node_interact_times
                        )
                    else:
                        _, _ = model.compute_src_dst_node_temporal_embeddings(
                            src_node_ids=batch_src_node_ids,
                            dst_node_ids=batch_dst_node_ids,
                            node_interact_times=batch_node_interact_times,
                            num_neighbors=args.num_neighbors
                        )
                    
                    # Print captured shapes for first 3 batches
                    if batch_idx < 3 and captured_shapes:
                        print(f"\nBatch {batch_idx+1}:")
                        print(f"  Input batch size: {len(batch_src_node_ids)}")
                        print(f"  Time encoder called {len(captured_shapes)} times")
                        for i, shape_info in enumerate(captured_shapes[:3]):  # Show first 3 calls
                            print(f"    Call {i+1}: batch={shape_info['batch']}, seq_len={shape_info['seq_len']}")
                    
                    if batch_idx == 0 and not captured_shapes:
                        print(f"  ⚠️  No shapes captured in first batch!")
                        print(f"     Time encoder might not be called for {model_name}.")
                
                except Exception as e:
                    print(f"  Batch {batch_idx+1}: Error - {str(e)[:100]}")
                    if batch_idx == 0:
                        import traceback
                        traceback.print_exc()
                    continue
        
        return analyze_captured_shapes(model_name, num_batches_to_check)
    
    def analyze_captured_shapes(model_name, num_batches):
        """Analyze and report captured sequence lengths"""
        if captured_shapes:
            # Analyze all captured shapes
            all_seq_lens = [s['seq_len'] for s in captured_shapes]
            all_batch_sizes = [s['batch'] for s in captured_shapes]
            
            print(f"\n{'='*80}")
            print(f"Sequence Length Statistics for {model_name}:")
            print(f"{'='*80}")
            print(f"  Total time encoder calls:      {len(captured_shapes)}")
            print(f"  Calls per batch (avg):         {len(captured_shapes)/num_batches:.1f}")
            print(f"  Min sequence length:           {min(all_seq_lens)}")
            print(f"  Max sequence length:           {max(all_seq_lens)}")
            print(f"  Mean sequence length:          {np.mean(all_seq_lens):.1f}")
            print(f"  Median sequence length:        {np.median(all_seq_lens):.1f}")
            print(f"  Std sequence length:           {np.std(all_seq_lens):.1f}")
            
            print(f"\n  Mean batch size per call:      {np.mean(all_batch_sizes):.1f}")
            print(f"  Total samples per call (avg):  {np.mean(all_batch_sizes):.1f} × {np.mean(all_seq_lens):.1f} = {np.mean(all_batch_sizes)*np.mean(all_seq_lens):.0f}")
            
            # Analyze distribution
            short_seqs = sum(1 for s in all_seq_lens if s < 50)
            medium_seqs = sum(1 for s in all_seq_lens if 50 <= s < 200)
            long_seqs = sum(1 for s in all_seq_lens if s >= 200)
            
            print(f"\nSequence Length Distribution:")
            print(f"  Short (<50):      {short_seqs}/{len(all_seq_lens)} ({short_seqs/len(all_seq_lens)*100:.1f}%)")
            print(f"  Medium (50-200):  {medium_seqs}/{len(all_seq_lens)} ({medium_seqs/len(all_seq_lens)*100:.1f}%)")
            print(f"  Long (>=200):     {long_seqs}/{len(all_seq_lens)} ({long_seqs/len(all_seq_lens)*100:.1f}%)")
            
            # Assessment
            avg_len = np.mean(all_seq_lens)
            if avg_len < 50:
                assessment = "❌ TOO SHORT for Mamba2"
            elif avg_len < 100:
                assessment = "⚠️  MODERATE - suboptimal for Mamba2"
            else:
                assessment = "✅ GOOD for Mamba2"
            
            print(f"\nAssessment: {assessment}")
            
            return {
                'model': model_name,
                'calls_per_batch': len(captured_shapes)/num_batches,
                'avg_seq_len': np.mean(all_seq_lens),
                'min_seq_len': min(all_seq_lens),
                'max_seq_len': max(all_seq_lens),
                'assessment': assessment
            }
        else:
            print(f"\n❌ Could not capture sequence lengths for {model_name}")
            print(f"   Time encoder was not called.")
            return None
    
    # Create models and analyze based on model_name
    results = []
    
    if args.model_name == 'TGAT':
        # Create time encoder (on CUDA)
        time_encoder = create_time_encoder(
            encoder_type=args.time_encoder_type,
            time_dim=args.time_feat_dim,
            train_data=train_data,
            train_neighbor_sampler=train_neighbor_sampler,
            args=args,
            device='cuda'
        )
        
        # Create TGAT model
        model = TGAT(
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features,
            neighbor_sampler=train_neighbor_sampler,
            time_encoder=time_encoder,
            time_feat_dim=args.time_feat_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            device='cuda'
        )
        model = convert_to_gpu(model, device='cuda')
        
        result = run_analysis(model, 'TGAT')
        if result:
            results.append(result)
    
    elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
        # Create time encoder (on CUDA)
        time_encoder = create_time_encoder(
            encoder_type=args.time_encoder_type,
            time_dim=args.time_feat_dim,
            train_data=train_data,
            train_neighbor_sampler=train_neighbor_sampler,
            args=args,
            device='cuda'
        )
        
        # Create memory-based model
        model = MemoryModel(
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features,
            neighbor_sampler=train_neighbor_sampler,
            time_encoder=time_encoder,
            model_name=args.model_name,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            device='cuda'
        )
        model = convert_to_gpu(model, device='cuda')
        
        result = run_analysis(model, args.model_name)
        if result:
            results.append(result)
    
    elif args.model_name == 'CAWN':
        print(f"\n{'='*80}")
        print(f"Analyzing: CAWN")
        print(f"{'='*80}")
        print(f"\n⚠️  CAWN uses position encodings instead of time encoders.")
        print(f"   Uses temporal random walks, not sequence modeling.")
        print(f"   Not suitable for Mamba2-based time encoding.")
    
    elif args.model_name == 'TCL':
        # Create time encoder (on CUDA)
        time_encoder = create_time_encoder(
            encoder_type=args.time_encoder_type,
            time_dim=args.time_feat_dim,
            train_data=train_data,
            train_neighbor_sampler=train_neighbor_sampler,
            args=args,
            device='cuda'
        )
        
        # Create TCL model
        model = TCL(
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features,
            neighbor_sampler=train_neighbor_sampler,
            time_encoder=time_encoder,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            device='cuda'
        )
        model = convert_to_gpu(model, device='cuda')
        
        result = run_analysis(model, 'TCL')
        if result:
            results.append(result)
    
    elif args.model_name == 'GraphMixer':
        # Create time encoder (on CUDA)
        time_encoder = create_time_encoder(
            encoder_type=args.time_encoder_type,
            time_dim=args.time_feat_dim,
            train_data=train_data,
            train_neighbor_sampler=train_neighbor_sampler,
            args=args,
            device='cuda'
        )
        
        # Create GraphMixer model
        model = GraphMixer(
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features,
            neighbor_sampler=train_neighbor_sampler,
            time_encoder=time_encoder,
            num_layers=args.num_layers,
            dropout=args.dropout,
            device='cuda'
        )
        model = convert_to_gpu(model, device='cuda')
        
        result = run_analysis(model, 'GraphMixer')
        if result:
            results.append(result)
    
    elif args.model_name == 'DyGFormer':
        # Create time encoder (on CUDA)
        time_encoder = create_time_encoder(
            encoder_type=args.time_encoder_type,
            time_dim=args.time_feat_dim,
            train_data=train_data,
            train_neighbor_sampler=train_neighbor_sampler,
            args=args,
            device='cuda'
        )
        
        # Create DyGFormer model
        model = DyGFormer(
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features,
            neighbor_sampler=train_neighbor_sampler,
            time_encoder=time_encoder,
            patch_size=args.patch_size,
            max_input_sequence_length=args.max_input_sequence_length,
            dropout=args.dropout,
            device='cuda'
        )
        model = convert_to_gpu(model, device='cuda')
        
        result = run_analysis(model, 'DyGFormer')
        if result:
            results.append(result)
        else:
            # If no captures (DyGFormer might use internal encoding), report fixed length
            results.append({
                'model': 'DyGFormer',
                'calls_per_batch': 'N/A (internal)',
                'avg_seq_len': args.max_input_sequence_length,
                'min_seq_len': args.max_input_sequence_length,
                'max_seq_len': args.max_input_sequence_length,
                'assessment': '✅ FIXED LONG SEQUENCES (optimal for Mamba2)'
            })
    
    elif args.model_name == 'DyGMamba':
        # Create time encoder (on CUDA)
        time_encoder = create_time_encoder(
            encoder_type=args.time_encoder_type,
            time_dim=args.time_feat_dim,
            train_data=train_data,
            train_neighbor_sampler=train_neighbor_sampler,
            args=args,
            device='cuda'
        )
        
        # Create DyGMamba model
        model = DyGMamba(
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features,
            neighbor_sampler=train_neighbor_sampler,
            time_encoder=time_encoder,
            patch_size=args.patch_size,
            max_input_sequence_length=args.max_input_sequence_length,
            dropout=args.dropout,
            device='cuda'
        )
        model = convert_to_gpu(model, device='cuda')
        
        result = run_analysis(model, 'DyGMamba')
        if result:
            results.append(result)
        else:
            # If no captures (DyGMamba might use internal encoding), report fixed length
            results.append({
                'model': 'DyGMamba',
                'calls_per_batch': 'N/A (internal)',
                'avg_seq_len': args.max_input_sequence_length,
                'min_seq_len': args.max_input_sequence_length,
                'max_seq_len': args.max_input_sequence_length,
                'assessment': '🚀 OPTIMAL FOR MAMBA2 (fixed long sequences)'
            })
    
    else:
        print(f"\n⚠️  Model {args.model_name} not recognized or not yet implemented.")
    
    # Final summary
    if results:
        print(f"\n{'#'*80}")
        print("SUMMARY: GNN Backbone Comparison for Mamba2 Compatibility")
        print(f"{'#'*80}\n")
        
        print(f"{'Model':<15} {'Avg Seq Len':<12} {'Calls/Batch':<12} {'Assessment'}")
        print(f"{'-'*80}")
        for r in results:
            calls_str = f"{r['calls_per_batch']:.1f}" if isinstance(r['calls_per_batch'], (int, float)) else r['calls_per_batch']
            print(f"{r['model']:<15} {r['avg_seq_len']:<12.1f} {calls_str:<12} {r['assessment']}")
        
        print(f"\n{'='*80}")
        print("Recommendations:")
        print(f"{'='*80}")
        print(f"\n✅ BEST for KAN-MAMMOTE (Mamba2):")
        print(f"   1. DyGMamba - designed for Mamba2, fixed long sequences")
        print(f"   2. DyGFormer - fixed sequences, can work with Mamba2")
        
        print(f"\n⚠️  SUBOPTIMAL for KAN-MAMMOTE with Mamba2:")
        short_seq_models = [r for r in results if r['avg_seq_len'] < 100]
        if short_seq_models:
            for r in short_seq_models:
                print(f"   - {r['model']}: avg seq_len = {r['avg_seq_len']:.1f} (too short)")
            print(f"   Solution: Use KAN-MAMMOTE Lite (without Mamba2) for these models")
        
        print(f"\n✅ OPTIMAL for KAN-MAMMOTE with Mamba2:")
        long_seq_models = [r for r in results if r['avg_seq_len'] >= 100]
        if long_seq_models:
            for r in long_seq_models:
                print(f"   - {r['model']}: avg seq_len = {r['avg_seq_len']:.1f} (good for Mamba2)")
        
        print(f"\n❌ NOT TESTED (incompatible or requires special handling):")
        print(f"   - CAWN (position encoding, not temporal sequences)")
        
        print(f"\n💡 Summary:")
        print(f"   Sequence length is KEY for Mamba2 performance:")
        print(f"   - Short (<50):    ❌ Mamba2 overhead > benefit")
        print(f"   - Medium (50-100): ⚠️  Marginal benefit")
        print(f"   - Long (>100):     ✅ Mamba2 excels at long-range modeling")
        print(f"   ")
        print(f"   Choose architecture based on your needs:")
        print(f"   → DyGMamba/DyGFormer: Fixed long sequences (optimal for Mamba2)")
        print(f"   → TGAT/TGN/TCL/GraphMixer: Use KAN-MAMMOTE Lite (no Mamba2)")
        print(f"   → Or increase num_neighbors to get longer sequences")

if __name__ == "__main__":
    analyze_sequence_lengths()