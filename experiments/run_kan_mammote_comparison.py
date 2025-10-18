#!/usr/bin/env python3
"""
Run KAN-MAMMOTE variant comparison on Event-based MNIST

This script compares all KAN-MAMMOTE fusion strategies:
1. Mamba fusion (ControllableMamba2) - Default, most powerful
2. Mamba fusion (Vanilla Mamba2) - Ablation without FiLM modulation
3. Concat fusion - Lightweight alternative
4. Weighted fusion - Interpretable alternative
5. Attention fusion - Cross-attention alternative
6. SM-Kernel (legacy) - For comparison with old implementation

Usage:
    python run_kan_mammote_comparison.py --epochs 10 --batch_size 64
"""

import argparse
import os
import sys
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def main():
    parser = argparse.ArgumentParser(description='Compare KAN-MAMMOTE variants on MNIST')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--expert_dim', type=int, default=64, help='Expert dimension (must be multiple of 16)')
    parser.add_argument('--hidden_dim', type=int, default=256, help='LSTM hidden dimension')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--output_dir', type=str, default='mnist_experiments/kan_mammote_comparison',
                       help='Output directory for results')
    parser.add_argument('--skip_warmup', action='store_true', help='Skip CUDA warmup')
    
    args = parser.parse_args()
    
    # Validate expert_dim
    if args.expert_dim % 16 != 0:
        print(f"❌ Error: expert_dim must be multiple of 16, got {args.expert_dim}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define variants to test
    variants = [
        {
            'name': 'kan_mammote_full',
            'description': 'Mamba + ControllableMamba2 + K-MOTE (default)',
            'expected_perf': 'Best'
        },
        {
            'name': 'kan_mammote_vanilla_mamba',
            'description': 'Mamba + Vanilla Mamba2 + K-MOTE',
            'expected_perf': 'High (no FiLM modulation)'
        },
        {
            'name': 'kan_mammote_concat',
            'description': 'Concat Fusion + K-MOTE',
            'expected_perf': 'Medium-High (lightweight)'
        },
        {
            'name': 'kan_mammote_weighted',
            'description': 'Weighted Fusion + K-MOTE',
            'expected_perf': 'Medium (interpretable)'
        },
        {
            'name': 'kan_mammote_attention',
            'description': 'Attention Fusion + K-MOTE',
            'expected_perf': 'High (expressive)'
        },
        {
            'name': 'kan_mammote_sm_kernel',
            'description': 'Mamba + ControllableMamba2 + SM-Kernel (legacy)',
            'expected_perf': 'High (baseline)'
        },
    ]
    
    print(f"{'='*80}")
    print(f"🧪 KAN-MAMMOTE Variant Comparison on Event-based MNIST")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Embedding dim: {args.embedding_dim}")
    print(f"  Expert dim: {args.expert_dim}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Device: {args.device}")
    print(f"  Output dir: {args.output_dir}")
    print(f"\nVariants to test: {len(variants)}")
    for i, v in enumerate(variants, 1):
        print(f"  {i}. {v['name']}: {v['description']}")
    print(f"{'='*80}\n")
    
    # Import experiment module
    from experiments.event_based_mnist_experiment import (
        EventBasedMNIST, TimeEncoderClassifier, train_model, custom_collate_fn
    )
    from torch.utils.data import DataLoader, random_split
    import torch
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"📱 Using device: {device}\n")
    
    # Load dataset (do this once)
    print("📥 Loading Event-based MNIST dataset...")
    full_dataset = EventBasedMNIST()
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"✅ Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val\n")
    
    # Results storage
    results = {}
    
    # Train each variant
    for i, variant in enumerate(variants, 1):
        print(f"\n{'='*80}")
        print(f"Training Variant {i}/{len(variants)}: {variant['name']}")
        print(f"Description: {variant['description']}")
        print(f"Expected Performance: {variant['expected_perf']}")
        print(f"{'='*80}\n")
        
        try:
            # Create model
            model = TimeEncoderClassifier(
                encoder_type=variant['name'],
                embedding_dim=args.embedding_dim,
                hidden_dim=args.hidden_dim,
                num_classes=10,
                expert_dim=args.expert_dim
            ).to(device)
            
            # Warmup if Mamba variant and not skipped
            if 'mamba' in variant['name'] and not args.skip_warmup:
                print("🔥 Warming up model (compiling CUDA kernels)...")
                model.time_encoder.warmup(device=str(device), num_iterations=3)
            
            # Train
            history, best_val_acc = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=args.epochs,
                device=device,
                encoder_name=variant['name'],
                models_dir=os.path.join(args.output_dir, 'models'),
                checkpoint_dir=os.path.join(args.output_dir, 'checkpoints')
            )
            
            # Store results
            results[variant['name']] = {
                'description': variant['description'],
                'best_val_acc': best_val_acc,
                'final_train_acc': history['train_acc'][-1],
                'final_train_loss': history['train_loss'][-1],
                'final_val_loss': history['val_loss'][-1],
                'history': history,
                'status': 'success'
            }
            
            print(f"\n✅ {variant['name']} completed: Best Val Acc = {best_val_acc:.2f}%\n")
            
        except Exception as e:
            print(f"\n❌ {variant['name']} failed: {str(e)}\n")
            import traceback
            traceback.print_exc()
            
            results[variant['name']] = {
                'description': variant['description'],
                'status': 'failed',
                'error': str(e)
            }
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"📊 COMPARISON SUMMARY")
    print(f"{'='*80}\n")
    
    successful = [(name, res) for name, res in results.items() if res['status'] == 'success']
    failed = [(name, res) for name, res in results.items() if res['status'] == 'failed']
    
    if successful:
        # Sort by best_val_acc
        successful.sort(key=lambda x: x[1]['best_val_acc'], reverse=True)
        
        print("✅ Successful Runs (sorted by validation accuracy):\n")
        print(f"{'Rank':<6} {'Encoder':<30} {'Val Acc':<12} {'Train Acc':<12} {'Description':<50}")
        print(f"{'-'*6} {'-'*30} {'-'*12} {'-'*12} {'-'*50}")
        
        for rank, (name, res) in enumerate(successful, 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            print(f"{medal} {rank:<4} {name:<30} {res['best_val_acc']:>10.2f}% {res['final_train_acc']:>10.2f}% {res['description'][:50]}")
    
    if failed:
        print(f"\n❌ Failed Runs:\n")
        for name, res in failed:
            print(f"  - {name}: {res['error']}")
    
    # Save results
    import json
    results_file = os.path.join(args.output_dir, f'comparison_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    
    # Convert history arrays to lists for JSON serialization
    results_json = {}
    for name, res in results.items():
        res_copy = res.copy()
        if 'history' in res_copy:
            res_copy['history'] = {k: [float(x) for x in v] for k, v in res_copy['history'].items()}
        results_json[name] = res_copy
    
    with open(results_file, 'w') as f:
        json.dump({
            'config': vars(args),
            'results': results_json,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
