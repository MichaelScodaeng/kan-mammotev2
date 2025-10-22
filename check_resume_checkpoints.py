#!/usr/bin/env python3
"""
Check which experiments have checkpoints available for resuming training from epoch 100 to 200.
"""

import os
from pathlib import Path

def check_resume_capabilities():
    """Check which experiments can be resumed from checkpoints."""
    
    # List of experiments that need to be extended to 200 epochs
    experiments_to_resume = [
        ("DyGMamba", "lastfm", "time2vec", "seed0"),
        ("DyGMamba", "mooc", "time2vec", "seed0"),
        ("DyGMamba", "uci", "time2vec", "seed0"),
        ("JODIE", "Contacts", "time2vec", "seed0"),
        ("JODIE", "SocialEvo", "time2vec", "seed0"),
        ("JODIE", "uci", "time2vec", "seed0"),
        ("JODIE", "wikipedia", "time2vec", "seed0"),
        ("TCL", "Contacts", "time2vec", "seed0"),
        ("TCL", "lastfm", "time2vec", "seed0"),
        ("TCL", "mooc", "time2vec", "seed0"),
        ("TCL", "reddit", "time2vec", "seed0"),
        ("TGN", "Flights", "time2vec", "seed0")
    ]
    
    print("=" * 80)
    print("CHECKING CHECKPOINT AVAILABILITY FOR RESUMING TRAINING")
    print("=" * 80)
    
    available_for_resume = []
    missing_checkpoints = []
    
    for model, dataset, time_encoder, seed in experiments_to_resume:
        # Construct expected checkpoint path
        experiment_name = f"{model}_{time_encoder}_{seed}"
        checkpoint_dir = f"/home/s2516027/kan-mammotev2/saved_models/{model}/{dataset}/{experiment_name}"
        
        # Check for checkpoint files
        checkpoint_100 = os.path.join(checkpoint_dir, "checkpoint_epoch_100.pth")
        checkpoint_95 = os.path.join(checkpoint_dir, "checkpoint_epoch_95.pth")
        checkpoint_90 = os.path.join(checkpoint_dir, "checkpoint_epoch_90.pth")
        
        if os.path.exists(checkpoint_100):
            print(f"✓ RESUME READY: {model}/{dataset}/{time_encoder}/{seed} - checkpoint_epoch_100.pth")
            available_for_resume.append((model, dataset, time_encoder, seed, checkpoint_100))
        elif os.path.exists(checkpoint_95):
            print(f"✓ RESUME READY: {model}/{dataset}/{time_encoder}/{seed} - checkpoint_epoch_95.pth (resume from 95)")
            available_for_resume.append((model, dataset, time_encoder, seed, checkpoint_95))
        elif os.path.exists(checkpoint_90):
            print(f"✓ RESUME READY: {model}/{dataset}/{time_encoder}/{seed} - checkpoint_epoch_90.pth (resume from 90)")
            available_for_resume.append((model, dataset, time_encoder, seed, checkpoint_90))
        else:
            print(f"✗ NO CHECKPOINT: {model}/{dataset}/{time_encoder}/{seed} - {checkpoint_dir}")
            missing_checkpoints.append((model, dataset, time_encoder, seed))
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Experiments that can be resumed: {len(available_for_resume)}")
    print(f"Experiments missing checkpoints: {len(missing_checkpoints)}")
    
    if available_for_resume:
        print(f"\n✓ CAN RESUME ({len(available_for_resume)} experiments):")
        for model, dataset, time_encoder, seed, checkpoint_path in available_for_resume:
            epoch = checkpoint_path.split("_")[-1].replace(".pth", "")
            print(f"  {model}/{dataset}/{time_encoder}/{seed} - from epoch {epoch}")
    
    if missing_checkpoints:
        print(f"\n✗ NEED FULL RETRAIN ({len(missing_checkpoints)} experiments):")
        for model, dataset, time_encoder, seed in missing_checkpoints:
            print(f"  {model}/{dataset}/{time_encoder}/{seed}")
    
    # Generate resume command examples
    if available_for_resume:
        print("\n" + "=" * 80)
        print("EXAMPLE RESUME COMMANDS")
        print("=" * 80)
        print("# To resume training, you would typically use commands like:")
        for i, (model, dataset, time_encoder, seed, checkpoint_path) in enumerate(available_for_resume[:3]):
            epoch = checkpoint_path.split("_")[-1].replace(".pth", "")
            print(f"# Example {i+1}:")
            print(f"python train_{model.lower()}.py --dataset {dataset} --time_encoder {time_encoder} --seed 0 \\")
            print(f"       --resume_from_checkpoint {checkpoint_path} --start_epoch {int(epoch)+1} --num_epochs 200")
            print()
    
    return available_for_resume, missing_checkpoints

if __name__ == "__main__":
    available, missing = check_resume_capabilities()