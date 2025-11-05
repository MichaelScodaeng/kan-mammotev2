#!/usr/bin/env python3
"""
Wrap a model state_dict (OrderedDict) into a lightweight checkpoint dict so training scripts
that expect keys like 'model_state_dict', 'optimizer_state_dict', 'epoch', 'seed' can resume.

Usage:
  python scripts/wrap_state_dict_checkpoint.py \
      /path/to/original_state_dict.pkl \
      --out /path/to/wrapped_checkpoint.pth \
      --epoch 10 --seed 0

If the input file already looks like a full checkpoint (has keys like 'model_state_dict'),
the script will copy it to the output path (optionally updating epoch/seed).
"""
import argparse
import os
import torch
from collections.abc import Mapping


def is_state_dict_like(obj):
    # Heuristic: mapping whose values are tensors or have .shape/.numel
    if not isinstance(obj, Mapping):
        return False
    vals = list(obj.values())
    if not vals:
        return False
    return all(hasattr(v, 'shape') or hasattr(v, 'numel') for v in vals)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('input', help='Input pth/pkl file (state_dict or checkpoint)')
    p.add_argument('--out', '-o', default=None, help='Output checkpoint file path (default: input_wrapped.pth)')
    p.add_argument('--epoch', type=int, default=0, help='Epoch number to set in wrapped checkpoint')
    p.add_argument('--seed', type=int, default=0, help='Seed value to set in wrapped checkpoint')
    p.add_argument('--optimizer', default=None, help='Optional path to an optimizer_state_dict file to include')
    args = p.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"Input file not found: {args.input}")

    out_path = args.out or (os.path.splitext(args.input)[0] + '_wrapped.pth')

    print(f"Loading {args.input}...")
    obj = torch.load(args.input, map_location='cpu')

    # If input already looks like a full checkpoint with model_state_dict, update fields and save
    if isinstance(obj, Mapping) and ('model_state_dict' in obj or 'state_dict' in obj):
        ckpt = dict(obj)
        ckpt.setdefault('model_state_dict', ckpt.get('model_state_dict', ckpt.get('state_dict')))
        # Overwrite epoch/seed if provided
        ckpt['epoch'] = args.epoch
        ckpt['seed'] = args.seed
        # add optimizer if provided
        if args.optimizer and os.path.exists(args.optimizer):
            print(f"Loading optimizer state from {args.optimizer}")
            opt = torch.load(args.optimizer, map_location='cpu')
            ckpt['optimizer_state_dict'] = opt

        torch.save(ckpt, out_path)
        print(f"Saved wrapped checkpoint to {out_path}")
        return

    # If it's a plain state_dict (OrderedDict mapping to tensors)
    if is_state_dict_like(obj):
        ckpt = {
            'model_state_dict': obj,
            'optimizer_state_dict': None,
            'epoch': args.epoch,
            'seed': args.seed,
        }
        # optionally attach optimizer
        if args.optimizer and os.path.exists(args.optimizer):
            print(f"Loading optimizer state from {args.optimizer}")
            opt = torch.load(args.optimizer, map_location='cpu')
            ckpt['optimizer_state_dict'] = opt

        torch.save(ckpt, out_path)
        print(f"Wrapped state_dict saved to {out_path}")
        return

    # Unknown format: save as-is into wrapper under 'model_state_dict'
    print("Input file not recognized as state_dict or checkpoint; wrapping under 'model_state_dict' anyway.")
    ckpt = {
        'model_state_dict': obj,
        'optimizer_state_dict': None,
        'epoch': args.epoch,
        'seed': args.seed,
    }
    torch.save(ckpt, out_path)
    print(f"Saved fallback wrapped checkpoint to {out_path}")


if __name__ == '__main__':
    main()
