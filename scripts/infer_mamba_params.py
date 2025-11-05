#!/usr/bin/env python3
"""
Infer KAN-MAMMOTE / Mamba parameters from a PyTorch checkpoint (state_dict or full checkpoint)
Does NOT modify the checkpoint. Prints candidate values and assumptions.

Usage:
  python3 scripts/infer_mamba_params.py path/to/checkpoint.pth

This script is conservative: it only reads shapes and tries a small combinatorial search to
find plausible (mamba_d_state, mamba_d_conv, mamba_headdim) values that explain the
observed in_proj output dimension inside Mamba2.
"""
import argparse
import torch
import re
import sys
from collections import OrderedDict


def load_obj(path):
    try:
        obj = torch.load(path, map_location='cpu')
        return obj
    except Exception as e:
        print(f"ERROR: failed to load {path}: {e}")
        sys.exit(2)


def find_keys(dct, pat):
    return [k for k in dct.keys() if pat in k]


def shape_of_key(dct, key):
    v = dct[key]
    if hasattr(v, 'shape'):
        return tuple(v.shape)
    return None


def infer_from_state_dict(state_dict: OrderedDict):
    results = {}
    keys = list(state_dict.keys())

    # 1) Try to infer expert_dim from time encoder output_projection or typical names
    expert_dim = None
    time_feat_dim = None

    for k in keys:
        if 'time_encoder' in k and ('output_projection' in k or 'output_proj' in k or 'output_projection.0.weight' in k):
            shp = shape_of_key(state_dict, k)
            if shp and len(shp) == 2:
                # weight shape: (out_dim, in_dim)
                time_feat_dim, expert_dim = shp[0], shp[1]
                break

    # fallback: search for any weight whose name contains 'output_projection' or 'out_proj'
    if expert_dim is None:
        for k in keys:
            if 'output_projection' in k or 'out_proj' in k or 'output_proj' in k:
                shp = shape_of_key(state_dict, k)
                if shp and len(shp) == 2:
                    time_feat_dim, expert_dim = shp[0], shp[1]
                    break

    # 2) Try to infer num_mixtures from any 'fourier_weight' shape where last dim looks like mixtures
    num_mixtures = None
    for k in keys:
        if 'fourier_weight' in k or 'fourier' in k and 'weight' in k:
            shp = shape_of_key(state_dict, k)
            if shp:
                # heuristics: if last dim is small (<=32) treat as mixtures
                last = shp[-1]
                if isinstance(last, int) and 1 <= last <= 128:
                    num_mixtures = last
                    break

    # 3) Try to find Mamba tensors
    mamba_keys = [k for k in keys if 'mamba2' in k or 'mamba' in k and ('in_proj' in k or 'dt_bias' in k or 'D' in k)]
    mamba_keys = sorted(mamba_keys)

    in_proj_out = None
    in_proj_in = None
    nheads = None

    # find in_proj weight
    for k in keys:
        if ('mamba2.in_proj.weight' in k) or ('.mamba2.in_proj.weight' in k) or ('mamba.in_proj.weight' in k):
            shp = shape_of_key(state_dict, k)
            if shp and len(shp) == 2:
                in_proj_out, in_proj_in = shp[0], shp[1]
                break

    # find dt_bias which often has shape (nheads,)
    for k in keys:
        if ('mamba2.dt_bias' in k) or ('mamba.dt_bias' in k) or ('.dt_bias' in k):
            shp = shape_of_key(state_dict, k)
            if shp and len(shp) == 1:
                nheads = shp[0]
                break

    # If we didn't find in_proj but found something like in_proj.weight with different naming, try regex
    if in_proj_out is None:
        for k in keys:
            if re.search(r'in_proj.*weight', k):
                shp = shape_of_key(state_dict, k)
                if shp and len(shp) == 2 and shp[1] <= 1024:
                    in_proj_out, in_proj_in = shp[0], shp[1]
                    break

    # Print gathered raw facts
    print('\n=== Raw facts from checkpoint ===')
    print(f'found keys: {len(keys)} parameters')
    print(f'expert_dim (inferred): {expert_dim}')
    print(f'time_feat_dim (inferred): {time_feat_dim}')
    print(f'num_mixtures (inferred, maybe None): {num_mixtures}')
    print(f'mamba in_proj out: {in_proj_out}, in: {in_proj_in}')
    print(f'mamba nheads (from dt_bias): {nheads}')

    # Heuristic: if in_proj_in equals expert_dim, good sign
    if in_proj_in is not None and expert_dim is not None and in_proj_in != expert_dim:
        print('\nWARNING: in_proj input dim != inferred expert_dim. This may indicate a different layout or multiple layers.')

    # 4) If we have in_proj_out and nheads, try to solve for d_state and d_ssm
    candidates = []
    if in_proj_out is not None and nheads is not None:
        # assume ngroups == nheads (common in some Mamba variants). Note: this is an assumption.
        ngroups = nheads
        # Try reasonable ranges for d_state and d_ssm
        possible_d_state = [8, 16, 32, 48, 64, 96, 128, 192, 256]
        possible_d_ssm = [1, 2, 4, 8, 16, 32, 64]
        for d_state in possible_d_state:
            for d_ssm in possible_d_ssm:
                rem = in_proj_out - nheads - 2*d_ssm - 2*ngroups*d_state
                # rem should be even and >= 0 because rem = 2*d_mlp
                if rem >= 0 and rem % 2 == 0:
                    d_mlp = rem // 2
                    candidates.append((d_state, d_ssm, d_mlp))

    # compute headdim guess if possible
    headdim = None
    if in_proj_in is not None and nheads is not None and in_proj_in % nheads == 0:
        headdim = in_proj_in // nheads

    print('\n=== Inference / candidate Mamba parameters ===')
    if headdim is not None:
        print(f'Inferred mamba_headdim (heuristic): {headdim}  (computed as in_proj_in // nheads)')
    else:
        print('Could not compute mamba_headdim (in_proj_in not divisible by nheads or missing)')

    if candidates:
        print('\nCandidate (mamba_d_state, mamba_d_conv, implied d_mlp):')
        for (d_state, d_ssm, d_mlp) in candidates:
            print(f'  - d_state={d_state}, d_ssm={d_ssm} -> d_mlp={d_mlp}')
    else:
        print('No candidate (d_state, d_ssm) found in the tried ranges. Consider expanding ranges or confirm Mamba layout assumptions (ngroups==nheads).')

    # Recommend argparse values (best-effort)
    print('\n=== Recommended argparse flags (best-effort) ===')
    if expert_dim is not None:
        print(f"--expert_dim {expert_dim}")
    if time_feat_dim is not None:
        print(f"--time_feat_dim {time_feat_dim}")
    if num_mixtures is not None:
        print(f"--num_mixtures {num_mixtures}")
    if candidates:
        # pick smallest d_state candidate that looks reasonable
        chosen = sorted(candidates, key=lambda x: x[0])[0]
        chosen_d_state, chosen_d_ssm, chosen_d_mlp = chosen
        print(f"--mamba_d_state {chosen_d_state}    # chosen from candidate list (smallest d_state)")
        print(f"--mamba_d_conv {chosen_d_ssm}     # chosen candidate for d_ssm (convolution dim)")
    else:
        print('# Could not pick mamba_d_state/mamba_d_conv; keep defaults or run with --debug_encoder and inspect model logs')

    if headdim is not None:
        print(f"--mamba_headdim {headdim}    # inferred from in_proj input / nheads")
    else:
        print('# Could not infer mamba_headdim reliably')

    print('\nNotes:')
    print('- This script only inspects shapes and uses heuristics and a small search. It avoids modifying the checkpoint.')
    print('- Assumption: ngroups == nheads was used for the candidate search. If your Mamba variant uses a different ngroups, the candidates may be wrong.')
    print('- If you want, run the model with these args in evaluation/warmup (no optimizer resume) to confirm shapes match. Do NOT overwrite the original checkpoint.')

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help='Path to checkpoint/state_dict file (pth/pkl)')
    args = parser.parse_args()

    obj = load_obj(args.checkpoint)
    # If it's a dict with model_state_dict, extract that
    sd = None
    if isinstance(obj, dict) and 'model_state_dict' in obj and isinstance(obj['model_state_dict'], OrderedDict):
        sd = obj['model_state_dict']
    elif isinstance(obj, OrderedDict):
        sd = obj
    elif isinstance(obj, dict) and all(isinstance(v, torch.Tensor) or hasattr(v, 'shape') for v in obj.values()):
        # sometimes a plain dict of tensors but not OrderedDict; still OK
        sd = OrderedDict(obj)
    else:
        # try common keys
        possible = ['state_dict', 'model_state_dict', 'model']
        for k in possible:
            if isinstance(obj, dict) and k in obj and isinstance(obj[k], (dict, OrderedDict)):
                sd = OrderedDict(obj[k])
                break

    if sd is None:
        print('Could not locate a state_dict-like mapping in the checkpoint. The file may be a custom format.')
        sys.exit(3)

    infer_from_state_dict(sd)
