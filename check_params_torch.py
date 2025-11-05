import torch
import argparse
import os
from collections.abc import Mapping


def tensor_info(tensor):
    try:
        shape = tuple(tensor.shape)
    except Exception:
        shape = getattr(tensor, 'size', lambda: 'unknown')()
    dtype = getattr(tensor, 'dtype', None)
    try:
        numel = int(tensor.numel())
    except Exception:
        numel = None
    return shape, dtype, numel


def print_param_summary(params_iterable):
    total_params = 0
    total_bytes = 0
    for name, tensor in params_iterable:
        shape, dtype, numel = tensor_info(tensor)
        itemsize = getattr(tensor, 'element_size', lambda: (torch.tensor(0, dtype=dtype).element_size() if dtype is not None else 4))()
        if numel is None:
            numel = 0
        total_params += numel
        total_bytes += numel * itemsize
        print(f"  {name}: shape={shape}, dtype={dtype}, params={numel}")

    mb = total_bytes / (1024 ** 2)
    print(f"\n  Total parameters: {total_params:,}  (~{mb:.3f} MB assuming actual dtype sizes)")


def inspect_loaded(obj):
    print(f"Loaded object type: {type(obj)}")

    # If it's a mapping/dict-like object, print keys and try to find model/state_dict inside
    if isinstance(obj, Mapping):
        print("Top-level keys:", list(obj.keys()))

        # Common keys that may contain parameters
        candidate_keys = ['model_state_dict', 'state_dict', 'model', 'net', 'model_state', 'state']
        for key in candidate_keys:
            if key in obj:
                print(f"\nFound '{key}' in checkpoint. Inspecting...")
                inner = obj[key]
                if isinstance(inner, Mapping):
                    # state_dict-like
                    items = list(inner.items())
                    print(f"  {len(items)} tensors in '{key}'")
                    print_param_summary(items)
                    return
                else:
                    # Could be a full module pickled
                    try:
                        print(f"  Inspecting '{key}' as a torch.nn.Module or object with named_parameters()")
                        for name, param in inner.named_parameters():
                            print(f"  {name}: {tuple(param.shape)}")
                        return
                    except Exception as e:
                        print(f"  Could not iterate named_parameters() for '{key}': {e}")

        # If no candidate key, maybe the dict itself is a state_dict
        # Check if all values are tensors
        values = list(obj.values())
        if values and all(hasattr(v, 'shape') or hasattr(v, 'numel') for v in values):
            print("Assuming top-level mapping is a state_dict. Listing tensors:")
            items = list(obj.items())
            print_param_summary(items)
            return

        # Otherwise print a brief repr
        print("No obvious state_dict found. You can inspect keys above and check appropriate key manually.")
        return

    # If it's a torch.nn.Module instance
    try:
        import torch.nn as nn
        if isinstance(obj, nn.Module):
            print("Object is an nn.Module. Listing named parameters:")
            print_param_summary(list(obj.named_parameters()))
            return
    except Exception:
        pass

    # If it's an OrderedDict / mapping of tensors
    if isinstance(obj, Mapping):
        print("Mapping-like object. Listing entries:")
        print_param_summary(list(obj.items()))
        return

    # Fallback: just print repr
    print("Unknown object type; printing repr:")
    print(obj)


def main():
    parser = argparse.ArgumentParser(description='Inspect torch checkpoint / saved model parameters')
    parser.add_argument('path', nargs='?', default=("saved_models/TGN/enron/TGN_kan_mammote_dual_kmote_seed0_trial_9_trial_9/TGN_kan_mammote_dual_kmote_seed0_trial_9_trial_9.pkl"),
                        help='Path to checkpoint file (default: the path used in project)')
    args = parser.parse_args()

    path = args.path
    if not os.path.exists(path):
        print(f"ERROR: file not found: {path}")
        return

    print(f"Loading file: {path} (map_location=cpu)")
    # Try several loading strategies for compatibility
    load_exceptions = []
    for kwargs in [{}, {'map_location': 'cpu'}]:
        try:
            obj = torch.load(path, **kwargs)
            inspect_loaded(obj)
            return
        except Exception as e:
            load_exceptions.append((kwargs, e))

    print("All load attempts failed. Exceptions:\n")
    for kw, exc in load_exceptions:
        print(f"With {kw}: {exc}")


if __name__ == '__main__':
    main()