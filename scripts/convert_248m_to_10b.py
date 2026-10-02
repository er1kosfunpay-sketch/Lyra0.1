#!/usr/bin/env python3
"""Convert/transfer weights from 248M model to 10B model."""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path so `import lyra` works when running as script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
try:
    from lyra.config import LyraConfig
    from lyra.model import LyraModel
    HAS_LYRA = True
except ImportError:
    HAS_LYRA = False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-248m", required=True, help="Path to 248M model config")
    p.add_argument("--checkpoint-248m", required=True, help="Path to 248M model checkpoint")
    p.add_argument("--config-10b", required=True, help="Path to 10B model config")
    p.add_argument("--out", default="checkpoints/248m_to_10b", help="Output directory")
    args = p.parse_args()

    config_248m_path = Path(args.config_248m)
    checkpoint_248m_path = Path(args.checkpoint_248m)
    config_10b_path = Path(args.config_10b)
    out_path = Path(args.out)

    # Load configs
    from lyra.config import LyraConfig
    cfg_248m = LyraConfig.from_json(str(config_248m_path))
    cfg_10b = LyraConfig.from_json(str(config_10b_path))

    print(f"248M config: {cfg_248m.model_name}, params formula: {cfg_248m.parameter_count()}")
    print(f"10B config: {cfg_10b.model_name}, params formula: {cfg_10b.parameter_count()}")

    # Load 248M model
    model_248m = LyraModel(cfg_248m)
    ckpt = torch.load(checkpoint_248m_path, map_location="cpu", weights_only=False)
    model_248m.load_state_dict(ckpt["model"])
    print(f"Loaded 248M checkpoint: step {ckpt.get('step', 'unknown')}")

    # Create 10B model
    model_10b = LyraModel(cfg_10b)

    # Try to transfer matching weights
    transferred = []
    new_initialized = []
    skipped = []

    w_248m = dict(model_248m.named_parameters())
    w_10b = dict(model_10b.named_parameters())

    print(f"\n248M model params: {len(w_248m)}, 10B model params: {len(w_10b)}")

    # Transfer weights with matching shapes
    for name_10b, param_10b in w_10b.items():
        # Try to find matching name in 248M model
        # Handle potential prefix differences (e.g., "model." vs nothing)
        match_name = name_10b
        for name_248m, param_248m in w_248m.items():
            # Check if names correspond (same base name, possibly different prefixes)
            if name_10b == name_248m:
                if param_10b.shape == param_248m.shape:
                    # Transfer the weight
                    with torch.no_grad():
                        param_10b.copy_(param_248m)
                    transferred.append((name_10b, param_10b.shape, param_248m.shape))
                else:
                    skipped.append((name_10b, f"248M: {param_248m.shape}", f"10B: {param_10b.shape}"))
                break
            # Also try without prefix
            elif name_10b.lstrip('.') == name_248m.lstrip('.'):
                if param_10b.shape == param_248m.shape:
                    with torch.no_grad():
                        param_10b.copy_(param_248m)
                    transferred.append((name_10b, param_10b.shape, param_248m.shape))
                else:
                    skipped.append((name_10b, f"248M: {param_248m.shape}", f"10B: {param_10b.shape}"))
                break

    # For any 10B parameters not transferred, they remain randomly initialized
    print(f"\nTransferred tensors: {len(transferred)}")
    for t in transferred[:10]:
        print(f"  {t[0]}: {t[1]} -> {t[2]}")
    if len(transferred) > 10:
        print(f"  ... and {len(transferred) - 10} more")

    print(f"\nNewly initialized tensors: {len([k for k in w_10b if k not in [t[0] for t in transferred]])}")
    print(f"Skipped tensors (shape mismatch): {len(skipped)}")
    for s in skipped[:5]:
        print(f"  {s[0]}: {s[1]} vs {s[2]}")

    # Save the converted model
    out_path.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model': model_10b.state_dict(),
        'config': cfg_10b.to_dict(),
        'source_checkpoint': str(checkpoint_248m_path),
    }, out_path / "converted_10b.pt")

    print(f"\nConverted model saved to: {out_path / 'converted_10b.pt'}")


if __name__ == "__main__":
    main()