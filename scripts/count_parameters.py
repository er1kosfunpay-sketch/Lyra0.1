#!/usr/bin/env python3
"""Count parameters for a model config file."""
import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path so `import lyra` works when running as script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from lyra.config import LyraConfig
    from lyra.model import LyraModel
    HAS_LYRA = True
except ImportError:
    HAS_LYRA = False


def compute_param_count(config_dict):
    """Compute parameter count using the exact formula from LyraConfig."""
    h = config_dict["hidden_size"]
    d = config_dict["head_dim"]
    m = config_dict["intermediate_size"]
    v = config_dict["vocab_size"]
    n = config_dict["num_layers"]
    kv = config_dict.get("num_kv_heads", config_dict.get("num_attention_heads", 16))

    # Q/O projections plus grouped K/V projections; SwiGLU has gate/up/down.
    attention = h * h + 2 * kv * d * h + h * h
    mlp = 3 * h * m
    block_norms = 2 * h
    final_norm = h
    embedding = v * h
    lm_head = 0 if config_dict.get("tie_embeddings", True) else v * h

    total = embedding + n * (attention + mlp + block_norms) + final_norm + lm_head
    return total


def main(argv=None):
    p = argparse.ArgumentParser(description="Count parameters for a Lyra model config.")
    p.add_argument("--config", required=True, help="Path to a JSON config")
    p.add_argument("--target", type=int, default=None,
                   help="Optional parameter target to compare the result with")
    args = p.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}")
        return 1

    raw = json.loads(config_path.read_text(encoding="utf-8-sig"))

    cfg = None
    if HAS_LYRA:
        try:
            cfg = LyraConfig.from_json(str(config_path))
        except Exception as e:
            print(f"Warning: Could not load LyraConfig, using formula: {e}")

    # Exact architecture formula (see compute_param_count).
    formula_count = compute_param_count(raw)

    # Authoritative number: instantiate the model and count its elements.
    model_count = None
    if cfg is not None:
        try:
            model_count = LyraModel(cfg).parameter_count()
            print("Model instantiated successfully")
        except Exception as e:
            print(f"Could not instantiate model: {e}")

    print(f"Parameters (formula): {formula_count:,}")
    if model_count is not None:
        print(f"Parameters (model.parameter_count()): {model_count:,}")
    if args.target:
        got = model_count if model_count is not None else formula_count
        diff = got - args.target
        print(f"Target: {args.target:,}   Difference: {diff:,} ({diff / args.target * 100:+.2f}%)")

    if cfg is not None:
        print("\nArchitecture config:")
        for k, v in cfg.to_dict().items():
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())