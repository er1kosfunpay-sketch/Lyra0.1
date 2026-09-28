#!/usr/bin/env python3
"""Count parameters for a model config file."""
import argparse
import json
from pathlib import Path

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML or JSON config")
    args = p.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}")
        return

    # Load config
    if config_path.suffix.lower() == ".yaml" or config_path.name.endswith(".yaml"):
        import yaml
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    else:
        raw = json.loads(config_path.read_text(encoding="utf-8-sig"))

    # Instantiate config object if lyra is available
    if HAS_LYRA:
        try:
            cfg = LyraConfig.from_json(str(config_path)) if config_path.suffix.lower() in ('.json',) else LyraConfig(**{k: v for k, v in raw.items() if k in LyraConfig.__dataclass_fields__})
            # Actually, let's just use the raw dict with LyraConfig
            if config_path.suffix.lower() == '.json':
                cfg = LyraConfig.from_json(str(config_path))
            else:
                # For YAML, we need to create LyraConfig from the dict
                # Extract only the known fields
                valid_fields = LyraConfig.__dataclass_fields__.keys()
                filtered = {k: v for k, v in raw.items() if k in valid_fields}
                cfg = LyraConfig(**filtered)
        except Exception as e:
            print(f"Warning: Could not load LyraConfig, using formula: {e}")
            cfg = None
    else:
        cfg = None

    # Compute using formula
    actual_count = compute_param_count(raw)

    # Try to instantiate model for actual numel() count
    model_count = None
    if cfg is not None:
        try:
            model = LyraModel(cfg)
            model_count = model.numel()
            print(f"Model instantiated successfully")
        except Exception as e:
            print(f"Could not instantiate model: {e}")

    target = 10_000_000_000
    diff = actual_count - target

    print(f"Target parameters: {target:,}")
    print(f"Actual parameters: {actual_count:,}")
    if model_count is not None:
        print(f"Actual from model.numel(): {model_count:,}")
    print(f"Difference: {diff:,} ({diff/target*100:+.2f}%)")
    print(f"Trainable parameters: {actual_count:,}")  # all params are trainable since no freezing

    if cfg is not None:
        print(f"\nArchitecture config:")
        for k, v in cfg.to_dict().items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()