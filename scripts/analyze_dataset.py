#!/usr/bin/env python3
"""Analyze Hugging Face dataset for training."""
import argparse
import sys
from pathlib import Path

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/dataset.yaml")
    args = p.parse_args()

    # Load dataset config
    import yaml
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8-sig"))
    hf_cfg = raw.get("huggingface", {})

    repo_id = hf_cfg.get("repo_id", "")
    config_name = hf_cfg.get("config_name")
    split = hf_cfg.get("split", "train")
    revision = hf_cfg.get("revision", "main")
    streaming = hf_cfg.get("streaming", False)
    token = hf_cfg.get("token")

    if not HAS_DATASETS:
        print("Error: 'datasets' library not installed. Install with: pip install datasets")
        sys.exit(1)

    print(f"Loading dataset: {repo_id}")
    print(f"Split: {split}, Revision: {revision}")
    print(f"Streaming: {streaming}, Token: {'*' * 8 if token else 'None'}")

    try:
        ds = load_dataset(repo_id, config_name, split=split, revision=revision, streaming=streaming, token=token)

        if streaming:
            # For streaming datasets, get first few entries
            lengths = []
            for i, example in enumerate(ds):
                lengths.append(len(example.get('text', example.get('content', ''))))
                if i >= 100:
                    break
            n_samples = min(i + 1, 1000000)
        else:
            n_samples = len(ds)
            lengths = [len(item.get('text', item.get('content', ''))) for item in ds]

        # Get columns
        cols = ds.column_names if not streaming else list(ds.features.keys())

        # Stats
        total_chars = sum(lengths)
        total_tokensapprox = total_chars // 3  # rough approx: 1 char ≈ 0.3 tokens
        avg_len = total_chars / n_samples if n_samples > 0 else 0
        max_len = max(lengths) if lengths else 0

        # Count empty/invalid
        empty_count = sum(1 for l in lengths if l == 0)
        # Count very short
        very_short = sum(1 for l in lengths if l < 10)

        print(f"\nDataset Statistics:")
        print(f"  Number of samples: {n_samples}")
        print(f"  Columns: {cols}")
        print(f"  Average length (chars): {avg_len:.1f}")
        print(f"  Maximum length (chars): {max_len}")
        print(f"  Total characters: {total_chars:,}")
        print(f"  Estimated total tokens: {total_tokensapprox:,}")
        print(f"  Empty samples: {empty_count}")
        print(f"  Very short samples (<10 chars): {very_short}")

        # Train/val split check
        if hasattr(ds, 'train_test_split'):
            print("\nNote: Dataset has train/test split capability")
        elif isinstance(ds, dict) and 'train' in ds:
            print("\nNote: Dataset appears to have train/validation splits")

        # If no validation split, suggest creating one
        if not (isinstance(ds, dict) and 'validation' in ds or (hasattr(ds, 'column_names') and 'validation' in str(ds))):
            print(f"\nValidation split: Not explicitly found. Consider using 99% train / 1% val split (seed=42).")

    except Exception as e:
        print(f"\nError loading dataset: {e}")
        print("\nAvailable columns (if public dataset):")
        try:
            ds2 = load_dataset(repo_id, split=split, revision=revision, token=token)
            cols2 = ds2.column_names
            print(f"  Columns: {cols2}")
        except Exception as e2:
            print(f"  Could not even load dataset: {e2}")


if __name__ == "__main__":
    main()