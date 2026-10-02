#!/usr/bin/env python3
"""Analyze Hugging Face dataset for training (fast, sampled)."""
import argparse
import sys
from pathlib import Path

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

SAMPLE_N = 500


def _extract_text(row):
    """Extract text from different dataset formats.

    Priority: messages (list of {role, content}) first, because datasets
    like UltraChat store the real content there while `prompt` may be empty.
    """
    # 1. messages field (list of dicts with role/content) — UltraChat, chat datasets
    if 'messages' in row and row['messages']:
        messages = row['messages']
        if isinstance(messages, list) and len(messages) > 0:
            parts = []
            for msg in messages:
                if isinstance(msg, dict) and msg.get('content'):
                    parts.append(str(msg['content']))
            if parts:
                return ' '.join(parts)
    # 2. Direct text field
    if 'text' in row and row['text']:
        return str(row['text'])
    # 3. prompt field (Alpaca, some chat datasets)
    if 'prompt' in row and row['prompt']:
        return str(row['prompt'])
    # 4. content field
    if 'content' in row and row['content']:
        return str(row['content'])
    # 5. instruction/input/output (Alpaca instruction format)
    if 'instruction' in row and row['instruction']:
        parts = [str(row['instruction'])]
        if row.get('input'):
            parts.append(str(row['input']))
        if row.get('output'):
            parts.append(str(row['output']))
        return ' '.join(parts)
    # 6. Any long string value as fallback
    if isinstance(row, dict):
        for v in row.values():
            if isinstance(v, str) and len(v) > 10:
                return v
    return ''


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/dataset.yaml")
    p.add_argument("--sample", type=int, default=SAMPLE_N,
                   help="How many rows to sample for stats (full scan is too slow on 200k+ rows)")
    args = p.parse_args()

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
    print(f"Streaming: {streaming}, Token: {'set' if token else 'None'}")

    try:
        ds = load_dataset(repo_id, config_name, split=split, revision=revision,
                          streaming=streaming, token=token)

        sample_n = args.sample
        lengths = []
        empty_count = 0
        if streaming:
            cols = list(ds.features.keys())
            n_samples = 0
            for i, example in enumerate(ds):
                if i >= sample_n:
                    break
                t = _extract_text(example)
                lengths.append(len(t))
                if not t:
                    empty_count += 1
                n_samples = i + 1
            print(f"\nDataset Statistics (sampled {n_samples} rows, streaming => total unknown):")
            total_chars_est = None
        else:
            n_samples = len(ds)
            cols = ds.column_names
            take = min(n_samples, sample_n)
            for i in range(take):
                t = _extract_text(ds[i])
                lengths.append(len(t))
                if not t:
                    empty_count += 1
            # Extrapolate totals from sample
            avg_len = sum(lengths) / len(lengths) if lengths else 0
            total_chars_est = int(avg_len * n_samples)
            print(f"\nDataset Statistics (extrapolated from {take}/{n_samples} rows):")
            print(f"  Number of samples: {n_samples}")

        print(f"  Columns: {cols}")
        if lengths:
            avg_len = sum(lengths) / len(lengths)
            max_len = max(lengths)
            print(f"  Average length (chars): {avg_len:.1f}")
            print(f"  Maximum length in sample (chars): {max_len}")
            if total_chars_est is not None:
                print(f"  Estimated total characters: {total_chars_est:,}")
                print(f"  Estimated total tokens (~chars/3): {total_chars_est // 3:,}")
            print(f"  Empty samples in sample: {empty_count}/{len(lengths)}")
            very_short = sum(1 for l in lengths if 0 < l < 10)
            print(f"  Very short samples (<10 chars, non-empty): {very_short}/{len(lengths)}")
            if empty_count == len(lengths):
                print("\n  WARNING: all sampled rows are empty — the extractor does not "
                      "understand this dataset format. Show columns above and adjust config.")
        print("\nValidation split: not checked here. Training uses 99% train / 1% val (seed=42).")

    except Exception as e:
        print(f"\nError loading dataset: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
