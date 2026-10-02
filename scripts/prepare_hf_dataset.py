#!/usr/bin/env python3
"""Download a Hugging Face dataset and convert it to local JSONL for training.

Writes:
    <out_dir>/train.jsonl
    <out_dir>/validation.jsonl   (1% of data, seed=42)
    <out_dir>/dataset_stats.json

Each row: {"messages": [{"role": "user"|"assistant"|"system", "content": "..."}]}
which is exactly what lyra.data.PackedTextDataset expects.

Supports: messages lists, text fields, instruction/input/output, prompt.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset


def normalize_row(row):
    """Return a list of {role, content} messages or None if unusable."""
    # 1. messages list (UltraChat / chat datasets)
    msgs = row.get("messages")
    if isinstance(msgs, list) and msgs:
        out = []
        aliases = {"user": "user", "human": "user", "prompter": "user",
                   "assistant": "assistant", "gpt": "assistant",
                   "system": "system"}
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = aliases.get(str(m.get("role", "")).lower())
            content = str(m.get("content", m.get("text", ""))).strip()
            if role is None or not content:
                continue
            out.append({"role": role, "content": content})
        # need at least one user turn and one assistant turn
        roles = {m["role"] for m in out}
        if len(out) >= 2 and "assistant" in roles:
            return out
    # 2. plain text
    text = row.get("text")
    if isinstance(text, str) and text.strip():
        return [{"role": "user", "content": text.strip()}]
    # 3. instruction format (Alpaca)
    instr = row.get("instruction")
    if isinstance(instr, str) and instr.strip():
        user = instr.strip()
        if row.get("input"):
            user += "\n" + str(row["input"]).strip()
        output = str(row.get("output", "")).strip()
        if output:
            return [{"role": "user", "content": user},
                    {"role": "assistant", "content": output}]
        return [{"role": "user", "content": user}]
    # 4. prompt + response-ish columns
    prompt = row.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return [{"role": "user", "content": prompt.strip()}]
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/dataset.yaml")
    p.add_argument("--repo-id", default=None)
    p.add_argument("--split", default=None)
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap on converted rows (default from config).")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    import yaml
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8-sig"))
    hf = raw.get("huggingface", {})

    repo_id = args.repo_id or hf.get("repo_id")
    config_name = hf.get("config_name")
    split = args.split or hf.get("split", "train")
    revision = hf.get("revision", "main")
    token = hf.get("token")
    max_samples = args.max_samples or int(hf.get("max_samples", 20000))
    out_dir = Path(args.out_dir or hf.get("out_dir", "data/hf"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if not repo_id:
        print("Error: no repo_id in config and --repo-id not given")
        sys.exit(1)

    print(f"Loading {repo_id} split={split} (streaming, cap {max_samples} rows)...")
    ds = load_dataset(repo_id, config_name, split=split, revision=revision,
                      streaming=True, token=token)

    rows, skipped = [], 0
    for i, raw_row in enumerate(ds):
        if len(rows) >= max_samples:
            break
        msgs = normalize_row(raw_row)
        if msgs is None:
            skipped += 1
            continue
        rows.append({"messages": msgs})
        if len(rows) % 2000 == 0:
            print(f"  ... {len(rows)} rows converted", flush=True)

    if not rows:
        print(f"Error: 0 usable rows from {repo_id} (skipped {skipped}). "
              f"Columns were: {list(ds.features.keys())}")
        sys.exit(1)

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * 0.01))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    def write_jsonl(path, subset):
        with open(path, "w", encoding="utf-8") as f:
            for r in subset:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "validation.jsonl", val_rows)
    stats = {
        "repo_id": repo_id, "split": split, "revision": revision,
        "requested_max": max_samples, "accepted": len(rows), "skipped": skipped,
        "train": len(train_rows), "validation": len(val_rows), "seed": args.seed,
    }
    (out_dir / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nDone: train={len(train_rows)}, validation={len(val_rows)}, "
          f"skipped={skipped}")
    print(f"Files: {out_dir}/train.jsonl, {out_dir}/validation.jsonl")


if __name__ == "__main__":
    main()
