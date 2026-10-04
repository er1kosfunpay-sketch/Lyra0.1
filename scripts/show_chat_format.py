#!/usr/bin/env python3
"""Show what training samples look like after chat formatting + tokenization.

Takes real rows from a JSONL dataset, runs them through the same
PackedTextDataset._conversation logic used in training, and prints:
  raw messages -> token ids (with special tokens decoded) -> labels
for both pretrain (labels = everything) and SFT (assistant-only labels).

Usage:
  python scripts/show_chat_format.py --tokenizer artifacts/tokenizer/tokenizer.json \
      --data data/processed/kaggle_stage1/train.jsonl --rows 2
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lyra.tokenizer import LyraTokenizer

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
except Exception:
    pass


def conversation(tok, row, assistant_only):
    ids, labels = [], []
    for m in row.get("messages", []):
        role = m.get("role", "user").upper()
        if role not in ("USER", "ASSISTANT", "SYSTEM"):
            continue
        segment = [tok.id(f"<{role}>")] + tok.encode(m.get("content", "")) + [tok.id("<END>")]
        ids.extend(segment)
        labels.extend([-100] * len(segment) if (assistant_only and role != "ASSISTANT") else segment)
    return ids, labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--rows", type=int, default=2)
    a = p.parse_args()

    tok = LyraTokenizer.from_file(a.tokenizer)
    shown = 0
    with open(a.data, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "messages" not in row or not row["messages"]:
                continue
            print("#" * 70)
            print("RAW:", json.dumps(row["messages"][:4], ensure_ascii=False)[:400])
            for mode in (False, True):
                ids, labs = conversation(tok, row, assistant_only=mode)
                print(f"--- mode={'sft/assistant-only' if mode else 'pretrain/all-tokens'} "
                      f"tokens={len(ids)} masked={sum(1 for x in labs if x == -100)}")
                print("TOKENS:", " ".join(
                    f"<{tok.backend.id_to_token(i)}>" if tok.backend.id_to_token(i) in
                    ("<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SYSTEM>", "<USER>",
                     "<ASSISTANT>", "<TOOL>", "<END>") else tok.backend.id_to_token(i) or f"[{i}]"
                    for i in ids[:80]))
                print("LABELS:", " ".join("XXXX" if x == -100 else str(x) for x in labs[:80]))
            shown += 1
            if shown >= a.rows:
                break


if __name__ == "__main__":
    main()
