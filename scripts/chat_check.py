#!/usr/bin/env python3
"""Quick chat-quality probe for a Lyra checkpoint.

Builds prompts in the exact training format (<USER>text<END><ASSISTANT>),
generates a continuation, and cuts it at <END>. Lexical check only:
it shows what the model actually says, it does not grade quality.

Usage:
  python scripts/chat_check.py --checkpoint checkpoints/stage1_1300/latest.pt
  python scripts/chat_check.py --checkpoint ... --tokenizer artifacts/tokenizer/tokenizer.json --max-tokens 64
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from lyra.config import LyraConfig
from lyra.model import LyraModel
from lyra.tokenizer import LyraTokenizer

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
except Exception:
    pass

DEFAULT_PROMPTS = [
    "Привет",
    "Как дела?",
    "Кто ты?",
    "Что такое Roblox?",
    "Расскажи короткую шутку.",
    "Помоги мне написать Lua-код.",
]


def build_prompt(tok, text):
    return [tok.id("<USER>")] + tok.encode(text) + [tok.id("<END>"), tok.id("<ASSISTANT>")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", default="artifacts/tokenizer/tokenizer.json")
    p.add_argument("--prompts", nargs="*", default=None)
    p.add_argument("--max-tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.9)
    a = p.parse_args()

    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    raw = dict(ckpt.get("config", {}))
    cfg = LyraConfig(**{k: v for k, v in raw.items() if k in LyraConfig.__dataclass_fields__})
    model = LyraModel(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = LyraTokenizer.from_file(a.tokenizer)
    end_id = tok.id("<END>")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print(f"checkpoint step: {ckpt.get('step')}, stage: {ckpt.get('stage')}")
    for text in (a.prompts or DEFAULT_PROMPTS):
        ids = torch.tensor([build_prompt(tok, text)], device=device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=a.max_tokens,
                                 temperature=a.temperature, top_k=a.top_k,
                                 top_p=a.top_p, eos_token_id=end_id)
        gen = out[0].tolist()[len(ids[0]):]
        if end_id in gen:
            gen = gen[:gen.index(end_id)]
        print("=" * 60)
        print(f"User: {text}")
        print(f"Assistant: {tok.decode(gen)}")


if __name__ == "__main__":
    main()
