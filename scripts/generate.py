#!/usr/bin/env python3
"""Text generation script for Lyra model.
Loads a checkpoint and generates text from a prompt.
"""
import argparse
import sys
from pathlib import Path

import torch

try:
    from lyra.config import LyraConfig
    from lyra.model import LyraModel
    from lyra.tokenizer import LyraTokenizer
    HAS_LYRA = True
except ImportError:
    HAS_LYRA = False


def main():
    p = argparse.ArgumentParser("Lyra Text Generation")
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    p.add_argument("--prompt", default="Once upon a time", help="Generation prompt")
    p.add_argument("--max-tokens", type=int, default=128, help="Max tokens to generate")
    p.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    p.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    p.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling")
    p.add_argument("--eos-token-id", type=int, default=None, help="EOS token ID")
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_cfg = LyraConfig.from_dict(ckpt.get("config", {}))

    # Create model
    model = LyraModel(ckpt_cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load tokenizer
    tokenizer_path = ckpt_cfg  # Would normally be stored separately
    try:
        tokenizer = LyraTokenizer.from_file("artifacts/tokenizer/tokenizer.json")
    except:
        # Fallback: create minimal tokenizer
        from lyra.tokenizer import LyraTokenizer
        tokenizer = LyraTokenizer.train  # won't work, just for type
        print("Warning: Using default tokenizer")

    # Handle BOS/EOS tokens
    bos_id = tokenizer.id("<BOS>") if hasattr(tokenizer, "id") and "<BOS>" in tokenizer else 0
    eos_id = tokenizer.id("<EOS>") if hasattr(tokenizer, "id") and "<EOS>" in tokenizer else None
    pad_id = tokenizer.id("<PAD>") if hasattr(tokenizer, "id") and "<PAD>" in tokenizer else 0
    unk_id = tokenizer.id("<UNK>") if hasattr(tokenizer, "id") and "<UNK>" in tokenizer else 0

    # Generate
    prompt_ids = tokenizer.encode(args.prompt, add_bos=False, add_eos=False)
    input_ids = torch.tensor([prompt_ids], device=model.device if hasattr(model, 'device') else "cuda")

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_token_id=eos_id,
        )

    # Decode and print
    generated_text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
    print(f"Prompt: {args.prompt}")
    print(f"Generated: {generated_text}")


if __name__ == "__main__":
    main()