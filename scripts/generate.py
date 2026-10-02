#!/usr/bin/env python3
"""Text generation script for Lyra model.
Loads a checkpoint and generates text from a prompt.
"""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path so `import lyra` works when running as script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    p.add_argument("--tokenizer", default="artifacts/tokenizer/tokenizer.json")
    p.add_argument("--prompt", default="Once upon a time", help="Generation prompt")
    p.add_argument("--max-tokens", type=int, default=128, help="Max tokens to generate")
    p.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    p.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    p.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling")
    p.add_argument("--eos-token-id", type=int, default=None, help="EOS token ID")
    args = p.parse_args()

    if not HAS_LYRA:
        print("Error: could not import lyra package. Run from the project root.")
        sys.exit(1)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_cfg = dict(ckpt.get("config", {}))
    cfg = LyraConfig(**{k: v for k, v in raw_cfg.items()
                        if k in LyraConfig.__dataclass_fields__})

    # Create model
    model = LyraModel(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load tokenizer
    tok_path = Path(args.tokenizer)
    if not tok_path.exists():
        print(f"Error: Tokenizer not found: {tok_path}")
        sys.exit(1)
    tokenizer = LyraTokenizer.from_file(str(tok_path))

    # EOS token for stopping; fall back to None if the vocab lacks it.
    try:
        eos_id = tokenizer.id("<EOS>")
    except KeyError:
        eos_id = None
    if args.eos_token_id is not None:
        eos_id = args.eos_token_id

    # Generate
    prompt_ids = tokenizer.encode(args.prompt, add_bos=False, add_eos=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    input_ids = torch.tensor([prompt_ids], device=device)

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