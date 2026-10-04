"""Saturn Cloud VRAM sizing for Lyra training from scratch.

Estimates single-GPU training memory for a config (from-scratch AdamW,
AMP, per-layer activation checkpointing as implemented in lyra/model.py)
and picks the biggest tier config that fits the detected GPU.

Memory model (conservative):
  static  = params * 16 bytes  (bf16/fp16 weights + grads + fp32 Adam m/v + margin)
  acts    = layers * micro_batch * seq_len * hidden * 8 bytes (checkpointed)
  logits  = micro_batch * seq_len * vocab * 2 bytes
  overhead = 1.5 GB (CUDA context, fragmentation, val batches)

A tier fits when estimate <= 0.85 * total VRAM.

Usage:
  python saturn/sizing.py                # detect GPU (needs torch+CUDA) and recommend
  python saturn/sizing.py --vram 24      # recommend for 24 GiB without a GPU
  python saturn/sizing.py --table        # print the full tier table
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.count_parameters import compute_param_count

# (tier name, min VRAM GiB, config file, default steps)
TIERS = [
    ("t4", 12, "configs/saturn_t4.json", 2000),
    ("a10g", 20, "configs/saturn_a10g.json", 2500),
    ("a100", 34, "configs/saturn_a100.json", 3000),
    ("h100", 70, "configs/saturn_h100.json", 4000),
]

BYTES_PER_PARAM = 16
OVERHEAD_GB = 1.5
FIT_RATIO = 0.85


def estimate_gb(cfg: dict, seq_len: int = 1024, micro_batch: int = 1) -> float:
    params = compute_param_count(cfg)
    static_gb = params * BYTES_PER_PARAM / 1e9
    acts_gb = (
        cfg["num_layers"] * micro_batch * seq_len * cfg["hidden_size"] * 8 / 1e9
    )
    logits_gb = micro_batch * seq_len * cfg["vocab_size"] * 2 / 1e9
    return static_gb + acts_gb + logits_gb + OVERHEAD_GB


def load_tier_table(project_root: Path) -> list[dict]:
    rows = []
    for name, min_vram, cfg_file, steps in TIERS:
        cfg = json.loads((project_root / cfg_file).read_text(encoding="utf-8-sig"))
        params = compute_param_count(cfg)
        rows.append(
            {
                "tier": name,
                "min_vram_gib": min_vram,
                "config": cfg_file,
                "params": params,
                "est_gb": estimate_gb(cfg),
                "steps": steps,
            }
        )
    return rows


def recommend(vram_gb: float, rows: list[dict]) -> dict:
    fitting = [r for r in rows if r["est_gb"] <= vram_gb * FIT_RATIO]
    if not fitting:
        raise RuntimeError(
            f"No tier fits {vram_gb:.1f} GiB VRAM (need >= 12 GiB for the t4 tier)."
        )
    return max(fitting, key=lambda r: r["params"])


def detect_vram_gb() -> float:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; pass --vram explicitly.")
    return torch.cuda.get_device_properties(0).total_memory / 1024**3


def main() -> None:
    p = argparse.ArgumentParser(description="Lyra Saturn Cloud VRAM sizing")
    p.add_argument("--vram", type=float, default=None, help="Total GPU VRAM in GiB")
    p.add_argument("--table", action="store_true", help="Print tier table and exit")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    rows = load_tier_table(root)
    print(f"{'tier':<6}{'minVRAM':>9}{'params':>14}{'est.VRAM':>10}  config")
    for r in rows:
        print(
            f"{r['tier']:<6}{r['min_vram_gib']:>7}GB{r['params']:>14,}"
            f"{r['est_gb']:>9.1f}GB  {r['config']}"
        )
    if args.table and args.vram is None:
        return
    vram = args.vram if args.vram is not None else detect_vram_gb()
    rec = recommend(vram, rows)
    print(f"\nDetected VRAM: {vram:.1f} GiB")
    print(
        f"Recommended (max fits): tier={rec['tier']} params={rec['params']:,} "
        f"est={rec['est_gb']:.1f}GB config={rec['config']} steps={rec['steps']}"
    )


if __name__ == "__main__":
    main()
