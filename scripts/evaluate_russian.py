#!/usr/bin/env python3
"""Automatic Russian conversational evaluation of a Lyra checkpoint.

Loads a checkpoint in the current (fresh) process and grades its replies on a
fixed Russian prompt set. Checks, per answer and overall:

  * russian language        - cyrillic share of the answer letters
  * decode quality          - no replacement chars, no control/mojibake garbage
  * <UNK> count             - byte-BPE must never emit <UNK>
  * repetition              - repeated n-gram loops / low unique-token ratio
  * empty answers           - answers must contain real text
  * special token leakage   - <USER>, <ASSISTANT>, <END>, ... must not surface
  * answer length           - too short / hitting the token cap without <END>
  * prompt copying          - the answer must not simply repeat the prompt
  * garbage symbols         - share of unexpected characters

Usage:
  python scripts/evaluate_russian.py --checkpoint checkpoints/run/final.pt \
      --tokenizer artifacts/tokenizer/tokenizer.json --out logs/russian_eval.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lyra.config import LyraConfig
from lyra.generation import GENERATION_DEFAULTS, format_chat, generate_response
from lyra.model import LyraModel
from lyra.tokenizer import SPECIAL_TOKENS, LyraTokenizer

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
except Exception:  # pragma: no cover
    pass

DEFAULT_PROMPTS = [
    "Привет",
    "Привет, как дела?",
    "Здравствуйте, рад тебя видеть",
    "Что ты сейчас делаешь?",
    "Как прошёл твой день?",
    "Мне сегодня грустно",
    "Давай немного поговорим",
    "Мне нравится играть в игры",
]

# Short multi-turn probes: role ordering and context use.
DEFAULT_CONVERSATIONS = [
    [{"role": "user", "content": "Привет"}, {"role": "assistant", "content": "Привет! Как дела?"},
     {"role": "user", "content": "Нормально, а у тебя как?"}],
    [{"role": "user", "content": "Чем занимаешься?"}, {"role": "assistant", "content": "Общаюсь с тобой."},
     {"role": "user", "content": "А что любишь делать свободным временем?"}],
]

# Russian language / quality thresholds.
MIN_CYRILLIC_SHARE = 0.50
MIN_UNIQUE_TOKEN_RATIO = 0.35
MAX_NGRAM_REPEAT = 3          # same n-gram repeated this many times in a row = loop
MAX_GARBAGE_SHARE = 0.20
MIN_ANSWER_CHARS = 1

GARBAGE_CATEGORIES = {"Cc", "Cf", "Co", "Cs"}


def cyrillic_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    # Case-insensitive: "ПРИВЕТ" is Russian too.
    lowered = [c.lower() for c in letters]
    return sum(("а" <= c <= "я") or c in "ёъыэ" for c in lowered) / len(letters)


def garbage_share(text: str) -> float:
    if not text:
        return 0.0
    bad = 0
    for ch in text:
        if ch in "\n\r\t ":
            continue
        if unicodedata.category(ch) in GARBAGE_CATEGORIES:
            bad += 1
        elif ch == "\ufffd":
            bad += 1
    return bad / len(text)


def ngram_loop(text: str, n: int = 6, repeats: int = MAX_NGRAM_REPEAT) -> bool:
    """True when one n-gram (word level) is repeated back to back, or a long
    character block is repeated - both are degenerate generations."""
    tokens = text.split()
    if len(tokens) >= n * repeats:
        for i in range(len(tokens) - n * repeats + 1):
            chunk = tokens[i:i + n]
            if all(tuple(tokens[i + k * n:i + (k + 1) * n]) == tuple(chunk) for k in range(repeats)):
                return True
    if len(set(tokens)) <= 2 and len(tokens) >= 12:
        return True
    return _char_loop(text)


def _char_loop(text: str, n: int = 20, repeats: int = 4) -> bool:
    if len(text) < n * repeats:
        return False
    for i in range(0, len(text) - n * repeats + 1, max(1, n // 2)):
        chunk = text[i:i + n]
        if text[i:i + n * repeats] == chunk * repeats:
            return True
    return False


def repeated_ngram_ratio(text: str, n: int = 3) -> float:
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(grams)
    return (len(grams) - len(counts)) / len(grams)


def evaluate_answer(prompt: str, answer: str, generated_ids: list[int], tok: LyraTokenizer,
                    max_new_tokens: int) -> dict:
    answer = answer.strip()
    tokens = tok.encode(answer)
    unk_ids = {tok.id("<UNK>")}
    unk_count = sum(1 for i in generated_ids if i in unk_ids)
    leaks = [s for s in SPECIAL_TOKENS if s in answer]
    prompt_norm = re.sub(r"\W+", " ", prompt.lower(), flags=re.UNICODE).split()
    answer_norm = re.sub(r"\W+", " ", answer.lower(), flags=re.UNICODE).split()
    echo = bool(answer_norm) and (
        answer_norm == prompt_norm
        or (set(prompt_norm) <= set(answer_norm) and len(answer_norm) <= len(prompt_norm) + 1)
    )
    unique_ratio = (len(set(tokens)) / len(tokens)) if tokens else 0.0
    cyr = cyrillic_share(answer)
    garbage = garbage_share(answer)
    checks = {
        "not_empty": len(answer) >= MIN_ANSWER_CHARS,
        "no_unk": unk_count == 0 and "<UNK>" not in answer,
        "no_special_tokens": not leaks,
        "russian": cyr >= MIN_CYRILLIC_SHARE,
        "decode_clean": "\ufffd" not in answer and garbage <= MAX_GARBAGE_SHARE,
        "no_repetition_loop": (not ngram_loop(answer)) and unique_ratio >= MIN_UNIQUE_TOKEN_RATIO,
        "not_prompt_copy": not echo,
        "reasonable_length": len(answer) >= 4,
    }
    return {
        "prompt": prompt,
        "answer": answer,
        "answer_tokens": len(tokens),
        "generated_ids": len(generated_ids),
        "cyrillic_share": round(cyr, 3),
        "garbage_share": round(garbage, 4),
        "unique_token_ratio": round(unique_ratio, 3),
        "repeated_ngram_ratio": round(repeated_ngram_ratio(answer), 3),
        "unk_ids": unk_count,
        "special_tokens_leaked": leaks,
        "hit_token_cap": len(generated_ids) >= max_new_tokens,
        "checks": checks,
        "passed": all(checks.values()),
    }


def load_model(checkpoint: Path, tokenizer_path: Path, device: str):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = LyraConfig(**{k: v for k, v in ckpt.get("config", {}).items() if k in LyraConfig.__dataclass_fields__})
    tok = LyraTokenizer.from_file(tokenizer_path)
    if tok.vocab_size != cfg.vocab_size:
        raise SystemExit(
            f"ERROR: tokenizer vocabulary {tok.vocab_size} != model vocabulary {cfg.vocab_size}; "
            "the checkpoint and the tokenizer are not compatible."
        )
    model = LyraModel(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    return model, cfg, tok, ckpt


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", default="artifacts/tokenizer/tokenizer.json")
    p.add_argument("--out", default="logs/russian_eval.json")
    p.add_argument("--prompts", nargs="*", default=None, help="Override the Russian probe prompt list")
    p.add_argument("--max-new-tokens", type=int, default=GENERATION_DEFAULTS["max_new_tokens"])
    p.add_argument("--temperature", type=float, default=GENERATION_DEFAULTS["temperature"])
    p.add_argument("--top-k", type=int, default=GENERATION_DEFAULTS["top_k"])
    p.add_argument("--top-p", type=float, default=GENERATION_DEFAULTS["top_p"])
    p.add_argument("--repetition-penalty", type=float, default=GENERATION_DEFAULTS["repetition_penalty"])
    p.add_argument("--multi-turn", action="store_true", default=True)
    p.add_argument("--no-multi-turn", dest="multi_turn", action="store_false")
    a = p.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, tok, ckpt = load_model(Path(a.checkpoint), Path(a.tokenizer), device)
    prompts = a.prompts if a.prompts else list(DEFAULT_PROMPTS)

    results = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        prompt_ids = format_chat(messages, tok, add_assistant_trigger=True)
        out = model.generate(
            torch.tensor([prompt_ids], device=device),
            max_new_tokens=a.max_new_tokens,
            temperature=a.temperature,
            top_k=a.top_k,
            top_p=a.top_p,
            repetition_penalty=a.repetition_penalty,
            eos_token_id=tok.id("<END>"),
        )
        gen = out[0, len(prompt_ids):].tolist()
        end = tok.id("<END>")
        if end in gen:
            gen = gen[: gen.index(end)]
        answer = tok.decode(gen)
        results.append(evaluate_answer(prompt, answer, gen, tok, a.max_new_tokens))

    multi_turn_results = []
    if a.multi_turn:
        for messages in DEFAULT_CONVERSATIONS:
            answer = generate_response(
                model, tok, messages, device,
                max_new_tokens=a.max_new_tokens, temperature=a.temperature,
                top_k=a.top_k, top_p=a.top_p, repetition_penalty=a.repetition_penalty,
            )
            prompt = " | ".join(m["content"] for m in messages)
            multi_turn_results.append(evaluate_answer(prompt, answer, tok.encode(answer), tok, a.max_new_tokens))

    all_results = results + multi_turn_results
    total = len(all_results)
    check_rates = {}
    for name in all_results[0]["checks"] if all_results else []:
        check_rates[name] = round(sum(1 for r in all_results if r["checks"][name]) / total, 3) if total else 0.0

    report = {
        "checkpoint": str(a.checkpoint),
        "tokenizer": str(a.tokenizer),
        "step": ckpt.get("step"),
        "tokens_seen": ckpt.get("tokens_seen"),
        "best_validation_loss": ckpt.get("best_validation_loss"),
        "model_parameters": model.parameter_count(),
        "vocab_size": tok.vocab_size,
        "device": device,
        "generation": {
            "temperature": a.temperature, "top_k": a.top_k, "top_p": a.top_p,
            "repetition_penalty": a.repetition_penalty, "max_new_tokens": a.max_new_tokens,
            "eos_token": "<END>",
        },
        "answers": total,
        "check_pass_rates": check_rates,
        "mean_cyrillic_share": round(sum(r["cyrillic_share"] for r in all_results) / total, 3) if total else 0,
        "mean_unique_token_ratio": round(sum(r["unique_token_ratio"] for r in all_results) / total, 3) if total else 0,
        "total_unk_ids": sum(r["unk_ids"] for r in all_results),
        "empty_answers": sum(1 for r in all_results if not r["checks"]["not_empty"]),
        "special_token_leaks": sum(len(r["special_tokens_leaked"]) for r in all_results),
        "repetition_loops": sum(1 for r in all_results if not r["checks"]["no_repetition_loop"]),
        "prompt_copies": sum(1 for r in all_results if not r["checks"]["not_prompt_copy"]),
        "single_turn": results,
        "multi_turn": multi_turn_results,
    }
    required = ("not_empty", "no_unk", "no_special_tokens", "russian", "decode_clean",
                "not_prompt_copy", "reasonable_length")
    failed_required = [name for name in required if check_rates.get(name, 0.0) < 1.0]
    repetition_ok = check_rates.get("no_repetition_loop", 0.0) >= 0.75
    report["failed_checks"] = failed_required + ([] if repetition_ok else ["no_repetition_loop"])
    report["overall"] = "PASS" if not report["failed_checks"] else "FAIL"

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print(f"Russian evaluation: {report['overall']}  (checkpoint step {report['step']})")
    print(json.dumps({"check_pass_rates": check_rates,
                      "mean_cyrillic_share": report["mean_cyrillic_share"],
                      "total_unk_ids": report["total_unk_ids"],
                      "empty_answers": report["empty_answers"],
                      "special_token_leaks": report["special_token_leaks"],
                      "repetition_loops": report["repetition_loops"],
                      "failed_checks": report["failed_checks"]}, ensure_ascii=False, indent=2))
    print("-" * 70)
    for r in results:
        print(f"you> {r['prompt']}")
        print(f"lyra> {r['answer']}")
        print(f"   cyr={r['cyrillic_share']} uniq={r['unique_token_ratio']} "
              f"passed={r['passed']} cap={r['hit_token_cap']}")
    for r in multi_turn_results:
        print(f"dialog> {r['prompt']}")
        print(f"lyra> {r['answer']}")
        print(f"   cyr={r['cyrillic_share']} uniq={r['unique_token_ratio']} passed={r['passed']}")
    print("=" * 70)
    print(f"Report: {out_path}")
    return 0 if report["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
