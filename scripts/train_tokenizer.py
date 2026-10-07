"""Train byte-BPE on the Den4ikAI/russian_dialogues_2 train split only.

The tokenizer is learned from Russian conversation text produced by
``scripts/prepare_data.py`` (single source, no other corpus). After training it
is verified on real Russian probe sentences: round-trip encode/decode must be
lossless, no <UNK> may appear and every special token must be a single id.

Token counts for the prepared splits are written to
``artifacts/tokenizer/tokenizer_stats.json``.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lyra.tokenizer import SPECIAL_TOKENS, LyraTokenizer

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
except Exception:  # pragma: no cover
    pass

RUSSIAN_PROBES = [
    "Привет",
    "Привет, как дела?",
    "Здравствуйте! Рад тебя видеть.",
    "Что ты сейчас делаешь?",
    "Мне сегодня очень грустно",
    "Давай немного поговорим",
    "Я играю в Roblox",
    "А что ты думаешь про эту погоду? Дождь с самого утра, совсем выйти некуда.",
    "Сегодня был такой день, что хочется забыть его поскорее.",
]


def build_corpus(files: list[str], out_path: Path) -> int:
    """Stream message bodies from JSONL conversations into one UTF-8 corpus."""
    messages = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        for src in files:
            with open(src, encoding="utf-8-sig", errors="replace") as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    for m in row.get("messages", []):
                        # Keep original newlines: byte-level BPE must see real
                        # line breaks or multiline inference meets rare bytes.
                        if m.get("content"):
                            out.write(m["content"] + "\n")
                            messages += 1
    return messages


def verify(tok: LyraTokenizer) -> dict:
    """Round-trip and special-token verification on Russian probe sentences."""
    results = {"probes": [], "ok": True}
    for special in SPECIAL_TOKENS:
        try:
            tok.id(special)
        except KeyError:
            results["ok"] = False
            results.setdefault("missing_special_tokens", []).append(special)
    # Every special token must encode to exactly one id.
    for special in SPECIAL_TOKENS:
        ids = tok.backend.encode(special).ids
        if ids != [tok.id(special)]:
            results["ok"] = False
            results.setdefault("non_atomic_specials", {})[special] = ids
    for text in RUSSIAN_PROBES:
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        tokens = tok.tokenize(text)
        unknown = sum(1 for t in tokens if t == "<UNK>")
        entry = {
            "text": text,
            "tokens": len(ids),
            "token_sample": tokens[:12],
            "roundtrip_ok": decoded == text,
            "unk_tokens": unknown,
            "contains_cyrillic_after_decode": any("а" <= c <= "я" or c in "ё" for c in decoded),
        }
        if not entry["roundtrip_ok"] or unknown:
            results["ok"] = False
        results["probes"].append(entry)
    return results


def count_split(path: Path, tok: LyraTokenizer) -> dict:
    """Tokenize one split and count conversations/messages/tokens/characters."""
    conversations = messages = tokens = characters = 0
    batch: list[str] = []
    started = time.time()

    def flush() -> None:
        nonlocal tokens
        if not batch:
            return
        for enc in tok.backend.encode_batch(batch):
            tokens += len(enc.ids)
        batch.clear()

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            conversations += 1
            for m in row.get("messages", []):
                text = m.get("content", "")
                if not text:
                    continue
                messages += 1
                characters += len(text)
                batch.append(text)
                if len(batch) >= 4096:
                    flush()
    flush()
    return {
        "file": str(path),
        "conversations": conversations,
        "messages": messages,
        "characters": characters,
        "tokens": tokens,
        "seconds": round(time.time() - started, 1),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="JSONL glob of the TRAIN split (Russian conversations)")
    p.add_argument("--vocab-size", type=int, default=16384)
    p.add_argument("--expect-vocab-size", type=int, default=None,
                   help="Fail unless the trained vocabulary matches this size (use the model config value).")
    p.add_argument("--out", default="artifacts/tokenizer/tokenizer.json")
    p.add_argument("--stats-out", default="artifacts/tokenizer/tokenizer_stats.json")
    p.add_argument("--count-splits", default="auto",
                   help="'auto' = train/validation/test next to --data, 'none' = skip counting, or a JSONL glob")
    a = p.parse_args(argv)

    files = sorted(glob.glob(a.data, recursive=True))
    if not files:
        raise FileNotFoundError(f"No training files matched {a.data!r}")
    if not any("train" in Path(f).name for f in files):
        print("WARNING: tokenizer input does not look like a train split; "
              "training on validation/test text leaks eval data into the vocabulary.")

    with tempfile.TemporaryDirectory() as td:
        corpus = Path(td) / "corpus.txt"
        message_count = build_corpus(files, corpus)
        started = time.time()
        tok = LyraTokenizer.train([corpus], a.vocab_size)
        train_seconds = round(time.time() - started, 1)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tok.save(a.out)

    report = {
        "tokenizer_file": str(a.out),
        "trained_on_files": files,
        "trained_on_messages": message_count,
        "train_seconds": train_seconds,
        "vocab_size": tok.vocab_size,
        "requested_vocab_size": a.vocab_size,
        "special_tokens": list(SPECIAL_TOKENS),
        "verification": verify(tok),
    }

    if a.expect_vocab_size and tok.vocab_size != a.expect_vocab_size:
        print(f"ERROR: tokenizer vocab {tok.vocab_size} != expected {a.expect_vocab_size}", file=sys.stderr)
        report["verification"]["ok"] = False

    # Token accounting for the prepared splits (drives the training step plan).
    if a.count_splits != "none":
        if a.count_splits == "auto":
            parent = Path(files[0]).parent
            count_files = [str(parent / name) for name in ("train.jsonl", "validation.jsonl", "test.jsonl")
                           if (parent / name).is_file()]
        else:
            count_files = sorted(glob.glob(a.count_splits, recursive=True))
        report["splits"] = {}
        for path in count_files:
            stats = count_split(Path(path), tok)
            report["splits"][Path(path).stem] = stats
            print(json.dumps({"counted": Path(path).name, **{k: stats[k] for k in
                                                             ("conversations", "messages", "tokens", "seconds")}},
                             ensure_ascii=False), flush=True)
        report["total_tokens"] = sum(s["tokens"] for s in report["splits"].values())

    Path(a.stats_out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.stats_out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"vocab_size": tok.vocab_size, "file": a.out,
                      "special_tokens": SPECIAL_TOKENS,
                      "verification_ok": report["verification"]["ok"],
                      "stats": a.stats_out}, ensure_ascii=False))

    for probe in report["verification"]["probes"]:
        print(json.dumps(probe, ensure_ascii=False))

    if not report["verification"]["ok"]:
        print("ERROR: tokenizer verification failed (see report above).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
