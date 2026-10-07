"""Single-source dataset builder: Den4ikAI/russian_dialogues_2 only.

The training chain is intentionally restricted to exactly one corpus:

    Den4ikAI/russian_dialogues_2 -> clean -> dedup -> train/validation/test

There is NO other source and NO fallback. If the raw file cannot be obtained
(local cache missing and Hugging Face download fails) the build stops with an
explicit error instead of silently switching to a different dataset.

The whole available corpus is used: no `max_samples`, no reservoir sampling and
no per-source caps in the default path. `--limit` exists only for tests and
debugging and is recorded in the build status whenever it is used.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import re
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lyra.conversations import clean_text, duplicate_key, stats

# --- The one and only training source -------------------------------------
SOURCE_NAME = "ru_chat"
DATASET_REPO = "Den4ikAI/russian_dialogues_2"
DATASET_LICENSE = "mit"
DATASET_REVISION = "8ce8d669a3f749ec8aa03ea01c475012ef210866"
RAW_FILENAME = "dataset.jsonl.gz"
LOCAL_RAW = Path("data/cache/russian_dialogues_2.jsonl.gz")
CACHE_DIR = Path("data/cache")
ALLOWED_SOURCES = frozenset({SOURCE_NAME})
BUILD_SCHEMA = 1
DATASET_VERSION = "lyra-russian-dialogues-0.1"

# --- Cleaning policy -------------------------------------------------------
# Deliberately conservative: only structurally broken or meaningless rows are
# dropped so that as much of the Russian corpus as possible survives.
MIN_MESSAGES = 2
MAX_CONV_CHARS = 8000
MAX_MESSAGE_CHARS = 3000
MIN_CYRILLIC_SHARE = 0.25      # of letters; the corpus must stay Russian
MAX_URL_MESSAGE_CHARS = 40     # a message that is only a link is technical junk
MIN_DISTINCT_CHARS = 5         # "ааааааа..." style degeneration
MIN_LETTER_RATIO = 0.35        # letters / characters of the whole conversation
BUFFER_ROWS = 20000            # bounded shuffle window, not a data limit

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)


class SourceUnavailableError(RuntimeError):
    """Raised when Den4ikAI/russian_dialogues_2 cannot be obtained."""


def resolve_raw_file(refresh: bool = False) -> tuple[Path, str]:
    """Return the raw dataset file and how it was obtained.

    Local cache first, then a pinned Hugging Face download. Failure raises
    SourceUnavailableError - never a different dataset.
    """
    if not refresh and LOCAL_RAW.is_file() and LOCAL_RAW.stat().st_size > 0:
        return LOCAL_RAW, "local-cache"
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - environment problem
        raise SourceUnavailableError(
            f"Cannot download {DATASET_REPO}: huggingface-hub is not installed "
            "(pip install huggingface-hub)."
        ) from exc
    try:
        path = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=RAW_FILENAME,
            repo_type="dataset",
            revision=DATASET_REVISION,
            cache_dir=str(CACHE_DIR),
        )
    except Exception as exc:
        raise SourceUnavailableError(
            f"Dataset {DATASET_REPO} is unavailable: {type(exc).__name__}: {exc}\n"
            "The training pipeline has exactly one source, so there is no fallback "
            "dataset. Restore the local copy at "
            f"{LOCAL_RAW} or enable network access and retry."
        ) from exc
    return Path(path), f"huggingface:{DATASET_REVISION}"


def iter_raw_records(path: Path) -> Iterator[list[str]]:
    """Yield the raw `sample` utterance lists from the gzipped JSONL corpus."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample = rec.get("sample") if isinstance(rec, dict) else None
            if isinstance(sample, list):
                yield sample


def clean_sample(sample: list[Any]) -> tuple[list[dict[str, str]] | None, str | None]:
    """Clean one raw dialogue. Returns (messages, None) or (None, reason)."""
    if len(sample) < MIN_MESSAGES:
        return None, "too_few_messages"
    if any(not isinstance(t, str) for t in sample):
        return None, "non_string_message"
    if any("\ufffd" in t for t in sample):
        return None, "encoding_damage"

    texts: list[str] = []
    for raw_text in sample:
        text = clean_text(raw_text)
        if not text:
            return None, "empty_message_after_clean"
        # A message that is only a link is technical junk, not conversation.
        if len(text) <= MAX_URL_MESSAGE_CHARS and URL_RE.fullmatch(text):
            return None, "link_only_message"
        texts.append(text)

    joined = " ".join(texts)
    letters = [c for c in joined if c.isalpha()]
    if not letters:
        return None, "no_letters"
    # Case-insensitive: full-uppercase Cyrillic messages are still Russian.
    lowered = [c.lower() for c in letters]
    cyrillic = sum(("а" <= c <= "я") or c in "ёъыэ" for c in lowered) / len(letters)
    if cyrillic < MIN_CYRILLIC_SHARE:
        return None, "not_russian"
    if len(letters) / len(joined) < MIN_LETTER_RATIO:
        return None, "symbol_soup"
    if len(set(joined)) < MIN_DISTINCT_CHARS and len(joined) > 20:
        return None, "meaningless_repetition"
    if len(joined) > MAX_CONV_CHARS:
        return None, "conversation_too_long"
    if any(len(t) > MAX_MESSAGE_CHARS for t in texts):
        return None, "message_too_long"

    # Alternate roles: the raw corpus is an ordered chain of utterances.
    messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": text}
        for i, text in enumerate(texts)
    ]
    return {"messages": messages, "lang": "ru", "source": SOURCE_NAME}, None


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


class SplitWriter:
    """Bounded shuffle window + deterministic hash routing into three splits."""

    def __init__(self, out: Path, seed: int, val_frac: float, test_frac: float):
        self.paths = {name: out / f"{name}.jsonl" for name in ("train", "validation", "test")}
        self.handles = {name: None for name in self.paths}
        self.seed = seed
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.buffer: list[tuple[str, dict[str, Any]]] = []
        self.flushes = 0
        self.written = Counter()
        self.chars = Counter()
        self.messages = Counter()

    @staticmethod
    def _bucket(key: str) -> int:
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % 100_000

    def _split_for(self, key: str) -> str:
        bucket = self._bucket(key)
        if bucket < self.test_frac * 100_000:
            return "test"
        if bucket < (self.test_frac + self.val_frac) * 100_000:
            return "validation"
        return "train"

    def _handle(self, name: str):
        if self.handles[name] is None:
            self.paths[name].parent.mkdir(parents=True, exist_ok=True)
            self.handles[name] = self.paths[name].open("w", encoding="utf-8", newline="\n")
        return self.handles[name]

    def add(self, key: str, row: dict[str, Any]) -> None:
        self.buffer.append((key, row))
        if len(self.buffer) >= BUFFER_ROWS:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        random.Random(self.seed + self.flushes).shuffle(self.buffer)
        for key, row in self.buffer:
            split = self._split_for(key)
            self._handle(split).write(json.dumps(row, ensure_ascii=False) + "\n")
            self.written[split] += 1
            self.messages[split] += len(row["messages"])
            self.chars[split] += sum(len(m["content"]) for m in row["messages"])
        self.buffer.clear()
        self.flushes += 1

    def close(self) -> None:
        self.flush()
        for fh in self.handles.values():
            if fh is not None:
                fh.close()


def build_status_path(out: Path) -> Path:
    return out / "build_status.json"


def existing_build_ok(out: Path, seed: int, val_frac: float, test_frac: float, limit: int) -> bool:
    """Reuse a completed build with identical parameters (fast Kaggle re-runs)."""
    status_path = build_status_path(out)
    if not status_path.is_file():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    expected = {
        "schema": BUILD_SCHEMA,
        "build_state": "SUCCESS",
        "source": SOURCE_NAME,
        "repo": DATASET_REPO,
        "seed": seed,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "limit": limit,
    }
    if any(status.get(k) != v for k, v in expected.items()):
        return False
    return all((out / f"{name}.jsonl").is_file() for name in ("train", "validation", "test"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/processed")
    parser.add_argument(
        "--sources",
        default=SOURCE_NAME,
        help="Must be 'ru_chat' (Den4ikAI/russian_dialogues_2). This pipeline is single-source.",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--val-frac", type=float, default=0.01)
    parser.add_argument("--test-frac", type=float, default=0.01)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="DEBUG ONLY: stop after N accepted dialogues (0 = the full corpus, the default).",
    )
    parser.add_argument("--refresh-raw", action="store_true", help="Re-download the raw corpus.")
    parser.add_argument("--force", action="store_true", help="Rebuild even if a completed build exists.")
    args = parser.parse_args(argv)

    requested = [s.strip() for s in args.sources.split(",") if s.strip()]
    if set(requested) != ALLOWED_SOURCES:
        parser.error(
            "This project trains on exactly one dataset: Den4ikAI/russian_dialogues_2 "
            f"(source key 'ru_chat'). Got --sources={args.sources!r}. "
            "Remove every other source; mixing datasets is not supported."
        )
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    if not 0 < args.val_frac < 1 or not 0 < args.test_frac < 1 or args.val_frac + args.test_frac >= 1:
        parser.error("--val-frac and --test-frac must be positive and sum to less than 1")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not args.force and existing_build_ok(out, args.seed, args.val_frac, args.test_frac, args.limit):
        status = json.loads(build_status_path(out).read_text(encoding="utf-8-sig"))
        print(json.dumps({"status": "SUCCESS_REUSED", "splits": status.get("splits", {})}, ensure_ascii=False))
        return 0

    started = time.time()
    try:
        raw_path, origin = resolve_raw_file(refresh=args.refresh_raw)
    except SourceUnavailableError as exc:
        failure = {
            "schema": BUILD_SCHEMA,
            "build_state": "FAILED_SOURCE_UNAVAILABLE",
            "source": SOURCE_NAME,
            "repo": DATASET_REPO,
            "error": str(exc),
            "requested_sources": requested,
            "failed_sources": [SOURCE_NAME],
            "fallback_used": False,
            "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        atomic_json(build_status_path(out), failure)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(json.dumps({"raw_file": str(raw_path), "origin": origin, "dataset": DATASET_REPO}, ensure_ascii=False))

    writer = SplitWriter(out, args.seed, args.val_frac, args.test_frac)
    rejections: Counter[str] = Counter()
    seen_keys: set[int] = set()
    duplicates = 0
    original = 0
    accepted = 0
    stop = False

    for sample in iter_raw_records(raw_path):
        original += 1
        if original % 200_000 == 0:
            print(json.dumps({"scanned": original, "accepted": accepted, "duplicates": duplicates}), flush=True)
        row, reason = clean_sample(sample)
        if row is None:
            rejections[reason] += 1
            continue
        key = duplicate_key(row["messages"])
        key_hash = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big")
        if key_hash in seen_keys:
            duplicates += 1
            continue
        seen_keys.add(key_hash)
        writer.add(key, row)
        accepted += 1
        if args.limit and accepted >= args.limit:
            stop = True
            break

    writer.close()
    if accepted == 0:
        atomic_json(build_status_path(out), {
            "schema": BUILD_SCHEMA, "build_state": "FAILED_NO_VALID_DATA",
            "source": SOURCE_NAME, "repo": DATASET_REPO, "origin": origin,
            "original_dialogues": original, "rejections": dict(rejections),
            "fallback_used": False,
            "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        print("ERROR: no dialogue survived cleaning; the splits were not published.", file=sys.stderr)
        return 3

    # All three splits must exist even when a hash bucket received no row.
    for path in writer.paths.values():
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8", newline="\n")

    splits = {}
    for name in ("train", "validation", "test"):
        path = writer.paths[name]
        splits[name] = {
            "conversations": writer.written[name],
            "messages": writer.messages[name],
            "characters": writer.chars[name],
            "bytes": path.stat().st_size if path.is_file() else 0,
        }

    removed = sum(rejections.values()) + duplicates
    metadata = {
        "dataset_version": DATASET_VERSION,
        "schema": BUILD_SCHEMA,
        "source_names": [SOURCE_NAME],
        "dataset": {
            "name": DATASET_REPO,
            "repo": DATASET_REPO,
            "revision": DATASET_REVISION,
            "license": DATASET_LICENSE,
            "raw_file": str(raw_path),
            "raw_origin": origin,
            "raw_bytes": raw_path.stat().st_size if raw_path.is_file() else 0,
        },
        "build_state": "SUCCESS",
        "requested_sources": requested,
        "failed_sources": [],
        "fallback_used": False,
        "single_source": True,
        "seed": args.seed,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "limit": args.limit,
        "limit_note": None if args.limit == 0 else "DEBUG LIMIT ACTIVE: this build is NOT the full corpus",
        "original_dialogues": original,
        "rejections_by_reason": dict(rejections),
        "duplicates_removed": duplicates,
        "removed_total": removed,
        "clean_dialogues": accepted,
        "removed_percent": round(removed / original * 100, 3) if original else 0.0,
        "clean_percent": round(accepted / original * 100, 3) if original else 0.0,
        "messages_total": sum(s["messages"] for s in splits.values()),
        "characters_total": sum(s["characters"] for s in splits.values()),
        "splits": splits,
        "split_stats": {},
        "elapsed_seconds": round(time.time() - started, 1),
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Per-split conversation statistics (turn counts, languages, lengths).
    for name in ("train", "validation", "test"):
        rows = []
        with writer.paths[name].open(encoding="utf-8") as fh:
            for line in fh:
                rows.append(json.loads(line))
                if len(rows) >= 20000:
                    break
        metadata["split_stats"][name] = stats(rows)

    atomic_json(out / "dataset_stats.json", metadata)
    status = {
        "schema": BUILD_SCHEMA,
        "build_state": "SUCCESS",
        "source": SOURCE_NAME,
        "repo": DATASET_REPO,
        "revision": DATASET_REVISION,
        "license": DATASET_LICENSE,
        "requested_sources": requested,
        "skipped_sources": [],
        "failed_sources": [],
        "fallback_used": False,
        "single_source": True,
        "seed": args.seed,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "limit": args.limit,
        "original_dialogues": original,
        "clean_dialogues": accepted,
        "duplicates_removed": duplicates,
        "rejections_by_reason": dict(rejections),
        "splits": {k: v["conversations"] for k, v in splits.items()},
        "elapsed_seconds": metadata["elapsed_seconds"],
        "stop_after_limit": stop,
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(build_status_path(out), status)

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
