"""Failure-tolerant, cached and atomic bilingual conversation dataset builder.

Completed source caches are written to data/cache/processed/<source>.jsonl.
Each complete source is immediately reflected in data/processed/intermediate/
and in the output splits. A failed optional source never discards successful
sources; use --refresh-sources to deliberately fetch sources again.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import re
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from lyra.conversations import clean_text, detect_lang, duplicate_key, stats, valid_messages

SOURCES = {
    "oasst": ("OpenAssistant/oasst1", "apache-2.0", "Human-rated conversation trees."),
    "siberian": ("SiberiaSoft/SiberianPersonaChat-2", "mit", "Russian persona/dialogue data; strict template and turn filters."),
    "ru_everyday": ("kukunechka/russian-everyday-dialogues", "cc-by-4.0", "Small native-authored everyday dialogue seed; retain attribution."),
    "ultra": ("HuggingFaceH4/ultrachat_200k", "mit", "Synthetic English assistant conversations; capped optional supplement."),
    "daily": ("roskoN/dailydialog", "cc-by-nc-sa-4.0", "Optional non-commercial/share-alike English daily dialogue."),
    "curated": ("lyra/identity_and_honesty_v0.1", "project-authored", "Repository-authored bilingual examples."),
}
DEFAULT_SOURCES = "oasst,siberian,ru_everyday,ultra"
CACHE_SCHEMA = 3
REVISION: dict[str, str] = {}
SIBERIAN_MARKER = re.compile(r"(?:\u041d\u0435\u0434\u0430\u0432\u043d\u043e\s*,?\s*\u0443\s*\u043c\u0435\u043d\u044f\s*\u0431\u044b\u043b\s*\u0441\u043b\u0435\u0434\u0443\u044e\u0449\u0438\u0439\s*\u0434\u0438\u0430\u043b\u043e\u0433\s*:|\u0434\u0438\u0430\u043b\u043e\u0433\s*:)", re.I)
SIBERIAN_TURN = re.compile(r"(?<!\w)(\u0422\u044b|\u042f)\s*:\s*", re.I)
SIBERIAN_TEMPLATE = re.compile(
    r"(?:\u044f\s+(?:\u043e\u0447\u0435\u043d\u044c\s+\u0443\u043c\u043d\u0430\u044f|\u043f\u0430\u0440\u0435\u043d\u044c\s*,?\s*\u043a\u043e\u043d\u0441\u0443\u043b\u044c\u0442\u0430\u043d\u0442)|\u0432\s+\u044d\u0442\u043e\u043c\s+\u0440\u0430\u0437\u0433\u043e\u0432\u043e\u0440\u0435\s+\u0442\u044b\s+\u0431\u0443\u0434\u0435\u0448\u044c)",
    re.I,
)


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


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_status(path: Path, status: dict[str, Any]) -> None:
    status["updated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_json(path, status)


def pin(repo: str) -> str:
    from huggingface_hub import HfApi

    if repo not in REVISION:
        REVISION[repo] = HfApi().dataset_info(repo).sha
    return REVISION[repo]


def normalize(messages: Any) -> list[dict[str, str]] | None:
    if not isinstance(messages, list):
        return None
    out = []
    aliases = {"prompter": "user", "human": "user", "user": "user", "assistant": "assistant", "system": "system"}
    for message in messages:
        if not isinstance(message, dict):
            return None
        role = aliases.get(str(message.get("role", "")).lower())
        text = clean_text(message.get("content", message.get("text", "")))
        if role is None or not text:
            return None
        out.append({"role": role, "content": text})
    return out


def parse_siberian_dialogue(prompt: str, answer: str, category: str = "") -> list[dict[str, str]] | None:
    """Strip persona boilerplate and recover alternating Ты/Я dialogue turns."""
    if category.lower() == "qa":
        return None
    match = SIBERIAN_MARKER.search(prompt)
    body = prompt[match.end():] if match else prompt
    markers = list(SIBERIAN_TURN.finditer(body))
    if not markers:
        return None
    messages: list[dict[str, str]] = []
    for i, marker in enumerate(markers):
        end = markers[i + 1].start() if i + 1 < len(markers) else len(body)
        content = clean_text(body[marker.end():end])
        role = "user" if marker.group(1).lower() == "\u0442\u044b" else "assistant"
        if content:
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"] = clean_text(messages[-1]["content"] + " " + content)
            else:
                messages.append({"role": role, "content": content})
    answer = clean_text(answer)
    if answer:
        if messages and messages[-1]["role"] == "assistant":
            messages[-1]["content"] = clean_text(messages[-1]["content"] + " " + answer)
        else:
            messages.append({"role": "assistant", "content": answer})
    return messages or None


def rows(source: str, limit: int) -> Iterable[dict[str, Any]]:
    """Yield source records; source-specific network and parser errors bubble to caller."""
    if source == "oasst":
        repo = SOURCES[source][0]
        tree_file = Path("data/cache/oasst/all.trees.jsonl.gz")
        if tree_file.exists():
            REVISION[repo] = "fdf72ae0827c1cda404aff25b6603abec9e3399b"
        else:
            from huggingface_hub import hf_hub_download

            revision = pin(repo)
            tree_file = Path(hf_hub_download(repo_id=repo, filename="2023-04-12_oasst_all.trees.jsonl.gz", repo_type="dataset", revision=revision, cache_dir="data/cache"))

        def quality(node: dict[str, Any]) -> tuple[int, float, int]:
            votes = node.get("emojis") or {}
            labels = node.get("labels") or {}
            helpful = labels.get("helpfulness", {}).get("value", 0)
            rated = labels.get("quality", {}).get("value", 0)
            return int(votes.get("+1", 0)) - int(votes.get("-1", 0)), float(helpful or 0) + float(rated or 0), int(node.get("review_count") or 0)

        emitted = 0
        with gzip.open(tree_file, "rt", encoding="utf-8") as f:
            for line in f:
                tree = json.loads(line)
                root = tree.get("prompt") or {}
                if tree.get("tree_state") != "ready_for_export" or root.get("lang") not in ("ru", "en") or root.get("deleted") or root.get("synthetic"):
                    continue
                chain, node = [root], root
                while node.get("replies") and len(chain) < 100:
                    children = [r for r in node["replies"] if not r.get("deleted") and not r.get("synthetic") and r.get("lang") == root["lang"]]
                    if not children:
                        break
                    node = max(children, key=quality)
                    chain.append(node)
                messages = normalize([{"role": r.get("role", ""), "content": r.get("text", "")} for r in chain])
                if messages:
                    yield {"messages": messages, "lang": root["lang"], "source": source}
                    emitted += 1
                    if emitted >= limit:
                        break

    elif source == "ru_everyday":
        repo = SOURCES[source][0]
        local = Path("data/curated/russian_everyday_dialogues.jsonl")
        if local.exists():
            REVISION[repo] = "3d9c43ca85a50e7a32fa7b05e27c0637a710e988"
            with local.open(encoding="utf-8-sig") as f:
                for i, line in enumerate(f):
                    if i >= limit:
                        break
                    yield json.loads(line)
            return
        from huggingface_hub import hf_hub_download

        revision = pin(repo)
        path = hf_hub_download(repo_id=repo, filename="russian_everyday_dialogues.jsonl", repo_type="dataset", revision=revision, cache_dir="data/cache")
        with open(path, encoding="utf-8-sig") as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                row = json.loads(line)
                messages = normalize([{"role": "user", "content": row.get("user", "")}, {"role": "assistant", "content": row.get("assistant", "")}])
                if messages:
                    yield {"messages": messages, "lang": "ru", "source": source, "attribution": row.get("attribution", "")}

    elif source in ("ultra", "daily", "siberian"):
        from datasets import load_dataset

        repo = SOURCES[source][0]
        revision = pin(repo)
        if source == "ultra":
            ds = load_dataset(repo, split="train_sft", streaming=True, revision=revision, cache_dir="data/cache").shuffle(seed=2026, buffer_size=10000)
        elif source == "daily":
            ds = load_dataset(repo, split="train", streaming=True, revision=revision, cache_dir="data/cache")
        else:
            ds = load_dataset(repo, split="train", streaming=True, revision=revision, cache_dir="data/cache")

        emitted = 0
        seen_daily = set()
        for raw in ds:
            if source == "ultra":
                messages = normalize(raw.get("messages", []))
                messages = [m for m in messages if m["role"] != "system"] if messages else None
                if not messages or len(messages) > 20:
                    continue
                text = " ".join(m["content"] for m in messages)
                if len(text) > 9000 or any(x in text.lower() for x in ("write a 1000 word", "create a comprehensive", "as an ai language model")):
                    continue
                row = {"messages": messages, "lang": detect_lang(text), "source": source}
            elif source == "daily":
                utterances = raw.get("dialog") or raw.get("dialogue")
                if not utterances or len(utterances) < 4:
                    continue
                key = tuple(utterances)
                if key in seen_daily:
                    continue
                seen_daily.add(key)
                messages = normalize([{"role": "user" if i % 2 == 0 else "assistant", "content": text} for i, text in enumerate(utterances)])
                if not messages:
                    continue
                row = {"messages": messages, "lang": "en", "source": source}
            else:
                messages = parse_siberian_dialogue(str(raw.get("input", "")), str(raw.get("output", "")), str(raw.get("name", "")))
                if not messages:
                    continue
                row = {"messages": messages, "lang": "ru", "source": source, "category": raw.get("name", "")}
            yield row
            emitted += 1
            if emitted >= limit:
                break

    elif source == "curated":
        path = Path("data/curated/identity_and_honesty.jsonl")
        if path.exists():
            with path.open(encoding="utf-8-sig") as f:
                for i, line in enumerate(f):
                    if i >= limit:
                        break
                    yield json.loads(line)


def conversation_language(messages: list[dict[str, str]], declared: str | None) -> str | None:
    text = " ".join(m["content"] for m in messages)
    detected = detect_lang(text)
    if detected in ("ru", "en"):
        return detected
    letters = [c.lower() for c in text if c.isalpha()]
    if not letters:
        return None
    cyr = sum("\u0430" <= c <= "\u044f" or c in "\u0451\u044a\u044b\u044d" for c in letters) / len(letters)
    if cyr >= 0.5:
        return "ru"
    if cyr <= 0.12:
        return "en"
    if declared in ("ru", "en"):
        return declared
    return None


def clean_source_row(raw: dict[str, Any], source: str) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    messages = normalize(raw.get("messages"))
    if not messages or not valid_messages(messages):
        return None
    text = " ".join(m["content"] for m in messages)
    lang = conversation_language(messages, raw.get("lang"))
    if lang not in ("ru", "en"):
        return None
    if source == "siberian":
        # The upstream also contains QA and persona-instruction rows. Retain
        # only real-looking multi-turn exchanges; never train on its character
        # profile scaffolding or repeated/template-only fragments.
        if len(messages) < 4 or len(messages) > 20 or len(text) > 12000:
            return None
        if SIBERIAN_TEMPLATE.search(text):
            return None
        normalized_messages = [re.sub(r"\W+", " ", m["content"].lower()).strip() for m in messages]
        if len(set(normalized_messages)) / len(normalized_messages) < 0.8:
            return None
        if lang != "ru":
            return None
    if source == "ultra" and lang != "en":
        return None
    row = {"messages": messages, "lang": lang, "source": source}
    for key in ("attribution", "category"):
        if raw.get(key):
            row[key] = raw[key]
    return row


def cache_paths(cache_dir: Path, source: str) -> tuple[Path, Path]:
    return cache_dir / f"{source}.jsonl", cache_dir / f"{source}.meta.json"


def read_complete_cache(cache_dir: Path, source: str, limit: int | None) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    data_path, meta_path = cache_paths(cache_dir, source)
    if not data_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("schema") != CACHE_SCHEMA or meta.get("status") != "SUCCESS" or (limit is not None and int(meta.get("limit", 0)) != limit):
            return None
        with data_path.open(encoding="utf-8") as f:
            cached = [json.loads(line) for line in f if line.strip()]
        return (cached[:limit] if limit is not None else cached), meta
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def write_source_cache(cache_dir: Path, source: str, rows_: list[dict[str, Any]], limit: int,
                       candidates: int, rejected: int, source_duplicates: int) -> dict[str, Any]:
    data_path, meta_path = cache_paths(cache_dir, source)
    atomic_jsonl(data_path, rows_)
    repo, license_name, note = SOURCES[source]
    meta = {"schema": CACHE_SCHEMA, "status": "SUCCESS", "source": source, "repo": repo, "revision": REVISION.get(repo, "local"),
            "license": license_name, "note": note, "limit": limit, "candidates": candidates, "accepted": len(rows_),
            "filtered": rejected, "source_duplicates": source_duplicates, "duplicates": source_duplicates,
            "cache_path": str(data_path), "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    atomic_json(meta_path, meta)
    return meta


def balance_rows(rows_: list[dict[str, Any]], mode: str, ru_share: float | None, seed: int) -> list[dict[str, Any]]:
    if mode == "auto":
        return list(rows_)
    if ru_share is None or not 0 < ru_share < 1:
        raise ValueError("Fixed balance requires --ru-share to be a number strictly between 0 and 1.")
    by_lang = {lang: [r for r in rows_ if r["lang"] == lang] for lang in ("ru", "en")}
    count = min(int(len(by_lang["ru"]) / ru_share), int(len(by_lang["en"]) / (1 - ru_share)))
    if count <= 0:
        return []
    rng = random.Random(seed)
    n_ru = int(count * ru_share)
    selected = rng.sample(by_lang["ru"], n_ru) + rng.sample(by_lang["en"], count - n_ru)
    rng.shuffle(selected)
    return selected


def build_dataset(rows_: list[dict[str, Any]], out: Path, mode: str, ru_share: float | None, seed: int,
                  source_status: dict[str, Any], source_names: list[str], max_per_source: int,
                  noncommercial: bool) -> dict[str, Any] | None:
    if not rows_:
        return None
    selected = balance_rows(rows_, mode, ru_share, seed)
    if not selected:
        return None
    noncommercial = noncommercial and any(row.get("source") == "daily" for row in selected)
    rng = random.Random(seed)
    rng.shuffle(selected)
    n = len(selected)
    ntest = max(1, int(n * 0.05)) if n >= 3 else 0
    nval = max(1, int(n * 0.05)) if n >= 3 else 0
    if n - ntest - nval < 1:
        ntest = nval = 0
    splits = {"train": selected[:n - ntest - nval],
              "validation": selected[n - ntest - nval:n - ntest] if nval else [],
              "test": selected[n - ntest:] if ntest else []}
    for name, subset in splits.items():
        atomic_jsonl(out / f"{name}.jsonl", subset)
    all_counts = Counter(r["lang"] for r in selected)
    metadata = {
        "dataset_version": "lyra-conversation-0.2", "source_names": source_names,
        "source_status": source_status,
        "successful_sources": [s for s, value in source_status.items() if value.get("status", "").startswith("SUCCESS")],
        "failed_sources": [s for s, value in source_status.items() if value.get("status") == "SOURCE_FAILED"],
        "source_revisions": {name: (source_status.get(name) or {}).get("revision") for name in source_names},
        "licenses": {name: SOURCES[name][1] for name in source_names},
        "non_commercial_data_included": noncommercial,
        "license_notice": "NON-COMMERCIAL DATASET INCLUDED" if noncommercial else None,
        "seed": seed, "max_per_source": max_per_source, "balance_mode": mode,
        "ru_share_target": ru_share if mode == "fixed" else None,
        "language_counts_before_balance": dict(Counter(r["lang"] for r in rows_)),
        "language_counts_after_balance": dict(all_counts),
        "source_conversations_before_balance": dict(Counter(r["source"] for r in rows_)),
        "source_conversations_after_balance": dict(Counter(r["source"] for r in selected)),
        "accepted_before_balance": len(rows_), "accepted_after_balance": len(selected),
        "filtered_after_extraction": sum(int((source_status.get(s) or {}).get("filtered", 0)) for s in source_names),
        "duplicates_after_normalization": sum(int((source_status.get(s) or {}).get("duplicates", 0)) for s in source_names),
        "splits": {name: stats(subset) for name, subset in splits.items()},
        "note": "Auto balance retains all unique accepted conversations without oversampling or language-based downsampling. Fixed balance samples without replacement. Per-source download failures are reported in source_status."
    }
    atomic_json(out / "dataset_stats.json", metadata)
    return metadata


def parse_names(value: str) -> list[str]:
    return list(dict.fromkeys(x.strip() for x in value.split(",") if x.strip()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--sources", default=DEFAULT_SOURCES)
    parser.add_argument("--skip-sources", default="", help="Comma-separated optional sources to omit, e.g. ultra")
    parser.add_argument("--max-per-source", type=int, default=50000)
    parser.add_argument("--balance", choices=("auto", "fixed"), default=None)
    parser.add_argument("--ru-share", default="auto", help="auto by default; use a float such as 0.5 to request a fixed ratio")
    parser.add_argument("--include-daily-nc", action="store_true")
    parser.add_argument("--include-siberian", action="store_true", help="Compatibility option; Siberian is in the default source mix")
    parser.add_argument("--resume-data", action="store_true", help="Reuse completed source caches; this is also the safe default")
    parser.add_argument("--refresh-sources", action="store_true", help="Ignore completed source caches and fetch/process those sources again")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    if args.max_per_source <= 0:
        parser.error("--max-per-source must be positive")

    names = parse_names(args.sources)
    if args.include_siberian and "siberian" not in names:
        names.append("siberian")
    if args.include_daily_nc and "daily" not in names:
        names.append("daily")
    unknown = set(names) - SOURCES.keys()
    if unknown:
        parser.error(f"unknown source(s): {sorted(unknown)}")
    skipped = set(parse_names(args.skip_sources))
    unknown_skipped = skipped - SOURCES.keys()
    if unknown_skipped:
        parser.error(f"unknown skipped source(s): {sorted(unknown_skipped)}")
    names = [name for name in names if name not in skipped]
    # Always include repository-authored examples unless explicitly omitted.
    if "curated" not in names and "curated" not in skipped:
        names.append("curated")
    if not names:
        parser.error("No sources remain after --skip-sources")

    if str(args.ru_share).lower() == "auto":
        ru_share = None
        mode = args.balance or "auto"
        if mode == "fixed":
            parser.error("--balance fixed requires numeric --ru-share, for example --ru-share 0.5")
    else:
        try:
            ru_share = float(args.ru_share)
        except ValueError:
            parser.error("--ru-share must be 'auto' or a number strictly between 0 and 1")
        if not 0 < ru_share < 1:
            parser.error("--ru-share must be strictly between 0 and 1")
        if args.balance == "auto":
            parser.error("--ru-share with a numeric ratio conflicts with --balance auto; use --balance fixed")
        mode = "fixed"

    out = Path(args.out)
    cache_dir = Path("data/cache/processed")
    intermediate = out / "intermediate"
    status_path = out / "build_status.json"
    out.mkdir(parents=True, exist_ok=True)
    intermediate.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {"requested_sources": names, "skipped_sources": sorted(skipped), "sources": {}, "errors": {}, "build_state": "IN_PROGRESS"}
    status["resume_data_requested"] = args.resume_data
    if "daily" in names:
        print("WARNING: DailyDialog has CC BY-NC-SA terms; this build is non-commercial/share-alike restricted.")
    atomic_status(status_path, status)
    source_rows: dict[str, list[dict[str, Any]]] = {}

    def current_unique_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
        all_rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        duplicates = Counter()
        counts = Counter()
        for source in names:
            for row in source_rows.get(source, []):
                key = duplicate_key(row["messages"])
                if key in seen:
                    duplicates[source] += 1
                    continue
                seen.add(key)
                all_rows.append(row)
                counts[source] += 1
        for source, record in status["sources"].items():
            record["duplicates"] = int(record.get("source_duplicates", 0)) + int(duplicates.get(source, 0))
        return all_rows, dict(counts)

    def publish_partial() -> None:
        all_rows, _ = current_unique_rows()
        metadata = build_dataset(all_rows, out, mode, ru_share, args.seed, status["sources"], names,
                                 args.max_per_source, "daily" in names)
        if metadata:
            status["dataset_state"] = "PARTIAL" if any(s.get("status") == "SOURCE_FAILED" for s in status["sources"].values()) else "READY"
            status["dataset_version"] = metadata["dataset_version"]
        atomic_status(status_path, status)

    for source in names:
        cached = None if args.refresh_sources else read_complete_cache(cache_dir, source, args.max_per_source)
        if cached is not None:
            cached_rows, cache_meta = cached
            source_rows[source] = cached_rows
            status["sources"][source] = {**cache_meta, "status": "SUCCESS_CACHED", "filtered": cache_meta.get("filtered", 0), "duplicates": cache_meta.get("source_duplicates", 0)}
            status[source] = "SUCCESS"
            atomic_jsonl(intermediate / f"{source}.jsonl", cached_rows)
            print(f"Source {source}: resumed {len(cached_rows)} cached conversations")
            publish_partial()
            continue

        staged: list[dict[str, Any]] = []
        rejected = candidates = source_duplicates = 0
        seen_siberian_prompts: set[str] = set()
        seen_source_conversations: set[str] = set()
        source_error: Exception | None = None
        try:
            for raw in rows(source, args.max_per_source):
                candidates += 1
                normalized = clean_source_row(raw, source)
                if normalized is None:
                    rejected += 1
                else:
                    key = duplicate_key(normalized["messages"])
                    if key in seen_source_conversations:
                        source_duplicates += 1
                        continue
                    seen_source_conversations.add(key)
                    if source == "siberian":
                        first_user = next(m["content"] for m in normalized["messages"] if m["role"] == "user")
                        prompt_key = re.sub(r"\W+", " ", first_user.lower()).strip()
                        if prompt_key in seen_siberian_prompts:
                            rejected += 1
                            continue
                        seen_siberian_prompts.add(prompt_key)
                    staged.append(normalized)
        except MemoryError:
            raise
        except Exception as exc:
            source_error = exc

        if source_error is None:
            # Cache and intermediate artifact become visible only after the
            # entire source completed. A kill during download leaves prior
            # sources intact and an incomplete temporary file is never reused.
            cache_meta = write_source_cache(cache_dir, source, staged, args.max_per_source, candidates, rejected, source_duplicates)
            source_rows[source] = staged
            atomic_jsonl(intermediate / f"{source}.jsonl", staged)
            status["sources"][source] = {**cache_meta, "status": "SUCCESS", "filtered": rejected, "duplicates": source_duplicates}
            status[source] = "SUCCESS"
            print(f"Source {source}: SUCCESS ({len(staged)} accepted, {rejected} filtered)")
        else:
            cached_fallback = read_complete_cache(cache_dir, source, None)
            detail = f"{type(source_error).__name__}: {source_error}"
            print(f"WARNING:\nSource {source} failed: {detail}\nContinue with remaining sources.")
            record: dict[str, Any] = {"status": "SOURCE_FAILED", "error": detail, "filtered": rejected, "duplicates": 0}
            if cached_fallback is not None:
                fallback_rows, cache_meta = cached_fallback
                # A source that failed during refresh does not invalidate its
                # previous successful copy. Keep that data in the partial build.
                fallback_rows = fallback_rows[:args.max_per_source]
                source_rows[source] = fallback_rows
                record.update({"using_cached_data": True, "revision": cache_meta.get("revision"),
                               "license": cache_meta.get("license"), "accepted": len(fallback_rows)})
                atomic_jsonl(intermediate / f"{source}.jsonl", fallback_rows)
            status["sources"][source] = record
            status[source] = "SOURCE_FAILED"
            status["errors"][source] = detail
        # Persist both status and currently usable train/validation/test after
        # every source, including a failure.
        publish_partial()

    all_rows, _ = current_unique_rows()
    if not all_rows:
        status["build_state"] = "FAILED_NO_DATA"
        atomic_status(status_path, status)
        raise RuntimeError("Dataset build completed without any valid conversations; no output splits were overwritten.")
    failed = [name for name, record in status["sources"].items() if record.get("status") == "SOURCE_FAILED"]
    status["build_state"] = "PARTIAL" if failed else "SUCCESS"
    status["failed_sources"] = failed
    metadata = build_dataset(all_rows, out, mode, ru_share, args.seed, status["sources"], names,
                             args.max_per_source, "daily" in names)
    if metadata is None:
        status["build_state"] = "FAILED_NO_SPLIT_DATA"
        atomic_status(status_path, status)
        raise RuntimeError("No data survived the requested balance; choose --balance auto or change the fixed share.")
    status["build_state"] = "PARTIAL" if failed else "SUCCESS"
    status["final_conversations"] = metadata["accepted_after_balance"]
    atomic_status(status_path, status)
    # A concise machine-readable summary belongs beside the splits as well.
    metadata["build_status"] = status
    atomic_json(out / "dataset_stats.json", metadata)
    if failed:
        print(f"Build completed with optional source failures: {', '.join(failed)}")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
