# Lyra 0.1 — Dataset Report

**Single source:** `Den4ikAI/russian_dialogues_2` (Hugging Face, MIT license)
**Revision:** `8ce8d669a3f749ec8aa03ea01c475012ef210866`
**Other datasets:** NONE
**Builder:** `python scripts/prepare_data.py --out data/processed`
**Build state:** `SUCCESS` · `single_source: true` · `fallback_used: false` · `failed_sources: []`

Generated from `data/processed/dataset_stats.json`, `build_status.json` and
`artifacts/tokenizer/tokenizer_stats.json`. Machine-readable copy:
`data/dataset_report.json`.

## 1. Source rules

- Exactly one source key: `ru_chat`. `--sources` accepts nothing else — any other
  value aborts with an argparse error before any data is touched.
- The full corpus is used. There is no `first N`, no `max_samples`, no per-source
  cap and no reservoir sampling. `--limit` exists only for tests; when used it is
  recorded in the stats (`limit`, `limit_note`).
- If the raw corpus cannot be obtained the build fails with an explicit error and
  writes `build_state: FAILED_SOURCE_UNAVAILABLE`. It never downloads or mixes in
  another dataset.

## 2. Cleaning (conservative)

Only demonstrably broken rows are dropped. Russian text, short answers and
ordinary chat formatting are kept.

| Reason | Dialogues removed |
|---|---:|
| `symbol_soup` | 58 |
| `no_letters` | 7 |
| `not_russian` (cyrillic share < 0.25, case-insensitive) | 6 |
| `duplicates` (exact normalized duplicates) | 109 |
| **Total removed** | **180 (0.011%)** |

Guards that fired zero times on this corpus: `too_few_messages`,
`non_string_message`, `encoding_damage`, `empty_message_after_clean`,
`link_only_message`, `meaningless_repetition`, `conversation_too_long`,
`message_too_long`.

## 3. Counts

| Metric | Value |
|---|---:|
| Original dialogues | 1,701,649 |
| Clean dialogues | 1,701,469 (99.989%) |
| Removed dialogues | 180 (0.011%) |
| Messages | 7,703,297 |
| Characters | 464,782,132 |
| Build time | 711.4 s |

## 4. Splits

Hash-based deterministic split, seed 17, 98 / 1 / 1.

| Split | Dialogues | Messages | Characters | Bytes | Tokens | Share of corpus |
|---|---:|---:|---:|---:|---:|---:|
| train | 1,667,208 | 7,547,875 | 455,434,219 | 1,168,186,616 | 122,821,577 | 97.976% |
| validation | 17,207 | 78,156 | 4,711,488 | 12,084,827 | 1,271,269 | 1.011% |
| test | 17,054 | 77,266 | 4,636,425 | 11,911,388 | 1,250,688 | 1.002% |
| **total** | **1,701,469** | **7,703,297** | **464,782,132** | **1,192,182,831** | **125,343,534** | **99.989%** |

Token counts are produced by `scripts/train_tokenizer.py --count-splits auto`
with the shipped tokenizer (16,384 vocab), so they are the numbers training
actually sees.

## 5. Tokenizer

| Item | Value |
|---|---|
| File | `artifacts/tokenizer/tokenizer.json` |
| Algorithm | byte-level BPE (`tokenizers`) |
| Vocabulary | 16,384 (exactly the requested size) |
| Trained on | `data/processed/train.jsonl` only (7,547,875 messages, 178.3 s) |
| Special tokens | `<PAD> <UNK> <BOS> <EOS> <SYSTEM> <USER> <ASSISTANT> <TOOL> <END>` |
| EOS | `<END>` |
| Verification | `verification.ok = true` |

Verification checks every Russian probe for exact round-trip and zero `<UNK>`:

| Probe | Tokens | Round-trip | `<UNK>` |
|---|---:|---|---:|
| `Привет` | 1 | OK | 0 |
| `Привет, как дела?` | 5 | OK | 0 |
| `Здравствуйте! Рад тебя видеть.` | 6 | OK | 0 |
| `Что ты сейчас делаешь?` | 5 | OK | 0 |
| `Мне сегодня очень грустно` | 4 | OK | 0 |
| `Давай немного поговорим` | 4 | OK | 0 |
| `Я играю в Roblox` | 9 | OK | 0 |
| `А что ты думаешь про эту погоду? Дождь с самого утра, совсем выйти некуда.` | 19 | OK | 0 |

The tokenizer is stable across Python restarts (plain JSON on disk, no pickled
state) and its vocabulary size matches `configs/kaggle.json` (`vocab_size`), so
the model embedding table lines up with it.

## 6. Corpus usage

| Metric | Value |
|---|---:|
| Dialogues written to splits / original | 99.9894% |
| train / original | 97.976% |
| validation / original | 1.011% |
| test / original | 1.002% |

Nothing beyond the 0.011% of provably broken rows is discarded: the whole usable
corpus reaches training.

## 7. Reproduce

```bash
python scripts/prepare_data.py --out data/processed     # ~12 min, reuses the raw cache
python scripts/train_tokenizer.py --data data/processed/train.jsonl \
    --vocab-size 16384 --expect-vocab-size 16384 \
    --out artifacts/tokenizer/tokenizer.json \
    --stats-out artifacts/tokenizer/tokenizer_stats.json --count-splits auto
python -m pytest tests/ -q                              # includes single-source guards
```

A re-run with the same parameters reuses the existing splits
(`SUCCESS_REUSED`); `--force` rebuilds them from scratch.
