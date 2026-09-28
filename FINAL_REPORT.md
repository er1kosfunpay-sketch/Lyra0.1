# Lyra 0.1 — run status

This report reflects artifacts actually present on 2026-09-26. The project is **not a trained language model yet**.

| Field | Value |
|---|---|
| Model | Lyra |
| Version | 0.1.0 |
| Initialization | Random PyTorch initialization; no external pretrained weights |
| Parameters | **257,991,680**, calculated exactly from config/model dimensions; model-instantiation comparison NOT TESTED |
| Tokenizer | Lyra byte-level BPE implementation present; tokenizer training NOT TESTED |
| Vocabulary / context | 16,384 / 1,024 |
| Layers / hidden | 20 / 1,024 |
| Attention / KV heads / head dimension | 16 / 4 / 64 |
| SwiGLU intermediate | 3,072 |
| Prepared train / validation / test | 1,332 / 73 / 73 conversations; OASST + 20 Russian everyday dialogues + 12 repo-authored examples, filtered/deduplicated and balanced 50/50 ru/en before split |
| Training tokens / steps | NOT TESTED / 0 |
| Train loss / validation loss / perplexity | NOT TESTED |
| Russian / English chat score | NOT TESTED |
| Context / naturalness / repetition score | NOT TESTED |
| Checkpoints | 0 |
| Hardware / precision used for training | No training run yet. Local environment has no CUDA; full GPU training is configured for Google Colab |
| Approximate final-model training VRAM | 7–11 GiB at sequence 1024, microbatch 1, AMP + checkpointing (estimate, not measured) |

## Actual data actions

- Downloaded OASST1 message tree JSONL.GZ (53,625,064 bytes compressed; Apache-2.0) and message JSONL.GZ (53,622,827 bytes). The clean builder selects ready, non-deleted, non-synthetic Russian/English message trees and one response branch per tree using positive/negative vote margin, then helpfulness/quality labels and review count.
- Downloaded and cleaned 20 Russian Everyday Dialogues examples (CC BY 4.0; attribution preserved in each JSONL row). The original JSON contained mojibake; the cleaner repaired it and verified Russian Unicode.
- Built a **partial** bilingual dataset at `data/processed/oasst_ru/`: train 1,332 conversations / 4,923 messages; validation 73 / 292; test 73 / 272. Train is exactly 666 ru / 666 en. Extracted candidates: 4,462; post-extraction quality filter excluded 224 (5.02%); normalized duplicates removed 0; language balance selected 1,478 unique conversations total without replacement. See `DATASET_REPORT.md` and `data/processed/oasst_ru/dataset_stats.json` for measured values. Train assistant replies average 794 characters (median 616; p90 1,677), confirming the current mixture skews long-form.
- Tested the revised failure-tolerant builder using locally cached OASST, Russian Everyday Dialogues and curated records while skipping Siberian and UltraChat: 4,238 conversations retained under `--balance auto` (3,502 English / 736 Russian). Source caches and intermediate JSONL files were reused on a second invocation. Injected an UltraChat failure in tests and confirmed the source was recorded as `SOURCE_FAILED` while earlier data remained in published splits.
- UltraChat 200k was researched and selected as a capped SFT supplement, but not downloaded/processed here because the local Python environment lacks Hugging Face `datasets`; DailyDialog was also not added because it carries CC BY-NC-SA non-commercial/share-alike terms.

## Checks actually run

- Python compileall: PASS for `lyra/`, `scripts/`, `inference/`, `training/` and `tests/`.
- TOML/config parsing and exact parameter formula: PASS; 257,991,680.
- Prepared training data structure/roles/languages: PASS for all 1,332 training conversations.
- All project tests: **15 PASS** in a clean Python 3.12 virtual environment with installed runtime and dev dependencies. This includes the actual model parameter count, tokenizer, checkpoint components, conversation filtering, source cache/resume, auto/fixed language balance, Siberian turn extraction and injected source outage/refresh-failure recovery.
- Editable installs `python -m pip install -e .` and `python -m pip install -e ".[data,dev]"`: PASS in that clean environment; `pip check`: no broken requirements. The bundled desktop Python itself still lacks these packages.
- Colab notebook JSON and Colab config parsing: PASS; actual GPU execution has not been observed from this local environment.

## Available next steps

1. Open `notebooks/train_colab.ipynb`, set its repository URL, select a Colab GPU runtime, and run the cells in order.
2. The notebook installs dependencies, checks GPU/VRAM, builds the pinned default mixture including capped UltraChat, trains/reuses the tokenizer on the train split, and executes health checks before model training.
3. Run the pretraining cell; rerunning after a disconnect resumes from `/content/drive/MyDrive/Lyra/checkpoints/latest.pt`. The optional SFT cell has its own persistent checkpoint directory and resume logic.

The training pipeline, evaluation prompts, Colab notebook, and export/inference/API code are in place. Local CPU unit tests pass; local environment has no CUDA, and full GPU training is configured for Google Colab. Semantic near-duplicate detection and precise file/row cursor checkpointing are not implemented. A Colab GPU profile, full training run, or benchmark output has not yet been observed.

## Audit 2026-09-28 (full pipeline review, no GPU run)

- Fixed critical train/inference mismatch: `inference/chat.py`, `inference/api.py` and `scripts/evaluate_chat.py` built prompts as `ROLE body` without the closing `<END>` the training packer emits. All inference now goes through `lyra.generation.format_chat` (`<ROLE> body <END> … <ASSISTANT>`), with shared defaults in `lyra.generation.GENERATION_DEFAULTS`.
- `lyra.conversations.clean_text` now strips literal `<USER>`/`<ASSISTANT>`-style markers from message bodies (train and inference alike), closing a role-injection channel through the BPE backend.
- `LyraConfig` defaults aligned to the real 258M profile (was a ~2B-parameter default config that would OOM any notebook GPU if instantiated bare).
- `LyraModel.generate` repetition penalty made batch-safe and vectorized (was hardcoded to batch row 0 with a per-token Python loop).
- `load_checkpoint` returns `epoch`; `scripts/train.py` warns if the replayed data epoch differs, and uses `torch.amp.GradScaler` with fallback for old PyTorch.
- `scripts/train_tokenizer.py` preserves newlines in the BPE corpus (was collapsing them to spaces) and warns when input is not a train split.
- All CLI scripts are import-safe (`main(argv)` + `__main__` guard); `inference/api.py` uses lifespan instead of deprecated `on_event`.
- Added `tests/test_smoke.py` (tokenizer RU/EN/mixed/URL/newlines roundtrip, train/inference format parity, forward/loss, weight update, generation bounds, checkpoint fingerprint, fresh-process reload+generate, SFT masking) and `tests/run_all.py`.
- Kaggle export cell now also writes `generation_config.json`, matching `scripts/export_model.py`.
- Removed committed build garbage (`lyra.egg-info`, `__pycache__`, `.pytest_cache`); `*.egg-info/` added to `.gitignore`.
- Local shell has no Python toolchain, so the suite was NOT executed here: run `python tests/run_all.py` on Kaggle/CI before training. Architecture (RMSNorm/RoPE/GQA/SwiGLU, assistant-only SFT masking, packing, checkpoint fingerprints) reviewed and kept as-is.

## Kaggle training path (added 2026-09-27)

- Added `kaggle/train_lyra.ipynb`, `kaggle/README.md`, and `configs/kaggle.json`; the notebook reuses the existing `scripts/train.py` / `LyraModel`, GPU 0, BF16 where supported or FP16 + GradScaler, 1,024 context, microbatch 1, gradient accumulation 8, and 2,000-step pretrain defaults.
- Kaggle output goes under `/kaggle/working/Lyra0.1/`; notebook versions can persist up to 20 GB of `/kaggle/working` output. A later session attaches the saved Notebook Output as input to restore `latest.pt` and the training JSONL history. Actual Kaggle GPU execution and persistence have **not** been tested from this development environment.
- Checkpoints now verify the serialized temporary file before atomic replace and include stage and dataset metadata. Step archives rotate to the requested last-N count; `latest.pt` and `best.pt` are never pruned. Dataset-changed resume is opt-in via `--allow-dataset-change` and still enforces model/config/tokenizer compatibility.
- Verified locally with a two-step tiny debug-model CPU run, checkpoint loading, same-stage resume through step 3, and `--reset-stage --stage sft` starting at step 1. This is pipeline verification, not Lyra 258M training.
- Fixed the invalid `utf8-sig` codec name in `scripts/train_tokenizer.py`, which surfaced during the tokenizer smoke run.
- Project suite: **15 passed** after the changes. Kaggle notebook JSON and every code cell parse successfully; Kaggle execution remains NOT TESTED.

## DATASET SWAP 2026-09-28 (conversational dataset replacement)

Goal: replace the previous tiny OASST-only mixture with a larger, higher-quality **conversational chat** dataset (Russian primary, English secondary).

- **Added sources** (both Apache-2.0/MIT, human-written, casual dialogue):
  - `Den4ikAI/russian_dialogues_2` (MIT): 1.6M multi-turn Russian Telegram dialogue chains; after quality filtering 149,341 conversations kept. Short-form natural chat (median 67 assistant chars).
  - `mookiezi/Discord-Dialogues` (Apache-2.0): human-only English Discord conversations, upstream-filtered for ToS, links, commands, deduplicated; 48,203 conversations kept.
- **Rejected candidates** (quality/license): `Den4ikAI/russian_dialogues` (low-quality forum pairs, irrelevant answers), `lmsys/lmsys-chat-1m` (gated / too large), `Anthropic/hh-rlhf` (not meant for SFT), `facebook/empathetic_dialogues` + `li2017dailydialog/daily_dialog` (CC BY-NC-SA non-commercial), `Winreee/russian_chat` (unknown license).
- **Result**: 268,336 conversations, 170,999 Russian (63.7%) / 97,337 English; splits train 241,504 / validation 13,416 / test 13,416; 0 duplicates; 36,676 invalid rows rejected (mostly Siberian template/short filters). Conversation length stayed chat-like (median 286 chars, p90 4,470; median assistant reply 67 chars).
- Pipeline changes: `--max-per-source` now accepts per-source overrides (e.g. `50000,ru_chat=150000,discord=50000`), per-source rejection tracking (`rejections_by_source`), and richer `stats()` fields (`conversation_chars_*`, `user/assistant_messages`).
- Old dataset preserved at `data/processed/oasst_ru_legacy/`. New canonical splits at `data/processed/oasst_ru/`. Full stats in `data/dataset_report.json` and `data/processed/dataset_stats.json`.
- **Next required step before training**: retrain the tokenizer on the new train split (`python scripts/train_tokenizer.py --data data/processed/oasst_ru/train.jsonl --out artifacts/tokenizer.json --vocab-size 16384`), then run `python tests/run_all.py`.
