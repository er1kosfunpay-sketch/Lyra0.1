# Lyra 0.1

Lyra is a small, conversation-first decoder-only language model trained from random initialization. It does not load Qwen/Llama/GPT or any other pretrained model weights. Its goal is ordinary Russian/English dialogue, short-term context and natural response length; its capabilities depend on actual training and are not implied by its parameter count.

## Why this model size?

The final configuration is **257,991,680 parameters** (programmatically calculated), chosen as a compromise between useful conversational capacity and repeated training iterations on 16 GB class notebook GPUs. Smaller 90–130M profiles reduce training cost but constrain bilingual conversational/context patterns; ~360M and ~500M profiles increase activation and optimizer memory enough to make free-host sessions less forgiving. Full rationale and memory estimate are in [ARCHITECTURE_DECISION.md](docs/ARCHITECTURE_DECISION.md).

The final architecture uses 20 layers, hidden size 1024, 16 query heads, 4 KV heads, head dimension 64, SwiGLU intermediate 3072, 16,384 vocabulary, 1024-token context and tied embeddings. It is a custom PyTorch implementation with RMSNorm, RoPE, causal GQA/SDPA, SwiGLU and residual blocks. The independent small `configs/debug.json` exists only for pipeline validation.

Estimated weights: ~0.48 GiB BF16 or ~0.96 GiB FP32. Full FP32 AdamW state is ~3.84 GiB before activations, workspaces and fragmentation. Estimated 7–11 GiB peak at context 1024, microbatch 1, mixed precision and activation checkpointing; this is not measured hardware data. Free GPU availability and session lengths vary, and a useful from-scratch bilingual training run can take many sessions. CPU is for debug and small inference only.

## Dataset research and build

Dataset selection, source quality, advertised upstream sizes, licenses, exclusions and known risks are documented in [DATASET_REPORT.md](DATASET_REPORT.md) and [data/dataset_report.json](data/dataset_report.json). A 268k-conversation bilingual chat mix (OASST + Den4ikAI/russian_dialogues_2 + mookiezi/Discord-Dialogues + SiberianPersonaChat-2 + UltraChat + Russian Everyday + curated) has been built and measured locally; the ingestion pins exact Hub commits and writes actual counts/rates to `data/processed/dataset_stats.json`.

Install (Colab/Kaggle or local venv):

```bash
python -m pip install -e ".[data,dev]"
```

For runtime-only use without data preparation or tests, install the core package with `python -m pip install -e .`. The Colab training notebook uses the full command above.

Build selected data:

```bash
python scripts/prepare_data.py --out data/processed --max-per-source 50000 --balance auto
```

Default sources are OASST, SiberianPersonaChat-2, Russian Everyday Dialogues, and capped UltraChat, plus the small repo-authored examples. Each source is independently cached and saved before the next source starts. A recoverable source failure is recorded in `build_status.json` and remaining sources continue. UltraChat is optional and can be skipped with `--skip-sources ultra`; for example `python scripts/prepare_data.py --sources oasst,siberian,ultra --skip-sources ultra`. Siberian rows require parsed multi-turn Russian exchanges and pass repetition/template filters.

Language balancing defaults to `--balance auto`: use every accepted unique example without oversampling or throwing away the majority language. To request strict no-replacement 50/50 sampling, pass `--balance fixed --ru-share 0.5`. Preparation writes per-source resumable caches under `data/cache/processed/`, intermediate files and partial split snapshots after each source, atomically replaces JSONL/JSON outputs, and records source status/errors in `build_status.json` and `dataset_stats.json`. Completed caches are reused automatically; `--resume-data` makes that intent explicit, while `--refresh-sources` deliberately reprocesses sources.

DailyDialog is human-written everyday English dialogue, but its CC BY-NC-SA 4.0 license is non-commercial/share-alike. It is excluded unless explicitly opted in:

```bash
python scripts/prepare_data.py --include-daily-nc
```

This emits a warning and marks `NON-COMMERCIAL DATASET INCLUDED` in generated statistics because of non-commercial-use restrictions and possible share-alike obligations. Do not publish a model trained on a mixed-license corpus without reviewing each source's terms.

Tokenizer must be trained on the training split only:

```bash
python scripts/train_tokenizer.py --data "data/processed/train.jsonl" --vocab-size 16384 --out artifacts/tokenizer.json
```

The tokenizer is Lyra's byte-level BPE with explicit `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`, `<SYSTEM>`, `<USER>`, `<ASSISTANT>`, `<TOOL>` and `<END>` IDs. It does text/token conversion only.

## Training

Stage 1 uses causal language modeling over the packed conversation stream. Stage 2 masks user/system targets and optimizes assistant responses only. Defaults support AdamW, warmup + cosine decay, gradient clipping, autocast, gradient accumulation and optional gradient checkpointing.

```bash
python scripts/train.py --config configs/lyra_0_1.json \
  --tokenizer artifacts/tokenizer.json \
  --data "data/processed/train.jsonl" \
  --validation "data/processed/validation.jsonl" \
  --stage pretrain --steps 10000 --batch-size 1 --grad-accum 16 \
  --save-every 100 --eval-every 100 --out checkpoints/pretrain
```

Then start the supervised conversation stage with reviewed data. `--reset-stage` starts a new step/scheduler/data-cursor phase from the checkpoint weights and optimizer state; config and tokenizer still have to match. `--steps` is the total target for that phase, so resuming the same phase uses the same target value.

```bash
python scripts/train.py --config configs/lyra_0_1.json \
  --tokenizer artifacts/tokenizer.json \
  --data "data/processed/train.jsonl" \
  --validation "data/processed/validation.jsonl" \
  --stage sft --steps 3000 --reset-stage --batch-size 1 --grad-accum 16 \
  --resume checkpoints/pretrain/latest.pt --out checkpoints/sft
```

Checkpoints contain weights, optimizer, scheduler, scaler, config, RNG states, epoch, step, tokens seen, best validation loss, git revision and strict config/tokenizer/dataset compatibility fingerprints. `--resume latest` resolves `OUT/latest.pt`; periodic `latest.pt`, `best.pt` (when validation is supplied), and numbered checkpoints are written. The deterministic stream is replayed and skipped to recover the next microbatch; this can make resume slow for long runs. Copy checkpoints to Google Drive or a Kaggle output before a session ends.

The training script takes batch size from the user and **does not currently auto-probe VRAM or search for a safe size**. Profiles in `configs/gpu_profiles.json` are starting heuristics only and never alter final model dimensions. Streaming source iteration and bounded caches reduce data RAM; preprocessing currently buffers the selected OASST tree table and balanced accepted conversations in host memory, so it is not fully constant-memory.

### Google Colab GPU training

The complete GPU-first workflow is [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb). Set its `REPO_URL`, choose a Colab GPU runtime, and run cells in order. It mounts Drive, updates the checkout, installs dependencies, checks GPU/VRAM, prepares the dataset, persists the tokenizer, runs health checks, and trains with periodic validation and checkpoints in `/content/drive/MyDrive/Lyra/`. The Colab profile refuses to start without CUDA; local CPU is for development and small debug checks only.

`configs/colab.json` contains both the 258M model dimensions and Colab training defaults. The direct command works after the notebook has prepared the Drive dataset and tokenizer:

```bash
python training/train.py --config configs/colab.json
```

Explicit resume after reconnecting Colab:

```bash
python training/train.py --config configs/colab.json \
  --resume /content/drive/MyDrive/Lyra/checkpoints/latest.pt
```

The notebook selects microbatch/gradient accumulation from detected GPU: conservative T4/P100/V100 profile 1/32; L4/A10 profile 1/16; A100 40GB profile 2/8. All use sequence length 1024 and gradient checkpointing; trainer picks BF16 where supported, otherwise FP16 with scaling. Treat these as starting values and monitor actual VRAM.

### Kaggle Notebook GPU training

Kaggle is also supported with the existing training pipeline. Open [`kaggle/train_lyra.ipynb`](kaggle/train_lyra.ipynb) in Kaggle, set **Settings → Accelerator → GPU** (T4 x2 where available), enable Internet, and run the cells in order. It clones/updates this repository, installs `.[data,dev]`, prepares or discovers data, checks the 16,384-token tokenizer, runs GPU/model sanity checks, then starts pretraining. Defaults are 2,000 steps, microbatch 1, gradient accumulation 8, 1,024 context, BF16 where supported or FP16 with GradScaler, and checkpoint/evaluation intervals of 100/250 steps. See [`kaggle/README.md`](kaggle/README.md) for exact UI and resume instructions.

The Kaggle trainer uses GPU 0 (it does not automatically distribute across both T4s). Checkpoints go to `/kaggle/working/Lyra0.1/checkpoints/stage1_1300/`, with three rolling step checkpoints plus `latest.pt` and `best.pt`. Kaggle retains notebook output after saving a notebook version (up to 20 GB); attach that prior Notebook Output as an input in the next session to resume automatically. A changed dataset may be resumed with `--allow-dataset-change`, which preserves optimizer/scheduler/step while continuing to enforce model and tokenizer compatibility.

## Evaluation / inference

`lyra/evaluation/chat_benchmark_0_1.jsonl` has 134 fixed bilingual test conversations covering greetings, daily chat, context memory, emotion, uncertainty, identity, concise flow and disagreement. `scripts/evaluate_chat.py` saves genuine generated outputs for human rating. It does not manufacture naturalness scores. Never train on this benchmark file.

```bash
pytest
python tests/run_all.py
python scripts/evaluate_chat.py --model exports/lyra --out logs/benchmark_outputs.jsonl
python scripts/export_model.py --config configs/lyra_0_1.json \
  --checkpoint checkpoints/sft/best.pt --tokenizer artifacts/tokenizer.json --out exports/lyra
python inference/chat.py --model exports/lyra
```

`inference/chat.py` retains recent dialogue and streams decoded output; oldest user/assistant turns are discarded when the model context fills, keeping the system prompt. `inference/api.py` provides FastAPI `/health`, `/info`, `/generate`, `/chat`; it reports unavailable until a real compatible trained export is present. No fallback LLM is called.

Chat format parity: training packs every turn as `<ROLE> body <END>` and inference builds the identical stream plus a trailing `<ASSISTANT>` trigger through `lyra.generation.format_chat` (defaults in `lyra.generation.GENERATION_DEFAULTS`). Message bodies are sanitized by `lyra.conversations.clean_text`, which also strips literal `<USER>`/`<ASSISTANT>`-style markers so neither scraped rows nor user input can inject fake role boundaries.

Run the API after exporting a model with `python -m uvicorn inference.api:app --host 0.0.0.0 --port 8000`.

## Reports and current status

See [FINAL_REPORT.md](FINAL_REPORT.md). No tokenizer/model checkpoint or training metrics are claimed until the Colab cells actually produce them. The local environment has no CUDA; full GPU training is configured for Google Colab. Reports must retain `NOT TESTED` for metrics without real runs.
