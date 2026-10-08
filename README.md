# Lyra 0.1 - Language Model Training Pipeline

## Overview

Lyra 0.1 is a 258M-parameter conversational decoder-only Transformer trained **from scratch on a single
dataset: `Den4ikAI/russian_dialogues_2`**. The whole pipeline - download, cleaning, tokenizer training,
splits, training, checkpointing and evaluation - uses that one Russian dialogue corpus. No OASST,
UltraChat, Discord, PersonaChat, English corpora, extra Russian datasets or hand-written demo samples
are mixed in, and there is no fallback source: a failed download is a hard error.

Full training runs on Kaggle GPU (`kaggle/train_lyra.ipynb`); the same scripts work locally.

## Project Structure

```
project/
├── configs/
│   ├── kaggle.json           # 258M model + training profile for Kaggle GPU
│   ├── lyra_0_1.json         # Lyra 0.1 model architecture (258M)
│   ├── small.json / medium.json  # smaller CPU-friendly profiles
│   ├── colab.json / saturn*.json # other platform profiles
│   └── debug.json            # tiny model for tests
├── lyra/
│   ├── model.py              # Model architecture (RMSNorm, RoPE, GQA, SwiGLU)
│   ├── config.py             # Model configuration dataclass
│   ├── tokenizer.py          # byte-level BPE tokenizer (trained on the dataset)
│   ├── data.py               # conversation packing, labels, masking
│   ├── conversations.py      # cleaning, language check, duplicate fingerprints
│   ├── generation.py         # chat formatting shared by training and inference
│   └── checkpoint.py         # atomic save/load with compatibility checks
├── scripts/
│   ├── prepare_data.py       # Den4ikAI/russian_dialogues_2 -> clean -> splits
│   ├── train_tokenizer.py    # tokenizer trained on that train split + verification
│   ├── train.py              # training loop (resume, checkpoints, validation)
│   ├── evaluate_russian.py   # automated Russian quality report for a checkpoint
│   ├── chat_check.py         # quick generation probe
│   ├── colab_healthcheck.py  # environment / model sanity checks
│   ├── count_parameters.py   # parameter count from a config
│   ├── show_chat_format.py   # prints the exact prompt/label format
│   ├── model_info.py         # config / tokenizer summary
│   └── export_model.py       # export trained weights
├── inference/
│   ├── chat.py               # interactive chat with an exported model
│   └── api.py                # local HTTP API (fastapi)
├── tests/                    # pytest suite (single-source guards included)
├── kaggle/train_lyra.ipynb   # full Kaggle training notebook
├── saturn/                   # Saturn Cloud notebook + VRAM sizing
├── notebooks/train_colab.ipynb  # Colab training + Google Drive persistence (LOW_STORAGE option)
├── docs/ARCHITECTURE_DECISION.md
├── checkpoints/              # model checkpoints (auto-generated)
├── artifacts/tokenizer/      # tokenizer + token statistics
├── logs/                     # training logs, evaluation reports
├── DATASET_REPORT.md / FINAL_REPORT.md
├── pyproject.toml
└── README.md
```

## Quick Start

### 1. Environment Setup

```bash
# Clone and navigate to project
git clone <your-repo>
cd Lyra0.1

# Install dependencies
pip install -e ".[data,dev]"

# Verify GPU / environment
python scripts/colab_healthcheck.py --config configs/kaggle.json --sanity
```

### 2. Dataset: Den4ikAI/russian_dialogues_2 (the only source)

Lyra is trained on **one** dataset: [`Den4ikAI/russian_dialogues_2`](https://huggingface.co/datasets/Den4ikAI/russian_dialogues_2).
No OASST, no UltraChat, no Discord, no PersonaChat, no English corpora, no demo samples.

```bash
# Download (or reuse the local raw copy), clean, deduplicate, split
python scripts/prepare_data.py --out data/processed

# Train the tokenizer on the Russian train split and verify it
python scripts/train_tokenizer.py --data data/processed/train.jsonl \
    --vocab-size 16384 --expect-vocab-size 16384 \
    --out artifacts/tokenizer/tokenizer.json --count-splits auto
```

If the dataset cannot be obtained, the build exits with a clear error and
**never** falls back to a different corpus. The source (repository, revision,
license) is hardcoded in `scripts/prepare_data.py`; `--sources` accepts nothing
but `ru_chat`.

### 3. Smoke test (CPU, a few steps)

```bash
python -m pytest tests/ -q
python scripts/train.py --config configs/debug.json \
    --tokenizer <tiny tokenizer> --data data/processed/train.jsonl \
    --steps 5 --save-every 5 --out checkpoints/smoke
```

### 4. Full training

Use `kaggle/train_lyra.ipynb` on Kaggle (GPU), or locally:

```bash
bash run_training.sh                 # configs/kaggle.json, auto-resume
```

The notebook prepares the full corpus, trains the tokenizer, runs the planned
number of optimizer steps (one epoch over the whole train split by default),
saves `latest.pt` / `best.pt` / `final.pt` and finishes with an automated
Russian evaluation (`scripts/evaluate_russian.py`).


## Model Architecture

### 258M Model (Lyra 0.1)
- Parameters: 257,991,680
- hidden_size: 1024
- num_layers: 20
- num_attention_heads: 16
- num_kv_heads: 4 (GQA)
- head_dim: 64
- intermediate_size: 3072
- vocab_size: 16,384
- Context length: 1,024
- Activations: SwiGLU
- Normalization: RMSNorm
- RoPE theta: 10,000

## Training

`scripts/train.py` is the training loop used everywhere (local, Colab, Kaggle):

```bash
python scripts/train.py --config configs/kaggle.json \
    --tokenizer artifacts/tokenizer/tokenizer.json \
    --data data/processed/train.jsonl \
    --validation data/processed/validation.jsonl \
    --stage pretrain --steps <N> --batch-size 1 --grad-accum 8 \
    --lr 3e-4 --warmup-steps 100 --save-every 500 --archive-every 500 \
    --keep-last-checkpoints 2 --eval-every 500 --out checkpoints/<run>
```

- Weights start from random initialization: no pretrained model, no second corpus.
- The run **auto-resumes**: a plain re-run picks the newest valid checkpoint
  (`latest.pt`, then the newest `checkpoint_step_*.pt`, then `best.pt`) and restores
  model, optimizer, scheduler, scaler, step, epoch and RNG state. A corrupt file
  falls back to an older archive; all-corrupt is a loud error, never a silent restart.
- `--reset-stage` restarts the step/scheduler/data cursor while keeping the weights.
- The loop runs until `--steps` is reached; on completion it writes `latest.pt`,
  `best.pt` (best validation loss) and `final.pt`.

### Conversation format

Each dialogue is streamed as role-marked tokens, in the original turn order:

```
<USER> текст <END> <ASSISTANT> текст <END> <USER> текст <END> <ASSISTANT> текст <END>
```

`lyra/data.py` packs those streams into fixed `context_length` windows.
`--stage pretrain` trains on all tokens; `--stage sft` masks non-assistant targets
with `-100` (assistant-only loss). Inference builds prompts through
`lyra/generation.py::format_chat`, so train and inference formats are identical.

## Checkpoint System

Checkpoints are written atomically and verified by reading them back:

```bash
python scripts/train.py --config configs/kaggle.json --resume checkpoints/<run>/latest.pt
```

Contents of every checkpoint:

- Model state dictionary
- Optimizer state dictionary
- Scheduler state dictionary
- Scaler state (AMP)
- RNG state (Python, NumPy, Torch, CUDA)
- Global step, epoch, tokens seen
- Best validation loss
- Config hash for compatibility checking
- Tokenizer fingerprint and dataset fingerprint
- Model config (`config`) and stage

Files per run: rolling archives `checkpoint_step_*.pt`, `latest.pt` (resume pointer),
`best.pt` (best validation loss), `final.pt` (completed planned training volume).

## Validation

Validation runs every `--eval-every` steps on the held-out validation split and
writes `validation_loss` records to `checkpoints/<run>/training.jsonl`.
Whenever validation improves, `best.pt` is written.

## Generation and chat probing

```bash
# Quick generation probe for a checkpoint
python scripts/chat_check.py --checkpoint checkpoints/<run>/final.pt

# Export the weights, then chat with them
python scripts/export_model.py --config configs/kaggle.json \
    --checkpoint checkpoints/<run>/final.pt \
    --tokenizer artifacts/tokenizer/tokenizer.json --out exports/lyra
python inference/chat.py --model exports/lyra

# Optional local HTTP API: pip install -e ".[serve]"
python -m uvicorn inference.api:app --host 127.0.0.1 --port 8000
```

Generation defaults live in `lyra/generation.py` (temperature 0.8, top-k 50,
top-p 0.9, repetition penalty 1.08, EOS `<END>`).

## Automated Russian evaluation

```bash
python scripts/evaluate_russian.py \
    --checkpoint checkpoints/<run>/final.pt \
    --tokenizer artifacts/tokenizer/tokenizer.json \
    --out logs/russian_eval.json
```

The report grades every answer for: Russian language (cyrillic share), decode
quality, `<UNK>` count, repetition loops, empty answers, special-token leakage,
answer length, prompt copying and garbage symbols, and finishes with
`"overall": "PASS"` or `"FAIL"` (non-zero exit code).

## Parameter Count Verification

```bash
python scripts/count_parameters.py --config configs/kaggle.json
```

The actual parameter count is derived from `model.parameter_count()`, not from
a formula in the config; `tests/test_count_parameters.py` keeps the two in sync.

## Dataset: one source only

```bash
python scripts/prepare_data.py --out data/processed
```

- Source: `Den4ikAI/russian_dialogues_2` (Hugging Face), revision pinned in
  `scripts/prepare_data.py`.
- The **full** corpus is used - no `max_samples`, no per-source caps, no reservoir
  sampling (`--limit` exists only for tests and is recorded in the stats).
- Cleaning removes only broken rows: empty messages, encoding damage, meaningless
  repetition, link-only messages, non-Russian rows and exact duplicates.
- Output: `train.jsonl` / `validation.jsonl` / `test.jsonl` + `dataset_stats.json`
  (original, removed, remaining, messages, characters, per-split statistics).
- If the download fails, the build exits with an explicit error and **never**
  switches to another dataset.

`scripts/prepare_data.py` names this repository and nothing else;
`tests/test_single_source.py` fails the suite if any other dataset name
reappears in `scripts/`, `lyra/`, `configs/` or `kaggle/`, and the build
records `single_source: true` / `fallback_used: false` in `build_status.json`.

## License

This project is open source; the training corpus is
[`Den4ikAI/russian_dialogues_2`](https://huggingface.co/datasets/Den4ikAI/russian_dialogues_2)
under the MIT license.
