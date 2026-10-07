# Lyra 0.1 on Kaggle — full training on ONE dataset

**Dataset: `Den4ikAI/russian_dialogues_2`. Other datasets: NONE.**

The notebook uses this repository's own `scripts/prepare_data.py`,
`scripts/train_tokenizer.py`, `scripts/train.py` and `lyra` package. It does not
create a separate model or a separate training loop.

Model: Lyra 0.1 — 257,991,680 parameters, vocabulary 16,384, context length 1,024,
initialized from random weights.

## Pipeline in the notebook

```
Den4ikAI/russian_dialogues_2
  -> scripts/prepare_data.py        (download/use local raw file, clean, dedup, train/validation/test)
  -> scripts/train_tokenizer.py     (byte-BPE trained on the Russian train split + verification)
  -> scripts/train.py               (training to the planned number of optimizer steps)
  -> checkpoints/<run>/latest.pt | best.pt | final.pt
  -> scripts/evaluate_russian.py    (automated Russian quality report)
```

`scripts/prepare_data.py` has exactly one source. If it cannot obtain
`Den4ikAI/russian_dialogues_2` it stops with an explicit error and never
substitutes another corpus. `tests/test_single_source.py` fails the test suite if
any foreign dataset name returns to `scripts/`, `lyra/`, `configs/` or `kaggle/`.

## Planned training volume

The notebook computes the planned volume from the tokenized train split:

```
steps_per_epoch = ceil(train_tokens / (BATCH_SIZE * GRAD_ACCUM * context_length))
TRAIN_STEPS     = EPOCHS * steps_per_epoch      # EPOCHS = 1 by default
```

`scripts/train.py` runs until `TRAIN_STEPS` optimizer steps are reached (it does
not stop early because the loss is falling), validating every `EVAL_EVERY` steps
and saving checkpoints every `SAVE_EVERY` steps. If a Kaggle session ends before
the target, attaching the saved output resumes from `latest.pt` and continues to
the same target.

## Kaggle UI setup

1. Sign in to Kaggle and open **Code → New Notebook**.
2. **Settings → Accelerator → GPU** (T4 x2 if available; the trainer uses `cuda:0`).
3. Turn **Internet** on: the notebook clones the repository and downloads
   `Den4ikAI/russian_dialogues_2` from Hugging Face.
4. Push this repository to `https://github.com/er1kosfunpay-sketch/Lyra0.1`, open
   `kaggle/train_lyra.ipynb` and run all cells in order.
5. Cells check CUDA/VRAM, clone or fast-forward the repo, install `.[data,dev]`,
   compile-check the pipeline, prepare the dataset, train/verify the tokenizer,
   compute `TRAIN_STEPS`, run the preflight, check the disk budget (< 19 GB),
   train, verify checkpoints and run the Russian evaluation.
6. Outputs live under `/kaggle/working/Lyra0.1/checkpoints/<RUN_NAME>/`.
7. **Save Version → Quick Save** to persist `/kaggle/working` (up to 20 GB).
8. To resume in a new session: **Input → Add Input → Notebook Output Files**,
   attach the previous training notebook output, rerun the cells. With
   `AUTO_RESUME = True` the newest valid `latest.pt` restores model, optimizer,
   scheduler, step and random state. Use `RESUME_CHECKPOINT` when several
   `latest.pt` files are attached.

## Notebook parameters (first code cell)

| Parameter | Default | Meaning |
|---|---|---|
| `RUN_NAME` | `ru_full_v1` | checkpoint directory name |
| `EPOCHS` | `1` | planned passes over the full train split |
| `BATCH_SIZE` / `GRAD_ACCUM` | `1` / `8` | micro-batch and gradient accumulation (effective 8) |
| `LR` | `3e-4` | AdamW learning rate, cosine schedule, warmup 100 |
| `SAVE_EVERY` / `ARCHIVE_EVERY` | `1000` / `5000` | `latest.pt` interval / rolling archive interval (checkpoint I/O is ~3.1 GB per write) |
| `EVAL_EVERY` | `1000` | validation loss + `best.pt` interval |
| `KEEP_LAST_N_CHECKPOINTS` | `2` | rolling archives (plus latest, best, final) |
| `USE_ATTACHED_DATA` | `True` | reuse attached splits **after verifying they are single-source** |
| `ALLOW_DATASET_CHANGE` | `False` | the corpus is fixed |

## Checkpoints

| File | Content |
|---|---|
| `latest.pt` | newest state, used for auto-resume |
| `best.pt` | lowest validation loss |
| `final.pt` | state after the planned training volume completed |
| `checkpoint_step_*.pt` | rolling archives (keep-last-N) |

Every checkpoint stores model config, tokenizer fingerprint, dataset fingerprint,
optimizer/scheduler/scaler state, step, epoch, tokens seen and RNG state.

## Verification cells

```bash
# What training samples look like after formatting + tokenization
python scripts/show_chat_format.py --tokenizer artifacts/tokenizer/tokenizer.json \
  --data data/processed/train.jsonl --rows 2

# Quick generation probe
python scripts/chat_check.py --checkpoint checkpoints/<run>/final.pt

# Automated Russian report (language, <UNK>, repetition, emptiness, leaks, echo, garbage)
python scripts/evaluate_russian.py --checkpoint checkpoints/<run>/final.pt \
  --out logs/russian_eval.json
```

## Disk budget (< 19 GB)

One checkpoint ≈ model fp32 (4 B/param) + Adam m/v fp32 (8 B/param) ≈ 12 B/param
≈ **3.1 GB** for 258M parameters. Planned files: 2 archives + `latest.pt` +
`best.pt` + `final.pt` = 5 × 3.1 ≈ **15.5 GB** worst case, plus the prepared
dataset (~1–2 GB). The notebook checks the budget before training, prunes old
archives when space runs low and refuses to write a partial checkpoint.
