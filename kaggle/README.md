# Lyra 256M on Kaggle (0 → 10 000 steps, auto-resume, < 19 GB)

The notebook uses the repository's existing `scripts/train.py` and `LyraModel`; it does not create a separate model or training loop. Lyra has 257,991,680 parameters, vocabulary 16,384 and context length 1,024. It starts from random weights.

Target: **10 000 optimizer steps**, checkpoints **every 500 steps**, **3 rolling archives** + `latest.pt` + `best.pt`. The whole project stays **under 19 GB**.

## Kaggle UI setup

1. Sign in to Kaggle and open **Code → New Notebook**.
2. In Notebook **Settings**, set **Accelerator → GPU**. Select **T4 x2** if that option is available to your account; the trainer intentionally uses `cuda:0` only.
3. Turn **Internet** on so Kaggle can clone GitHub and fetch the public Hugging Face dataset sources.
4. Upload this repository to `https://github.com/er1kosfunpay-sketch/Lyra0.1` if the latest changes are not there yet. Open `kaggle/train_lyra.ipynb` in Kaggle, or create a notebook and copy its cells.
5. Run all cells in order. The notebook checks CUDA/VRAM, clones or fast-forwards the repo, installs `.[data,dev]`, checks the code, prepares or reuses the dataset, trains the tokenizer if needed, runs preflight, checks the **disk budget (< 19 GB)**, then trains.
6. Confirm the printed run summary. Defaults: pretrain, **10,000 optimizer steps**, microbatch 1, accumulation 8, effective batch 8, LR `3e-4`, warmup 100, **save every 500, validation every 500**, context 1,024. Precision is BF16 where GPU supports it, otherwise FP16 with GradScaler.
7. Training outputs are under `/kaggle/working/Lyra0.1/checkpoints/stage1_1300/`; tokenizer/data/logs are also inside `/kaggle/working/Lyra0.1`. The export is `/kaggle/working/lyra_export/`.
8. When a session ends, use **Save Version → Quick Save** and ensure output files are included. Kaggle documents up to 20 GB of notebook output storage under `/kaggle/working`, and our budget keeps the run under 19 GB.
9. To resume in a new session, open the notebook, choose **Input → Add Input → Notebook Output Files**, and attach the latest saved version of the training notebook. Run cells again. With `AUTO_RESUME=True`, the notebook finds a unique `latest.pt` and restores model, optimizer, scheduler, scaler, step and random state. If multiple runs are attached, set `RESUME_CHECKPOINT` to the exact intended file. `train.py` itself also auto-resumes: a plain `python train.py` (no `--resume`) continues from the newest **valid** checkpoint, and falls back to older archives if the newest file is corrupt.
10. To use a larger dataset, attach it as an input and set `USE_ATTACHED_DATA=True`, or set it to `False` to rebuild with the existing data-preparation pipeline. `ALLOW_DATASET_CHANGE=True` retains optimizer/scheduler/step while allowing the changed data fingerprint; model config and tokenizer fingerprint remain strict.

Kaggle notebook outputs are persistent only after saving a notebook version. An interrupted unsaved interactive session can still lose its newest progress; keep the 500-step save interval and save notebook versions regularly. The rotating set holds three step snapshots plus `latest.pt` and, when validation improves, `best.pt`.

## Auto-resume logic (`scripts/train.py`)

```
start → scan out dir (latest.pt, then checkpoint_step_*.pt newest-first, then best.pt)
  → none found → train from step 0
  → newest valid found → restore model/optimizer/scheduler/scaler/RNG/step, continue
  → newest corrupt → try older archives automatically
  → all corrupt or incompatible → loud error, never a silent restart from 0
  → restored step >= 10000 → print "Training complete" and exit
```

Manual resume of a specific checkpoint (file or directory):

```bash
python train.py --resume checkpoints/stage1_1300/latest.pt
python train.py --resume checkpoints/stage1_1300/checkpoint_step_00002000.pt
```

Plain auto-resume (finds the newest valid checkpoint by itself):

```bash
python train.py
```

## Commands used by the notebook

```bash
python -m pip install -e ".[data,dev]"
python scripts/colab_healthcheck.py --config configs/kaggle.json --require-cuda --instantiate --sanity
python scripts/train.py --config configs/kaggle.json \
  --tokenizer artifacts/tokenizer/tokenizer.json \
  --data data/processed/kaggle_stage1/train.jsonl \
  --validation data/processed/kaggle_stage1/validation.jsonl \
  --stage pretrain --steps 10000 --batch-size 1 --grad-accum 8 \
  --lr 3e-4 --warmup-steps 100 --save-every 500 --archive-every 500 \
  --keep-last-checkpoints 3 --eval-every 500 \
  --out checkpoints/stage1_1300
```

To resume manually, add `--resume /kaggle/input/<attached-notebook-output>/Lyra0.1/checkpoints/stage1_1300/latest.pt`. To reset the phase for SFT while keeping pretrained weights, pass `--resume ... --reset-stage --stage sft` and use a separate `--out checkpoints/stage3_sft` directory.

## Chat-format check and inference test

```bash
# Show what 2 real samples become after tokenization (pretrain vs SFT labels)
python scripts/show_chat_format.py --tokenizer artifacts/tokenizer/tokenizer.json \
  --data data/processed/kaggle_stage1/train.jsonl --rows 2

# Probe generation quality ("Привет", "Кто ты?", Roblox, Lua, ...)
python scripts/chat_check.py --checkpoint checkpoints/stage1_1300/latest.pt
```

## Disk budget (< 19 GB)

One checkpoint ≈ model bf16 (2 B/param) + Adam m/v fp32 (8 B/param) ≈ 12 B/param ≈ **3.1 GB** for 258M params. Planned files: 3 archives + `latest.pt` + `best.pt` = 5 × 3.1 ≈ **15.5 GB** worst case, plus dataset (~1–3 GB). `train.py` prints the exact budget at startup, prunes extra archives when space is low, and refuses to write a partial checkpoint instead of crashing with `No space left on device`. The notebook re-checks the budget in cell 11 before the long run.
