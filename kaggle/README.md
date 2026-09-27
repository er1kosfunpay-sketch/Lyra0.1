# Lyra 0.1 on Kaggle

The notebook uses the repository's existing `scripts/train.py` and `LyraModel`; it does not create a separate model or training loop. Lyra has 257,991,680 parameters, vocabulary 16,384 and context length 1,024. It starts from random weights.

## Kaggle UI setup

1. Sign in to Kaggle and open **Code → New Notebook**.
2. In Notebook **Settings**, set **Accelerator → GPU**. Select **T4 x2** if that option is available to your account; the trainer intentionally uses `cuda:0` only.
3. Turn **Internet** on so Kaggle can clone GitHub and fetch the public Hugging Face dataset sources.
4. Upload this repository to `https://github.com/er1kosfunpay-sketch/Lyra0.1` if the latest changes are not there yet. Open `kaggle/train_lyra.ipynb` in Kaggle, or create a notebook and copy its cells.
5. Run all cells in order. The notebook checks CUDA/VRAM, clones or fast-forwards the repo, installs `.[data,dev]`, checks the code, prepares or reuses the dataset, trains the tokenizer if needed, and runs preflight before training.
6. Confirm the printed run summary. Defaults: pretrain, 2,000 optimizer steps, microbatch 1, accumulation 8, effective batch 8, LR `3e-4`, warmup 100, save every 100, validation every 250, context 1,024. Precision is BF16 where GPU supports it, otherwise FP16 with GradScaler.
7. Training outputs are under `/kaggle/working/Lyra0.1/checkpoints/stage1_1300/`; tokenizer/data/logs are also inside `/kaggle/working/Lyra0.1`. The export is `/kaggle/working/lyra_export/`.
8. When a session ends, use **Save Version → Quick Save** and ensure output files are included. Kaggle documents up to 20 GB of notebook output storage under `/kaggle/working`.
9. To resume in a new session, open the notebook, choose **Input → Add Input → Notebook Output Files**, and attach the latest saved version of the training notebook. Run cells again. With `AUTO_RESUME=True`, the notebook finds a unique `latest.pt` and restores model, optimizer, scheduler, scaler, step and random state. If multiple runs are attached, set `RESUME_CHECKPOINT` to the exact intended file.
10. To use a larger dataset, attach it as an input and set `USE_ATTACHED_DATA=True`, or set it to `False` to rebuild with the existing data-preparation pipeline. `ALLOW_DATASET_CHANGE=True` retains optimizer/scheduler/step while allowing the changed data fingerprint; model config and tokenizer fingerprint remain strict.

Kaggle notebook outputs are persistent only after saving a notebook version. An interrupted unsaved interactive session can still lose its newest progress; keep the 100-step save interval and save notebook versions regularly. The rotating set holds three step snapshots plus `latest.pt` and, when validation improves, `best.pt`.

## Commands used by the notebook

```bash
python -m pip install -e ".[data,dev]"
python scripts/colab_healthcheck.py --config configs/kaggle.json --require-cuda --instantiate --sanity
python scripts/train.py --config configs/kaggle.json \
  --tokenizer artifacts/tokenizer/tokenizer.json \
  --data data/processed/kaggle_stage1/train.jsonl \
  --validation data/processed/kaggle_stage1/validation.jsonl \
  --stage pretrain --steps 2000 --batch-size 1 --grad-accum 8 \
  --lr 3e-4 --warmup-steps 100 --save-every 100 --archive-every 100 \
  --keep-last-checkpoints 3 --eval-every 250 \
  --out checkpoints/stage1_1300
```

To resume manually, add `--resume /kaggle/input/<attached-notebook-output>/Lyra0.1/checkpoints/stage1_1300/latest.pt`. To reset the phase for SFT while keeping pretrained weights, pass `--resume ... --reset-stage --stage sft` and use a separate `--out checkpoints/stage3_sft` directory.
