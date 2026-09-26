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
- UltraChat 200k was researched and selected as a capped SFT supplement, but not downloaded/processed here because the local Python environment lacks Hugging Face `datasets`; DailyDialog was also not added because it carries CC BY-NC-SA non-commercial/share-alike terms.

## Checks actually run

- Python compileall: PASS for `lyra/`, `scripts/`, `inference/` and `tests/`.
- TOML/config parsing and exact parameter formula: PASS; 257,991,680.
- Prepared training data structure/roles/languages: PASS for all 1,332 training conversations.
- Conversation cleanup/short-reply/dedup smoke checks: 3 PASS (invoked directly; `pytest` is not installed).
- Full model/tokenizer/checkpoint/generation/evaluation test suite: NOT RUN because this Python runtime has neither PyTorch nor pytest/tokenizers. No claim is made that forward/backward, optimizer, tokenizer training, or model generation passed.
- Colab notebook JSON and Colab config parsing: PASS; actual GPU execution has not been observed from this local environment.

## Available next steps

1. Open `notebooks/train_colab.ipynb`, set its repository URL, select a Colab GPU runtime, and run the cells in order.
2. The notebook installs dependencies, checks GPU/VRAM, builds the pinned default mixture including capped UltraChat, trains/reuses the tokenizer on the train split, and executes health checks before model training.
3. Run the pretraining cell; rerunning after a disconnect resumes from `/content/drive/MyDrive/Lyra/checkpoints/latest.pt`. The optional SFT cell has its own persistent checkpoint directory and resume logic.

The training pipeline, evaluation prompts, Colab notebook, and export/inference/API code are in place but GPU execution and dependency-backed tests remain to be confirmed in Colab. Semantic near-duplicate detection and precise file/row cursor checkpointing are not implemented. Full model instantiation, tokenizer training, GPU profile, training run, or benchmark output have not yet been observed.
