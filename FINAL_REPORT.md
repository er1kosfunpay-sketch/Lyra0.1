# Lyra 0.1 — Final Report

Status: **pipeline complete and verified; full training run is pending on Kaggle.**
Rows marked `PENDING` are filled in from the run's `checkpoints/<RUN>/run_report.json`
(written by the last cell of `kaggle/train_lyra.ipynb`). Nothing below is estimated —
every value shown was produced by a command in this repository.

## Summary

| Item | Value |
|---|---|
| Dataset | `Den4ikAI/russian_dialogues_2` (revision `8ce8d669a3f749ec8aa03ea01c475012ef210866`) |
| Other datasets | **NONE** (single source, `fallback_used: false`) |
| Original dialogues | 1,701,649 |
| Clean dialogues | 1,701,469 (99.989%) |
| Removed dialogues | 180 (0.011%): symbol_soup 58, no_letters 7, not_russian 6, duplicates 109 |
| Messages / characters | 7,703,297 / 464,782,132 |
| train / validation / test | 1,667,208 / 17,207 / 17,054 dialogues |
| Corpus actually used | **99.9894%** of the original corpus (train 97.976% + val 1.011% + test 1.002%) |
| Tokenizer vocabulary | 16,384 (byte-level BPE, trained on the train split only, verification OK) |
| Tokens | train 122,821,577 · validation 1,271,269 · test 1,250,688 (total 125,343,534) |
| Model parameters | 257,991,680 (`scripts/count_parameters.py`, formula == `model.parameter_count()`) |
| Context / dtype | 1,024 / bfloat16 |
| Epochs planned | 1 |
| Optimizer steps planned | 14,993 = ⌈122,821,577 / (1 × 8 × 1,024)⌉ |
| Optimizer steps done | **PENDING** (Kaggle run) |
| Final train loss | **PENDING** |
| Final / best validation loss | **PENDING** |
| Checkpoints | `latest.pt` + `best.pt` + `final.pt` + rolling `checkpoint_step_*.pt` (cadence: save 1000, archive 5000, eval 1000, keep last 2) |
| Russian inference (final checkpoint, new process) | **PENDING** — must be `PASS` from `scripts/evaluate_russian.py` |

## What is already verified locally

| Check | Command | Result |
|---|---|---|
| Test suite | `python -m pytest tests/ -q` | 42 passed |
| Full suite runner | `python tests/run_all.py` | SMOKE PASS |
| Dataset build | `python scripts/prepare_data.py --out data/processed --force` | `SUCCESS`, 1,701,469 clean, 711.4 s |
| Single-source guards | `tests/test_single_source.py` | no foreign dataset name in `scripts/`, `lyra/`, `configs/`, `kaggle/` |
| Tokenizer | `scripts/train_tokenizer.py … --expect-vocab-size 16384` | vocab 16,384, `verification.ok = true`, 0 `<UNK>`, all Russian probes round-trip |
| Token counting | `--count-splits auto` | 125,343,534 tokens across the three splits |
| Parameter count | `python scripts/count_parameters.py --config configs/kaggle.json` | 257,991,680 (formula and instantiated model agree) |
| Training smoke (fresh) | `scripts/train.py --config configs/small.json … --steps 5` | `Training complete`, validation_loss 9.676, `latest/best/final` written |
| Resume | re-run with `--resume checkpoints/smoke/latest.pt --steps 10` | resumed from step 5 → 10, loss 9.329, val 9.364 |
| Fresh-process reload | `scripts/chat_check.py --checkpoint checkpoints/smoke/final.pt` | loads checkpoint step 10 and generates Cyrillic text |
| Automated Russian evaluation | `scripts/evaluate_russian.py --checkpoint checkpoints/smoke/final.pt` | `FAIL` — **expected**: the model saw only 10 optimizer steps. Diagnostics: `<UNK>` 0, special-token leaks 0, prompt copies 0, mean cyrillic share 0.80, 2 empty answers, 2 repetition loops |
| Inference CLI | `python inference/chat.py --help` | OK |
| Inference API | `from inference.api import app` | routes `/health`, `/info`, `/generate`, `/chat` |

## Remaining step

1. Open `kaggle/train_lyra.ipynb` on Kaggle (Internet on): it clones this repo,
   rebuilds the dataset, retrains/verifies the tokenizer, computes `TRAIN_STEPS`
   from `tokenizer_stats.json`, trains to the planned volume with auto-resume,
   saves `latest/best/final`, exports the model and runs
   `scripts/evaluate_russian.py` on the final checkpoint in a fresh process.
2. Copy the printed `run_report.json` values into the `PENDING` rows above.
   The report is acceptable only if `Russian inference` is `PASS`; a `FAIL` must
   be diagnosed through the evaluation checks (cyrillic share, `<UNK>`, leaks,
   empties, repetition, length, prompt copying) and never explained away by
   generation settings.

## Provenance

- Cleaning, splits and counters: `data/processed/dataset_stats.json`
- Build status (single source / no fallback): `data/processed/build_status.json`
- Tokenizer verification and token counts: `artifacts/tokenizer/tokenizer_stats.json`
- Machine-readable dataset summary: `data/dataset_report.json`
- Human-readable dataset report: `DATASET_REPORT.md`
