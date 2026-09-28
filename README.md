# Lyra 0.1 - Language Model Training Pipeline

## Overview

Lyra is a comprehensive framework for training language models from 248M to 10B parameters. This project migrated the training pipeline from Kaggle to Lightning AI Studio, added Hugging Face dataset support, and implemented full multi-GPU training with FSDP.

## Project Structure

```
project/
├── configs/
│   ├── model_248m.yaml      # Original 248M model configuration
│   ├── model_10b.yaml       # New 10B model configuration (real architecture)
│   ├── training.yaml        # Training parameters and hyperparameters
│   └── dataset.yaml         # Hugging Face dataset configuration
├── model/
│   ├── architecture.py      # Model architecture (RMSNorm, RoPE, GQA, SwiGLU)
│   ├── config.py            # Model configuration dataclass
│   └── initialization.py    # Weight initialization utilities
├── data/
│   ├── dataset_loader.py    # Dataset loading and preprocessing
│   ├── preprocessing.py     # Text preprocessing and formatting
│   └── tokenizer.py         # Tokenizer handling
├── training/
│   ├── trainer.py           # Main training loop with all modes
│   ├── distributed.py       # Multi-GPU FSDP setup
│   ├── checkpoint.py        # Checkpoint save/load with verification
│   └── validation.py        # Validation loop and perplexity
├── scripts/
│   ├── count_parameters.py  # Count actual parameter count
│   ├── analyze_dataset.py   # Dataset analysis tool
│   ├── convert_248m_to_10b.py  # Weight conversion from 248M to 10B
│   └── generate.py          # Text generation script
├── checkpoints/             # Model checkpoints (auto-generated)
├── logs/                    # Training logs
├── output/                  # Generated text output
├── hf_cache/                # Hugging Face cache directory
├── train.py                 # Entry point script
├── run_training.sh          # Training launcher script
├── setup.sh                 # Environment setup script
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variables template
└── README.md                # This file
```

## Quick Start

### 1. Environment Setup

```bash
# Clone and navigate to project
git clone <your-repo>
cd Lyra0.1

# Create backup of 248M model (first run)
python backup.py

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your HF_TOKEN, etc.

# Verify GPU
bash setup.sh
```

### 2. Dataset Configuration

Edit `configs/dataset.yaml`:

```yaml
huggingface:
  repo_id: "lyra/conversations-v0.1"
  config_name: null
  split: "train"
  revision: "main"
  streaming: false
  token: YOUR_HF_TOKEN  # optional for public datasets
```

### 3. Dry Run (First Time)

```bash
# Verify everything works before training
python train.py --config configs/training.yaml --dry-run
```

Expected output:
```
=== DRY RUN ===
Device: cuda, GPUs: 1, World size: 1
Model parameters: 10074939392 total, 10074939392 trainable
Target: 10000000000, Actual: 10074939392, Diff: 74939392 (+0.75%)

Tokenizer: artifacts/tokenizer/tokenizer.json, vocab_size: 100277

Model: Lyra-10B Architecture: hidden=4096, layers=64

Forward pass: loss=2.3045, logits shape=(2, 4096, 100277)

Backward pass: OK

Optimizer step: OK

Latest checkpoint exists: checkpoints/latest.pt

Distributed: Single GPU/CPU mode

=== DRY RUN PASSED ===
```

### 4. Start Training

```bash
# From scratch training (10B model)
bash run_training.sh configs/training.yaml from_scratch

# Continued pretraining (load existing 10B model)
bash run_training.sh configs/training.yaml continued_pretraining

# SFT mode (instruction fine-tuning)
bash run_training.sh configs/training.yaml sft
```

## Model Architecture

### 248M Model (Reference)
- Parameters: ~258M
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

### 10B Model (Target)
- Parameters: ~10.07B (within ±5% target)
- hidden_size: 4096
- num_layers: 64
- num_attention_heads: 32
- num_kv_heads: 4 (GQA) - maintains parameter efficiency
- head_dim: 128
- intermediate_size: 9216
- vocab_size: 100,277 (fine-tuned vocabulary)
- Context length: 4,096
- Activations: SwiGLU
- Normalization: RMSNorm
- RoPE theta: 100,000
- Grouped Query Attention (4 KV heads for 32 query heads)

The 10B architecture scales all dimensions appropriately while maintaining the core architectural concepts (RMSNorm, RoPE, SwiGLU, GQA) from the original 248M model.

## Training Modes

### MODE=from_scratch
- Trains a new 10B model from random initialization
- Uses proper weight initialization (normal_, std=0.02)
- Full training from step 0

### MODE=continued_pretraining
- Loads a pre-trained 10B open-weight model
- Continues pre-training on your dataset
- Preserves optimizer state and learning rate schedule

### MODE=sft
- Supervised Fine-Tuning on instruction/chat data
- Trains on instruction datasets with input/output format
- Lower learning rate, fewer steps

## Checkpoint System

Checkpoints are saved atomically with verification:

```bash
# Automatic checkpoint saving
# Every 100 steps by default

# Resume from checkpoint
python train.py --config configs/training.yaml --resume

# Or auto-resume (finds latest checkpoint)
python train.py --config configs/training.yaml --auto-resume
```

Checkpoint contents:
- Model state dictionary
- Optimizer state dictionary
- Scheduler state dictionary
- Scaler state (AMP)
- RNG state (Python, NumPy, Torch, CUDA)
- Global step and epoch
- Best validation loss
- Config hash for compatibility checking
- Tokenizer fingerprint
- Dataset metadata

## Multi-GPU Training

Supported GPU counts: 1, 2, 4, 8

```bash
# Automatic distributed detection
# If multiple GPUs available, training runs with FSDP

# Manual launch
python -m torch.distributed.launch --nproc_per_node=4 train.py ...
```

FSDP (FullyShardedDataParallel) is used for memory efficiency:
- Shards optimizer states, gradients, and parameters across GPUs
- Activation checkpointing to reduce memory usage
- Mixed precision (bf16) support

## OOM Protection

The pipeline includes automatic OOM protection:

1. **Pre-training VRAM estimation**: Checks available VRAM vs. estimated requirements
2. **Automatic fallback sequence**:
   - Reduce micro batch size
   - Increase gradient accumulation steps
   - Enable gradient checkpointing
   - Switch attention implementation (FlashAttention → SDPA fallback)
3. **Never silently fail**: Always reports the reason for any reduction
4. **Config changes logged**: All automatic hyperparameter changes are logged

## Validation

Validation runs every `eval_steps` (default: 100):
- Validation loss
- Perplexity
- Learning rate
- Tokens/second throughput
- GPU memory usage

Results saved to `logs/` directory.

## Generation

After training, generate text:

```bash
python scripts/generate.py --checkpoint checkpoints/latest --prompt "Once upon a time"
```

## Parameter Count Verification

The project explicitly verifies parameter counts:

```bash
python scripts/count_parameters.py --config configs/model_10b.yaml
```

Output:
```
Target parameters: 10000000000
Actual parameters: 10074939392
Difference: 74939392 (+0.75%)
Trainable parameters: 10074939392
```

The actual parameter count is derived from `model.numel()`, not from any formula in config.

## Hugging Face Integration

### Dataset Loading

Datasets are loaded directly from Hugging Face:

```python
from datasets import load_dataset

ds = load_dataset(
    repo_id="lyra/conversations-v0.1",
    split="train",
    token=HF_TOKEN
)
```

### Supported Formats

The preprocessing supports three dataset formats:

1. **text**: `{"text": "..."}`
2. **instruction**: `{"instruction": "...", "input": "...", "output": "..."}`
3. **chat**: `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`

Configuration in `dataset.yaml`:
```yaml
text_field: ""         # For "text" format
instruction_field: ""  # For "instruction" format
input_field: ""        # For "instruction" input
output_field: ""       # For "instruction" output
messages_field: ""     # For "chat" format
```

If format is unknown, the analyzer shows available columns:

```bash
python scripts/analyze_dataset.py --config configs/dataset.yaml
```

## License

This project is open source. The 248M model checkpoint and training code from the original project are preserved in `backup_248m/`.

## Key Differences from Original Kaggle Version

| Aspect | Original (Kaggle) | New (Lightning AI) |
|---|---|---|
| Model size | 248M | 10B (real architecture) |
| Dataset source | Kaggle input paths | Hugging Face `load_dataset()` |
| Distributed training | Not supported | FSDP for 1/2/4/8 GPUs |
| Checkpoint system | Basic | Atomic with verification |
| Resume support | Partial | Full (step, epoch, RNG state) |
| OOM protection | None | Automatic fallback |
| Validation loop | Optional | Required, perplexity tracked |
| WandB support | None | Optional, API key from .env |
| Parameter counting | Config formula | `model.numel()` actual count |
| Sequence length | 1024 | 4096 (configurable) |