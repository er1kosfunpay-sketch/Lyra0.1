# Lyra 0.1 architecture choice

Decision date: 2026-09-26. The chosen profile is **257,991,680 trainable parameters**, based on the implemented model's parameter formula (tied token embeddings; no linear biases). It targets 1024-token training/inference context and 16,384 tokenizer vocabulary.

| Candidate | Layers × width | Heads / KV | SwiGLU width | Vocab | Parameters (approx.) | Assessment |
|---|---|---|---|---:|---:|---|
| 100M class | 12 × 768 | 12 / 4 | 2048 | 16384 | 88.1M | Easiest iterations, but capacity too constrained for bilingual chat/context |
| 150M class | 16 × 768 | 12 / 4 | 2560 | 16384 | 132.2M | Affordable, but limited room for Russian+English response patterns |
| **Selected** | **20 × 1024** | **16 / 4** | **3072** | **16384** | **257,991,680** | Practical capacity/compute balance for 16 GB class notebook GPUs at seq 1024, microbatch 1 and checkpointing |
| 350M class | 40 × 768 | 12 / 4 | 3072 | 16384 | 358.8M | More depth, but ~40% more parameter and activation work with less useful batch throughput |
| 500M class | 32 × 1024 | 16 / 4 | 4096 | 16384 | 503.4M | Optimizer state and activation pressure leave too little room on common free GPUs |

The 250M-class choice is a capacity/compute compromise, not a quality guarantee. Free GPU time can support pipeline experiments and incremental training, but cannot guarantee enough tokens for a capable bilingual assistant. High-quality conversational data and repeated evaluation matter more than scaling parameters alone.

## Exact final config

- Vocabulary: 16,384 (byte-level BPE; includes special tokens)
- Context: 1,024 tokens
- Layers: 20 pre-norm decoder blocks
- Hidden size: 1,024
- Query heads: 16; KV heads: 4; head dimension: 64
- SwiGLU intermediate: 3,072
- RoPE theta: 10,000; RMSNorm epsilon: 1e-6
- Embedding/head: tied
- Parameters: 257,991,680 (computed by config and model implementations)

Each block has Q/O projections of 1024×1024, K/V projections of 256×1024, three 1024×3072 MLP matrices, two 1024-element RMSNorm scales. The vocabulary embedding contributes 16,777,216 parameters; final norm 1,024.

## Approximate memory

- BF16/FP16 inference weights: ~0.48 GiB; FP32: ~0.96 GiB.
- Full-precision AdamW training state (weights + gradients + two moments): ~3.84 GiB decimal before activations, temporary buffers and allocator fragmentation.
- Training estimate at sequence 1024, microbatch 1, mixed precision + activation checkpointing: **roughly 7–11 GiB peak**; this is an engineering estimate, not measured on this host. T4 16 GB / L4 / A10 should be plausible with conservative settings; smaller cards may need shorter context or CPU offload, which is not implemented.
- Dataset is iterated from files; it is not intentionally copied into RAM as a full token matrix.

No automatic change to final architecture occurs based on GPU. `configs/debug.json` is distinct and must not be exported or reported as final Lyra.
