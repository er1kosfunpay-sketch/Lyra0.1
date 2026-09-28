"""Multi-GPU training setup using PyTorch Distributed and FSDP.
Stable, battle-tested configuration for large model training.
"""
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import fallback_auto_wrap
from torch.distributed.fsdp import MixedPrecision


def find_free_port():
    """Find a free port for distributed training."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def setup_distributed(rank, world_size, backend="nccl"):
    """Initialize the distributed process group."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(find_free_port())
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank % torch.cuda.device_count())


def cleanup_distributed():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model_for_fsdp(model, mixed_precision="bf16", use_activation_checkpointing=True):
    """Wrap model with FSDP for multi-GPU training."""
    # Mixed precision configuration
    if mixed_precision == "bf16":
        mixed_precision_cfg = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        )
    elif mixed_precision == "fp16":
        mixed_precision_cfg = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float16,
        )
    else:
        mixed_precision_cfg = None

    # Activation checkpointing (gradient checkpointing)
    use_checkpoint = mixed_precision_cfg is not None  # or separate flag

    # Auto-wrap rule: wrap small modules (RMSNorm, embeddings, etc.)
    def auto_wrap_rule(module):
        return isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm, RMSNorm))

    wrapped = FSDP(
        model,
        mixed_precision=mixed_precision_cfg,
        use_activation_checkpointing=use_checkpoint,
        auto_wrap_policy=auto_wrap_rule,
        device_id=torch.cuda.current_device(),
    )
    return wrapped


def train_distributed(rank, world_size, main_fn, **kwargs):
    """Entry point for distributed training processes."""
    try:
        setup_distributed(rank, world_size)
        print(f"[Rank {rank}/{world_size}] Distributed training started")
        main_fn(rank=rank, world_size=world_size, **kwargs)
    except Exception as e:
        print(f"[Rank {rank}] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_distributed()


def launch_distributed(main_fn, world_size, backend="nccl", **kwargs):
    """Launch distributed training across multiple GPUs.
    
    Supports:
    - 1 GPU (single process)
    - 2 GPUs (spawn 2 processes)
    - 4 GPUs (spawn 4 processes)
    - 8 GPUs (spawn 8 processes)
    """
    n_gpus = torch.cuda.device_count()
    if world_size > n_gpus:
        print(f"Warning: Requested {world_size} GPUs but only {n_gpus} available")
        world_size = n_gpus

    if world_size <= 1:
        # Single GPU - just run main function
        print(f"Running on single GPU/CPU")
        main_fn(rank=0, world_size=1, **kwargs)
    else:
        # Multi-GPU - spawn processes
        mp.spawn(
            train_distributed,
            args=(world_size, main_fn, kwargs),
            nprocs=world_size,
            join=True,
        )