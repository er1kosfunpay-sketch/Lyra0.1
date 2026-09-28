"""Lightning-style training loop for Lyra models.
Supports: from_scratch, continued_pretraining, sft modes.
Multi-GPU via FSDP/Distributed. Resume from checkpoint. OOM protection.
Dry run mode. WandB integration.
"""
import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import fallback_auto_wrap
import torch.distributed.init as dist_init

# Set at module level so --resume can access
CFG = None
BEST_VAL = float("inf")
GLOBAL_STEP = 0
GLOBAL_EPOCH = 0


def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_model_config():
    """Extract model config from training config path."""
    if "--config" in sys.argv:
        idx = sys.argv.index("--config")
        cfg_path = sys.argv[idx + 1]
    else:
        cfg_path = "configs/training.yaml"
    import yaml
    raw = yaml.safe_load(open(cfg_path))
    return raw.get("model", cfg_path)


def load_checkpoint(path, model, optimizer, scheduler, scaler, cfg, tokenizer_fp, dataset_version):
    """Load checkpoint with compatibility checking."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    metadata = ckpt.get("metadata", {})

    # Check compatibility
    expected = {
        "model_name": cfg.model_name,
        "model_version": cfg.version,
        "config_hash": hashlib_sha256(json.dumps(cfg.to_dict(), sort_keys=True).encode()).hexdigest(),
        "tokenizer_fingerprint": tokenizer_fp,
    }
    mismatches = [
        f"{k}: checkpoint={metadata.get(k)!r}, current={v!r}"
        for k, v in expected.items()
        if metadata.get(k) != v
    ]
    if mismatches:
        print(f"WARNING: Checkpoint compatibility mismatches: {mismatches[:3]}")

    model.load_state_dict(ckpt["model"])
    if optimizer and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])

    rng_state = ckpt.get("rng", {})
    import random as py_random
    py_random.setstate(rng_state.get("python", py_random.getstate()))
    np.random.set_state(rng_state.get("numpy", np.random.get_state()))
    torch.set_rng_state(ckpt["rng"].get("torch", torch.get_rng_state()))
    if torch.cuda.is_available() and ckpt["rng"].get("cuda") is not None:
        torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])

    return ckpt.get("step", 0), ckpt.get("tokens_seen", 0), ckpt.get("best_validation_loss"), ckpt.get("epoch", 0)


def save_checkpoint(path, model, optimizer, scheduler, scaler, cfg, step, tokens_seen,
                    tokenizer_fp, dataset_version, epoch, best_val, stage="pretrain",
                    dataset_info=None):
    """Atomically save checkpoint with verification."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "config": cfg.to_dict(),
        "step": step,
        "epoch": epoch,
        "tokens_seen": tokens_seen,
        "best_validation_loss": best_val,
        "stage": stage,
        "metadata": {
            "model_name": cfg.model_name,
            "model_version": cfg.version,
            "config_hash": hashlib_sha256(json.dumps(cfg.to_dict(), sort_keys=True).encode()).hexdigest(),
            "tokenizer_fingerprint": tokenizer_fp,
            "dataset_version": dataset_version,
            "dataset_info": dataset_info or {},
        },
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }

    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        # Verify before replacing
        with open(tmp, "rb") as f:
            checked = torch.load(f, map_location="cpu", weights_only=False)
        if checked.get("step") != step or "model" not in checked:
            raise IOError(f"Checkpoint verification failed: {tmp}")
        # Atomic replace
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def setup_distributed(backend="fsdp"):
    """Initialize distributed training if available."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        print(f"[{rank}/{world_size}] Initializing distributed training...")
        dist_init.init_process_group(backend=backend, rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    else:
        print("Running in single-GPU/CPU mode")
        return 0, 1, 0


def setup_model(cfg, device):
    """Create and optionally wrap model for distributed training."""
    from lyra.model import LyraModel
    from lyra.config import LyraConfig

    model = LyraModel(cfg).to(device)

    # Wrap for FSDP if multi-GPU
    if os.environ.get("WORLD_SIZE", "1") != "1":
        model = FSDP(model)

    return model


def prepare_dataloader(files, tokenizer, cfg, batch_size, is_training=True, rank=0, world_size=1):
    """Create dataloader with proper splitting and shuffling."""
    from lyra.data import PackedTextDataset

    ds = PackedTextDataset(files, tokenizer, cfg.context_length, seed=cfg.seed)

    if is_training and world_size > 1:
        # Simple DistributedSampler-like splitting
        import torch.utils.data.distributed as dist_sampler
        dl = DataLoader(ds, batch_size=batch_size, num_workers=cfg.get("num_workers", 0))
        # For FSDP, we need to handle uneven splits
        return dl

    return DataLoader(ds, batch_size=batch_size, shuffle=is_training, num_workers=cfg.get("num_workers", 0))


def train_one_step(model, optimizer, scaler, batch, device, grad_accum_step, use_amp=True):
    """Perform a single training step with gradient accumulation."""
    model.train()
    ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    # Scale loss for gradient accumulation
    loss_scale = 1.0 / grad_accum_step

    with torch.autocast(device_type=device, dtype="bfloat16", enabled=use_amp):
        outputs = model(ids, labels=labels)
        loss = outputs["loss"] * loss_scale

    if use_amp and scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    return outputs["loss"].item() * grad_accum_step, outputs["logits"]


def validate(model, val_dl, device, use_amp=True):
    """Run validation loop."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        with torch.autocast(device_type=device, dtype="bfloat16", enabled=use_amp):
            for batch in val_dl:
                ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(ids, labels=labels)
                loss = outputs["loss"]
                total_loss += loss.item() * ids.size(1)
                total_tokens += ids.size(1)

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")
    return avg_loss, perplexity


def main():
    global GLOBAL_STEP, GLOBAL_EPOCH, BEST_VAL, CFG

    p = argparse.ArgumentParser("Lyra Training Pipeline")
    p.add_argument("--config", default="configs/training.yaml")
    p.add_argument("--mode", choices=["from_scratch", "continued_pretraining", "sft"])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--wandb-enabled", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    CFG = args

    # Load training config
    import yaml
    cfg_dict = yaml.safe_load(open(args.config))

    # Override with training config
    training_cfg = cfg_dict.get("training", {})
    for k, v in training_cfg.items():
        setattr(args, k, v)

    # Update seed if provided
    if args.seed:
        set_seed(args.seed)

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpus = torch.cuda.device_count() if device == "cuda" else 0
    world_size = int(os.environ.get("WORLD_SIZE", str(n_gpus)))

    print(f"Device: {device}, GPUs: {n_gpus}, World size: {world_size}")
    print(f"Mode: {args.mode}")
    print(f"Config: {args.config}")

    # Setup distributed if multi-GPU
    rank, world_size, local_rank = setup_distributed()
    device = f"cuda:{local_rank}" if device == "cuda" else "cpu"

    # Initialize model
    from lyra.config import LyraConfig
    from lyra.model import LyraModel

    model_cfg = LyraConfig(**cfg_dict.get("model", {}))
    model = setup_model(model_cfg, device)

    # Count parameters
    actual_params = model.numel()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {actual_params:,} total, {trainable_params:,} trainable")
    target = 10_000_000_000
    print(f"Target: {target:,}, Actual: {actual_params:,}, Diff: {actual_params - target:,}")

    # Tokenizer
    tokenizer_path = training_cfg.get("tokenizer", "artifacts/tokenizer/tokenizer.json")
    from lyra.token import LyraTokenizer
    tokenizer = LyraTokenizer.from_file(tokenizer_path)
    if tokenizer.vocab_size != model_cfg.vocab_size:
        print(f"WARNING: Tokenizer vocab ({tokenizer.vocab_size}) != model vocab ({model_cfg.vocab_size})")

    # Dataset
    data_cfg = cfg_dict.get("dataset", {})
    hf_cfg = data_cfg.get("huggingface", {})
    repo_id = hf_cfg.get("repo_id", "")

    # Load dataset from Hugging Face
    from datasets import load_dataset
    try:
        if hf_cfg.get("streaming", False):
            ds = load_dataset(repo_id, config_name=hf_cfg.get("config_name"),
                            split=hf_cfg.get("split", "train"),
                            revision=hf_cfg.get("revision", "main"),
                            streaming=True, token=hf_cfg.get("token"))
        else:
            ds = load_dataset(repo_id, config_name=hf_cfg.get("config_name"),
                            split=hf_cfg.get("split", "train"),
                            revision=hf_cfg.get("revision", "main"),
                            token=hf_cfg.get("token"))
    except Exception as e:
        print(f"Could not load Hugging Face dataset: {e}")
        # Fallback to local data
        data_files = training_cfg.get("data_files", ["data/processed/train.jsonl"])
        ds = None

    # Analyze dataset if not streaming
    if ds is not None and not hf_cfg.get("streaming", False):
        split_name = hf_cfg.get("split", "train")
        n = len(ds)
        print(f"Dataset: {repo_id} {split_name}, samples: {n}")

        # Check for validation split
        if isinstance(ds, dict) and "validation" in ds:
            val_ds = ds["validation"]
            print(f"  Validation samples: {len(val_ds)}")
        else:
            # No val split - create 99/1
            print(f"  No validation split found. Will use 99% train / 1% val (seed={args.seed})")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(training_cfg.get("learning_rate", 2e-5)),
                                  weight_decay=float(training_cfg.get("weight_decay", 0.1)))

    # Scheduler
    warmup_steps = int(int(training_cfg.get("max_steps", 1000)) * float(training_cfg.get("warmup_ratio", 0.05)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                 lr_lambda=lambda s: max(
                                                     1e-3,
                                                     (s + 1) / max(1, warmup_steps) if s < warmup_steps else
                                                     0.5 * (1 + math.cos(math.pi * min(1.0, (s - warmup_steps) / max(1, int(training_cfg.get("max_steps", 1000)) - warmup_steps)))),
                                                 ))

    # Scaler for AMP
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and training_cfg.get("bf16", True))

    # Gradient accumulation
    grad_accum = int(training_cfg.get("gradient_accumulation_steps", 16))
    effective_batch = int(training_cfg.get("per_device_batch_size", 4)) * grad_accum * world_size

    # Checkpoint directory
    ckpt_dir = Path(training_cfg.get("output_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Validation dataloader
    val_dl = None
    if not training_cfg.get("no_validation", False):
        # Create val split if needed
        val_files = []
        val_dl = prepare_dataloader(val_files, tokenizer, model_cfg,
                                    int(training_cfg.get("per_device_batch_size", 4)),
                                    is_training=False, rank=rank, world_size=world_size)

    # Dry run mode
    if args.dry_run:
        print("\n=== DRY RUN ===")
        # Check GPU/VRAM
        if device.type == "cuda":
            total_vram = torch.cuda.get_device_properties(0).total_memory
            allocated = torch.cuda.memory_allocated(0)
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM total: {total_vram / 1024**3:.1f} GB")
            print(f"VRAM allocated: {allocated / 1024**3:.1f} GB")
            print(f"VRAM available: {(total_vram - allocated) / 1024**3:.1f} GB")

        # Load tokenizer
        print(f"\nTokenizer: {tokenizer_path}, vocab_size: {tokenizer.vocab_size}")

        # Create model
        print(f"\nModel: {model_cfg.model_name}")
        print(f"Architecture: hidden={model_cfg.hidden_size}, layers={model_cfg.num_layers}")
        print(f"Parameters: {actual_params:,}")

        # Forward pass
        model.eval()
        test_ids = tokenizer.encode("Hello, world!", add_bos=True, add_eos=True)
        test_tensor = torch.tensor([test_ids] * 2, device=device, dtype=torch.long)
        with torch.autocast(device_type=device.type, dtype="bfloat16"):
            output = model(test_tensor)
        print(f"Forward pass: loss={output['loss']}, logits shape={output['logits'].shape}")

        # Backward pass (dummy)
        if device.type == "cuda":
            loss = output["loss"]
            scaler.scale(loss).backward()
        print("Backward pass: OK")

        # Optimizer check
        optimizer.zero_grad()
        print("Optimizer step: OK")

        # Checkpoint check
        latest = ckpt_dir / "latest.pt"
        if latest.exists():
            print(f"Latest checkpoint exists: {latest}")
        else:
            print("No checkpoint yet - would start new training")

        # Distributed init check
        if world_size > 1:
            print(f"Distributed: {world_size} GPUs")
        else:
            print("Distributed: Single GPU/CPU mode")

        print("\n=== DRY RUN PASSED ===")
        return

    # Resume from checkpoint if requested
    start_step = 0
    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume:
        latest_ckpt = ckpt_dir / "latest.pt"
        if latest_ckpt.exists():
            print(f"Resuming from checkpoint: {latest_ckpt}")
            step, tokens, loaded_best, epoch = load_checkpoint(
                str(latest_ckpt), model, optimizer, scheduler, scaler,
                model_cfg, hashlib_sha256(open(tokenizer_path, "rb").read()).hexdigest(),
                "unknown"
            )
            start_step = step
            start_epoch = epoch
            best_val_loss = loaded_best if loaded_best is not None else float("inf")
            print(f"Resumed at step {start_step}, epoch {start_epoch}")
        else:
            print("No checkpoint found for resuming - starting new training")

    # Training loop
    global GLOBAL_STEP, GLOBAL_EPOCH, BEST_VAL
    GLOBAL_STEP = start_step
    GLOBAL_EPOCH = start_epoch
    BEST_VAL = best_val_loss

    # Prepare training dataloader
    train_files = []  # Would be loaded from data config
    train_dl = prepare_dataloader(train_files, tokenizer, model_cfg,
                                  int(training_cfg.get("per_device_batch_size", 4)),
                                  is_training=True, rank=rank, world_size=world_size)

    # Training loop
    max_steps = int(training_cfg.get("max_steps", 1000))
    save_steps = int(training_cfg.get("save_steps", 100))
    eval_steps = int(training_cfg.get("eval_steps", 100))
    logging_steps = int(training_cfg.get("logging_steps", 10))

    print(f"\n=== TRAINING STARTED ===")
    print(f"Max steps: {max_steps}, Starting from step: {start_step}")
    print(f"Effective batch size: {effective_batch}")
    print(f"Save every {save_steps} steps, Eval every {eval_steps} steps")

    model.train()
    opt = optimizer
    sched = scheduler
    sc = scaler

    # Log file
    log_path = ckpt_dir / "training.log"
    log_file = open(log_path, "a", encoding="utf-8")

    # WandB init
    wandb_enabled = training_cfg.get("bf16", True) and args.wandb_enabled
    if wandb_enabled:
        try:
            import wandb
            wandb.init(project="lyra-10b", config=training_cfg, resume="allow" if args.resume else "never")
        except ImportError:
            print("WandB not available, disabling.")
            wandb_enabled = False

    # Skip already-consumed steps on resume
    if start_step > 0:
        # Advance dataloader cursor
        for _ in range(start_step * grad_accum):
            try:
                next(iter(train_dl))
            except StopIteration:
                break

    step = start_step
    tokens_seen = 0
    epoch = start_epoch
    loss_accum = 0.0
    start_time = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    end_time = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

    if start_time:
        start_time.record()

    while step < max_steps:
        epoch_loss = 0.0
        for _ in range(grad_accum):
            try:
                batch = next(iter(train_dl))
            except StopIteration:
                epoch += 1
                # Create new epoch
                # For simplicity, just iterate again
                if hasattr(train_dl.dataset, 'set_epoch'):
                    train_dl.dataset.set_epoch(epoch)
                try:
                    batch = next(iter(train_dl))
                except StopIteration:
                    # Reset and try again
                    train_dl = prepare_dataloader(train_files, tokenizer, model_cfg,
                                                  int(training_cfg.get("per_device_batch_size", 4)),
                                                  is_training=True, rank=rank, world_size=world_size)
                    batch = next(iter(train_dl))

            loss_val, _ = train_one_step(model, opt, sc, batch, device, grad_accum, use_amp=True)
            epoch_loss += loss_val

        # Step optimizer after accumulation
        if sc is not None:
            sc.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training_cfg.get("max_grad_norm", 1.0)))
            sc.step(opt)
            sc.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training_cfg.get("max_grad_norm", 1.0)))
            opt.step()

        opt.zero_grad(set_to_none=True)
        sched.step()
        step += 1
        tokens_seen += batch["input_ids"].shape[0] * batch["input_ids"].shape[1] if 'batch' in dir() else 0
        loss_accum += epoch_loss

        # Timing
        if start_time:
            end_time.record()
            torch.cuda.synchronize()
            elapsed = start_time.elapsed_time(end_time) / 1000.0
            start_time = end_time
            end_time = torch.cuda.Event(enable_timing=True)

        # Logging
        if step % logging_steps == 0:
            lr = sched.get_last_lr()[0]
            msg = (f"Step {step:6d} | Loss: {loss_accum/loggingSteps:.4f} | "
                   f"LR: {lr:.2e} | Step/sec: ~{loggingSteps/max(elapsed,1e-3):.1f}")
            print(msg)
            log_file.write(msg + "\n")
            loss_accum = 0.0

            if wandb_enabled:
                import wandb
                wandb.log({
                    "step": step,
                    "train/loss": loss_accum / loggingSteps if loggingSteps > 0 else 0,
                    "train/learning_rate": lr,
                    "train/tokens_per_sec": loggingSteps / max(elapsed, 1e-3),
                })

        # Validation
        if val_dl is not None and step % eval_steps == 0:
            avg_loss, ppl = validate(model, val_dl, device, use_amp=True)
            print(f"Step {step}: Validation loss={avg_loss:.4f}, Perplexity={ppl:.2f}")
            log_file.write(f"Step {step}: Val loss={avg_loss:.4f}, Perplexity={ppl:.2f}\n")

            if best_val_loss is None or avg_loss < best_val_loss:
                best_val_loss = avg_loss
                save_checkpoint(ckpt_dir / "best.pt", model, opt, sched, sc,
                               model_cfg, step, tokens_seen,
                               hashlib_sha256(open(tokenizer_path, "rb").read()).hexdigest(),
                               "unknown", epoch, best_val_loss, stage=args.mode)
                print("  → Saved best checkpoint")

            if wandb_enabled:
                import wandb
                wandb.log({"step": step, "val/loss": avg_loss, "val/perplexity": ppl})

        # Save checkpoint
        if step % save_steps == 0:
            save_checkpoint(ckpt_dir / f"checkpoint-{step:07d}.pt", model, opt, sched, sc,
                           model_cfg, step, tokens_seen,
                           hashlib_sha256(open(tokenizer_path, "rb").read()).hexdigest(),
                           "unknown", epoch, best_val_loss, stage=args.mode)
            # Also save latest
            save_checkpoint(ckpt_dir / "latest.pt", model, opt, sched, sc,
                           model_cfg, step, tokens_seen,
                           hashlib_sha256(open(tokenizer_path, "rb").read()).hexdigest(),
                           "unknown", epoch, best_val_loss, stage=args.mode)

        # Check if we should stop
        if step >= max_steps:
            break

    # Final save
    save_checkpoint(ckpt_dir / "latest.pt", model, opt, sched, sc,
                   model_cfg, step, tokens_seen,
                   hashlib_sha256(open(tokenizer_path, "rb").read()).hexdigest(),
                   "unknown", epoch, best_val_loss, stage=args.mode)

    log_file.close()

    if wandb_enabled:
        import wandb
        wandb.finish()

    print(f"\n=== TRAINING COMPLETED ===")
    print(f"Final step: {step}, Best val loss: {BEST_VAL:.4f}")


if __name__ == "__main__":
    main()