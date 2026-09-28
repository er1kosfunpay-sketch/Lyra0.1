"""Validation loop for Lyra model training.
Supports perplexity computation, loss tracking, and GPU memory monitoring.
"""
import math
import torch
import torch.nn.functional as F


def validate(model, dataloader, device, use_amp=True, max_batches=None):
    """Run validation loop and return average loss and perplexity.
    
    Args:
        model: The model to validate
        dataloader: Validation DataLoader
        device: torch device ('cuda' or 'cpu')
        use_amp: Whether to use automatic mixed precision
        max_batches: Maximum number of batches to run (None = all)
    
    Returns:
        Tuple of (average_loss, perplexity)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batches_seen = 0

    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
            for batch in dataloader:
                if max_batches is not None and batches_seen >= max_batches:
                    break

                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                # Forward pass
                outputs = model(input_ids, labels=labels)
                loss = outputs["loss"]

                # Accumulate
                batch_size = input_ids.shape[0]
                seq_len = input_ids.shape[1]
                total_loss += loss.item() * seq_len * batch_size
                total_tokens += seq_len * batch_size
                batches_seen += 1

    # Compute average loss per token
    avg_loss = total_loss / max(total_tokens, 1)

    # Perplexity = exp(avg_loss), clamp to avoid overflow
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")

    return avg_loss, perplexity


def compute_perplexity(loss):
    """Compute perplexity from cross-entropy loss."""
    return math.exp(loss) if loss < 100 else float("inf")


def format_validation_results(step, train_loss, val_loss, perplexity, lr,
                              tokens_per_sec, gpu_memory_gib=None):
    """Format validation results as a human-readable string."""
    lines = [
        f"=== Step {step} Validation ===",
        f"Train Loss: {train_loss:.4f}",
        f"Val Loss:   {val_loss:.4f}",
        f"Perplexity: {perplexity:.2f}",
        f"Learning Rate: {lr:.2e}",
        f"Tokens/sec:    {tokens_per_sec:.1f}",
    ]
    if gpu_memory_gib is not None:
        lines.append(f"GPU Memory:   {gpu_memory_gib:.1f} GiB")
    return "\n".join(lines)