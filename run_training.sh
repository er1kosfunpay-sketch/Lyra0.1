#!/bin/bash
# Lyra 0.1 - Training Script
# Runs training on Lightning AI Studio

set -e

echo "=== Lyra Training ==="

# Activate virtual environment if it exists
if [ -n "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
fi

# Read configuration
CONFIG="${1:-configs/training.yaml}"
MODE="${2:-from_scratch}"

echo "Config: $CONFIG"
echo "Mode: $MODE"
echo "GPUs available: $(python -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)")"

# Set environment variables from .env if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "Loaded .env variables"
fi

# Enable TF32 for better performance on Ampere+ GPUs
export PYTORCH_TF32_ENABLED=1

# Run training
echo "Starting training..."
python train.py \
    --config "$CONFIG" \
    --mode "$MODE" \
    ${WANDB_ENABLED:+--wandb-enabled} \
    ${RESUME:+--resume} \
    ${DRY_RUN:+--dry-run}

echo "Training finished."