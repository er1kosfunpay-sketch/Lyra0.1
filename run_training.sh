#!/bin/bash
# Lyra 0.1 - training launcher (single dataset: Den4ikAI/russian_dialogues_2)
#
# Usage:
#   bash run_training.sh                      # config from $CONFIG, auto-resume
#   CONFIG=configs/kaggle.json bash run_training.sh
#   DRY_RUN=1 bash run_training.sh            # verify the pipeline only

set -e

echo "=== Lyra 0.1 training (dataset: Den4ikAI/russian_dialogues_2) ==="

if [ -n "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
fi

CONFIG="${CONFIG:-configs/kaggle.json}"
echo "Config: $CONFIG"
echo "GPUs available: $(python -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)")"

export PYTORCH_TF32_ENABLED=1

if [ -n "$DRY_RUN" ]; then
    echo "Running pipeline preflight (no training)..."
    python scripts/colab_healthcheck.py --config "$CONFIG" --sanity
    exit 0
fi

python scripts/train.py --config "$CONFIG"

echo "Training finished."
