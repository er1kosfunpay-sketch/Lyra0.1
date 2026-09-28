#!/bin/bash
# Lyra 0.1 - Setup Script
# Prepares the environment for training on Lightning AI Studio

set -e

echo "=== Lyra Setup ==="

# 1. Create directory structure
echo "Creating directory structure..."
mkdir -p /teamspace/studios/this_studio/project
mkdir -p /teamspace/studios/this_studio/checkpoints
mkdir -p /teamspace/studios/this_studio/logs
mkdir -p /teamspace/studios/this_studio/output
mkdir -p /teamspace/studios/this_studio/cache
mkdir -p /teamspace/studios/this_studio/hf_cache

# 2. Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install lightning datasets transformers tqdm wandb accelerate

# 3. Verify GPU
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if torch.cuda.is_available(); then
    echo "GPU count: $(python -c "import torch; print(torch.cuda.device_count())")"
fi

# 4. Verify Hugging Face access
echo "Checking HF token..."
if [ -n "$HF_TOKEN" ]; then
    echo "HF_TOKEN set. Testing access..."
    python -c "from datasets import load_dataset; ds = load_dataset('hf-internal-testing/tiny_shakespeare', token='$HF_TOKEN')" 2>/dev/null && echo "HF access OK" || echo "HF access failed"
else
    echo "No HF_TOKEN set. Testing public dataset access..."
    python -c "from datasets import load_dataset; ds = load_dataset('hf-internal-testing/tiny_shakespeare')" 2>/dev/null && echo "Public HF access OK" || echo "Note: May need HF_TOKEN for private datasets"
fi

# 5. Verify model config
echo "Checking model configuration..."
python -c "
from lyra.config import LyraConfig
from lyra.model import LyraModel
import yaml

# Load 10B config
with open('configs/model_10b.yaml') as f:
    cfg = yaml.safe_load(f)
print(f'10B Config: vocab={cfg[\"vocab_size\"]}, hidden={cfg[\"hidden_size\"]}, layers={cfg[\"num_layers\"]}')

# Count parameters
from scripts.count_parameters import compute_param_count
count = compute_param_count(cfg)
print(f'Actual parameters: {count:,}')
target = 10_000_000_000
diff = count - target
pct = diff / target * 100
print(f'Target: {target:,}, Difference: {diff:,} ({pct:+.2f}%)')
"

echo ""
echo "=== Setup Complete ==="
echo "Directory structure created at /teamspace/studios/this_studio/"
echo "Config files in configs/ folder"
echo "Run 'bash run_training.sh' to start training"