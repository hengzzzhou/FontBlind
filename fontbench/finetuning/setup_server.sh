#!/bin/bash
# Setup script for remote GPU server (2×A800 80GB)
# Server: ssh -p 7148 root@101.126.156.90
# Models: /fs-computility-new/Uma4agi/shared/models/
#
# Usage: bash setup_server.sh

set -e

WORK_DIR="/root/fontvlm"
TRAIN_DATA_DIR="${WORK_DIR}/train_data"
CHECKPOINT_DIR="${WORK_DIR}/checkpoints"
MODEL_BASE="/fs-computility-new/Uma4agi/shared/models"

echo "=== FontBench Fine-tuning Server Setup ==="

# 1. Create working directories
mkdir -p "${WORK_DIR}" "${TRAIN_DATA_DIR}" "${CHECKPOINT_DIR}"

# 2. Install dependencies
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.45.0 peft>=0.13.0 accelerate>=0.34.0
pip install bitsandbytes>=0.44.0  # for 4-bit quantization
pip install Pillow>=10.0.0 tqdm>=4.65.0
pip install vllm>=0.6.0  # for serving fine-tuned models

# 3. Check available models on server
echo ""
echo "=== Checking available VL models ==="
for model_dir in "${MODEL_BASE}"/Qwen*VL* "${MODEL_BASE}"/Qwen*vl*; do
    if [ -d "$model_dir" ]; then
        echo "  Found: $model_dir"
    fi
done

# 4. Check GPU status
echo ""
echo "=== GPU Status ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

echo ""
echo "=== Setup Complete ==="
echo "Next steps:"
echo "  1. Upload training data to ${TRAIN_DATA_DIR}/"
echo "  2. Run: bash run_all_finetune.sh"
