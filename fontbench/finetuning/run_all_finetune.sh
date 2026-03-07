#!/bin/bash
# Batch fine-tuning script for all models on 2×A800 80GB server.
#
# Usage: bash run_all_finetune.sh
#
# Before running, ensure:
#   1. setup_server.sh has been run
#   2. Training data is at /root/fontvlm/train_data/train.jsonl

set -e

WORK_DIR="/root/fontvlm"
DATA="${WORK_DIR}/train_data/train.jsonl"
CKPT_DIR="${WORK_DIR}/checkpoints"
MODEL_BASE="/fs-computility-new/Uma4agi/shared/models"

if [ ! -f "$DATA" ]; then
    echo "ERROR: Training data not found at $DATA"
    echo "Please generate or upload training data first."
    exit 1
fi

echo "=== FontBench Batch Fine-tuning ==="
echo "Training data: $DATA"
echo ""

# Helper function
run_finetune() {
    local model_key=$1
    local model_path=$2
    local extra_args=$3
    local output="${CKPT_DIR}/${model_key}_lora"

    if [ -d "$output" ] && [ -f "$output/adapter_config.json" ]; then
        echo "[SKIP] ${model_key} — already trained at ${output}"
        return 0
    fi

    echo "============================================"
    echo "[START] Training ${model_key}"
    echo "  Model: ${model_path}"
    echo "  Output: ${output}"
    echo "============================================"

    python -m fontbench.finetuning.train_lora \
        --model "${model_key}" \
        --model-path "${model_path}" \
        --data "${DATA}" \
        --output "${output}" \
        ${extra_args}

    echo "[DONE] ${model_key} saved to ${output}"
    echo ""
}

# -------------------------------------------------------
# 1. Small models (existing, for completeness)
# -------------------------------------------------------
if [ -d "${MODEL_BASE}/Qwen2.5-VL-3B-Instruct" ]; then
    run_finetune "qwen2.5-vl-3b" "${MODEL_BASE}/Qwen2.5-VL-3B-Instruct"
fi

if [ -d "${MODEL_BASE}/Qwen3-VL-4B-Instruct" ]; then
    run_finetune "qwen3-vl-4b" "${MODEL_BASE}/Qwen3-VL-4B-Instruct"
fi

# -------------------------------------------------------
# 2. Medium models (bf16 on single A800)
# -------------------------------------------------------
if [ -d "${MODEL_BASE}/Qwen2.5-VL-7B-Instruct" ]; then
    run_finetune "qwen2.5-vl-7b" "${MODEL_BASE}/Qwen2.5-VL-7B-Instruct"
fi

if [ -d "${MODEL_BASE}/Qwen3-VL-8B-Instruct" ]; then
    run_finetune "qwen3-vl-8b" "${MODEL_BASE}/Qwen3-VL-8B-Instruct"
fi

# -------------------------------------------------------
# 3. Large models (4-bit quantization across 2×A800)
# -------------------------------------------------------
if [ -d "${MODEL_BASE}/Qwen2.5-VL-32B-Instruct" ]; then
    run_finetune "qwen2.5-vl-32b" "${MODEL_BASE}/Qwen2.5-VL-32B-Instruct" "--use-bnb-4bit"
fi

if [ -d "${MODEL_BASE}/Qwen2.5-VL-72B-Instruct" ]; then
    run_finetune "qwen2.5-vl-72b" "${MODEL_BASE}/Qwen2.5-VL-72B-Instruct" "--use-bnb-4bit"
fi

echo ""
echo "=== All fine-tuning runs complete ==="
echo "Checkpoints saved to: ${CKPT_DIR}/"
ls -la "${CKPT_DIR}/"
