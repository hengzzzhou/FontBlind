#!/bin/bash
# Download models via wget (bypasses xethub CDN) and train sequentially.
# Usage: nohup bash /root/fontvlm/code/download_and_train.sh > /root/fontvlm/finetune.log 2>&1 &

set -e
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1

WORK_DIR="/root/fontvlm"
DATA="${WORK_DIR}/train_data/train.jsonl"
CKPT_DIR="${WORK_DIR}/checkpoints"
MODEL_BASE="/fs-computility-new/Uma4agi/shared/models"

download_model_wget() {
    local repo=$1   # e.g. Qwen/Qwen3-VL-8B-Instruct
    local dir=$2     # e.g. /fs-computility-new/.../Qwen3-VL-8B-Instruct
    local mirror="https://hf-mirror.com"

    mkdir -p "$dir"

    # Get the file list from model.safetensors.index.json first
    echo "  Downloading file list from ${repo}..."
    wget -q -O "$dir/model.safetensors.index.json" "${mirror}/${repo}/resolve/main/model.safetensors.index.json" 2>/dev/null || true

    # Download all safetensors files
    local shards=$(python3 -c "
import json, sys
try:
    idx = json.load(open('$dir/model.safetensors.index.json'))
    files = sorted(set(idx['weight_map'].values()))
    for f in files:
        print(f)
except: pass
")

    for shard in $shards; do
        if [ -f "$dir/$shard" ]; then
            echo "  [EXISTS] $shard"
        else
            echo "  [DOWNLOAD] $shard ..."
            wget -q --show-progress -O "$dir/$shard" "${mirror}/${repo}/resolve/main/${shard}"
        fi
    done

    # Download config and tokenizer files
    for f in config.json generation_config.json preprocessor_config.json tokenizer.json tokenizer_config.json merges.txt vocab.json chat_template.json video_preprocessor_config.json special_tokens_map.json; do
        if [ ! -f "$dir/$f" ]; then
            wget -q -O "$dir/$f" "${mirror}/${repo}/resolve/main/${f}" 2>/dev/null || true
        fi
    done

    echo "  Download complete. Files:"
    ls -lh "$dir"/*.safetensors 2>/dev/null | wc -l
    echo "  safetensors files"
}

run_training() {
    local model_key=$1
    local model_dir=$2
    local extra=$3
    local output="${CKPT_DIR}/${model_key}_lora"

    if [ -f "$output/adapter_config.json" ]; then
        echo "[SKIP] ${model_key} — already trained"
        return 0
    fi

    echo ""
    echo "============================================"
    echo "[TRAIN] ${model_key}"
    echo "  Model: ${model_dir}"
    echo "  Output: ${output}"
    echo "============================================"

    python "${WORK_DIR}/code/train_lora.py" \
        --model "${model_key}" \
        --model-path "${model_dir}" \
        --data "${DATA}" \
        --output "${output}" \
        ${extra}

    echo "[DONE] ${model_key} adapter saved at ${output}"
}

echo "============================================"
echo "FontBench Fine-tuning Pipeline"
echo "============================================"

# 1. Qwen2.5-VL-7B (already done if checkpoint exists)
echo ""
echo "=== Model 1: Qwen2.5-VL-7B ==="
run_training "qwen2.5-vl-7b" "${MODEL_BASE}/Qwen2.5-VL-7B-Instruct" ""

# 2. Qwen3-VL-8B
echo ""
echo "=== Model 2: Qwen3-VL-8B ==="
DIR="${MODEL_BASE}/Qwen3-VL-8B-Instruct"
if [ ! -f "$DIR/model-00001-of-00004.safetensors" ]; then
    echo "  Downloading Qwen3-VL-8B-Instruct via wget..."
    download_model_wget "Qwen/Qwen3-VL-8B-Instruct" "$DIR"
fi
run_training "qwen3-vl-8b" "$DIR" ""

# Clean HF cache to free space
echo "  Cleaning HF cache..."
rm -rf ~/.cache/huggingface/hub/models--Qwen* 2>/dev/null

# 3. Qwen2.5-VL-32B (4-bit)
echo ""
echo "=== Model 3: Qwen2.5-VL-32B ==="
FREE=$(df / --output=avail | tail -1 | tr -d ' ')
FREE_GB=$((FREE / 1024 / 1024))
echo "  Disk free: ${FREE_GB}GB"
DIR="${MODEL_BASE}/Qwen2.5-VL-32B-Instruct"
if [ "$FREE_GB" -lt 66 ]; then
    echo "[SKIP] Qwen2.5-VL-32B — not enough disk (${FREE_GB}GB < 66GB)"
else
    if [ ! -f "$DIR/model-00001-of-00019.safetensors" ]; then
        echo "  Downloading Qwen2.5-VL-32B-Instruct via wget..."
        download_model_wget "Qwen/Qwen2.5-VL-32B-Instruct" "$DIR"
    fi
    run_training "qwen2.5-vl-32b" "$DIR" "--use-bnb-4bit"
fi

# 4. Qwen2.5-VL-72B (4-bit) — needs ~145GB, likely won't fit
echo ""
echo "=== Model 4: Qwen2.5-VL-72B ==="
FREE=$(df / --output=avail | tail -1 | tr -d ' ')
FREE_GB=$((FREE / 1024 / 1024))
echo "  Disk free: ${FREE_GB}GB"
if [ "$FREE_GB" -lt 146 ]; then
    echo "[SKIP] Qwen2.5-VL-72B — not enough disk (${FREE_GB}GB < 146GB)"
else
    DIR="${MODEL_BASE}/Qwen2.5-VL-72B-Instruct"
    if [ ! -f "$DIR/model-00001-of-00037.safetensors" ]; then
        echo "  Downloading Qwen2.5-VL-72B-Instruct via wget..."
        download_model_wget "Qwen/Qwen2.5-VL-72B-Instruct" "$DIR"
    fi
    run_training "qwen2.5-vl-72b" "$DIR" "--use-bnb-4bit"
fi

echo ""
echo "============================================"
echo "TRAINING COMPLETE"
echo "============================================"
echo "Checkpoints:"
ls -la "${CKPT_DIR}/"
echo ""
echo "Disk usage:"
df -h /
