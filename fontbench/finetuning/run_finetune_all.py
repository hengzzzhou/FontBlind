"""Master fine-tuning script: download models then train sequentially.

Downloads models to /fs-computility-new/Uma4agi/shared/models/ first,
then fine-tunes using local weights. Manages disk space between runs.

Run on GPU server (2×A800 80GB):
    nohup python /root/fontvlm/code/run_finetune_all.py > /root/fontvlm/finetune.log 2>&1 &
"""
import subprocess
import sys
import os
import shutil
import time

# Use Chinese HF mirror (huggingface.co is blocked on this server)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# Disable xet transport (xethub CDN is unreliable from this server)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"

WORK_DIR = "/root/fontvlm"
DATA_PATH = f"{WORK_DIR}/train_data/train.jsonl"
CKPT_DIR = f"{WORK_DIR}/checkpoints"
MODEL_BASE = "/fs-computility-new/Uma4agi/shared/models"
HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub")

# Models to train in order of size (download one, train, optionally clean, repeat)
MODELS = [
    {
        "key": "qwen2.5-vl-7b",
        "hf_repo": "Qwen/Qwen2.5-VL-7B-Instruct",
        "local_dir": f"{MODEL_BASE}/Qwen2.5-VL-7B-Instruct",
        "extra_args": [],
        "approx_gb": 15,
    },
    {
        "key": "qwen3-vl-8b",
        "hf_repo": "Qwen/Qwen3-VL-8B-Instruct",
        "local_dir": f"{MODEL_BASE}/Qwen3-VL-8B-Instruct",
        "extra_args": [],
        "approx_gb": 17,
    },
    {
        "key": "qwen2.5-vl-32b",
        "hf_repo": "Qwen/Qwen2.5-VL-32B-Instruct",
        "local_dir": f"{MODEL_BASE}/Qwen2.5-VL-32B-Instruct",
        "extra_args": ["--use-bnb-4bit"],
        "approx_gb": 65,
    },
    {
        "key": "qwen2.5-vl-72b",
        "hf_repo": "Qwen/Qwen2.5-VL-72B-Instruct",
        "local_dir": f"{MODEL_BASE}/Qwen2.5-VL-72B-Instruct",
        "extra_args": ["--use-bnb-4bit"],
        "approx_gb": 145,
    },
]


def get_disk_free_gb(path="/"):
    stat = os.statvfs(path)
    return (stat.f_bavail * stat.f_frsize) / (1024**3)


def clear_hf_cache():
    """Remove HuggingFace download cache to free temp space."""
    if os.path.exists(HF_CACHE):
        for item in os.listdir(HF_CACHE):
            path = os.path.join(HF_CACHE, item)
            if item.startswith("models--Qwen") and os.path.isdir(path):
                print(f"  Clearing HF cache: {path}")
                shutil.rmtree(path, ignore_errors=True)


def download_model(hf_repo, local_dir):
    """Download model from HuggingFace to local directory."""
    import glob
    if os.path.exists(local_dir) and os.path.exists(os.path.join(local_dir, "config.json")):
        safetensors = glob.glob(os.path.join(local_dir, "*.safetensors"))
        if safetensors:
            print(f"  Model already exists at {local_dir} ({len(safetensors)} safetensors files)")
            return True
        else:
            print(f"  Model dir exists but missing weight files, re-downloading...")

    print(f"  Downloading {hf_repo} -> {local_dir}")
    cmd = [
        "huggingface-cli", "download",
        hf_repo,
        "--local-dir", local_dir,
    ]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  Download failed with exit code {result.returncode}")
        return False

    # Clear HF cache after download (files are now in local_dir)
    clear_hf_cache()
    return True


def train_model(model_info):
    """Fine-tune a single model using local weights."""
    key = model_info["key"]
    output_dir = f"{CKPT_DIR}/{key}_lora"

    # Skip if already trained
    if os.path.exists(f"{output_dir}/adapter_config.json"):
        print(f"[SKIP] {key} — already trained at {output_dir}")
        return True

    # Check disk space (minimal margin - LoRA adapters are small)
    free_gb = get_disk_free_gb()
    needed = model_info["approx_gb"]
    if free_gb < needed + 1:
        print(f"[SKIP] {key} — need ~{needed}GB free but only {free_gb:.1f}GB available")
        return False

    print(f"\n{'='*60}")
    print(f"[START] {key}")
    print(f"  Disk free: {free_gb:.1f} GB (need ~{needed} GB)")
    print(f"{'='*60}")

    # Step 1: Download model
    if not download_model(model_info["hf_repo"], model_info["local_dir"]):
        return False

    print(f"  Disk after download: {get_disk_free_gb():.1f} GB")

    # Step 2: Fine-tune
    print(f"  Starting LoRA fine-tuning...")
    cmd = [
        sys.executable, "/root/fontvlm/code/train_lora.py",
        "--model", key,
        "--model-path", model_info["local_dir"],
        "--data", DATA_PATH,
        "--output", output_dir,
    ] + model_info["extra_args"]

    start = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"[FAIL] {key} failed after {elapsed/60:.1f} min")
        return False

    print(f"[DONE] {key} completed in {elapsed/60:.1f} min")
    print(f"  Adapter saved at: {output_dir}")
    print(f"  Disk free: {get_disk_free_gb():.1f} GB")
    return True


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)

    print("=" * 60)
    print("FontBench Batch Fine-tuning")
    print(f"Training data: {DATA_PATH}")
    print(f"Checkpoint dir: {CKPT_DIR}")
    print(f"Model dir: {MODEL_BASE}")
    print(f"Disk free: {get_disk_free_gb():.1f} GB")
    print(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', 'default')}")
    print("=" * 60)

    # Clear any leftover HF cache first
    clear_hf_cache()

    results = {}
    for model_info in MODELS:
        key = model_info["key"]
        success = train_model(model_info)
        results[key] = "success" if success else "skipped/failed"

    print("\n" + "=" * 60)
    print("BATCH TRAINING SUMMARY")
    print("=" * 60)
    for key, status in results.items():
        print(f"  {key}: {status}")
    print(f"\nCheckpoints:")
    subprocess.run(["ls", "-la", CKPT_DIR])


if __name__ == "__main__":
    main()
