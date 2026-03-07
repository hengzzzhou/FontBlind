"""Download Qwen VL models via ModelScope (fast in China).

Usage on server:
    python /root/fontvlm/code/download_models.py
"""
import os
import sys

MODEL_BASE = "/fs-computility-new/Uma4agi/shared/models"

MODELS_TO_DOWNLOAD = [
    {
        "modelscope_id": "Qwen/Qwen3-VL-8B-Instruct",
        "local_dir": f"{MODEL_BASE}/Qwen3-VL-8B-Instruct",
    },
    {
        "modelscope_id": "Qwen/Qwen2.5-VL-32B-Instruct",
        "local_dir": f"{MODEL_BASE}/Qwen2.5-VL-32B-Instruct",
    },
]


def has_safetensors(path):
    import glob
    return bool(glob.glob(os.path.join(path, "*.safetensors")))


def download_via_modelscope(model_id, local_dir):
    """Download model using modelscope SDK."""
    if os.path.exists(local_dir) and has_safetensors(local_dir):
        print(f"[SKIP] {model_id} — already downloaded at {local_dir}")
        return True

    print(f"[DOWNLOAD] {model_id} -> {local_dir}")
    try:
        from modelscope import snapshot_download
        snapshot_download(
            model_id,
            local_dir=local_dir,
        )
        print(f"[DONE] {model_id}")
        return True
    except Exception as e:
        print(f"[FAIL] {model_id}: {e}")
        return False


def main():
    for m in MODELS_TO_DOWNLOAD:
        # Check disk space
        stat = os.statvfs("/")
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        print(f"\nDisk free: {free_gb:.1f} GB")

        download_via_modelscope(m["modelscope_id"], m["local_dir"])


if __name__ == "__main__":
    main()
