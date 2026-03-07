"""Serve fine-tuned LoRA models via vLLM for evaluation.

Usage:
    # Serve a single model:
    python -m fontbench.finetuning.serve_lora \
        --base-model /fs-computility-new/Uma4agi/shared/models/Qwen2.5-VL-7B-Instruct \
        --lora-path /root/fontvlm/checkpoints/qwen2.5-vl-7b_lora \
        --port 8001

    # Then evaluate from local machine:
    python -m fontbench.finetuning.eval_finetuned \
        --api-base http://101.126.156.90:8001/v1 \
        --model-name "Qwen2.5-VL-7B+LoRA"
"""
import argparse
import subprocess
import sys


def serve_model(base_model, lora_path, port=8001, gpu_memory_utilization=0.9):
    """Launch vLLM server with LoRA adapter."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", base_model,
        "--enable-lora",
        "--lora-modules", f"font-lora={lora_path}",
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--trust-remote-code",
        "--dtype", "bfloat16",
        "--max-model-len", "4096",
    ]

    print(f"Starting vLLM server:")
    print(f"  Base model: {base_model}")
    print(f"  LoRA: {lora_path}")
    print(f"  Port: {port}")
    print(f"  API: http://0.0.0.0:{port}/v1")
    print()

    subprocess.run(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve fine-tuned LoRA model via vLLM")
    parser.add_argument("--base-model", required=True, help="Path to base model")
    parser.add_argument("--lora-path", required=True, help="Path to LoRA adapter checkpoint")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()

    serve_model(args.base_model, args.lora_path, args.port, args.gpu_memory_utilization)
