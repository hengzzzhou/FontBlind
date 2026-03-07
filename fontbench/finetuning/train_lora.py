"""LoRA fine-tuning script for Qwen VL models on font recognition.

Run on GPU server (2×A800 80GB):
    ssh -p 7148 root@101.126.156.90

Usage:
    python -m fontbench.finetuning.train_lora \
        --model qwen2.5-vl-7b \
        --data /path/to/train.jsonl \
        --output /path/to/checkpoints/qwen25vl7b_lora

    # Use 4-bit quantization for large models (32B, 72B):
    python -m fontbench.finetuning.train_lora \
        --model qwen2.5-vl-72b \
        --data /path/to/train.jsonl \
        --output /path/to/checkpoints/qwen25vl72b_lora \
        --use-bnb-4bit

    # Use local model weights:
    python -m fontbench.finetuning.train_lora \
        --model qwen2.5-vl-7b \
        --model-path /fs-computility-new/Uma4agi/shared/models/Qwen2.5-VL-7B-Instruct \
        --data /path/to/train.jsonl \
        --output /path/to/checkpoints/qwen25vl7b_lora
"""
import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


MODEL_CONFIGS = {
    # --- Small models (single GPU, ~16-24GB) ---
    "qwen2.5-vl-3b": {
        "model_name": "Qwen/Qwen2.5-VL-3B-Instruct",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "learning_rate": 2e-4,
        "num_epochs": 3,
        "batch_size": 2,
        "gradient_accumulation_steps": 8,
        "max_length": 512,
    },
    "qwen3-vl-4b": {
        "model_name": "Qwen/Qwen3-VL-4B-Instruct",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "learning_rate": 2e-4,
        "num_epochs": 3,
        "batch_size": 2,
        "gradient_accumulation_steps": 8,
        "max_length": 512,
    },
    # --- Medium models (single A800 80GB) ---
    "qwen2.5-vl-7b": {
        "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "learning_rate": 2e-4,
        "num_epochs": 3,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "max_length": 512,
    },
    "qwen3-vl-8b": {
        "model_name": "Qwen/Qwen3-VL-8B-Instruct",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "learning_rate": 2e-4,
        "num_epochs": 3,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "max_length": 512,
    },
    # --- Large models (2×A800 or 4-bit quantization) ---
    "qwen2.5-vl-32b": {
        "model_name": "Qwen/Qwen2.5-VL-32B-Instruct",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "learning_rate": 1e-4,
        "num_epochs": 3,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "max_length": 512,
    },
    "qwen2.5-vl-72b": {
        "model_name": "Qwen/Qwen2.5-VL-72B-Instruct",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "learning_rate": 5e-5,
        "num_epochs": 3,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "max_length": 512,
    },
}


class FontVLMDataset(Dataset):
    """Dataset for VLM fine-tuning with image+conversation pairs."""

    def __init__(self, data_path, processor, max_length=512):
        self.data = []
        self.data_dir = Path(data_path).parent
        with open(data_path) as f:
            for line in f:
                self.data.append(json.loads(line))
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = self.data_dir / item["image"]
        image = Image.open(image_path).convert("RGB")

        # Build conversation in Qwen VL format
        human_msg = item["conversations"][0]["value"].replace("<image>", "").strip()
        assistant_msg = item["conversations"][1]["value"]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": human_msg},
                ],
            },
        ]

        # Process input using the processor
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=False,
            return_tensors="pt",
        )

        # Tokenize the assistant response for labels
        answer_ids = self.processor.tokenizer(
            assistant_msg, add_special_tokens=False, return_tensors="pt"
        ).input_ids[0]

        # Concatenate input_ids with answer_ids + eos
        eos_id = self.processor.tokenizer.eos_token_id
        input_ids = inputs["input_ids"][0]
        full_ids = torch.cat([input_ids, answer_ids, torch.tensor([eos_id])])

        # Labels: -100 for input prefix, actual ids for answer
        labels = torch.full_like(full_ids, -100)
        labels[len(input_ids):] = full_ids[len(input_ids):]

        # Truncate if needed
        if len(full_ids) > self.max_length:
            full_ids = full_ids[:self.max_length]
            labels = labels[:self.max_length]

        return {
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(full_ids),
            "pixel_values": inputs.get("pixel_values", inputs.get("pixel_values_videos")),
            "image_grid_thw": inputs.get("image_grid_thw"),
        }


def collate_fn(batch):
    """Collate function that pads sequences."""
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids = []
    labels = []
    attention_mask = []

    for item in batch:
        pad_len = max_len - item["input_ids"].size(0)
        input_ids.append(torch.cat([item["input_ids"], torch.zeros(pad_len, dtype=torch.long)]))
        labels.append(torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
        attention_mask.append(torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))

    result = {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
    }

    # Handle pixel values (may have different shapes per image)
    if batch[0].get("pixel_values") is not None:
        result["pixel_values"] = torch.cat([item["pixel_values"] for item in batch if item["pixel_values"] is not None])
    if batch[0].get("image_grid_thw") is not None:
        result["image_grid_thw"] = torch.cat([item["image_grid_thw"] for item in batch if item["image_grid_thw"] is not None])

    return result


def train(model_key, data_path, output_dir, use_bnb_4bit=False):
    """Run LoRA fine-tuning on Qwen VL model."""
    from transformers import AutoModelForImageTextToText, AutoProcessor, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model, TaskType

    cfg = MODEL_CONFIGS[model_key]
    print(f"Loading model: {cfg['model_name']}")
    print(f"  LoRA r={cfg['lora_r']}, alpha={cfg['lora_alpha']}")
    print(f"  4-bit quantization: {use_bnb_4bit}")

    # Load processor
    processor = AutoProcessor.from_pretrained(cfg["model_name"], trust_remote_code=True)

    # Build model loading kwargs
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if use_bnb_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForImageTextToText.from_pretrained(cfg["model_name"], **model_kwargs)

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    dataset = FontVLMDataset(data_path, processor, max_length=cfg["max_length"])
    print(f"Loaded {len(dataset)} training samples")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg["num_epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=0.05,
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        bf16=True,
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )

    print(f"Starting training...")
    trainer.train()

    # Save the LoRA adapter
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_CONFIGS.keys()), required=True)
    parser.add_argument("--model-path", type=str, default=None,
                        help="Local path to model (overrides HuggingFace download)")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--use-bnb-4bit", action="store_true",
                        help="Load base model in 4-bit quantization (for 32B/72B models)")
    args = parser.parse_args()
    if args.model_path:
        MODEL_CONFIGS[args.model]["model_name"] = args.model_path
    train(args.model, args.data, args.output, use_bnb_4bit=args.use_bnb_4bit)
