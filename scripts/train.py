#!/usr/bin/env python3
"""train.py -- Fine-tune SmolLM2-1.7B on the Lyra dataset with LoRA.

Supports MPS (Apple Silicon), CUDA, and CPU backends with auto-detection.
Uses TRL SFTTrainer + PEFT LoRA for parameter-efficient fine-tuning.

The script loads the assembled Lyra dataset (tool-calling, code, knowledge),
configures LoRA adapters targeting all attention and MLP projection layers,
and trains with assistant-only loss. After training, the LoRA adapter is
saved and optionally merged into the base model for deployment.

Usage:
    python scripts/train.py                    # Train with defaults
    python scripts/train.py --help             # Show all flags
    python scripts/train.py --epochs 5 --lr 1e-4 --wandb
    python scripts/train.py --no-merge         # Skip model merging

Requirements:
    pip install torch peft trl accelerate
    # CUDA-only (optional): pip install bitsandbytes
"""
import argparse
import os
import sys
import time
from pathlib import Path

# MPS fallback MUST be set before torch import (Pitfall 1)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def detect_hardware():
    """Detect available hardware and return device string and quantization config.

    Returns:
        Tuple of (device_str, quantization_config_or_None):
        - CUDA: ("cuda", BitsAndBytesConfig with NF4 4-bit quantization)
        - MPS: ("mps", None) -- bitsandbytes not supported on MPS
        - CPU: ("cpu", None) -- fallback, no quantization
    """
    import torch

    if torch.cuda.is_available():
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        return "cuda", quant_config
    elif torch.backends.mps.is_available():
        return "mps", None
    else:
        return "cpu", None


def build_parser():
    """Build argparse parser with all training hyperparameters.

    All flags have documented defaults matching D-04 hyperparameters.
    Paths use D-08/D-10/D-12 conventions.

    Returns:
        argparse.ArgumentParser configured with all training flags.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune SmolLM2-1.7B on the Lyra dataset with LoRA/PEFT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model and data paths
    parser.add_argument(
        "--model-name",
        type=str,
        default="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="datasets/assembled",
        help="Path to assembled DatasetDict (from assemble_dataset.py)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/lyra-adapter",
        help="Output directory for LoRA adapter weights",
    )
    parser.add_argument(
        "--merged-dir",
        type=str,
        default="models/lyra-merged",
        help="Output directory for merged full model",
    )

    # LoRA hyperparameters (D-04, D-05)
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (r parameter)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha scaling factor",
    )

    # Training hyperparameters (D-04)
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device training batch size",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Maximum sequence length for training samples",
    )

    # Optional features (D-11)
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Enable Weights & Biases experiment tracking",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        default=False,
        help="Skip merging adapter into base model after training",
    )

    return parser


def get_lora_config(args):
    """Build LoRA configuration from CLI arguments.

    Targets all 7 projection layers per D-05:
    - Attention: q_proj, k_proj, v_proj, o_proj
    - MLP: gate_proj, up_proj, down_proj

    Args:
        args: Parsed CLI arguments with lora_r and lora_alpha.

    Returns:
        LoraConfig configured for causal language modeling.
    """
    from peft import LoraConfig

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


def get_training_args(args, device):
    """Build SFTConfig training arguments with device-specific settings.

    MPS-specific (Pitfalls 2, 3):
    - bf16=False, fp16=False (use fp32 for stability)
    - dataloader_num_workers=0 (avoid multiprocessing issues)

    CUDA-specific:
    - bf16=True for mixed precision
    - dataloader_num_workers=4 for data loading throughput

    Common (D-09, D-11):
    - assistant_only_loss=True (train only on assistant responses)
    - save_strategy/eval_strategy="epoch" with load_best_model_at_end
    - gradient_checkpointing for memory efficiency

    Args:
        args: Parsed CLI arguments with training hyperparameters.
        device: Device string ("mps", "cuda", or "cpu").

    Returns:
        SFTConfig with device-appropriate settings.
    """
    from trl import SFTConfig

    return SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_length=args.max_length,
        assistant_only_loss=True,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        logging_steps=10,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=(device == "cuda"),
        fp16=False,
        dataloader_num_workers=0 if device == "mps" else 4,
        report_to="wandb" if args.wandb else "none",
    )


def main():
    """Run the complete fine-tuning pipeline.

    Steps:
    1. Parse CLI arguments
    2. Detect hardware (MPS/CUDA/CPU)
    3. Load model with appropriate quantization config
    4. Load tokenizer
    5. Load assembled dataset from disk
    6. Configure LoRA adapters
    7. Configure training arguments
    8. Create SFTTrainer and train
    9. Save LoRA adapter
    10. Optionally merge adapter into base model and save
    """
    import torch
    from datasets import DatasetDict
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer

    parser = build_parser()
    args = parser.parse_args()

    # Validate paths (T-08-02: use Path objects for safety)
    output_dir = Path(args.output_dir)
    merged_dir = Path(args.merged_dir)
    dataset_dir = Path(args.dataset_dir)

    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    # Step 1: Detect hardware
    device, quant_config = detect_hardware()
    print(f"[Lyra] Hardware detected: {device}")
    if device == "cuda" and quant_config:
        print("[Lyra] Using QLoRA with NF4 4-bit quantization")
    elif device == "mps":
        print("[Lyra] Using LoRA with fp32 (bitsandbytes not available on MPS)")
    else:
        print("[Lyra] Using LoRA with fp32 (CPU fallback)")

    # Step 2: Load model
    print(f"[Lyra] Loading model: {args.model_name}")
    model_kwargs = {}
    if quant_config:
        model_kwargs["quantization_config"] = quant_config
    else:
        model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, **model_kwargs
    )

    # Step 3: Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Step 4: Load dataset
    print(f"[Lyra] Loading dataset from: {dataset_dir}")
    dataset = DatasetDict.load_from_disk(str(dataset_dir))
    print(
        f"[Lyra] Dataset loaded: "
        f"train={len(dataset['train'])}, "
        f"validation={len(dataset['validation'])}, "
        f"test={len(dataset['test'])}"
    )

    # Step 5: Configure LoRA
    lora_config = get_lora_config(args)
    print(
        f"[Lyra] LoRA config: r={lora_config.r}, "
        f"alpha={lora_config.lora_alpha}, "
        f"modules={lora_config.target_modules}"
    )

    # Step 6: Configure training
    training_args = get_training_args(args, device)

    # Step 7: Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # Step 8: Train
    print("[Lyra] Starting training...")
    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"[Lyra] Training complete in {minutes}m {seconds}s")

    # Step 9: Save adapter
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    print(f"[Lyra] Adapter saved to: {output_dir}")

    # Step 10: Merge and save (unless --no-merge)
    if not args.no_merge:
        print("[Lyra] Merging adapter into base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.float16
        )
        peft_model = PeftModel.from_pretrained(base_model, str(output_dir))
        merged = peft_model.merge_and_unload()

        merged_dir.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(str(merged_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(merged_dir))
        print(f"[Lyra] Merged model saved to: {merged_dir}")
    else:
        print("[Lyra] Skipping merge (--no-merge flag set)")

    print("[Lyra] Done.")


if __name__ == "__main__":
    main()
