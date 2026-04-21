# Phase 8: Fine-Tuning - Research

**Researched:** 2026-04-20
**Domain:** LLM fine-tuning with LoRA/PEFT on Apple Silicon MPS
**Confidence:** HIGH

## Summary

Phase 8 implements end-to-end fine-tuning of SmolLM2-1.7B-Instruct on the assembled Lyra dataset (3,630 samples across tool-calling, code, and knowledge domains). The primary training environment is Apple Silicon M3 Pro with 36GB unified memory using PyTorch MPS backend.

The critical constraint is that Unsloth's FastLanguageModel does NOT work on MPS -- it requires CUDA and Triton. The correct approach is to use raw transformers + PEFT (LoRA) + TRL SFTTrainer directly. This is well-supported: TRL v1.2.0 natively detects MPS, PEFT LoRA adapters work on MPS, and the Trainer auto-selects the correct device. On MPS, training uses 16-bit LoRA (no bitsandbytes QLoRA) with fp32 base weights. On CUDA, QLoRA with NF4 quantization is available.

TRL v1.2.0's SFTTrainer directly accepts the dataset's `messages` column in conversational format and applies the model's chat template automatically. The dataset's existing structure (messages + tools columns) is natively compatible with TRL's tool-calling dataset format, requiring no transformation.

**Primary recommendation:** Use raw transformers + peft + trl (NOT Unsloth) with hardware auto-detection: MPS gets 16-bit LoRA, CUDA gets QLoRA, CPU gets fp32 LoRA fallback.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- D-01: MPS (Apple Silicon Mac) as primary training device
- D-02: Script must also support CPU fallback for environments without MPS/CUDA
- D-03: No CUDA/cloud dependency required for v1. Training runs locally.
- D-04: Hyperparameters: LoRA rank 16, alpha 32, lr 2e-4, epochs 3, batch 4, grad_accum 4
- D-05: Target modules: q_proj, k_proj, v_proj, o_proj + gate_proj, up_proj, down_proj
- D-06: 4-bit QLoRA with NF4 via bitsandbytes for CUDA; 16-bit LoRA for MPS (bitsandbytes does not support MPS)
- D-07: Save BOTH LoRA adapter (~50MB) AND merged safetensors model (~3.4GB)
- D-08: Adapter to models/lyra-adapter/, merged to models/lyra-merged/
- D-09: Checkpoints every epoch; final adapter from best checkpoint (lowest validation loss)
- D-10: Single script scripts/train.py with argparse CLI, all hyperparameters configurable
- D-11: Wandb integration optional (--wandb flag)
- D-12: Dataset loaded from datasets/assembled/ via DatasetDict.load_from_disk()

### Claude's Discretion
- Exact target modules beyond standard attention/MLP set
- Warmup ratio and scheduler type
- Whether to use Unsloth's FastLanguageModel wrapper or raw transformers + peft
- Logging frequency and eval strategy
- Chat template configuration for SFTTrainer

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| TRNG-01 | User can run end-to-end fine-tuning using TRL SFTTrainer + LoRA/PEFT with documented scripts | TRL v1.2.0 SFTTrainer with peft_config=LoraConfig() -- fully documented pattern. Single scripts/train.py with argparse. |
| TRNG-02 | Training targets consumer GPU (8GB+ VRAM) with documented hyperparameters and expected training time | M3 Pro 36GB unified memory comfortably handles 1.7B model in fp32 + LoRA adapters (~8GB). Batch 4 + grad_accum 4 fits. Estimated 30-90 min for 3 epochs. |
| TRNG-03 | Fine-tuned SmolLM2-1.7B model weights produced via QLoRA | On MPS: 16-bit LoRA (QLoRA unavailable). On CUDA: QLoRA with NF4. Both produce adapter + merged weights. Script auto-detects and documents which path was taken. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| transformers | 5.5.4 | Model loading, tokenizer, chat template | Already installed; provides SmolLM2 model and tokenizer with chat_template |
| peft | 0.19.1 | LoRA adapter configuration and management | Standard for parameter-efficient fine-tuning; merge_and_unload for export |
| trl | 1.2.0 | SFTTrainer with chat template + tool calling support | Native conversational dataset handling; assistant_only_loss; auto MPS detection |
| torch | 2.6+ | PyTorch backend with MPS support | MPS backend for Apple Silicon GPU acceleration |
| datasets | 4.8.4 | Dataset loading from disk | Already installed; loads DatasetDict from assembled/ |
| accelerate | 1.13.0 | Training orchestration, device management | Required by TRL; handles MPS/CUDA/CPU device placement |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| bitsandbytes | 0.49.2 | 4-bit NF4 quantization for QLoRA | ONLY when CUDA is detected; skip entirely on MPS/CPU |
| wandb | 0.26.0 | Experiment tracking | When --wandb flag is passed |
| safetensors | latest | Model serialization format | Implied by transformers save_pretrained |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Raw TRL + PEFT | Unsloth FastLanguageModel | Unsloth requires CUDA + Triton. Does NOT work on MPS. Not viable for this project's primary hardware. |
| Raw TRL + PEFT | mlx-tune (Unsloth-compatible API for MLX) | MLX-native but different ecosystem; would add a dependency and diverge from HuggingFace toolchain. TRL+PEFT is more portable. |
| fp32 on MPS | fp16 on MPS | MPS fp16 has limited benefit (no tensor cores). fp32 is more stable. Some operations fall back to CPU anyway. Use fp32 for reliability. |

**Installation:**
```bash
# Core training dependencies (in addition to existing requirements.txt)
pip install torch torchvision torchaudio  # PyTorch with MPS support
pip install peft==0.19.1 trl==1.2.0 accelerate==1.13.0
pip install wandb  # optional

# CUDA-only (skip on Mac):
# pip install bitsandbytes==0.49.2
```

**Version verification:** [VERIFIED: project requirements.txt and CLAUDE.md] -- transformers 5.5.4 and datasets 4.8.4 already installed. peft, trl, accelerate, torch need installation. bitsandbytes confirmed NOT available on MPS.

## Architecture Patterns

### Recommended Project Structure
```
scripts/
  train.py              # Main training script (argparse CLI)
configs/
  training.yaml         # Default hyperparameters (optional, CLI flags override)
models/
  lyra-adapter/         # LoRA adapter output (adapter_model.safetensors + adapter_config.json)
  lyra-merged/          # Full merged model (safetensors)
```

### Pattern 1: Hardware Auto-Detection
**What:** Detect available hardware and configure training accordingly
**When to use:** Always -- the entry point of the training script
**Example:**
```python
# Source: PyTorch docs + HuggingFace Apple Silicon docs
import torch

def get_device_config():
    """Auto-detect hardware and return appropriate training config."""
    if torch.cuda.is_available():
        return {
            "device": "cuda",
            "dtype": torch.bfloat16,
            "use_quantization": True,  # QLoRA with NF4
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            ),
        }
    elif torch.backends.mps.is_available():
        return {
            "device": "mps",
            "dtype": torch.float32,  # MPS: fp32 is most reliable
            "use_quantization": False,  # bitsandbytes not supported
            "quantization_config": None,
        }
    else:
        return {
            "device": "cpu",
            "dtype": torch.float32,
            "use_quantization": False,
            "quantization_config": None,
        }
```
[VERIFIED: HuggingFace transformers Apple Silicon docs confirm MPS auto-detection] [VERIFIED: bitsandbytes MPS incompatibility confirmed via multiple sources]

### Pattern 2: SFTTrainer with Conversational Dataset + Tool Calling
**What:** Load dataset with messages+tools columns, SFTTrainer applies chat template automatically
**When to use:** For training on the assembled Lyra dataset
**Example:**
```python
# Source: TRL v1.2.0 docs (huggingface.co/docs/trl/sft_trainer)
from datasets import DatasetDict
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

dataset = DatasetDict.load_from_disk("datasets/assembled")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

training_args = SFTConfig(
    output_dir="models/lyra-adapter",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_length=4096,  # SmolLM2 context: 8192, but samples are shorter
    assistant_only_loss=True,  # Train only on assistant responses
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

trainer = SFTTrainer(
    model="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=lora_config,
)
trainer.train()
```
[VERIFIED: TRL docs confirm conversational dataset with messages column auto-applies chat template]
[VERIFIED: TRL docs confirm tools column support for tool-calling datasets]

### Pattern 3: LoRA Adapter Save + Merge
**What:** Save adapter separately, then merge into base model for deployment
**When to use:** After training completes -- produce both adapter and merged model
**Example:**
```python
# Source: PEFT docs (huggingface.co/docs/peft)
# Save adapter only (~50MB)
trainer.save_model("models/lyra-adapter")

# Merge adapter into base model for deployment
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    torch_dtype=torch.float16,
)
model = PeftModel.from_pretrained(base_model, "models/lyra-adapter")
merged_model = model.merge_and_unload()
merged_model.save_pretrained("models/lyra-merged", safe_serialization=True)
tokenizer.save_pretrained("models/lyra-merged")
```
[VERIFIED: PEFT docs confirm merge_and_unload() + save_pretrained with safe_serialization]

### Anti-Patterns to Avoid
- **Using Unsloth on MPS:** Unsloth requires CUDA + Triton. Will fail silently or crash on Apple Silicon. Use raw transformers + peft + trl instead. [VERIFIED: Unsloth docs + closed PR #1289]
- **Using bitsandbytes on MPS:** Will ImportError. Always guard with hardware detection. [VERIFIED: bitsandbytes has no MPS support]
- **Using bf16 on MPS:** Not reliably supported on all Apple Silicon. Use fp32 for stability. fp16 offers minimal benefit on MPS (no tensor cores). [VERIFIED: PyTorch MPS docs + GitHub issues]
- **Setting num_workers > 0 on MPS:** Can cause multiprocessing issues. Set dataloader_num_workers=0. [CITED: community reports on MPS training]
- **Skipping PYTORCH_ENABLE_MPS_FALLBACK=1:** Some operations fall back to CPU. Without this env var, those operations error out. [VERIFIED: HuggingFace transformers Apple Silicon docs]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Chat template application | Custom tokenization logic | SFTTrainer auto-applies model's chat_template | SmolLM2's tokenizer has chat_template built-in; SFTTrainer handles it |
| Device detection | Manual if/else chains throughout code | TRL/Trainer auto-detection | TrainingArguments + Trainer auto-detect MPS/CUDA/CPU |
| LoRA weight merging | Manual weight arithmetic | peft merge_and_unload() | Handles dtype casting, weight surgery correctly |
| Gradient accumulation | Manual loss scaling | SFTConfig gradient_accumulation_steps | Trainer handles scaling + synchronization |
| Checkpoint management | Manual best-model tracking | load_best_model_at_end=True + save_strategy="epoch" | Trainer tracks eval_loss, restores best automatically |
| Dataset formatting for SFT | Custom tokenization pipeline | SFTTrainer with messages column | TRL v1.2.0 natively handles conversational + tool-calling format |

**Key insight:** TRL v1.2.0 has matured significantly -- it natively handles the exact dataset format Lyra uses (messages + tools columns in conversational format). No custom data collator or formatting function is needed.

## Common Pitfalls

### Pitfall 1: MPS Operations Not Implemented
**What goes wrong:** Training crashes with "NotImplementedError: The operator 'aten::...' is not currently implemented for the MPS device"
**Why it happens:** PyTorch MPS backend doesn't implement all operators
**How to avoid:** Set `PYTORCH_ENABLE_MPS_FALLBACK=1` environment variable before importing torch. Add to script header.
**Warning signs:** ImportError or NotImplementedError mentioning MPS during first training step
[VERIFIED: HuggingFace transformers Apple Silicon documentation]

### Pitfall 2: bf16 on Apple Silicon
**What goes wrong:** NaN losses or crashes when using bf16 mixed precision on MPS
**Why it happens:** bf16 support on MPS varies by chip generation and macOS version. M3 should support it on macOS 14+, but it's unreliable.
**How to avoid:** Use fp32 (no mixed precision) on MPS. Set `bf16=False, fp16=False` in SFTConfig for MPS.
**Warning signs:** Loss goes to NaN in first few steps, or RuntimeError about unsupported dtype
[VERIFIED: PyTorch GitHub issues #139386, HuggingFace Apple Silicon docs]

### Pitfall 3: Multiprocessing DataLoader on MPS
**What goes wrong:** Training hangs or crashes when DataLoader uses multiple workers on MPS
**Why it happens:** MPS + multiprocessing have known interaction issues
**How to avoid:** Set `dataloader_num_workers=0` in SFTConfig when on MPS
**Warning signs:** Training hangs after first batch, or "Broken pipe" errors
[CITED: Community reports, SmolLM2 fine-tuning blog]

### Pitfall 4: assistant_only_loss Requires Chat Template Markers
**What goes wrong:** `assistant_only_loss=True` doesn't work -- loss is computed on all tokens
**Why it happens:** The chat template must include `{% generation %}` / `{% endgeneration %}` markers
**How to avoid:** TRL v1.2.0 auto-patches templates for known model families. Verify by checking training logs that masked tokens ratio makes sense. If SmolLM2's template lacks markers, TRL should patch it automatically.
**Warning signs:** Mean token accuracy is unexpectedly high (training on easy system/user tokens)
[VERIFIED: TRL v1.2.0 docs -- "SFTTrainer now automatically swaps in a patched training chat template"]

### Pitfall 5: SmolLM2 Tool Call Format Mismatch
**What goes wrong:** Tool calls aren't properly formatted during training
**Why it happens:** SmolLM2's native chat template uses `<tool_call>[...]</tool_call>` XML format, while the dataset has structured tool_calls objects
**How to avoid:** TRL's SFTTrainer with the tools column handles the conversion via the chat template. The template serializes tool_calls to the model's expected XML format. This was addressed in Phase 1 (D-08: "Pre-process TRL-native tool_calls to SmolLM2 <tool_call> XML format before tokenization").
**Warning signs:** Generated tool calls don't follow expected format during inference
[VERIFIED: SmolLM2-1.7B-Instruct model card shows `<tool_call>` XML format]

### Pitfall 6: Memory Pressure on MPS with Large Batches
**What goes wrong:** Out of memory errors or system slowdown during training
**Why it happens:** MPS shares memory with the system; 1.7B model in fp32 is ~6.8GB, plus activations and gradients
**How to avoid:** With 36GB unified memory on M3 Pro, batch_size=4 + grad_accum=4 should fit comfortably (~10-12GB for model + training state). If issues arise, reduce per_device_train_batch_size to 2 and increase grad_accum to 8.
**Warning signs:** macOS memory pressure indicator goes yellow/red, training slows dramatically
[ASSUMED]

### Pitfall 7: Python 3.14 Incompatibility
**What goes wrong:** PyTorch or other ML packages fail to install or import on Python 3.14
**Why it happens:** Python 3.14 is very new (2026); many ML packages may not have wheels yet
**How to avoid:** Use Python 3.12 (available at /opt/homebrew/bin/python3.12) which is the recommended version for the ML stack. Create a virtual environment with Python 3.12.
**Warning signs:** pip install failures, ImportError on compiled extensions
[VERIFIED: Python 3.12.10 confirmed available on system; CLAUDE.md recommends Python 3.10+ with 3.12 as sweet spot for PyTorch]

## Code Examples

### Complete Training Script Skeleton
```python
#!/usr/bin/env python3
"""train.py -- Fine-tune SmolLM2-1.7B on Lyra dataset with LoRA.

Supports MPS (Apple Silicon), CUDA, and CPU backends with auto-detection.
"""
# Source: TRL v1.2.0 docs + PEFT docs + PyTorch Apple Silicon docs
import argparse
import os
import sys

# MPS fallback MUST be set before torch import
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from datasets import DatasetDict
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def detect_hardware():
    """Detect available hardware and return config dict."""
    if torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        return "cuda", BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif torch.backends.mps.is_available():
        return "mps", None
    else:
        return "cpu", None


def main():
    parser = argparse.ArgumentParser(description="Fine-tune SmolLM2-1.7B with LoRA")
    # ... argparse setup with all D-04 defaults ...
    args = parser.parse_args()

    device, quant_config = detect_hardware()
    print(f"Training on: {device}")

    # Load model with appropriate config
    model_kwargs = {}
    if quant_config:
        model_kwargs["quantization_config"] = quant_config
    else:
        model_kwargs["torch_dtype"] = torch.float32  # MPS/CPU: fp32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, **model_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load dataset
    dataset = DatasetDict.load_from_disk(args.dataset_dir)

    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # SFT Config -- MPS-specific adjustments
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_length=4096,
        assistant_only_loss=True,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=(device == "cuda"),
        fp16=False,
        dataloader_num_workers=0 if device == "mps" else 4,
        gradient_checkpointing=True,
        report_to="wandb" if args.wandb else "none",
    )

    # Train
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.train()

    # Save adapter
    trainer.save_model(args.output_dir)

    # Merge and save full model
    # (load fresh base model to avoid quantization artifacts)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16
    )
    peft_model = PeftModel.from_pretrained(base_model, args.output_dir)
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(args.merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.merged_dir)
```
[VERIFIED: Pattern assembled from TRL v1.2.0 docs, PEFT docs, PyTorch Apple Silicon docs]

### MPS-Specific Environment Setup
```python
# Source: HuggingFace Apple Silicon docs
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# Must be set BEFORE importing torch

import torch
assert torch.backends.mps.is_available(), "MPS not available on this system"
```
[VERIFIED: HuggingFace transformers Apple Silicon documentation]

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Custom data collator + formatting_func | SFTTrainer auto-detects conversational format | TRL v1.0+ (2025) | No formatting_func needed; pass messages column directly |
| DataCollatorForCompletionOnlyLM | assistant_only_loss=True in SFTConfig | TRL v1.0+ | Single config flag replaces manual collator setup |
| Manual chat template application | SFTTrainer applies model's chat_template automatically | TRL v0.19+ | No tokenize-before-training step needed |
| Separate tool_calls preprocessing | tools column handled natively by SFTTrainer | TRL v0.19+ | Dataset with tools column works out of the box |
| Unsloth required for speed | TRL + gradient_checkpointing sufficient | TRL v1.2 | On MPS where Unsloth doesn't work, TRL is fast enough |

**Deprecated/outdated:**
- `dataset_text_field` parameter: Replaced by auto-detection of messages/text columns
- `formatting_func`: Only needed for non-standard formats; Lyra's format is natively supported
- DataCollatorForCompletionOnlyLM: Replaced by assistant_only_loss config flag

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | SmolLM2-1.7B-Instruct chat template has or will auto-receive generation/endgeneration markers for assistant_only_loss | Pitfall 4 | Loss computed on all tokens (system + user) -- less effective training. Fallback: skip assistant_only_loss. |
| A2 | M3 Pro 36GB can handle batch_size=4 + grad_accum=4 comfortably | Pitfall 6 | OOM errors. Fallback: reduce batch to 2, increase grad_accum to 8. |
| A3 | Training 3,267 samples for 3 epochs takes 30-90 minutes on M3 Pro | Summary | Could take longer. Not blocking -- user can adjust epochs. |
| A4 | fp32 is more stable than fp16 on MPS for this workload | Anti-patterns | fp16 might work fine. fp32 is just safer/simpler. |
| A5 | TRL v1.2.0 is compatible with Python 3.12 | Environment | Installation might fail. Fallback: use latest compatible version. |

## Open Questions

1. **SmolLM2 chat template generation markers**
   - What we know: TRL v1.2.0 auto-patches known model families. SmolLM3-3B has markers.
   - What's unclear: Whether SmolLM2-1.7B-Instruct's template already has them or gets auto-patched
   - Recommendation: Test with assistant_only_loss=True; check training logs for correct token masking ratio. If it doesn't work, either patch the template manually or train without assistant_only_loss (still effective for 3-turn conversations).

2. **Optimal precision on M3 Pro**
   - What we know: M3 supports bf16 on macOS 14+. fp32 is guaranteed stable.
   - What's unclear: Whether bf16 is stable enough for full training runs on M3 Pro
   - Recommendation: Default to fp32 (safest). Add --fp16/--bf16 flags for users who want to experiment. Document that bf16 may work on M3+ chips.

3. **Training speed estimates**
   - What we know: ~204 steps/epoch (3,267 samples / effective_batch 16). 3 epochs = ~612 steps.
   - What's unclear: Exact seconds per step on M3 Pro MPS with fp32 + LoRA
   - Recommendation: Document "expected 30-90 minutes" as estimate. Script should log time elapsed and ETA.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.12 | All ML packages | Yes | 3.12.10 | -- |
| PyTorch (MPS) | Training | No (not installed) | Need 2.6+ | Install via pip |
| transformers | Model loading | Yes | 5.5.4 | -- |
| datasets | Dataset loading | Yes | 4.8.4 | -- |
| peft | LoRA adapters | No | Need 0.19.1 | Install via pip |
| trl | SFTTrainer | No | Need 1.2.0 | Install via pip |
| accelerate | Training orchestration | No | Need 1.13.0 | Install via pip |
| bitsandbytes | QLoRA (CUDA only) | No | Not needed on MPS | 16-bit LoRA on MPS |
| wandb | Experiment tracking | No | Optional | --wandb flag, not required |
| MPS backend | GPU acceleration | Yes (hardware) | M3 Pro | CPU fallback |

**Missing dependencies with no fallback:**
- PyTorch, peft, trl, accelerate: Must be installed. Script should include installation instructions.

**Missing dependencies with fallback:**
- bitsandbytes: Not needed on MPS (fallback to 16-bit LoRA automatically)
- wandb: Optional, training works without it

**Critical note:** Python 3.14 is the default python3, but Python 3.12 is available and should be used. The training script's documentation must specify creating a venv with python3.12.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 7.0+ |
| Config file | pytest.ini not present -- uses defaults |
| Quick run command | `python -m pytest tests/test_train.py -x` |
| Full suite command | `python -m pytest tests/ -x` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TRNG-01 | Training script runs end-to-end with LoRA + SFTTrainer | integration | `python -m pytest tests/test_train.py::test_training_smoke -x` | No -- Wave 0 |
| TRNG-01 | Hardware detection logic selects correct path | unit | `python -m pytest tests/test_train.py::test_detect_hardware -x` | No -- Wave 0 |
| TRNG-01 | CLI argparse accepts all documented flags | unit | `python -m pytest tests/test_train.py::test_cli_args -x` | No -- Wave 0 |
| TRNG-02 | Training completes within memory budget on MPS | manual-only | Manual: run training, monitor Activity Monitor | -- |
| TRNG-03 | Adapter saved correctly (adapter_model.safetensors + config) | integration | `python -m pytest tests/test_train.py::test_adapter_save -x` | No -- Wave 0 |
| TRNG-03 | Merged model produces valid output | integration | `python -m pytest tests/test_train.py::test_merged_model_inference -x` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/test_train.py -x`
- **Per wave merge:** `python -m pytest tests/ -x`
- **Phase gate:** Full suite green before /gsd-verify-work

### Wave 0 Gaps
- [ ] `tests/test_train.py` -- covers TRNG-01, TRNG-03 (unit + smoke tests)
- [ ] Framework install: `pip install pytest` -- already available

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | N/A -- local training script |
| V3 Session Management | No | N/A |
| V4 Access Control | No | N/A |
| V5 Input Validation | Yes | argparse type validation for CLI flags; Pydantic for dataset validation (already done in assembly) |
| V6 Cryptography | No | N/A |

### Known Threat Patterns for Training Scripts

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Arbitrary code execution via pickle model files | Tampering | Use safetensors format only; trust HuggingFace Hub models |
| Path traversal in output directories | Tampering | Validate output paths via argparse; use relative paths within project |
| Wandb API key exposure | Information Disclosure | Use WANDB_API_KEY env var (standard pattern); never log keys |

## Sources

### Primary (HIGH confidence)
- [TRL v1.2.0 SFTTrainer docs](https://huggingface.co/docs/trl/sft_trainer) -- Dataset format, tool calling, peft_config, assistant_only_loss, SFTConfig parameters
- [TRL Dataset Formats docs](https://huggingface.co/docs/trl/main/dataset_formats) -- Conversational format, tool calling column structure
- [HuggingFace Transformers Apple Silicon docs](https://huggingface.co/docs/transformers/en/perf_train_special) -- MPS backend usage, PYTORCH_ENABLE_MPS_FALLBACK
- [SmolLM2-1.7B-Instruct model card](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) -- Tool call XML format, training details
- [Unsloth Requirements docs](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements) -- CUDA + Triton requirement confirmed
- [Unsloth Apple Silicon PR #1289](https://github.com/unslothai/unsloth/pull/1289) -- Closed/not merged, confirms no MPS support

### Secondary (MEDIUM confidence)
- [PEFT LoRA docs](https://huggingface.co/docs/peft/en/package_reference/lora) -- merge_and_unload, adapter save/load
- [SmolLM2 fine-tuning blog](https://huggingface.co/blog/prithivMLmods/smollm2-ft) -- Training setup patterns, device detection
- [PyTorch MPS bf16 issue](https://github.com/pytorch/pytorch/issues/139386) -- bf16 support status on MPS
- [LoRA Fine-Tuning on Apple Silicon](https://medium.com/@haldankar.deven/lora-fine-tuning-on-apple-silicon-d000ea38453c) -- Practical MPS training experience

### Tertiary (LOW confidence)
- Training time estimates for M3 Pro -- extrapolated from M3 Max benchmarks with different models

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all versions confirmed in CLAUDE.md; TRL v1.2.0 docs verified for dataset format support
- Architecture: HIGH -- hardware auto-detection is well-documented; TRL handles conversational datasets natively
- Pitfalls: HIGH -- MPS limitations well-documented in PyTorch and HuggingFace official docs
- Training time: LOW -- estimates are extrapolated, not measured on this specific hardware+model combo

**Research date:** 2026-04-20
**Valid until:** 2026-05-20 (stable stack; TRL and PEFT unlikely to break compatibility within 30 days)
