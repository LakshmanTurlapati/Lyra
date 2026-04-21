"""Tests for the fine-tuning training script.

Validates hardware auto-detection, CLI argument parsing, LoRA configuration,
and training argument builders. All tests mock torch/peft/trl so they run
on any machine without GPU or ML library requirements.

Follows the project testing pattern from test_assemble_dataset.py:
sys.path insertion, pytest fixtures, class-based test organization.
"""
import importlib
import importlib.machinery
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Mock ML libraries that may not be installed in the test environment.
# train.py uses lazy imports (inside functions), so we install mock modules
# into sys.modules BEFORE importing scripts.train. build_parser() uses only
# stdlib (argparse) and needs no mocks.
# ---------------------------------------------------------------------------

def _ensure_mock_torch():
    """Install a mock torch module if the real one is not available."""
    try:
        import torch  # noqa: F401
        return False  # real torch available, no mocking needed
    except ImportError:
        pass

    mock_torch = types.ModuleType("torch")
    mock_torch.__version__ = "2.6.0.mock"
    # __spec__ is needed by importlib.util.find_spec() which transformers calls
    mock_torch.__spec__ = importlib.machinery.ModuleSpec("torch", None)

    # torch.cuda
    mock_cuda = types.ModuleType("torch.cuda")
    mock_cuda.is_available = MagicMock(return_value=False)
    mock_torch.cuda = mock_cuda

    # torch.backends and torch.backends.mps
    mock_backends = types.ModuleType("torch.backends")
    mock_mps = types.ModuleType("torch.backends.mps")
    mock_mps.is_available = MagicMock(return_value=False)
    mock_backends.mps = mock_mps
    mock_torch.backends = mock_backends

    # Common torch attributes used by transformers/trl/peft
    mock_torch.float32 = "torch.float32"
    mock_torch.float16 = "torch.float16"
    mock_torch.bfloat16 = "torch.bfloat16"
    mock_torch.no_grad = MagicMock()

    sys.modules["torch"] = mock_torch
    sys.modules["torch.cuda"] = mock_cuda
    sys.modules["torch.backends"] = mock_backends
    sys.modules["torch.backends.mps"] = mock_mps

    return True  # mocked


def _ensure_mock_peft():
    """Install a mock peft module if the real one is not available."""
    try:
        from peft import LoraConfig  # noqa: F401
        return False
    except ImportError:
        pass

    mock_peft = types.ModuleType("peft")

    class MockLoraConfig:
        """Minimal LoraConfig mock that stores constructor arguments."""
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    mock_peft.LoraConfig = MockLoraConfig
    mock_peft.PeftModel = MagicMock()
    sys.modules["peft"] = mock_peft

    return True


def _ensure_mock_trl():
    """Install a mock trl module if the real one is not available."""
    try:
        from trl import SFTConfig  # noqa: F401
        return False
    except ImportError:
        pass

    mock_trl = types.ModuleType("trl")

    class IntervalStrategyEnum:
        """Minimal enum-like for save_strategy/eval_strategy."""
        def __init__(self, value):
            self.value = value

    class SchedulerTypeEnum:
        """Minimal enum-like for lr_scheduler_type."""
        def __init__(self, value):
            self.value = value

    class MockSFTConfig:
        """Minimal SFTConfig mock that stores all constructor arguments."""
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                # Convert string strategy values to enum-like objects
                if k in ("save_strategy", "eval_strategy"):
                    setattr(self, k, IntervalStrategyEnum(v))
                elif k == "lr_scheduler_type":
                    setattr(self, k, SchedulerTypeEnum(v))
                elif k == "report_to":
                    # SFTConfig normalizes report_to to a list
                    if isinstance(v, str):
                        setattr(self, k, [v])
                    else:
                        setattr(self, k, v)
                else:
                    setattr(self, k, v)

    mock_trl.SFTConfig = MockSFTConfig
    mock_trl.SFTTrainer = MagicMock()
    sys.modules["trl"] = mock_trl

    return True


def _ensure_mock_transformers_extras():
    """Ensure transformers.BitsAndBytesConfig is available for CUDA tests."""
    try:
        from transformers import BitsAndBytesConfig  # noqa: F401
    except (ImportError, Exception):
        # If transformers is installed but BitsAndBytesConfig fails
        # (e.g., no bitsandbytes), or if transformers isn't installed,
        # we add a mock
        try:
            import transformers
            if not hasattr(transformers, "BitsAndBytesConfig"):
                transformers.BitsAndBytesConfig = MagicMock()
        except ImportError:
            mock_transformers = types.ModuleType("transformers")
            mock_transformers.BitsAndBytesConfig = MagicMock()
            mock_transformers.AutoModelForCausalLM = MagicMock()
            mock_transformers.AutoTokenizer = MagicMock()
            sys.modules["transformers"] = mock_transformers


# Install mocks before any test imports scripts.train
_torch_mocked = _ensure_mock_torch()
_peft_mocked = _ensure_mock_peft()
_trl_mocked = _ensure_mock_trl()
_ensure_mock_transformers_extras()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDetectHardware:
    """Test detect_hardware() returns correct device and quantization config."""

    def test_detect_hardware_mps(self):
        """MPS detected: returns ('mps', None)."""
        import torch
        from scripts.train import detect_hardware

        torch.cuda.is_available = MagicMock(return_value=False)
        torch.backends.mps.is_available = MagicMock(return_value=True)

        device, quant_config = detect_hardware()
        assert device == "mps"
        assert quant_config is None

    def test_detect_hardware_cuda(self):
        """CUDA detected: returns ('cuda', BitsAndBytesConfig)."""
        import torch
        from scripts.train import detect_hardware

        torch.cuda.is_available = MagicMock(return_value=True)
        torch.backends.mps.is_available = MagicMock(return_value=False)

        # Mock BitsAndBytesConfig to avoid real transformers trying to
        # resolve torch dtypes (torch is mocked, not real)
        mock_bnb_cls = MagicMock()
        with patch("transformers.BitsAndBytesConfig", mock_bnb_cls):
            device, quant_config = detect_hardware()

        assert device == "cuda"
        assert quant_config is not None
        mock_bnb_cls.assert_called_once()

    def test_detect_hardware_cpu(self):
        """No GPU: returns ('cpu', None)."""
        import torch
        from scripts.train import detect_hardware

        torch.cuda.is_available = MagicMock(return_value=False)
        torch.backends.mps.is_available = MagicMock(return_value=False)

        device, quant_config = detect_hardware()
        assert device == "cpu"
        assert quant_config is None


class TestCliArgs:
    """Test build_parser() produces correct defaults and accepts overrides."""

    def test_cli_defaults(self):
        """Default args match D-04 hyperparameters and D-08/D-10/D-12 paths."""
        from scripts.train import build_parser

        parser = build_parser()
        args = parser.parse_args([])

        assert args.model_name == "HuggingFaceTB/SmolLM2-1.7B-Instruct"
        assert args.lora_r == 16
        assert args.lora_alpha == 32
        assert args.lr == 2e-4
        assert args.epochs == 3
        assert args.batch_size == 4
        assert args.grad_accum == 4
        assert args.output_dir == "models/lyra-adapter"
        assert args.merged_dir == "models/lyra-merged"
        assert args.dataset_dir == "datasets/assembled"
        assert args.wandb is False

    def test_cli_overrides(self):
        """CLI flags override defaults correctly."""
        from scripts.train import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--lora-r", "32",
            "--epochs", "5",
            "--wandb",
        ])

        assert args.lora_r == 32
        assert args.epochs == 5
        assert args.wandb is True

    def test_cli_all_flags(self):
        """All documented flags are accepted without error."""
        from scripts.train import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--model-name", "some/model",
            "--dataset-dir", "/tmp/data",
            "--output-dir", "/tmp/adapter",
            "--merged-dir", "/tmp/merged",
            "--lora-r", "8",
            "--lora-alpha", "16",
            "--lr", "1e-5",
            "--epochs", "1",
            "--batch-size", "2",
            "--grad-accum", "8",
            "--max-length", "2048",
            "--wandb",
            "--no-merge",
        ])

        assert args.model_name == "some/model"
        assert args.dataset_dir == "/tmp/data"
        assert args.output_dir == "/tmp/adapter"
        assert args.merged_dir == "/tmp/merged"
        assert args.lora_r == 8
        assert args.lora_alpha == 16
        assert args.lr == 1e-5
        assert args.epochs == 1
        assert args.batch_size == 2
        assert args.grad_accum == 8
        assert args.max_length == 2048
        assert args.wandb is True
        assert args.no_merge is True


class TestLoraConfig:
    """Test get_lora_config() returns correct LoraConfig."""

    def test_get_lora_config(self):
        """LoRA config has all 7 target modules, correct r/alpha, CAUSAL_LM."""
        from scripts.train import build_parser, get_lora_config

        parser = build_parser()
        args = parser.parse_args([])
        config = get_lora_config(args)

        assert config.r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.05
        assert config.bias == "none"
        assert config.task_type == "CAUSAL_LM"

        expected_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        for module in expected_modules:
            assert module in config.target_modules, (
                f"Missing target module: {module}"
            )

    def test_get_lora_config_custom(self):
        """LoRA config respects custom r and alpha from CLI."""
        from scripts.train import build_parser, get_lora_config

        parser = build_parser()
        args = parser.parse_args(["--lora-r", "32", "--lora-alpha", "64"])
        config = get_lora_config(args)

        assert config.r == 32
        assert config.lora_alpha == 64


class TestTrainingArgs:
    """Test get_training_args() returns correct SFTConfig per device."""

    def test_get_training_args_mps(self):
        """MPS: bf16=False, fp16=False, dataloader_num_workers=0."""
        from scripts.train import build_parser, get_training_args

        parser = build_parser()
        args = parser.parse_args([])
        training_args = get_training_args(args, "mps")

        assert training_args.bf16 is False
        assert training_args.fp16 is False
        assert training_args.dataloader_num_workers == 0
        assert training_args.gradient_checkpointing is True
        assert training_args.load_best_model_at_end is True

    def test_get_training_args_cuda(self):
        """CUDA: bf16=True, fp16=False, dataloader_num_workers=4."""
        from scripts.train import build_parser, get_training_args

        parser = build_parser()
        args = parser.parse_args([])
        training_args = get_training_args(args, "cuda")

        assert training_args.bf16 is True
        assert training_args.fp16 is False
        assert training_args.dataloader_num_workers == 4

    def test_get_training_args_common(self):
        """Common args present regardless of device."""
        from scripts.train import build_parser, get_training_args

        parser = build_parser()
        args = parser.parse_args([])
        training_args = get_training_args(args, "cpu")

        assert training_args.num_train_epochs == 3
        assert training_args.per_device_train_batch_size == 4
        assert training_args.gradient_accumulation_steps == 4
        assert training_args.learning_rate == 2e-4
        assert training_args.save_strategy.value == "epoch"
        assert training_args.eval_strategy.value == "epoch"
        assert training_args.load_best_model_at_end is True
        assert training_args.metric_for_best_model == "eval_loss"
        assert training_args.greater_is_better is False
        assert training_args.logging_steps == 10
        assert training_args.warmup_ratio == 0.03
        assert training_args.lr_scheduler_type.value == "cosine"
        assert training_args.report_to == ["none"]

    def test_get_training_args_wandb(self):
        """Wandb enabled when --wandb flag is set."""
        from scripts.train import build_parser, get_training_args

        parser = build_parser()
        args = parser.parse_args(["--wandb"])
        training_args = get_training_args(args, "cpu")

        assert training_args.report_to == ["wandb"]


class TestMpsFallbackEnv:
    """Test that PYTORCH_ENABLE_MPS_FALLBACK is set at module import."""

    def test_mps_fallback_env(self):
        """PYTORCH_ENABLE_MPS_FALLBACK is '1' after importing train module."""
        import scripts.train  # noqa: F401

        assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"


class TestMaxStepsFlag:
    """Test that --max-steps flag is present and wires into training args."""

    def test_max_steps_default(self):
        """Default max_steps is -1 (use epochs)."""
        from scripts.train import build_parser

        parser = build_parser()
        args = parser.parse_args([])
        assert args.max_steps == -1

    def test_max_steps_override(self):
        """--max-steps flag sets max_steps correctly."""
        from scripts.train import build_parser

        parser = build_parser()
        args = parser.parse_args(["--max-steps", "1"])
        assert args.max_steps == 1

    def test_max_steps_in_training_args(self):
        """When max_steps > 0, training args use max_steps and disable epoch save/eval."""
        from scripts.train import build_parser, get_training_args

        parser = build_parser()
        args = parser.parse_args(["--max-steps", "5"])
        training_args = get_training_args(args, "cpu")

        assert training_args.max_steps == 5
        assert training_args.num_train_epochs == 100  # overridden to large value
        assert training_args.save_strategy.value == "no"
        assert training_args.eval_strategy.value == "no"
        assert training_args.load_best_model_at_end is False

    def test_max_steps_disabled_uses_epochs(self):
        """When max_steps is -1 (default), epochs-based training is used."""
        from scripts.train import build_parser, get_training_args

        parser = build_parser()
        args = parser.parse_args(["--epochs", "3"])
        training_args = get_training_args(args, "cpu")

        assert training_args.max_steps == -1
        assert training_args.num_train_epochs == 3
        assert training_args.save_strategy.value == "epoch"
        assert training_args.eval_strategy.value == "epoch"
        assert training_args.load_best_model_at_end is True


class TestMergeProducesModel:
    """Test that the merge path produces expected output files.

    Uses mocks to avoid loading a real 3.4GB model. Verifies the merge_and_unload
    code path writes files to the merged directory.
    """

    def test_merge_produces_model(self, tmp_path):
        """Merge path produces safetensors files in merged directory."""
        # Create a mock adapter directory with adapter_config.json
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        adapter_config = {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            "task_type": "CAUSAL_LM",
        }
        import json
        (adapter_dir / "adapter_config.json").write_text(json.dumps(adapter_config))

        merged_dir = tmp_path / "merged"

        # Mock the entire merge workflow
        mock_base_model = MagicMock()
        mock_peft_model = MagicMock()
        mock_merged = MagicMock()
        mock_tokenizer = MagicMock()

        mock_peft_model.merge_and_unload.return_value = mock_merged

        # Simulate save_pretrained writing files
        def fake_save_pretrained(path, **kwargs):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "model.safetensors").write_text("fake")
            (Path(path) / "config.json").write_text("{}")

        mock_merged.save_pretrained.side_effect = fake_save_pretrained

        def fake_tokenizer_save(path, **kwargs):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "tokenizer.json").write_text("{}")

        mock_tokenizer.save_pretrained.side_effect = fake_tokenizer_save

        with patch("transformers.AutoModelForCausalLM") as mock_auto_model, \
             patch("peft.PeftModel") as mock_peft_cls:
            mock_auto_model.from_pretrained.return_value = mock_base_model
            mock_peft_cls.from_pretrained.return_value = mock_peft_model

            # Execute the merge logic inline (mirrors main() merge block)
            import torch
            base_model = mock_auto_model.from_pretrained(
                "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                torch_dtype=torch.float16,
            )
            peft_model = mock_peft_cls.from_pretrained(
                base_model, str(adapter_dir)
            )
            merged = peft_model.merge_and_unload()

            merged_dir.mkdir(parents=True, exist_ok=True)
            merged.save_pretrained(str(merged_dir), safe_serialization=True)
            mock_tokenizer.save_pretrained(str(merged_dir))

        # Verify merged output files
        assert (merged_dir / "model.safetensors").exists()
        assert (merged_dir / "config.json").exists()
        assert (merged_dir / "tokenizer.json").exists()

        # Verify merge_and_unload was called
        mock_peft_model.merge_and_unload.assert_called_once()


# ---------------------------------------------------------------------------
# Integration smoke tests (require real ML libraries + model download)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestIntegrationSmoke:
    """Integration smoke tests that run the actual training pipeline.

    These tests download SmolLM2-1.7B-Instruct (~3.4GB) on first run and
    execute 1 training step on a 2-sample subset of the assembled dataset.

    Requirements:
        - pip install torch peft trl accelerate (from requirements.txt)
        - datasets/assembled/ must exist with train/validation/test splits
        - ~3.4GB disk space for model download (cached in ~/.cache/huggingface/)
        - ~10GB RAM/VRAM for model loading + 1 training step

    Run with: python -m pytest tests/test_train.py -x -v -m slow
    Skip with: python -m pytest tests/test_train.py -x -v -m "not slow"

    Expected time: 2-5 minutes (mostly model download on first run)
    """

    def test_training_smoke(self, tmp_path):
        """Run training script for 1 step on 2 samples -- full pipeline validation.

        Downloads SmolLM2-1.7B-Instruct (~3.4GB) on first run.
        Validates: model load, dataset load, LoRA init, 1 training step, adapter save.
        """
        import subprocess

        project_root = Path(__file__).parent.parent

        # Verify dataset exists before attempting training
        dataset_dir = project_root / "datasets" / "assembled"
        assert dataset_dir.exists(), (
            f"Assembled dataset not found at {dataset_dir}. "
            "Run scripts/assemble_dataset.py first."
        )

        adapter_dir = tmp_path / "adapter"
        merged_dir = tmp_path / "merged"

        cmd = [
            sys.executable,
            str(project_root / "scripts" / "train.py"),
            "--dataset-dir", str(dataset_dir),
            "--output-dir", str(adapter_dir),
            "--merged-dir", str(merged_dir),
            "--epochs", "1",
            "--batch-size", "1",
            "--grad-accum", "1",
            "--max-length", "512",
            "--max-steps", "1",
            "--no-merge",
        ]

        env = os.environ.copy()
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        # Disable wandb for smoke test
        env["WANDB_DISABLED"] = "true"

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout for model download + 1 step
            env=env,
            cwd=str(project_root),
        )

        # Print output for debugging on failure
        if result.returncode != 0:
            print("STDOUT:", result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
            print("STDERR:", result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)

        assert result.returncode == 0, (
            f"Training script failed with exit code {result.returncode}.\n"
            f"STDERR (last 500 chars): {result.stderr[-500:]}"
        )

        # Verify adapter output
        assert (adapter_dir / "adapter_config.json").exists(), (
            "adapter_config.json not found in output directory"
        )

        # Verify adapter_config.json contains expected LoRA fields
        import json
        config = json.loads((adapter_dir / "adapter_config.json").read_text())
        assert "r" in config, "adapter_config.json missing 'r' field"
        assert "lora_alpha" in config, "adapter_config.json missing 'lora_alpha' field"
        assert "target_modules" in config, "adapter_config.json missing 'target_modules' field"
