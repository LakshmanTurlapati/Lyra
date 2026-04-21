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
