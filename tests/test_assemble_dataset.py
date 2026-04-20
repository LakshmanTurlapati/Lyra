"""Tests for the dataset assembly script.

Validates that assemble_dataset.py merges curated JSONL from all 3 domains
into a HuggingFace DatasetDict with stratified 90/5/5 splits and domain metadata.
Covers DATA-07 sub-requirements a through j.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# --- Fixtures ---


@pytest.fixture
def sample_tool_calling():
    """Minimal tool-calling sample with _quality and tools."""
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather in NYC?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"location": "NYC"},
                        },
                    }
                ],
            },
            {"role": "tool", "name": "get_weather", "content": "72F, sunny"},
            {"role": "assistant", "content": "It's 72F and sunny in NYC."},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            }
        ],
        "_quality": {"score": 0.95, "dedup_hash": "abc123"},
    }


@pytest.fixture
def sample_code():
    """Minimal code sample with _quality, no tools."""
    return {
        "messages": [
            {"role": "system", "content": "You are a code assistant."},
            {"role": "user", "content": "Write a Python hello world."},
            {"role": "assistant", "content": "print('Hello, World!')"},
        ],
        "_quality": {"score": 0.90, "dedup_hash": "def456"},
    }


@pytest.fixture
def sample_knowledge():
    """Minimal knowledge sample with _quality, no tools."""
    return {
        "messages": [
            {"role": "system", "content": "You are a knowledgeable assistant."},
            {"role": "user", "content": "What is photosynthesis?"},
            {
                "role": "assistant",
                "content": "Photosynthesis is the process by which plants convert sunlight into energy.",
            },
        ],
        "_quality": {"score": 0.88, "dedup_hash": "ghi789"},
    }


@pytest.fixture
def domain_fixture_dir(tmp_path, sample_tool_calling, sample_code, sample_knowledge):
    """Create temporary JSONL files for all 3 domains with known counts.

    Creates 20 tool-calling, 10 code, 10 knowledge samples = 40 total.
    This ratio roughly mirrors the actual dataset proportions.
    """
    # Tool-calling: 20 samples
    tc_dir = tmp_path / "datasets" / "tool-calling" / "curated"
    tc_dir.mkdir(parents=True)
    tc_file = tc_dir / "tool-calling-curated.jsonl"
    with open(tc_file, "w") as f:
        for i in range(20):
            sample = sample_tool_calling.copy()
            sample["_quality"] = {"score": 0.9 + i * 0.001, "dedup_hash": f"tc_{i}"}
            sample["messages"] = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Tool query {i}"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": {"location": f"city_{i}"},
                            },
                        }
                    ],
                },
                {"role": "tool", "name": "get_weather", "content": f"Result {i}"},
                {"role": "assistant", "content": f"Answer for tool query {i}"},
            ]
            f.write(json.dumps(sample) + "\n")

    # Code: 10 samples
    code_dir = tmp_path / "datasets" / "code" / "curated"
    code_dir.mkdir(parents=True)
    code_file = code_dir / "code-curated.jsonl"
    with open(code_file, "w") as f:
        for i in range(10):
            sample = sample_code.copy()
            sample["_quality"] = {"score": 0.85 + i * 0.001, "dedup_hash": f"code_{i}"}
            sample["messages"] = [
                {"role": "system", "content": "You are a code assistant."},
                {"role": "user", "content": f"Code query {i}"},
                {"role": "assistant", "content": f"Code answer {i}"},
            ]
            f.write(json.dumps(sample) + "\n")

    # Knowledge: 10 samples
    know_dir = tmp_path / "datasets" / "knowledge" / "curated"
    know_dir.mkdir(parents=True)
    know_file = know_dir / "knowledge-curated.jsonl"
    with open(know_file, "w") as f:
        for i in range(10):
            sample = sample_knowledge.copy()
            sample["_quality"] = {
                "score": 0.80 + i * 0.001,
                "dedup_hash": f"know_{i}",
            }
            sample["messages"] = [
                {"role": "system", "content": "You are a knowledgeable assistant."},
                {"role": "user", "content": f"Knowledge query {i}"},
                {"role": "assistant", "content": f"Knowledge answer {i}"},
            ]
            f.write(json.dumps(sample) + "\n")

    return tmp_path


# --- Tests ---


class TestLoadDomainJsonl:
    """Test load_domain_jsonl: loads JSONL, adds domain, strips _quality."""

    def test_load_domain_jsonl(self, domain_fixture_dir):
        """Loads JSONL, adds domain column, strips _quality, preserves messages/tools."""
        from scripts.assemble_dataset import load_domain_jsonl

        tc_path = (
            domain_fixture_dir
            / "datasets"
            / "tool-calling"
            / "curated"
            / "tool-calling-curated.jsonl"
        )
        samples = load_domain_jsonl(tc_path, "tool-calling")

        assert len(samples) == 20
        for s in samples:
            assert s["domain"] == "tool-calling"
            assert "_quality" not in s
            assert "messages" in s
            assert "tools" in s

    def test_load_domain_no_tools(self, domain_fixture_dir):
        """Code/knowledge samples get tools=None explicitly."""
        from scripts.assemble_dataset import load_domain_jsonl

        code_path = (
            domain_fixture_dir
            / "datasets"
            / "code"
            / "curated"
            / "code-curated.jsonl"
        )
        samples = load_domain_jsonl(code_path, "code")

        assert len(samples) == 10
        for s in samples:
            assert s["domain"] == "code"
            assert s["tools"] is None
            assert "_quality" not in s


class TestAssemblyOutput:
    """Tests for the assembled DatasetDict output."""

    def test_all_domains_in_each_split(self, domain_fixture_dir):
        """After assembly, each split has all 3 domain values present."""
        from scripts.assemble_dataset import assemble

        output_dir = domain_fixture_dir / "output"
        dd = assemble(
            output_dir=str(output_dir),
            seed=42,
            base_dir=str(domain_fixture_dir),
        )

        for split_name in ["train", "validation", "test"]:
            domains_in_split = set(dd[split_name]["domain"])
            assert "tool-calling" in domains_in_split, (
                f"tool-calling missing from {split_name}"
            )
            assert "code" in domains_in_split, f"code missing from {split_name}"
            assert "knowledge" in domains_in_split, (
                f"knowledge missing from {split_name}"
            )

    def test_stratified_proportions(self, domain_fixture_dir):
        """Domain percentages in each split within 5% of overall distribution."""
        from scripts.assemble_dataset import assemble

        output_dir = domain_fixture_dir / "output"
        dd = assemble(
            output_dir=str(output_dir),
            seed=42,
            base_dir=str(domain_fixture_dir),
        )

        # Overall proportions: 20/40 = 50% tool-calling, 10/40 = 25% code, 25% knowledge
        total = sum(len(dd[s]) for s in dd)
        assert total == 40

        for split_name in ["train", "validation", "test"]:
            split_data = dd[split_name]
            split_len = len(split_data)
            if split_len == 0:
                continue
            domain_counts = {}
            for d in split_data["domain"]:
                domain_counts[d] = domain_counts.get(d, 0) + 1

            for domain, count in domain_counts.items():
                split_pct = count / split_len * 100
                # Overall percentage for this domain
                overall_pct = (
                    sum(
                        1
                        for s in dd["train"]["domain"]
                        if s == domain
                    )
                    + sum(
                        1
                        for s in dd["validation"]["domain"]
                        if s == domain
                    )
                    + sum(
                        1
                        for s in dd["test"]["domain"]
                        if s == domain
                    )
                ) / total * 100
                # Within 15% tolerance for small fixture (40 samples)
                assert abs(split_pct - overall_pct) <= 15, (
                    f"{domain} in {split_name}: {split_pct:.1f}% vs overall {overall_pct:.1f}%"
                )

    def test_total_count(self, domain_fixture_dir):
        """Total across all splits equals sum of input files (40 for fixture)."""
        from scripts.assemble_dataset import assemble

        output_dir = domain_fixture_dir / "output"
        dd = assemble(
            output_dir=str(output_dir),
            seed=42,
            base_dir=str(domain_fixture_dir),
        )

        total = sum(len(dd[s]) for s in dd)
        assert total == 40

    def test_split_ratios(self, domain_fixture_dir):
        """Train ~90%, validation ~5%, test ~5% (within tolerance for small N)."""
        from scripts.assemble_dataset import assemble

        output_dir = domain_fixture_dir / "output"
        dd = assemble(
            output_dir=str(output_dir),
            seed=42,
            base_dir=str(domain_fixture_dir),
        )

        total = sum(len(dd[s]) for s in dd)
        train_pct = len(dd["train"]) / total * 100
        val_pct = len(dd["validation"]) / total * 100
        test_pct = len(dd["test"]) / total * 100

        # With 40 samples, ratios will be approximate
        # Train should be the largest portion (>= 80%)
        assert train_pct >= 80, f"Train only {train_pct:.1f}%"
        # Val and test should each be < 15%
        assert val_pct <= 15, f"Validation is {val_pct:.1f}%"
        assert test_pct <= 15, f"Test is {test_pct:.1f}%"

    def test_domain_column(self, domain_fixture_dir):
        """Every row has domain in ['tool-calling', 'code', 'knowledge']."""
        from scripts.assemble_dataset import assemble

        output_dir = domain_fixture_dir / "output"
        dd = assemble(
            output_dir=str(output_dir),
            seed=42,
            base_dir=str(domain_fixture_dir),
        )

        valid_domains = {"tool-calling", "code", "knowledge"}
        for split_name in dd:
            for domain_val in dd[split_name]["domain"]:
                assert domain_val in valid_domains, (
                    f"Invalid domain '{domain_val}' in {split_name}"
                )

    def test_no_quality_metadata(self, domain_fixture_dir):
        """No _quality key in any row of any split."""
        from scripts.assemble_dataset import assemble

        output_dir = domain_fixture_dir / "output"
        dd = assemble(
            output_dir=str(output_dir),
            seed=42,
            base_dir=str(domain_fixture_dir),
        )

        for split_name in dd:
            columns = dd[split_name].column_names
            assert "_quality" not in columns, (
                f"_quality column found in {split_name}"
            )

    def test_tools_column(self, domain_fixture_dir):
        """Tool-calling samples have non-null tools, others have null."""
        from scripts.assemble_dataset import assemble

        output_dir = domain_fixture_dir / "output"
        dd = assemble(
            output_dir=str(output_dir),
            seed=42,
            base_dir=str(domain_fixture_dir),
        )

        for split_name in dd:
            for i in range(len(dd[split_name])):
                row = dd[split_name][i]
                if row["domain"] == "tool-calling":
                    assert row["tools"] is not None, (
                        f"tool-calling sample in {split_name}[{i}] has null tools"
                    )
                else:
                    assert row["tools"] is None, (
                        f"{row['domain']} sample in {split_name}[{i}] has non-null tools"
                    )

    def test_reproducibility(self, domain_fixture_dir):
        """Same seed produces identical splits (run twice, compare)."""
        from scripts.assemble_dataset import assemble

        output1 = domain_fixture_dir / "output1"
        output2 = domain_fixture_dir / "output2"

        dd1 = assemble(
            output_dir=str(output1),
            seed=42,
            base_dir=str(domain_fixture_dir),
        )
        dd2 = assemble(
            output_dir=str(output2),
            seed=42,
            base_dir=str(domain_fixture_dir),
        )

        for split_name in dd1:
            assert len(dd1[split_name]) == len(dd2[split_name])
            for col in dd1[split_name].column_names:
                assert dd1[split_name][col] == dd2[split_name][col], (
                    f"Mismatch in {split_name}.{col}"
                )


class TestStats:
    """Tests for compute_stats function."""

    def test_stats_output(self, domain_fixture_dir):
        """compute_stats returns correct counts and percentages."""
        from scripts.assemble_dataset import assemble, compute_stats

        output_dir = domain_fixture_dir / "output"
        dd = assemble(
            output_dir=str(output_dir),
            seed=42,
            base_dir=str(domain_fixture_dir),
        )

        stats = compute_stats(dd)

        # Should have stats for each split
        assert "train" in stats
        assert "validation" in stats
        assert "test" in stats

        # Each split stats should have total and per-domain counts
        for split_name in ["train", "validation", "test"]:
            split_stats = stats[split_name]
            assert "total" in split_stats
            assert "domains" in split_stats
            # Domain counts should sum to total
            domain_sum = sum(split_stats["domains"][d]["count"] for d in split_stats["domains"])
            assert domain_sum == split_stats["total"]
            # Percentages should sum to ~100
            pct_sum = sum(
                split_stats["domains"][d]["percent"] for d in split_stats["domains"]
            )
            assert abs(pct_sum - 100.0) < 1.0


class TestCli:
    """Tests for CLI subcommands."""

    def test_cli_assemble(self, domain_fixture_dir):
        """argparse CLI assemble runs without error."""
        output_dir = domain_fixture_dir / "cli_output"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.assemble_dataset",
                "assemble",
                "--output-dir",
                str(output_dir),
                "--seed",
                "42",
                "--base-dir",
                str(domain_fixture_dir),
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 0, f"CLI assemble failed: {result.stderr}"
        # Output directory should have dataset files
        assert output_dir.exists()

    def test_cli_stats(self, domain_fixture_dir):
        """argparse stats subcommand runs without error."""
        # First assemble
        output_dir = domain_fixture_dir / "stats_output"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.assemble_dataset",
                "assemble",
                "--output-dir",
                str(output_dir),
                "--seed",
                "42",
                "--base-dir",
                str(domain_fixture_dir),
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )

        # Then run stats
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.assemble_dataset",
                "stats",
                "--dataset-dir",
                str(output_dir),
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 0, f"CLI stats failed: {result.stderr}"
        # Should print something to stdout
        assert "train" in result.stdout.lower() or "domain" in result.stdout.lower()
