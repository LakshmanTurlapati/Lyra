#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# scripts/convert_gguf.sh -- reproducible GGUF conversion pipeline (Phase 10 D-04).
#
# Usage:
#   scripts/convert_gguf.sh <model_dir> <output_prefix>
#
# Requires: llama.cpp CLI tools on $PATH (brew install llama.cpp) + Python 3.10+
# Default model_dir: models/lyra-merged; default output_prefix: lyra-v1.0
#
# Threat mitigations (mirror scripts/eval_runner.py T-03-05/T-03-07):
#   T-10-01: positional args validated before substitution
#   T-10-02: post-conversion verify_gguf.py asserts chat_template present
set -euo pipefail

MODEL_DIR="${1:-models/lyra-merged}"
OUTPUT_PREFIX="${2:-lyra-v1.0}"
OUTPUT_DIR="build/gguf"

_valid_path='^[a-zA-Z0-9._/~\-]+$'
if ! [[ "$MODEL_DIR" =~ $_valid_path ]]; then
    echo "Error: invalid model dir: $MODEL_DIR" >&2
    exit 1
fi
if ! [[ "$OUTPUT_PREFIX" =~ $_valid_path ]]; then
    echo "Error: invalid output prefix: $OUTPUT_PREFIX" >&2
    exit 1
fi
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: model dir not found: $MODEL_DIR" >&2
    exit 1
fi

# Precondition: chat_template must be present in tokenizer_config.json (Pitfall 1 mitigation)
python3 -c "import json,sys; d=json.load(open('$MODEL_DIR/tokenizer_config.json')); sys.exit(0 if d.get('chat_template') else 1)" \
    || { echo "Error: $MODEL_DIR/tokenizer_config.json has no chat_template field" >&2; exit 1; }

mkdir -p "$OUTPUT_DIR"

# Locate convert_hf_to_gguf.py (brew default: /opt/homebrew/share/llama.cpp; override via LLAMA_CPP_DIR env)
CONVERT_SCRIPT="${LLAMA_CPP_DIR:-/opt/homebrew/share/llama.cpp}/convert_hf_to_gguf.py"
if [ ! -f "$CONVERT_SCRIPT" ]; then
    echo "Error: convert_hf_to_gguf.py not found at $CONVERT_SCRIPT. Install llama.cpp or set LLAMA_CPP_DIR." >&2
    exit 1
fi

# Step 1: HF -> f16 GGUF (chat_template auto-embedded from tokenizer_config.json)
python3 "$CONVERT_SCRIPT" "$MODEL_DIR" \
    --outfile "$OUTPUT_DIR/${OUTPUT_PREFIX}-f16.gguf" \
    --outtype f16

# Step 2: f16 -> Q4_K_M
llama-quantize "$OUTPUT_DIR/${OUTPUT_PREFIX}-f16.gguf" \
    "$OUTPUT_DIR/${OUTPUT_PREFIX}-q4_k_m.gguf" Q4_K_M

# Step 3: f16 -> Q8_0
llama-quantize "$OUTPUT_DIR/${OUTPUT_PREFIX}-f16.gguf" \
    "$OUTPUT_DIR/${OUTPUT_PREFIX}-q8_0.gguf" Q8_0

# Step 4: Verify chat_template embedded in both outputs (D-07 non-negotiable)
python3 -m scripts.verify_gguf "$OUTPUT_DIR/${OUTPUT_PREFIX}-q4_k_m.gguf"
python3 -m scripts.verify_gguf "$OUTPUT_DIR/${OUTPUT_PREFIX}-q8_0.gguf"

echo "GGUF conversion complete:"
ls -lh "$OUTPUT_DIR/"*.gguf
