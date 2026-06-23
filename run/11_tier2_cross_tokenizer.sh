#!/usr/bin/env bash
# Tier 2: Cross-tokenizer transfer (Pythia-160M -> TinyLlama-1.1B).
# Uses word-boundary canonicalization.
# 3 conditions x 3 seeds = 9 runs.
# Prerequisite: results/source_memory_wb/memory.pt (run/02_source_memories.sh).

set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f results/source_memory_wb/memory.pt ]; then
    echo "ERROR: results/source_memory_wb/memory.pt not found."
    echo "Run: bash run/02_source_memories.sh   (word-boundary block)"
    exit 1
fi

uv run python scripts/run_tier2.py \
    --source-memory results/source_memory_wb/memory.pt \
    --memory-config results/source_memory_wb/memory_config.json
