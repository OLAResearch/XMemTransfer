#!/usr/bin/env bash
# Tier 1: Same-tokenizer dimension transfer (Pythia-160M -> Pythia-410M).
# 3 conditions (baseline, transferred, random_memory) x 3 seeds = 9 runs.
# Prerequisite: results/source_memory/memory.pt (run/02_source_memories.sh).

set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f results/source_memory/memory.pt ]; then
    echo "ERROR: results/source_memory/memory.pt not found."
    echo "Run: bash run/02_source_memories.sh   (Pythia-160M block)"
    exit 1
fi

uv run python scripts/run_tier1.py \
    --source-memory results/source_memory/memory.pt \
    --memory-config results/source_memory/memory_config.json
