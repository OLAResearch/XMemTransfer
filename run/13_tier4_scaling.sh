#!/usr/bin/env bash
# Tier 4: Scaling and CKA analysis across target sizes.
# Scales target from QWen-2B to QWen-9B using the QWen-0.8B source memory.
# Prerequisite: results/qwen35_source_memory/memory.pt.

set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f results/qwen35_source_memory/memory.pt ]; then
    echo "ERROR: results/qwen35_source_memory/memory.pt not found."
    echo "Run: bash run/02_source_memories.sh   (QWen-0.8B block)"
    exit 1
fi

uv run python scripts/run_tier4.py \
    --source-memory results/qwen35_source_memory/memory.pt \
    --memory-config results/qwen35_source_memory/memory_config.json
