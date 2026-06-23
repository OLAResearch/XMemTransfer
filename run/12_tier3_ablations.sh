#!/usr/bin/env bash
# Tier 3: Ablation studies on the source of transfer benefit.
# Conditions: transferred, permuted_keys, no_gate, train_from_scratch, ffn_only,
#             affine_stitch (6 conditions x 3 seeds = 18 runs).
# Prerequisite: results/source_memory/memory.pt.
#
# Optional: pass --array-index 0..17 to run one condition/seed pair at a time
# (useful on a job scheduler or when parallelizing across GPUs).

set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f results/source_memory/memory.pt ]; then
    echo "ERROR: results/source_memory/memory.pt not found."
    exit 1
fi

ARRAY_ARGS=()
if [ "${1:-}" = "--array-index" ] && [ -n "${2:-}" ]; then
    ARRAY_ARGS=(--array-index "$2")
fi

uv run python scripts/run_tier3.py \
    --source-memory results/source_memory/memory.pt \
    --memory-config results/source_memory/memory_config.json \
    "${ARRAY_ARGS[@]}"
