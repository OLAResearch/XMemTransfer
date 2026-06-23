#!/usr/bin/env bash
# Pilot smoke test: 2M tokens, 1 seed, 4 conditions on Pythia-160M -> Pythia-410M.
# Validates: frozen-backbone gradients, random-memory control, checkpoint round-trip.
# Expected runtime: ~30 min on a single consumer GPU (RTX 4090 / A6000).
#
# Run from the repository root:
#   bash run/01_pilot.sh

set -euo pipefail
cd "$(dirname "$0")/.."

echo "==== Phase 1: Source memory training (pilot, Pythia-160M, 2M tokens) ===="
uv run python scripts/train_source_memory.py \
    --config configs/pilot.yaml \
    --output-dir results/pilot_source

echo ""
echo "==== Phase 2: Pilot validation (4 conditions, 1 seed, 2M tokens) ===="
uv run python scripts/run_pilot.py \
    --source-memory results/pilot_source/memory.pt \
    --memory-config results/pilot_source/memory_config.json \
    --max-tokens 2000000 \
    --output-dir results/pilot

echo ""
echo "==== Pilot DONE ===="
echo "Inspect: results/pilot/*/results.json"
