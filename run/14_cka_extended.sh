#!/usr/bin/env bash
# Extended CKA: 5×5 representational similarity matrix at relative depth L/3.
# Models: Pythia-{160M,410M}, TinyLlama-1.1B, QWen3.5-{0.8B,4B}.
# Expected runtime: ~1 h on a single GPU.

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p results/cka_extended

uv run python scripts/run_cka_extended.py \
    --output-dir results/cka_extended \
    --n-samples 2560

echo ""
echo "==== CKA DONE ===="
echo "Matrix saved: results/cka_extended/cka_results.json"
cat results/cka_extended/cka_results.json 2>/dev/null || true
