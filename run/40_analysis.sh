#!/usr/bin/env bash
# Post-experiment analysis: aggregate results, generate figures, analyze gates/collisions.
# Requires that one or more experiments from run/10*..30* have completed.

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p results/figures

# Aggregate tier 1-4 into a single summary JSON
uv run python scripts/analyze_results.py \
    --results-root results \
    --output results/figures/summary.json || true

# Generate paper figures (scaling, bar charts, CKA heatmap)
uv run python scripts/generate_figures.py \
    --input-dir results \
    --output-dir results/figures || true

# Gate statistics for any completed adaptor
for d in results/*matched_to_*/transferred_seed*; do
    [ -d "$d" ] || continue
    uv run python scripts/analyze_gates.py \
        --adaptor-dir "$d" \
        --output "${d}/gate_stats.json" 2>/dev/null || true
done

# Collision analysis (memory table hash occupancy)
for m in results/*source*/memory.pt; do
    [ -f "$m" ] || continue
    uv run python scripts/analyze_collisions.py \
        --memory "$m" \
        --output "$(dirname "$m")/collision_stats.json" 2>/dev/null || true
done

echo ""
echo "==== Analysis DONE ===="
echo "Figures:   results/figures/"
echo "Summary:   results/figures/summary.json"
